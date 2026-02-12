"""GRPO trainer: group sampling, advantage normalization, PPO-style updates."""

import importlib
import logging
import math
import random
from pathlib import Path
from typing import Optional, List, Tuple, Any, cast, Protocol, TypeVar

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler

try:
    _liger = importlib.import_module("liger_kernel.transformers")
except Exception:
    from transformers import AutoModelForCausalLM
    _USING_LIGER = False
else:
    AutoModelForCausalLM = _liger.AutoLigerKernelForCausalLM
    _USING_LIGER = True

from transformers import AutoTokenizer

from rlfusion.inference.hf_utils import sample_completions_batch_hf
from rlfusion.trainers.data import Trajectory
from rlfusion.trainers.types import AttentionMask, LogProbs, TokenIds
from rlfusion.trainers.utils import (
    get_device,
    set_seed,
    configure_torch_backends,
    truncate_text,
    format_prompt,
    resolve_attention_implementation,
    get_tokenizer_compat_kwargs,
    build_full_attention_mask,
)
from rlfusion.trainers.common import configure_logging, is_main_process, unwrap_model_for_saving
from rlfusion.envs import EnvBase

logger = logging.getLogger(__name__)
if _USING_LIGER:
    logger.info("Using liger kernel")
else:
    logger.info("Liger kernel not installed, defaulting to HF")

_SampleT = TypeVar("_SampleT", covariant=True)


class DatasetLike(Protocol[_SampleT]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> _SampleT: ...


class GRPOTrainer():
    @staticmethod
    def _validate_init_args(
        *,
        num_steps: int,
        num_epochs: int | None = None,
        saving_steps: int,
        logging_steps: int,
        eval_steps: Optional[int],
        sampling_temperature: float,
        max_new_tokens: int,
        batch_size: int,
        group_size: int,
        ppo_steps: int,
        clip_eps: float,
        max_grad_norm: Optional[float],
    ) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be >= 1.")
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError("num_epochs must be >= 1 or None.")
        if saving_steps <= 0:
            raise ValueError("saving_steps must be >= 1.")
        if logging_steps <= 0:
            raise ValueError("logging_steps must be >= 1.")
        if eval_steps is not None and eval_steps <= 0:
            raise ValueError("eval_steps must be >= 1 or None.")
        if sampling_temperature <= 0.0:
            raise ValueError("sampling_temperature must be > 0.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be >= 1.")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1.")
        if group_size <= 1:
            raise ValueError("group_size must be >= 2.")
        if ppo_steps <= 0:
            raise ValueError("ppo_steps must be >= 1.")
        if clip_eps < 0.0:
            raise ValueError("clip_eps must be >= 0.")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 or None.")

    def __init__(self,
                 model: str,
                 train_dataset: Optional[DatasetLike[Any]],
                 num_steps: int = 100,
                 num_epochs: int | None = None,
                 saving_steps: int = 10,
                 logging_steps: int = 10,
                 eval_steps: Optional[int] = None,
                 eval_dataset: Optional[DatasetLike[EnvBase]] = None,
                 enable_wandb: bool = False,
                 wandb_project: str = "grpo",
                 wandb_run_name: Optional[str] = None,
                 sampling_temperature: float = 1.0,
                 kl_penalty: float = 0.0,
                 output_dir: str = "./outputs",
                 generation_args: Optional[dict[str, Any]] = None,
                 optimizer: type[Optimizer] = AdamW,
                 optimizer_args: Optional[dict[str, Any]] = None,
                 lr_scheduler: Optional[type[LRScheduler] | LRScheduler] = None,
                 lr_scheduler_args: Optional[dict[str, Any]] = None,
                 seed: int = 42,
                 max_new_tokens: int = 1024,
                 batch_size: int = 1,
                 group_size: int = 4,
                 ppo_steps: int = 2,
                 clip_eps: float = 0.2,
                 log_completions: bool = False,
                 max_log_chars: Optional[int] = 320,
                 max_error: float = 100.0,
                 invalid_penalty: float = 1.0,
                 max_grad_norm: Optional[float] = None,
                 log_level: int = logging.INFO,
                 use_accelerate: bool = False,
                 ):
        self._validate_init_args(
            num_steps=num_steps,
            num_epochs=num_epochs,
            saving_steps=saving_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            sampling_temperature=sampling_temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            group_size=group_size,
            ppo_steps=ppo_steps,
            clip_eps=clip_eps,
            max_grad_norm=max_grad_norm,
        )

        self.accelerator = None
        if use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator()

        # Set random seed for reproducibility
        seed_offset = 0 if self.accelerator is None else int(self.accelerator.process_index)
        set_seed(seed + seed_offset)
        self.seed = seed
        self._configure_logging(log_level)

        self.sampling_temperature = sampling_temperature
        self.kl_penalty = kl_penalty
        self.output_dir = Path(output_dir)
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.saving_steps = saving_steps
        self.logging_steps = logging_steps
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.generation_args = generation_args or {}
        self.group_size = group_size
        self.ppo_steps = ppo_steps
        self.clip_eps = clip_eps
        self.log_completions = log_completions
        self.max_log_chars = max_log_chars
        self.max_error = max_error
        self.invalid_penalty = invalid_penalty
        self.max_grad_norm = max_grad_norm

        self.eval_steps = eval_steps
        self.eval_dataset = eval_dataset

        device = get_device()

        if device == "cuda":
            if self.accelerator is None:
                device_map: str | dict[str, int] = "auto"
            else:
                # In distributed mode each rank owns one device.
                device_map = {"": int(self.accelerator.local_process_index)}

            # Configure torch backends for performance
            configure_torch_backends()

        else:

            device_map = device

        self.device = device
        self.device_map = device_map
        self.train_dataset = train_dataset

        attn_device_map = "auto" if device_map == "auto" else "manual"
        attn_implementation = resolve_attention_implementation(attn_device_map)
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }
        if device == "cuda" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif device == "mps":
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        self.model.config.use_cache = False 

        tokenizer_kwargs = get_tokenizer_compat_kwargs(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if kl_penalty > 0.0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

            self.ref_model.config.use_cache = False
            self.ref_model.eval()

            for param in self.ref_model.parameters():
                param.requires_grad_(False)
        else:
            self.ref_model = None 


        self._wandb = None 

        should_enable_wandb = enable_wandb and (
            self.accelerator is None or self.accelerator.is_main_process
        )
        if should_enable_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model": model,
                        "device": device,
                        "device_map": device_map,
                        "num_steps": num_steps,
                        "num_epochs": num_epochs,
                        "saving_steps": saving_steps,
                        "logging_steps": logging_steps,
                        "eval_steps": eval_steps,
                        "sampling_temperature": sampling_temperature,
                        "kl_penalty": kl_penalty,
                        "use_ref_model": kl_penalty > 0.0,
                        "use_eval_dataset": eval_dataset is not None,
                        "output_dir": output_dir,
                        "optimizer": optimizer.__name__ if hasattr(optimizer, "__name__") else optimizer.__class__.__name__,
                        "optimizer_args": optimizer_args,
                        "lr_scheduler": None if lr_scheduler is None else (lr_scheduler.__name__ if hasattr(lr_scheduler, "__name__") else lr_scheduler.__class__.__name__),
                        "lr_scheduler_args": lr_scheduler_args,
                        "attn_implementation": attn_implementation,
                        "dtype": str(self.model.dtype),
                        "seed": seed,
                        "max_new_tokens": self.max_new_tokens,
                        "batch_size": self.batch_size,
                        "group_size": self.group_size,
                        "ppo_steps": self.ppo_steps,
                        "clip_eps": self.clip_eps,
                        "max_error": self.max_error,
                        "invalid_penalty": self.invalid_penalty,
                    },
                )

                self._wandb = wandb

            except Exception as exc:
                logger.warning(f"wandb could not be enabled: {exc}")


        if optimizer_args is None:
            optimizer_args = {"lr": 1e-5}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        if lr_scheduler is None:
            self.lr_scheduler = None
        elif isinstance(lr_scheduler, LRScheduler):
            self.lr_scheduler = lr_scheduler
        else:
            if lr_scheduler_args is None:
                lr_scheduler_args = {}
            self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_args)

        if self.accelerator is not None:
            if self.lr_scheduler is None:
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._num_steps: Optional[int] = None


    def _configure_logging(self, log_level: int) -> None:
        configure_logging(logger, log_level)

    def _is_main_process(self) -> bool:
        return is_main_process(self.accelerator)

    def _unwrap_model_for_saving(self) -> torch.nn.Module:
        return unwrap_model_for_saving(self.model, self.accelerator)

    def sample_completions_batch(self, envs: List[EnvBase]) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
        model = cast(Any, self.model)
        return sample_completions_batch_hf(
            model=model,
            tokenizer=self.tokenizer,
            envs=envs,
            do_sample=True,
            sampling_temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
            generation_args=self.generation_args,
        )

    def _sample_completions_batch_with_mask(
        self, envs: List[EnvBase]
    ) -> Tuple[torch.Tensor, List[str], List[int], List[int], torch.Tensor]:
        model = cast(Any, self.model)
        return sample_completions_batch_hf(
            model=model,
            tokenizer=self.tokenizer,
            envs=envs,
            do_sample=True,
            sampling_temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
            generation_args=self.generation_args,
            return_attention_mask=True,
        )

    def _compute_reward(self, env: EnvBase, completion_text: Optional[str]) -> float:
        if completion_text is None:
            return -(self.max_error + self.invalid_penalty)
        reward_value = env.get_reward(completion_text)
        if reward_value is None:
            return -(self.max_error + self.invalid_penalty)
        return float(reward_value)


    def compute_advantage(
        self,
        trajectories: List[Trajectory],
        eps: float = 1e-8,
        group_size: Optional[int] = None,
    ) -> None:
        for traj in trajectories:
            if traj.reward is None:
                raise ValueError("Trajectory reward is required to compute advantage.")
        if not trajectories:
            return

        group = len(trajectories) if group_size is None else int(group_size)
        if group <= 0:
            raise ValueError("group_size must be >= 1.")
        if len(trajectories) % group != 0:
            raise ValueError("Number of trajectories must be divisible by group_size.")

        # GRPO advantages are normalized within each prompt group.
        for start in range(0, len(trajectories), group):
            group_trajectories = trajectories[start : start + group]
            rewards = torch.tensor(
                [traj.reward for traj in group_trajectories],
                dtype=torch.float32,
            )
            mean_reward = float(rewards.mean().item())
            std = float(rewards.std(unbiased=False).item())
            for traj in group_trajectories:
                assert traj.reward is not None
                traj.advantage = (traj.reward - mean_reward) / (std + eps)

    def _build_full_attention_mask(
        self,
        input_attention_mask: AttentionMask,
        completion_lengths: List[int],
        sequence_ids: TokenIds,
    ) -> AttentionMask:
        return build_full_attention_mask(input_attention_mask, completion_lengths, sequence_ids)

    def get_log_probs(
        self,
        sequence_ids: TokenIds,
        model: Optional[torch.nn.Module] = None,
        attention_mask: Optional[AttentionMask] = None,
    ) -> LogProbs:
        if sequence_ids.ndim == 1:
            sequence_ids = sequence_ids.unsqueeze(0)
        if model is None:
            model = self.model

        if attention_mask is None:
            # Fall back to pad-based masking only when an explicit mask is unavailable.
            if self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                attention_mask = (sequence_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(sequence_ids)
        else:
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.shape != sequence_ids.shape:
                raise ValueError("attention_mask must match sequence_ids shape.")
            attention_mask = attention_mask.to(sequence_ids.device).long()

        outputs = model(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

        logits = outputs.logits / max(self.sampling_temperature, 1e-6)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = sequence_ids[:, 1:]

        return torch.gather(logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    def grpo_loss(self,
                  old_log_probs: torch.Tensor,
                  new_log_probs: torch.Tensor,
                  ref_log_probs: torch.Tensor,
                  mask: torch.Tensor,
                  advantage: float,
                  eps: float,
                  kl_beta: float) -> torch.Tensor:
        ratio = torch.exp(new_log_probs - old_log_probs)
        advantage_tensor = torch.tensor(advantage, device=new_log_probs.device, dtype=new_log_probs.dtype)

        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage_tensor
        obj = torch.minimum(unclipped, clipped) * mask

        denom = mask.sum().clamp_min(1.0)
        kl = (new_log_probs - ref_log_probs) * mask

        # KL penalty keeps the policy close to the reference when enabled.
        return -(obj.sum() / denom) + (kl_beta * (kl.sum() / denom))

    def grpo_loss_batch(self,
                        old_log_probs: torch.Tensor,
                        new_log_probs: torch.Tensor,
                        ref_log_probs: torch.Tensor,
                        mask: torch.Tensor,
                        advantages: torch.Tensor,
                        eps: float,
                        kl_beta: float) -> torch.Tensor:
        ratio = torch.exp(new_log_probs - old_log_probs)
        advantages = advantages.view(-1, 1)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
        obj = torch.minimum(unclipped, clipped) * mask

        denom = mask.sum(dim=1).clamp_min(1.0)
        kl = (new_log_probs - ref_log_probs) * mask

        return -(obj.sum(dim=1) / denom) + (kl_beta * (kl.sum(dim=1) / denom))

    def generate_mask(self, trajectory: Trajectory, input_length: Optional[int] = None) -> torch.Tensor:
        if trajectory.prompt_len is None and input_length is None:
            raise ValueError("Trajectory prompt_len or input_length is required to build the mask.")
        if trajectory.sequence_ids is None:
            raise ValueError("Trajectory sequence_ids is required to build the mask.")

        sequence_length = int(trajectory.sequence_ids.numel())
        mask = torch.zeros((sequence_length - 1,), device=trajectory.sequence_ids.device, dtype=torch.float32)

        # Use padded input length when provided so the mask starts at the generation boundary.
        prompt_len_value = trajectory.prompt_len if input_length is None else input_length
        if prompt_len_value is None:
            raise ValueError("Prompt length must be defined to build the mask.")
        prompt_len = int(prompt_len_value)
        completion_len = trajectory.completion_len
        if completion_len is None:
            completion_len = max(sequence_length - prompt_len, 0)
        end_len = min(prompt_len + completion_len, sequence_length)

        start = max(prompt_len - 1, 0)
        end = max(end_len - 2, -1)

        if end >= start:
            mask[start:end + 1] = 1.0

        trajectory.mask = mask
        return mask

    def _log_step(self, step: int, env: EnvBase, trajectories: List[Trajectory], batch_stats: Optional[dict] = None) -> None:
        if not self._is_main_process():
            return
        if step % self.logging_steps != 0:
            return

        rewards = torch.tensor([traj.reward for traj in trajectories], dtype=torch.float32)
        advantages = torch.tensor([traj.advantage for traj in trajectories], dtype=torch.float32)
        reward_mean = float(rewards.mean().item())
        reward_std = float(rewards.std(unbiased=False).item())
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())

        if batch_stats is None:
            batch_stats = {}
        loss_mean = float(batch_stats.get("loss_mean", 0.0))
        loss_std = float(batch_stats.get("loss_std", 0.0))
        ratio_mean = float(batch_stats.get("ratio_mean", 0.0))
        ratio_min = float(batch_stats.get("ratio_min", 0.0))
        ratio_max = float(batch_stats.get("ratio_max", 0.0))
        mask_tokens_mean = float(batch_stats.get("mask_tokens_mean", 0.0))
        completion_tokens_mean = float(batch_stats.get("completion_tokens_mean", 0.0))

        logger.info(
            "step %d reward_mean=%.4f reward_std=%.4f adv_mean=%.4f adv_std=%.4f loss_mean=%.4f loss_std=%.4f",
            step,
            reward_mean,
            reward_std,
            adv_mean,
            adv_std,
            loss_mean,
            loss_std,
        )
        logger.info(
            "step %d ratio_mean=%.4f ratio_min=%.4f ratio_max=%.4f mask_tokens_mean=%.1f completion_tokens_mean=%.1f",
            step,
            ratio_mean,
            ratio_min,
            ratio_max,
            mask_tokens_mean,
            completion_tokens_mean,
        )
        if self.log_completions:
            prompt_text = format_prompt(env.prompt)
            logger.info("prompt: %s", truncate_text(prompt_text, self.max_log_chars))
            answer_text = None if env.answer is None else str(env.answer)
            logger.info("gt_answer: %s", truncate_text(answer_text, self.max_log_chars))
            for idx, traj in enumerate(trajectories):
                completion_preview = truncate_text(traj.completion_text, self.max_log_chars)
                logger.info(
                    "generated_%d reward=%.4f: %s",
                    idx,
                    traj.reward,
                    completion_preview,
                )

        if self._wandb is not None:
            self._wandb.log(
                {
                    "reward/mean": reward_mean,
                    "reward/std": reward_std,
                    "adv/mean": adv_mean,
                    "adv/std": adv_std,
                    "loss/mean": loss_mean,
                    "loss/std": loss_std,
                    "ratio/mean": ratio_mean,
                    "ratio/min": ratio_min,
                    "ratio/max": ratio_max,
                    "mask_tokens/mean": mask_tokens_mean,
                    "completion_tokens/mean": completion_tokens_mean,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                step=step,
            )

    def train(self) -> None:
        if self.train_dataset is None:
            raise ValueError("Dataset is required for training.")
        dataset_len = len(self.train_dataset)
        if dataset_len == 0:
            raise ValueError("Dataset is empty.")
        if self.eval_steps is not None:
            if self.eval_dataset is None:
                raise ValueError("eval_dataset is required when eval_steps is set.")
            if len(self.eval_dataset) == 0:
                raise ValueError("Eval dataset is empty.")

        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        if self.num_epochs is None:
            self._num_steps = self.num_steps
            steps_per_epoch = 0
            epoch_indices: list[int] = []
        else:
            steps_per_epoch = max(1, math.ceil(dataset_len / self.batch_size))
            self._num_steps = steps_per_epoch * self.num_epochs
            epoch_indices = list(range(dataset_len))

        assert self._num_steps is not None

        for step in range(self._num_steps):
            if self.num_epochs is None:
                env_batch = [
                    self.train_dataset[random.randint(0, dataset_len - 1)]
                    for _ in range(self.batch_size)
                ]
            else:
                if step % steps_per_epoch == 0:
                    random.shuffle(epoch_indices)
                start = (step % steps_per_epoch) * self.batch_size
                batch_indices = epoch_indices[start : start + self.batch_size]
                if not batch_indices:
                    raise ValueError("Epoch sampling produced an empty batch.")
                env_batch = [self.train_dataset[idx] for idx in batch_indices]

            envs = []
            for env in env_batch:
                envs.extend([env for _ in range(self.group_size)])

            sequences, texts, prompt_lens, completion_lens, input_attention_mask = (
                self._sample_completions_batch_with_mask(envs)
            )
            # prompt_lens are true prompt lengths; input_length is the padded generation boundary.

            trajectories = []
            for i, env_instance in enumerate(envs):
                trajectory = Trajectory(
                    env=env_instance,
                    sequence_ids=sequences[i],
                    completion_text=texts[i],
                    prompt_len=prompt_lens[i],
                    completion_len=completion_lens[i],
                )
                trajectory.reward = self._compute_reward(env_instance, texts[i])
                trajectories.append(trajectory)

            self.compute_advantage(trajectories, group_size=self.group_size)

            input_length = int(input_attention_mask.shape[1])
            # Build a full mask that keeps prompt padding holes but enables generated tokens.
            full_attention_mask = self._build_full_attention_mask(
                input_attention_mask, completion_lens, sequences
            )
            with torch.no_grad():
                old_log_probs = self.get_log_probs(sequences, attention_mask=full_attention_mask)
                if self.ref_model is not None:
                    ref_log_probs = self.get_log_probs(
                        sequences, model=self.ref_model, attention_mask=full_attention_mask
                    )
                else:
                    ref_log_probs = torch.zeros_like(old_log_probs)

            for i, traj in enumerate(trajectories):
                traj.old_log_probs = old_log_probs[i]
                traj.ref_log_probs = ref_log_probs[i]
                self.generate_mask(traj, input_length=input_length)

            masks = torch.stack([traj.mask for traj in trajectories]).to(sequences.device)
            advantages = torch.tensor(
                [traj.advantage for traj in trajectories],
                device=sequences.device,
                dtype=old_log_probs.dtype,
            )
            completion_lengths = torch.tensor(
                completion_lens,
                device=sequences.device,
                dtype=old_log_probs.dtype,
            )
            batch_stats = None
            for ppo_step in range(self.ppo_steps):
                self.optimizer.zero_grad(set_to_none=True)
                new_log_probs = self.get_log_probs(sequences, attention_mask=full_attention_mask)

                loss_per = self.grpo_loss_batch(
                    old_log_probs,
                    new_log_probs,
                    ref_log_probs,
                    masks,
                    advantages,
                    eps=self.clip_eps,
                    kl_beta=self.kl_penalty,
                )
                loss = loss_per.mean()
                if self.accelerator is None:
                    loss.backward()
                else:
                    self.accelerator.backward(loss)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if ppo_step == self.ppo_steps - 1:
                    ratio = torch.exp(new_log_probs.detach() - old_log_probs)
                    mask_bool = masks.bool()
                    mask_counts = masks.sum(dim=1)
                    denom = mask_counts.clamp_min(1.0)
                    ratio_mean_per = (ratio * masks).sum(dim=1) / denom

                    ratio_min_per = ratio.masked_fill(~mask_bool, float("inf")).min(dim=1).values
                    ratio_max_per = ratio.masked_fill(~mask_bool, float("-inf")).max(dim=1).values
                    empty_mask = mask_counts == 0
                    if empty_mask.any():
                        ratio_min_per = torch.where(empty_mask, torch.zeros_like(ratio_min_per), ratio_min_per)
                        ratio_max_per = torch.where(empty_mask, torch.zeros_like(ratio_max_per), ratio_max_per)

                    batch_stats = {
                        "loss_mean": float(loss_per.mean().item()),
                        "loss_std": float(loss_per.std(unbiased=False).item()),
                        "ratio_mean": float(ratio_mean_per.mean().item()),
                        "ratio_min": float(ratio_min_per.min().item()),
                        "ratio_max": float(ratio_max_per.max().item()),
                        "mask_tokens_mean": float(mask_counts.mean().item()),
                        "completion_tokens_mean": float(completion_lengths.mean().item()),
                    }

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            log_env = env_batch[0] if env_batch else envs[0]
            self._log_step(step + 1, log_env, trajectories, batch_stats)

            if self.eval_steps is not None and (step + 1) % self.eval_steps == 0:
                if self._is_main_process():
                    self.test(dataset=self.eval_dataset, step=step + 1)
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

            if (step + 1) % self.saving_steps == 0:
                step_output_dir = self.output_dir / f"step_{step + 1}"
                step_output_dir.mkdir(parents=True, exist_ok=True)
                if self._is_main_process():
                    model = cast(Any, self._unwrap_model_for_saving())
                    model.save_pretrained(step_output_dir)
                    self.tokenizer.save_pretrained(step_output_dir)
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

        # Save final model
        final_output_dir = self.output_dir / "final"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        if self._is_main_process():
            model = cast(Any, self._unwrap_model_for_saving())
            model.save_pretrained(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            logger.info("Saved final model to %s", final_output_dir)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if self._wandb is not None:
            self._wandb.finish()

    def test(
        self,
        dataset: Optional[DatasetLike[EnvBase]] = None,
        num_batches: Optional[int] = None,
        step: Optional[int] = None,
        eval_temperature: Optional[float] = None,
    ) -> dict:
        if dataset is None:
            raise ValueError("dataset is required for testing.")
        if num_batches is not None and num_batches <= 0:
            raise ValueError("num_batches must be >= 1 or None.")
        eval_dataset = dataset
        dataset_len = len(eval_dataset)
        if dataset_len == 0:
            raise ValueError("Eval dataset is empty.")

        # Save original temperature and set eval temperature
        original_temperature = self.sampling_temperature
        if eval_temperature is not None:
            self.sampling_temperature = eval_temperature

        self.model.eval()

        try:
            if num_batches is None:
                indices = list(range(dataset_len))
            else:
                indices = [random.randint(0, dataset_len - 1) for _ in range(num_batches * self.batch_size)]

            all_rewards = []
            all_completion_lengths = []
            first_env = None
            first_completions = None

            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                env_batch = [eval_dataset[i] for i in batch_indices]

                _, texts, _, completion_lens = self.sample_completions_batch(env_batch)

                rewards = [self._compute_reward(env, text) for env, text in zip(env_batch, texts)]
                all_rewards.extend(rewards)
                all_completion_lengths.extend(completion_lens)

                if first_completions is None:
                    first_env = env_batch[0]
                    first_completions = texts

            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
            reward_mean = float(rewards_tensor.mean().item())
            reward_std = float(rewards_tensor.std(unbiased=False).item())
            completion_tokens_mean = float(torch.tensor(all_completion_lengths, dtype=torch.float32).mean().item())

            logger.info(
                "eval reward_mean=%.4f reward_std=%.4f completion_tokens_mean=%.1f",
                reward_mean,
                reward_std,
                completion_tokens_mean,
            )
            if self.log_completions and first_env is not None and first_completions is not None:
                prompt_text = format_prompt(first_env.prompt)
                logger.info("prompt: %s", truncate_text(prompt_text, self.max_log_chars))
                answer_text = None if first_env.answer is None else str(first_env.answer)
                logger.info("gt_answer: %s", truncate_text(answer_text, self.max_log_chars))
                for idx, completion in enumerate(first_completions):
                    completion_preview = truncate_text(completion, self.max_log_chars)
                    logger.info("generated_%d: %s", idx, completion_preview)

            if self._wandb is not None:
                log_data = {
                    "eval/reward_mean": reward_mean,
                    "eval/reward_std": reward_std,
                    "eval/completion_tokens_mean": completion_tokens_mean,
                }
                if step is not None:
                    self._wandb.log(log_data, step=step)
                else:
                    self._wandb.log(log_data)

            return {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "completion_tokens_mean": completion_tokens_mean,
            }
        finally:
            # Restore original temperature and model state
            self.sampling_temperature = original_temperature
            self.model.train()
