import importlib
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, List, Tuple, Any, cast

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

from rlfusion.trainers.data import Trajectory
from rlfusion.trainers.utils import (
    get_device,
    set_seed,
    configure_torch_backends,
    truncate_text,
    format_prompt,
    resolve_attention_implementation,
)
from rlfusion.trainers.types import GenerateOutput
from rlfusion.envs import EnvBase

logger = logging.getLogger(__name__)
if _USING_LIGER:
    logger.info("Using liger kernel")
else:
    logger.info("Liger kernel not installed, defaulting to HF")

class GRPOTrainer():
    def __init__(self,
                 model: str,
                 train_dataset: Optional[Sequence[Any]],
                 eval_dataset: Optional[Sequence[Any]] = None,
                 num_steps: int = 100,
                 saving_steps: int = 10,
                 logging_steps: int = 10,
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
                 ):

        # Set random seed for reproducibility
        set_seed(seed)
        self.seed = seed
        self._configure_logging(log_level)

        self.sampling_temperature = sampling_temperature
        self.kl_penalty = kl_penalty
        self.output_dir = Path(output_dir)
        self.num_steps = num_steps
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

        device = get_device()

        if device == "cuda":
            device_map = "auto"

            # Configure torch backends for performance
            configure_torch_backends()

        else:

            device_map = device

        self.device = device
        self.device_map = device_map
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        attn_implementation = resolve_attention_implementation(device_map)
        self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map=device_map,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16
            )

        self.model.config.use_cache = False 

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if kl_penalty > 0.0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map=device_map,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16
            )

            self.ref_model.config.use_cache = False
            self.ref_model.eval()

            for param in self.ref_model.parameters():
                param.requires_grad_(False)
        else:
            self.ref_model = None 


        self._wandb = None 

        if enable_wandb:
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
                        "saving_steps": saving_steps,
                        "logging_steps": logging_steps,
                        "sampling_temperature": sampling_temperature,
                        "kl_penalty": kl_penalty,
                        "use_ref_model": kl_penalty > 0.0,
                        "output_dir": output_dir,
                        "optimizer": optimizer.__name__ if hasattr(optimizer, "__name__") else optimizer.__class__.__name__,
                        "optimizer_args": optimizer_args,
                        "lr_scheduler": None if lr_scheduler is None else (lr_scheduler.__name__ if hasattr(lr_scheduler, "__name__") else lr_scheduler.__class__.__name__),
                        "lr_scheduler_args": lr_scheduler_args,
                        "attn_implementation": "flash_attention_2",
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

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def _configure_logging(self, log_level: int) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        logger.setLevel(log_level)

    def sample_completions_batch(self, envs: List[EnvBase]) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:

        formatted_prompts = []

        for env in envs:
            formatted_prompt = self.tokenizer.apply_chat_template(
                    env.prompt,
                    add_generation_prompt=True,
                    tokenize=False
                )
            
            formatted_prompts.append(formatted_prompt)

        input_tokens = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        model_device = next(self.model.parameters()).device
        input_tokens = input_tokens.to(model_device)

        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": True,
            "temperature": self.sampling_temperature,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": False,
        }
        if self.generation_args:
            gen_kwargs.update(self.generation_args)
        gen_kwargs["return_dict_in_generate"] = True

        outputs = cast(GenerateOutput, self.model.generate(**gen_kwargs))

        generated_sequences = outputs.sequences
        ret_texts = []
        completion_lengths = []
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        for i, prompt_len in enumerate(prompt_lengths):

            output_token_ids = generated_sequences[i]
            end_index = output_token_ids.shape[0]

            if eos_token_id is not None:
                eos_positions = (output_token_ids[prompt_len:] == eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    end_index = prompt_len + int(eos_positions[0]) + 1
            elif pad_token_id is not None:
                pad_positions = (output_token_ids[prompt_len:] == pad_token_id).nonzero(as_tuple=True)[0]
                if pad_positions.numel() > 0:
                    end_index = prompt_len + int(pad_positions[0]) + 1

            completion_token_ids = output_token_ids[prompt_len:end_index]

            text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=True)

            ret_texts.append(text)
            completion_lengths.append(max(end_index - prompt_len, 0))
        
        return generated_sequences, ret_texts, prompt_lengths, completion_lengths

    def _compute_reward(self, env: EnvBase, completion_text: Optional[str]) -> float:
        if completion_text is None:
            return -(self.max_error + self.invalid_penalty)

        if env.answer is None:
            raise ValueError("Environment has no defined answer.")
        else:
            return float(env.get_reward(completion_text))


    def compute_advantage(self, trajectories: List[Trajectory], eps: float = 1e-8) -> None:
        for traj in trajectories:
            if traj.reward is None:
                raise ValueError("Trajectory reward is required to compute advantage.")
        rewards = torch.tensor([traj.reward for traj in trajectories], dtype=torch.float32)
        mean_reward = float(rewards.mean().item())
        std = float(rewards.std(unbiased=False).item())

        for traj in trajectories:
            assert traj.reward is not None
            traj.advantage = (traj.reward - mean_reward) / (std + eps)

    def get_log_probs(self, sequence_ids: torch.Tensor, model: Optional[torch.nn.Module] = None) -> torch.Tensor:
        if sequence_ids.ndim == 1:
            sequence_ids = sequence_ids.unsqueeze(0)
        if model is None:
            model = self.model

        if self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
            attention_mask = (sequence_ids != self.tokenizer.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(sequence_ids)

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

    def generate_mask(self, trajectory: Trajectory) -> torch.Tensor:
        if trajectory.prompt_len is None:
            raise ValueError("Trajectory prompt_len is required to build the mask.")
        if trajectory.sequence_ids is None:
            raise ValueError("Trajectory sequence_ids is required to build the mask.")

        sequence_length = int(trajectory.sequence_ids.numel())
        mask = torch.zeros((sequence_length - 1,), device=trajectory.sequence_ids.device, dtype=torch.float32)

        completion_len = trajectory.completion_len
        if completion_len is None:
            completion_len = max(sequence_length - trajectory.prompt_len, 0)
        end_len = min(trajectory.prompt_len + completion_len, sequence_length)

        start = max(trajectory.prompt_len - 1, 0)
        end = max(end_len - 2, -1)

        if end >= start:
            mask[start:end + 1] = 1.0

        trajectory.mask = mask
        return mask

    def _log_step(self, step: int, env: EnvBase, trajectories: List[Trajectory], batch_stats: Optional[dict] = None) -> None:
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

        self.model.train()
        if self.ref_model is not None:
            self.ref_model.eval()

        for step in range(self.num_steps):
            env_batch = [self.train_dataset[random.randint(0, dataset_len - 1)] for _ in range(self.batch_size)]
            envs = []
            for env in env_batch:
                envs.extend([env for _ in range(self.group_size)])

            sequences, texts, prompt_lens, completion_lens = self.sample_completions_batch(envs)

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

            self.compute_advantage(trajectories)

            with torch.no_grad():
                old_log_probs = self.get_log_probs(sequences)
                if self.ref_model is not None:
                    ref_log_probs = self.get_log_probs(sequences, model=self.ref_model)
                else:
                    ref_log_probs = torch.zeros_like(old_log_probs)

            for i, traj in enumerate(trajectories):
                traj.old_log_probs = old_log_probs[i]
                traj.ref_log_probs = ref_log_probs[i]
                self.generate_mask(traj)

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
                new_log_probs = self.get_log_probs(sequences)

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
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

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

            log_env = env_batch[0] if env_batch else envs[0]
            self._log_step(step, log_env, trajectories, batch_stats)

            if step % self.saving_steps == 0:
                step_output_dir = self.output_dir / f"step_{step}"
                step_output_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(step_output_dir)
                self.tokenizer.save_pretrained(step_output_dir)

        if self._wandb is not None:
            self._wandb.finish()

    def test(self, dataset: Optional[Sequence[Any]] = None, num_batches: Optional[int] = None) -> dict:
        eval_dataset = dataset if dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Eval dataset is required for testing.")
        dataset_len = len(eval_dataset)
        if dataset_len == 0:
            raise ValueError("Eval dataset is empty.")

        self.model.eval()

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
            "test reward_mean=%.4f reward_std=%.4f completion_tokens_mean=%.1f",
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

        self.model.train()

        return {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "completion_tokens_mean": completion_tokens_mean,
        }
    

if __name__ == "__main__":

    envs = [EnvBase(
        prompt=[
            {
                "role": "user",
                "content": "What is 5 + 6?"
            }
        ],
        answer="11"
    )]

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        train_dataset=None,
        num_steps=1,
        saving_steps=1,
        logging_steps=1,
    )


    _sequences, _texts, _prompt_lens, _completion_lens = trainer.sample_completions_batch(envs)

    print("done")
