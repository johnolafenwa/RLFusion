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

from rlfusion.evaluation.evaluator import Evaluator
from rlfusion.envs import EnvBase
from rlfusion.inference.hf_utils import sample_completions_batch_hf
from rlfusion.trainers.utils import (
    get_device,
    set_seed,
    configure_torch_backends,
    truncate_text,
    format_prompt,
    resolve_attention_implementation,
)

logger = logging.getLogger(__name__)
if _USING_LIGER:
    logger.info("Using liger kernel")
else:
    logger.info("Liger kernel not installed, defaulting to HF")


class OnPolicyDistillationTrainer:
    def __init__(
        self,
        model: str,
        teacher_model: str,
        train_dataset: Sequence[Any],
        num_steps: int = 100,
        saving_steps: int = 10,
        logging_steps: int = 10,
        eval_steps: Optional[int] = None,
        enable_wandb: bool = False,
        wandb_project: str = "onpolicy_distill",
        wandb_run_name: Optional[str] = None,
        sampling_temperature: float = 1.0,
        output_dir: str = "./outputs",
        generation_args: Optional[dict[str, Any]] = None,
        optimizer: type[Optimizer] = AdamW,
        optimizer_args: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[type[LRScheduler] | LRScheduler] = None,
        lr_scheduler_args: Optional[dict[str, Any]] = None,
        seed: int = 42,
        max_new_tokens: int = 1024,
        batch_size: int = 1,
        ppo_steps: int = 1,
        clip_eps: float = 0.2,
        log_completions: bool = False,
        max_log_chars: Optional[int] = 320,
        max_grad_norm: Optional[float] = None,
        log_level: int = logging.INFO,
        evaluator: Optional[Evaluator] = None,
        use_accelerate: bool = False,
    ):
        self.accelerator = None
        if use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator()

        seed_offset = 0 if self.accelerator is None else int(self.accelerator.process_index)
        set_seed(seed + seed_offset)
        self.seed = seed
        self._configure_logging(log_level)

        self.train_dataset = train_dataset
        self.num_steps = num_steps
        self.saving_steps = saving_steps
        self.logging_steps = logging_steps
        self.sampling_temperature = sampling_temperature
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.generation_args = generation_args or {}
        self.ppo_steps = ppo_steps
        self.clip_eps = clip_eps
        self.log_completions = log_completions
        self.max_log_chars = max_log_chars
        self.max_grad_norm = max_grad_norm

        if eval_steps is not None and eval_steps <= 0:
            raise ValueError("eval_steps must be >= 1 or None.")
        if eval_steps is not None and evaluator is None:
            raise ValueError("evaluator is required when eval_steps is set.")
        self.eval_steps = eval_steps
        self.evaluator = evaluator

        device = get_device()
        if device == "cuda":
            device_map = "auto"
            configure_torch_backends()
        else:
            device_map = device

        self.device = device
        self.device_map = device_map

        attn_implementation = resolve_attention_implementation(device_map)
        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "attn_implementation": attn_implementation,
        }
        if device == "cuda" and torch.cuda.is_available():
            model_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif device == "mps":
            model_kwargs["dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self.model.config.use_cache = False

        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **model_kwargs)
        self.teacher_model.config.use_cache = False
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self._wandb = None
        if enable_wandb and self._is_main_process():
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model": model,
                        "teacher_model": teacher_model,
                        "device": device,
                        "device_map": device_map,
                        "num_steps": num_steps,
                        "saving_steps": saving_steps,
                        "logging_steps": logging_steps,
                        "eval_steps": eval_steps,
                        "sampling_temperature": sampling_temperature,
                        "output_dir": output_dir,
                        "optimizer": optimizer.__name__ if hasattr(optimizer, "__name__") else optimizer.__class__.__name__,
                        "optimizer_args": optimizer_args,
                        "lr_scheduler": None if lr_scheduler is None else (lr_scheduler.__name__ if hasattr(lr_scheduler, "__name__") else lr_scheduler.__class__.__name__),
                        "lr_scheduler_args": lr_scheduler_args,
                        "dtype": str(self.model.dtype),
                        "seed": seed,
                        "max_new_tokens": max_new_tokens,
                        "batch_size": batch_size,
                        "ppo_steps": ppo_steps,
                        "clip_eps": clip_eps,
                        "use_evaluator": evaluator is not None,
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

    def _configure_logging(self, log_level: int) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        logger.setLevel(log_level)

    def _is_main_process(self) -> bool:
        return self.accelerator is None or bool(self.accelerator.is_main_process)

    def _unwrap_model_for_saving(self) -> torch.nn.Module:
        if self.accelerator is None:
            return self.model
        return cast(torch.nn.Module, self.accelerator.unwrap_model(self.model))

    def _evaluate_with_evaluator(self, step: int) -> dict[str, float]:
        if self.evaluator is None:
            raise RuntimeError("Evaluator is not set.")

        if self.evaluator.engine == "hf":
            metrics: dict[str, float] = {}
            if self._is_main_process():
                model = cast(Any, self._unwrap_model_for_saving())
                metrics = self.evaluator.evaluate_with_model(
                    model=model, tokenizer=self.evaluator.tokenizer
                )
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            if not self._is_main_process():
                return {}
        else:
            model_dir = self.output_dir / "eval_latest"
            model_dir.mkdir(parents=True, exist_ok=True)
            if self._is_main_process():
                model = cast(Any, self._unwrap_model_for_saving())
                model.save_pretrained(model_dir)
                self.tokenizer.save_pretrained(model_dir)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if not self._is_main_process():
                return {}

            self.evaluator.set_model(str(model_dir))
            metrics = self.evaluator.evaluate()

        if self._wandb is not None:
            self._wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

        return metrics

    def sample_completions_batch(
        self, envs: List[EnvBase]
    ) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
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
            use_cache=False,
        )

        logits = outputs.logits / max(self.sampling_temperature, 1e-6)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = sequence_ids[:, 1:]

        return torch.gather(logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    def _build_masks(
        self,
        prompt_lengths: List[int],
        completion_lengths: List[int],
        sequence_ids: torch.Tensor,
    ) -> torch.Tensor:
        sequence_lengths = [int(seq.numel()) for seq in sequence_ids]
        masks = []
        for prompt_len, completion_len, seq_len in zip(prompt_lengths, completion_lengths, sequence_lengths):
            if completion_len is None:
                completion_len = max(seq_len - prompt_len, 0)
            end_len = min(prompt_len + completion_len, seq_len)
            start = max(prompt_len - 1, 0)
            end = max(end_len - 2, -1)

            mask = torch.zeros((seq_len - 1,), device=sequence_ids.device, dtype=torch.float32)
            if end >= start:
                mask[start:end + 1] = 1.0
            masks.append(mask)

        return torch.stack(masks, dim=0)

    def _ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        clip_eps: float,
    ) -> torch.Tensor:
        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        obj = torch.minimum(unclipped, clipped) * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return -(obj.sum(dim=1) / denom)

    def _log_step(
        self,
        step: int,
        env: EnvBase,
        completions: List[str],
        loss_value: float,
        kl_value: float,
        mask_tokens_mean: float,
        completion_tokens_mean: float,
        reward_mean: Optional[float] = None,
        reward_std: Optional[float] = None,
    ) -> None:
        if step % self.logging_steps != 0:
            return

        if reward_mean is None or reward_std is None:
            logger.info(
                "step %d loss=%.6f reverse_kl=%.6f mask_tokens_mean=%.1f completion_tokens_mean=%.1f",
                step,
                loss_value,
                kl_value,
                mask_tokens_mean,
                completion_tokens_mean,
            )
        else:
            logger.info(
                "step %d loss=%.6f reverse_kl=%.6f reward_mean=%.4f reward_std=%.4f",
                step,
                loss_value,
                kl_value,
                reward_mean,
                reward_std,
            )
            logger.info(
                "step %d mask_tokens_mean=%.1f completion_tokens_mean=%.1f",
                step,
                mask_tokens_mean,
                completion_tokens_mean,
            )
        if self.log_completions:
            prompt_text = format_prompt(env.prompt)
            logger.info("prompt: %s", truncate_text(prompt_text, self.max_log_chars))
            answer_text = None if env.answer is None else str(env.answer)
            logger.info("gt_answer: %s", truncate_text(answer_text, self.max_log_chars))
            for idx, completion in enumerate(completions):
                completion_preview = truncate_text(completion, self.max_log_chars)
                logger.info("generated_%d: %s", idx, completion_preview)

        if self._wandb is not None:
            metrics = {
                "train/loss": loss_value,
                "train/reverse_kl": kl_value,
                "train/mask_tokens_mean": mask_tokens_mean,
                "train/completion_tokens_mean": completion_tokens_mean,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if reward_mean is not None and reward_std is not None:
                metrics["train/reward_mean"] = reward_mean
                metrics["train/reward_std"] = reward_std
            self._wandb.log(metrics, step=step)

    def _compute_reward(self, env: EnvBase, completion_text: Optional[str]) -> Optional[float]:
        if env.answer is None:
            return None
        if completion_text is None:
            return None
        return float(env.get_reward(completion_text))

    def train(self) -> None:
        dataset_len = len(self.train_dataset)
        if dataset_len == 0:
            raise ValueError("Dataset is empty.")
        if self.eval_steps is not None and self.evaluator is None:
            raise ValueError("evaluator is required when eval_steps is set.")

        self.model.train()
        self.teacher_model.eval()

        for step in range(self.num_steps):
            env_batch = [self.train_dataset[random.randint(0, dataset_len - 1)] for _ in range(self.batch_size)]

            sequences, texts, prompt_lens, completion_lens = self.sample_completions_batch(env_batch)
            with torch.no_grad():
                old_log_probs = self.get_log_probs(sequences)
                teacher_log_probs = self.get_log_probs(sequences, model=self.teacher_model)

            masks = self._build_masks(prompt_lens, completion_lens, sequences)
            mask_counts = masks.sum(dim=1).clamp_min(1.0)
            reverse_kl = (old_log_probs - teacher_log_probs) * masks
            advantages = -reverse_kl

            loss = None
            for ppo_step in range(self.ppo_steps):
                self.optimizer.zero_grad(set_to_none=True)
                new_log_probs = self.get_log_probs(sequences)
                loss_per = self._ppo_loss(
                    old_log_probs,
                    new_log_probs,
                    advantages,
                    masks,
                    clip_eps=self.clip_eps,
                )
                loss = loss_per.mean()
                if self.accelerator is None:
                    loss.backward()
                else:
                    self.accelerator.backward(loss)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            reverse_kl_mean = float((reverse_kl.sum(dim=1) / mask_counts).mean().item())

            rewards = [
                self._compute_reward(env, completion)
                for env, completion in zip(env_batch, texts)
            ]
            valid_rewards = [r for r in rewards if r is not None]
            if valid_rewards:
                rewards_tensor = torch.tensor(valid_rewards, dtype=torch.float32)
                reward_mean = float(rewards_tensor.mean().item())
                reward_std = float(rewards_tensor.std(unbiased=False).item())
            else:
                reward_mean = None
                reward_std = None

            if (step + 1) % self.logging_steps == 0 and self._is_main_process():
                self._log_step(
                    step + 1,
                    env_batch[0],
                    texts,
                    loss_value=float(loss.item() if loss is not None else 0.0),
                    kl_value=reverse_kl_mean,
                    mask_tokens_mean=float(mask_counts.mean().item()),
                    completion_tokens_mean=float(torch.tensor(completion_lens, dtype=torch.float32).mean().item()),
                    reward_mean=reward_mean,
                    reward_std=reward_std,
                )

            if self.eval_steps is not None and (step + 1) % self.eval_steps == 0:
                self._evaluate_with_evaluator(step=step + 1)

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
        dataset: Optional[Sequence[Any]] = None,
        num_batches: Optional[int] = None,
        step: Optional[int] = None,
        eval_temperature: Optional[float] = None,
    ) -> dict:
        if dataset is None:
            raise ValueError("dataset is required for testing.")
        eval_dataset = dataset
        dataset_len = len(eval_dataset)
        if dataset_len == 0:
            raise ValueError("Eval dataset is empty.")

        # Save original temperature and set eval temperature
        original_temperature = self.sampling_temperature
        if eval_temperature is not None:
            self.sampling_temperature = eval_temperature

        self.model.eval()
        self.teacher_model.eval()

        try:
            all_loss = []
            all_mask_counts = []
            all_completion_lengths = []
            all_rewards = []
            first_batch_completions: Optional[List[str]] = None
            first_batch_env: Optional[EnvBase] = None

            if num_batches is None:
                indices = list(range(dataset_len))
            else:
                indices = [random.randint(0, dataset_len - 1) for _ in range(num_batches * self.batch_size)]

            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                env_batch = [eval_dataset[i] for i in batch_indices]

                with torch.no_grad():
                    sequences, texts, prompt_lens, completion_lens = self.sample_completions_batch(env_batch)
                    student_log_probs = self.get_log_probs(sequences)
                    teacher_log_probs = self.get_log_probs(sequences, model=self.teacher_model)

                masks = self._build_masks(prompt_lens, completion_lens, sequences)
                mask_counts = masks.sum(dim=1).clamp_min(1.0)
                reverse_kl = (student_log_probs - teacher_log_probs) * masks
                loss_per = reverse_kl.sum(dim=1) / mask_counts

                all_loss.extend(loss_per.detach().cpu().tolist())
                all_mask_counts.extend(mask_counts.detach().cpu().tolist())
                all_completion_lengths.extend(completion_lens)

                rewards = [
                    self._compute_reward(env, completion)
                    for env, completion in zip(env_batch, texts)
                ]
                all_rewards.extend([r for r in rewards if r is not None])

                if first_batch_completions is None:
                    first_batch_completions = texts
                    first_batch_env = env_batch[0]

            loss_tensor = torch.tensor(all_loss, dtype=torch.float32)
            loss_mean = float(loss_tensor.mean().item())
            if all_rewards:
                rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
                reward_mean = float(rewards_tensor.mean().item())
                reward_std = float(rewards_tensor.std(unbiased=False).item())
            else:
                reward_mean = None
                reward_std = None

            mask_tokens_mean = float(torch.tensor(all_mask_counts, dtype=torch.float32).mean().item())
            completion_tokens_mean = float(torch.tensor(all_completion_lengths, dtype=torch.float32).mean().item())

            # Log eval results
            if reward_mean is None or reward_std is None:
                logger.info(
                    "eval loss=%.6f reverse_kl=%.6f mask_tokens_mean=%.1f completion_tokens_mean=%.1f",
                    loss_mean,
                    loss_mean,
                    mask_tokens_mean,
                    completion_tokens_mean,
                )
            else:
                logger.info(
                    "eval loss=%.6f reverse_kl=%.6f reward_mean=%.4f reward_std=%.4f",
                    loss_mean,
                    loss_mean,
                    reward_mean,
                    reward_std,
                )

            if self.log_completions and first_batch_completions is not None and first_batch_env is not None:
                prompt_text = format_prompt(first_batch_env.prompt)
                logger.info("prompt: %s", truncate_text(prompt_text, self.max_log_chars))
                answer_text = None if first_batch_env.answer is None else str(first_batch_env.answer)
                logger.info("gt_answer: %s", truncate_text(answer_text, self.max_log_chars))
                for idx, completion in enumerate(first_batch_completions):
                    completion_preview = truncate_text(completion, self.max_log_chars)
                    logger.info("generated_%d: %s", idx, completion_preview)

            if self._wandb is not None:
                log_data: dict[str, Any] = {
                    "eval/loss": loss_mean,
                    "eval/reverse_kl": loss_mean,
                    "eval/mask_tokens_mean": mask_tokens_mean,
                    "eval/completion_tokens_mean": completion_tokens_mean,
                }
                if reward_mean is not None and reward_std is not None:
                    log_data["eval/reward_mean"] = reward_mean
                    log_data["eval/reward_std"] = reward_std
                if step is not None:
                    self._wandb.log(log_data, step=step)
                else:
                    self._wandb.log(log_data)

            return {
                "loss": loss_mean,
                "reverse_kl": loss_mean,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "mask_tokens_mean": mask_tokens_mean,
                "completion_tokens_mean": completion_tokens_mean,
            }
        finally:
            # Restore original temperature and model state
            self.sampling_temperature = original_temperature
            self.model.train()
