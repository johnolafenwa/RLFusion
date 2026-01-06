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

from rlfusion.envs import EnvBase
from rlfusion.trainers.utils import (
    get_device,
    set_seed,
    configure_torch_backends,
    truncate_text,
    format_prompt,
    resolve_attention_implementation,
)
from rlfusion.trainers.types import GenerateOutput

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
        eval_dataset: Optional[Sequence[Any]] = None,
        num_steps: int = 100,
        saving_steps: int = 10,
        logging_steps: int = 10,
        enable_wandb: bool = False,
        wandb_project: str = "onpolicy_distill",
        wandb_run_name: Optional[str] = None,
        sampling_temperature: float = 1.0,
        output_dir: str = "./outputs",
        optimizer: type[Optimizer] = AdamW,
        optimizer_args: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[type[LRScheduler] | LRScheduler] = None,
        lr_scheduler_args: Optional[dict[str, object]] = None,
        seed: int = 42,
        max_new_tokens: int = 1024,
        batch_size: int = 1,
        log_completions: bool = False,
        max_log_chars: int = 320,
        max_grad_norm: Optional[float] = None,
        log_level: int = logging.INFO,
    ):
        set_seed(seed)
        self.seed = seed
        self._configure_logging(log_level)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_steps = num_steps
        self.saving_steps = saving_steps
        self.logging_steps = logging_steps
        self.sampling_temperature = sampling_temperature
        self.output_dir = Path(output_dir)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.log_completions = log_completions
        self.max_log_chars = max_log_chars
        self.max_grad_norm = max_grad_norm

        device = get_device()
        if device == "cuda":
            device_map = "auto"
            configure_torch_backends()
        else:
            device_map = device

        self.device = device
        self.device_map = device_map

        attn_implementation = resolve_attention_implementation(device_map)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        self.model.config.use_cache = False

        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            device_map=device_map,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        self.teacher_model.config.use_cache = False
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._wandb = None
        if enable_wandb:
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

    def sample_completions_batch(
        self, envs: List[EnvBase]
    ) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
        formatted_prompts = []
        for env in envs:
            formatted_prompt = self.tokenizer.apply_chat_template(
                env.prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            formatted_prompts.append(formatted_prompt)

        input_tokens = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        model_device = next(self.model.parameters()).device
        input_tokens = input_tokens.to(model_device)

        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        outputs = cast(GenerateOutput, self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=False,
        ))

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
            for idx, completion in enumerate(completions):
                completion_preview = truncate_text(completion, self.max_log_chars)
                logger.info("sample %d completion=%s", idx, completion_preview)

        if self._wandb is not None:
            metrics = {
                "loss": loss_value,
                "reverse_kl": kl_value,
                "mask_tokens_mean": mask_tokens_mean,
                "completion_tokens_mean": completion_tokens_mean,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if reward_mean is not None and reward_std is not None:
                metrics["reward/mean"] = reward_mean
                metrics["reward/std"] = reward_std
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

        self.model.train()
        self.teacher_model.eval()

        for step in range(self.num_steps):
            env_batch = [self.train_dataset[random.randint(0, dataset_len - 1)] for _ in range(self.batch_size)]

            sequences, texts, prompt_lens, completion_lens = self.sample_completions_batch(env_batch)
            student_log_probs = self.get_log_probs(sequences)
            with torch.no_grad():
                teacher_log_probs = self.get_log_probs(sequences, model=self.teacher_model)

            masks = self._build_masks(prompt_lens, completion_lens, sequences)
            mask_counts = masks.sum(dim=1).clamp_min(1.0)
            reverse_kl = (student_log_probs - teacher_log_probs) * masks
            loss_per = reverse_kl.sum(dim=1) / mask_counts
            loss = loss_per.mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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

            self._log_step(
                step,
                env_batch[0],
                texts,
                loss_value=float(loss.item()),
                kl_value=float(loss_per.mean().item()),
                mask_tokens_mean=float(mask_counts.mean().item()),
                completion_tokens_mean=float(torch.tensor(completion_lens, dtype=torch.float32).mean().item()),
                reward_mean=reward_mean,
                reward_std=reward_std,
            )

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
        self.teacher_model.eval()

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
        loss = loss_tensor.mean()
        if all_rewards:
            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
            reward_mean = float(rewards_tensor.mean().item())
            reward_std = float(rewards_tensor.std(unbiased=False).item())
        else:
            reward_mean = None
            reward_std = None

        mask_tokens_mean = float(torch.tensor(all_mask_counts, dtype=torch.float32).mean().item())
        completion_tokens_mean = float(torch.tensor(all_completion_lengths, dtype=torch.float32).mean().item())

        if first_batch_completions is None or first_batch_env is None:
            raise ValueError("Eval dataset did not yield any batches.")

        self._log_step(
            0,
            first_batch_env,
            first_batch_completions,
            loss_value=float(loss.item()),
            kl_value=float(loss_tensor.mean().item()),
            mask_tokens_mean=mask_tokens_mean,
            completion_tokens_mean=completion_tokens_mean,
            reward_mean=reward_mean,
            reward_std=reward_std,
        )

        self.model.train()

        return {
            "loss": float(loss.item()),
            "reverse_kl": float(loss_tensor.mean().item()),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "mask_tokens_mean": mask_tokens_mean,
            "completion_tokens_mean": completion_tokens_mean,
        }
