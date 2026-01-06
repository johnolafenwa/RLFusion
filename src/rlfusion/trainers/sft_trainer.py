import importlib
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Iterable, Tuple, Any

import torch
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

from rlfusion.trainers.utils import get_device, set_seed, configure_torch_backends, resolve_attention_implementation

logger = logging.getLogger(__name__)
if _USING_LIGER:
    logger.info("Using liger kernel")
else:
    logger.info("Liger kernel not installed, defaulting to HF")


class SFTTrainer:
    def __init__(
        self,
        model: str,
        train_dataset: Sequence[Any],
        eval_dataset: Optional[Sequence[Any]] = None,
        num_steps: int = 100,
        batch_size: int = 8,
        saving_steps: int = 10,
        logging_steps: int = 10,
        enable_wandb: bool = False,
        wandb_project: str = "sft",
        wandb_run_name: Optional[str] = None,
        output_dir: str = "./outputs",
        optimizer: type[Optimizer] = AdamW,
        optimizer_args: Optional[dict[str, Any]] = None,
        lr_scheduler: Optional[type[LRScheduler] | LRScheduler] = None,
        lr_scheduler_args: Optional[dict[str, object]] = None,
        seed: int = 42,
        max_seq_len: Optional[int] = None,
        mask_prompt: bool = True,
        max_grad_norm: Optional[float] = None,
        log_level: int = logging.INFO,
    ):
        set_seed(seed)
        self.seed = seed
        self._configure_logging(log_level)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.saving_steps = saving_steps
        self.logging_steps = logging_steps
        self.output_dir = Path(output_dir)
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt
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
                        "device": device,
                        "device_map": device_map,
                        "num_steps": num_steps,
                        "batch_size": batch_size,
                        "saving_steps": saving_steps,
                        "logging_steps": logging_steps,
                        "output_dir": output_dir,
                        "optimizer": optimizer.__name__ if hasattr(optimizer, "__name__") else optimizer.__class__.__name__,
                        "optimizer_args": optimizer_args,
                        "lr_scheduler": None if lr_scheduler is None else (lr_scheduler.__name__ if hasattr(lr_scheduler, "__name__") else lr_scheduler.__class__.__name__),
                        "lr_scheduler_args": lr_scheduler_args,
                        "dtype": str(self.model.dtype),
                        "seed": seed,
                        "max_seq_len": max_seq_len,
                        "mask_prompt": mask_prompt,
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

    def _extract_sample(self, sample: Any) -> Tuple[list[dict[str, object]], Optional[str]]:
        if isinstance(sample, dict):
            prompt = sample.get("prompt")
            answer = sample.get("answer")
        else:
            prompt = getattr(sample, "prompt", None)
            answer = getattr(sample, "answer", None)

        if prompt is None:
            raise ValueError("Sample must include a prompt list of role/content dicts.")
        return self._normalize_prompt(prompt), None if answer is None else str(answer)

    def _normalize_prompt(self, prompt: Any) -> list[dict[str, object]]:
        if not isinstance(prompt, list):
            raise ValueError("Prompt must be a list of role/content dicts.")
        normalized: list[dict[str, object]] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                raise ValueError("Each prompt message must be a dict with role/content.")
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each prompt message must include role and content.")
            role = msg["role"]
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role: {role}")
            content = msg["content"]
            if not isinstance(content, str):
                content = str(content)
            normalized.append({"role": role, "content": content})
        return normalized

    def _apply_chat_template_ids(self, messages: list[dict[str, object]]) -> list[int]:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        if isinstance(token_ids, torch.Tensor):
            return token_ids.tolist()
        return token_ids

    def _chat_ids_and_user_mask(self, messages: list[dict[str, object]]) -> Tuple[list[int], list[bool]]:
        full_ids = self._apply_chat_template_ids(messages)
        user_mask = [False] * len(full_ids)
        prev_len = 0
        for idx in range(len(messages)):
            current_ids = self._apply_chat_template_ids(messages[: idx + 1])
            cur_len = len(current_ids)
            if messages[idx].get("role") == "user":
                for j in range(prev_len, cur_len):
                    if j < len(user_mask):
                        user_mask[j] = True
            prev_len = cur_len
        return full_ids, user_mask

    def _build_batch(self, prompts: Iterable[Any], responses: Iterable[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for prompt, response in zip(prompts, responses):
            prompt = self._normalize_prompt(prompt)

            messages = list(prompt)
            if response is not None:
                messages.append({"role": "assistant", "content": response})
            input_ids, user_mask = self._chat_ids_and_user_mask(messages)
            if self.max_seq_len is not None and len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                user_mask = user_mask[: self.max_seq_len]
                
            attention_mask = [1] * len(input_ids)
            labels = list(input_ids)
            if self.mask_prompt:
                for i, mask in enumerate(user_mask):
                    if mask:
                        labels[i] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        padded = self.tokenizer.pad(
            {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask},
            padding=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )
        max_len = padded["input_ids"].size(1)
        labels_tensor = torch.full((len(batch_labels), max_len), -100, dtype=torch.long)
        for i, labels in enumerate(batch_labels):
            end = min(len(labels), max_len)
            labels_tensor[i, :end] = torch.tensor(labels[:end], dtype=torch.long)
        labels_tensor[padded["attention_mask"] == 0] = -100

        return padded["input_ids"], padded["attention_mask"], labels_tensor

    def train(self) -> None:
        dataset_len = len(self.train_dataset)
        if dataset_len == 0:
            raise ValueError("Dataset is empty.")

        self.model.train()

        for step in range(self.num_steps):
            indices = [random.randint(0, dataset_len - 1) for _ in range(self.batch_size)]
            batch_samples = [self.train_dataset[i] for i in indices]

            prompts = []
            responses = []
            for sample in batch_samples:
                prompt, response = self._extract_sample(sample)
                prompts.append(prompt)
                responses.append(None if response is None else str(response))

            input_ids, attention_mask, labels = self._build_batch(prompts, responses)

            model_device = next(self.model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            labels = labels.to(model_device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if step % self.logging_steps == 0:
                loss_val = float(loss.item())
                logger.info("step %d loss=%.6f", step, loss_val)
                if self._wandb is not None:
                    self._wandb.log(
                        {
                            "loss": loss_val,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=step,
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

        if num_batches is None:
            indices = list(range(dataset_len))
        else:
            indices = [random.randint(0, dataset_len - 1) for _ in range(num_batches * self.batch_size)]

        total_loss = 0.0
        total_tokens = 0
        batches = 0

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_samples = [eval_dataset[i] for i in batch_indices]

            prompts = []
            responses = []
            for sample in batch_samples:
                prompt, response = self._extract_sample(sample)
                prompts.append(prompt)
                responses.append(None if response is None else str(response))

            input_ids, attention_mask, labels = self._build_batch(prompts, responses)

            model_device = next(self.model.parameters()).device
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            labels = labels.to(model_device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            token_count = int((labels != -100).sum().item())
            if token_count > 0:
                total_loss += float(loss.item()) * token_count
                total_tokens += token_count
            batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        logger.info("test loss=%.6f tokens=%d batches=%d", avg_loss, total_tokens, batches)

        self.model.train()

        return {
            "loss": avg_loss,
            "tokens": total_tokens,
            "batches": batches,
        }
