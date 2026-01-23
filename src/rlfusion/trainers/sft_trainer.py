import importlib
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Iterable, Tuple, Any, Literal, cast

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

from rlfusion.envs import EnvBase
from rlfusion.inference.hf_utils import sample_completions_batch_hf
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
        num_steps: int = 100,
        batch_size: int = 8,
        saving_steps: int = 10,
        logging_steps: int = 10,
        eval_steps: Optional[int] = None,
        eval_dataset: Optional[Sequence[EnvBase]] = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        sampling_temperature: float = 1.0,
        generation_args: Optional[dict[str, Any]] = None,
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
        assistant_loss_mode: Literal["all", "last"] = "all",
        max_grad_norm: Optional[float] = None,
        log_level: int = logging.INFO,
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
        self.batch_size = batch_size
        self.saving_steps = saving_steps
        self.logging_steps = logging_steps
        self.output_dir = Path(output_dir)
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt
        if assistant_loss_mode not in {"all", "last"}:
            raise ValueError("assistant_loss_mode must be 'all' or 'last'.")
        self.assistant_loss_mode: Literal["all", "last"] = assistant_loss_mode
        self.max_grad_norm = max_grad_norm

        if eval_steps is not None and eval_steps <= 0:
            raise ValueError("eval_steps must be >= 1 or None.")
        self.eval_steps = eval_steps
        self.eval_dataset = eval_dataset
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.sampling_temperature = sampling_temperature
        self.generation_args = generation_args or {}

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
        if device == "cuda":
            model_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif device == "mps":
            model_kwargs["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            **model_kwargs,
        )
        self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

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
                        "batch_size": batch_size,
                        "saving_steps": saving_steps,
                        "logging_steps": logging_steps,
                        "eval_steps": eval_steps,
                        "output_dir": output_dir,
                        "optimizer": optimizer.__name__ if hasattr(optimizer, "__name__") else optimizer.__class__.__name__,
                        "optimizer_args": optimizer_args,
                        "lr_scheduler": None if lr_scheduler is None else (lr_scheduler.__name__ if hasattr(lr_scheduler, "__name__") else lr_scheduler.__class__.__name__),
                        "lr_scheduler_args": lr_scheduler_args,
                        "dtype": str(self.model.dtype),
                        "seed": seed,
                        "max_seq_len": max_seq_len,
                        "max_new_tokens": max_new_tokens,
                        "mask_prompt": mask_prompt,
                        "assistant_loss_mode": assistant_loss_mode,
                        "use_eval_dataset": eval_dataset is not None,
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

    def sample_completions_batch(
        self, envs: list[EnvBase]
    ) -> Tuple[torch.Tensor, list[str], list[int], list[int]]:
        model = cast(Any, self.model)
        return sample_completions_batch_hf(
            model=model,
            tokenizer=self.tokenizer,
            envs=envs,
            do_sample=self.do_sample,
            sampling_temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
            generation_args=self.generation_args,
        )

    def _compute_reward(self, env: EnvBase, completion_text: Optional[str]) -> float:
        if completion_text is None:
            return 0.0
        reward_value = env.get_reward(completion_text)
        if reward_value is None:
            return 0.0
        return float(reward_value)

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

    def _chat_template_message_spans(
        self, messages: list[dict[str, object]]
    ) -> Tuple[list[int], list[tuple[int, int, str]]]:
        """Return token ids and per-message token spans (start, end, role).

        Spans are computed by tokenizing prefixes of `messages` using `apply_chat_template` and
        taking deltas. This lets us mask tokens by role for multi-turn chats.
        """
        spans: list[tuple[int, int, str]] = []
        prev_len = 0
        full_ids = self._apply_chat_template_ids(messages)

        for idx, msg in enumerate(messages):
            prefix_ids = self._apply_chat_template_ids(messages[: idx + 1])
            end = len(prefix_ids)
            role = str(msg.get("role", ""))
            spans.append((prev_len, end, role))
            prev_len = end

        return full_ids, spans

    def _build_batch(self, prompts: Iterable[Any], responses: Iterable[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for prompt, response in zip(prompts, responses):
            prompt = self._normalize_prompt(prompt)

            messages = list(prompt)
            if response is not None:
                messages.append({"role": "assistant", "content": response})
            input_ids, spans = self._chat_template_message_spans(messages)
            labels = list(input_ids) if not self.mask_prompt else [-100] * len(input_ids)

            if self.mask_prompt:
                assistant_spans = [(s, e) for s, e, r in spans if r == "assistant"]
                if not assistant_spans:
                    raise ValueError("No assistant messages found; nothing to train on.")

                spans_to_train = assistant_spans if self.assistant_loss_mode == "all" else [assistant_spans[-1]]
                for start, end in spans_to_train:
                    labels[start:end] = input_ids[start:end]

            if self.max_seq_len is not None and len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                labels = labels[: self.max_seq_len]

            attention_mask = [1] * len(input_ids)

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
        if self.eval_steps is not None:
            if self.eval_dataset is None:
                raise ValueError("eval_dataset is required when eval_steps is set.")
            if len(self.eval_dataset) == 0:
                raise ValueError("Eval dataset is empty.")

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
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if (step + 1) % self.logging_steps == 0:
                loss_val = float(loss.item())
                if self._is_main_process():
                    logger.info("step %d loss=%.6f", step + 1, loss_val)
                if self._wandb is not None:
                    self._wandb.log(
                        {
                            "train/loss": loss_val,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=step + 1,
                    )

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
        dataset: Optional[Sequence[EnvBase]] = None,
        num_batches: Optional[int] = None,
        step: Optional[int] = None,
    ) -> dict:
        if dataset is None:
            raise ValueError("dataset is required for testing.")
        eval_dataset = dataset
        dataset_len = len(eval_dataset)
        if dataset_len == 0:
            raise ValueError("Eval dataset is empty.")

        self.model.eval()

        try:
            if num_batches is None:
                indices = list(range(dataset_len))
            else:
                indices = [random.randint(0, dataset_len - 1) for _ in range(num_batches * self.batch_size)]

            all_rewards: list[float] = []
            all_completion_lengths: list[int] = []

            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                env_batch = [eval_dataset[i] for i in batch_indices]

                _, texts, _, completion_lens = self.sample_completions_batch(env_batch)
                rewards = [self._compute_reward(env, text) for env, text in zip(env_batch, texts)]
                all_rewards.extend(rewards)
                all_completion_lengths.extend(completion_lens)

            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
            reward_mean = float(rewards_tensor.mean().item())
            reward_std = float(rewards_tensor.std(unbiased=False).item())
            completion_tokens_mean = float(
                torch.tensor(all_completion_lengths, dtype=torch.float32).mean().item()
            )

            logger.info(
                "eval reward_mean=%.4f reward_std=%.4f completion_tokens_mean=%.1f",
                reward_mean,
                reward_std,
                completion_tokens_mean,
            )

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
            self.model.train()
