from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Tuple, List

import torch
from tqdm import tqdm

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
from rlfusion.inference.vllm_utils import build_sampling_params, load_vllm_engine
from rlfusion.trainers.utils import (
    configure_torch_backends,
    format_prompt,
    get_device,
    resolve_attention_implementation,
    set_seed,
    truncate_text,
)

logger = logging.getLogger(__name__)
if _USING_LIGER:
    logger.info("Using liger kernel")
else:
    logger.info("Liger kernel not installed, defaulting to HF")


class Evaluator:
    def __init__(
        self,
        model: str,
        dataset: Sequence[Any],
        output_dir: str = "./outputs/evaluation",
        enable_wandb: bool = False,
        wandb_project: str = "evaluation",
        wandb_run_name: Optional[str] = None,
        engine: str = "hf",
        vllm_args: Optional[dict[str, Any]] = None,
        generation_args: Optional[dict[str, Any]] = None,
        seed: int = 42,
        max_new_tokens: int = 1024,
        batch_size: int = 1,
        do_sample: bool = True,
        sampling_temperature: float = 1.0,
        log_completions: bool = False,
        max_log_chars: Optional[int] = 320,
        show_progress: bool = True,
        log_level: int = logging.INFO,
        num_batches: Optional[int] = None,
    ) -> None:
        set_seed(seed)
        self._configure_logging(log_level)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.generation_args = generation_args or {}
        self.do_sample = do_sample
        self.sampling_temperature = sampling_temperature
        self.log_completions = log_completions
        self.max_log_chars = max_log_chars
        self.show_progress = show_progress
        if num_batches is not None and num_batches <= 0:
            raise ValueError("num_batches must be >= 1 or None.")
        self.num_batches = num_batches

        if engine not in {"hf", "vllm"}:
            raise ValueError("engine must be 'hf' or 'vllm'.")
        self.engine = engine
        self._model_id_or_path = model
        self._vllm_args = vllm_args or {}
        self._hf_model_kwargs: dict[str, Any] = {}
        self._vllm = None
        self._vllm_sampling_params_cls = None
        self._vllm_sampling_param_keys = None

        device = "vllm"
        device_map = "vllm"
        self.model: Any = None
        if self.engine == "hf":
            device = get_device()
            if device == "cuda":
                device_map = "auto"
                configure_torch_backends()
            else:
                device_map = device

            attn_implementation = resolve_attention_implementation(device_map)
            model_kwargs: dict[str, Any] = {
                "device_map": device_map,
                "attn_implementation": attn_implementation,
            }
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                model_kwargs["dtype"] = dtype
            elif device == "mps":
                model_kwargs["dtype"] = torch.float16
            self._hf_model_kwargs = model_kwargs
            self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        else:
            self._vllm, self._vllm_sampling_params_cls, self._vllm_sampling_param_keys = (
                load_vllm_engine(model, self._vllm_args)
            )
            self.model = None

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

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
                        "engine": engine,
                        "output_dir": output_dir,
                        "max_new_tokens": max_new_tokens,
                        "batch_size": batch_size,
                        "do_sample": do_sample,
                        "sampling_temperature": sampling_temperature,
                    },
                )
                self._wandb = wandb
            except Exception as exc:
                logger.warning("wandb could not be enabled: %s", exc)

    def set_model(self, model: str) -> None:
        """Update the evaluator to point at a new model checkpoint (path or HF id)."""
        self._model_id_or_path = model

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.engine == "hf":
            self.model = AutoModelForCausalLM.from_pretrained(model, **self._hf_model_kwargs)
            return

        self._vllm, self._vllm_sampling_params_cls, self._vllm_sampling_param_keys = (
            load_vllm_engine(model, self._vllm_args)
        )
        self.model = None

    def _configure_logging(self, log_level: int) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        logger.setLevel(log_level)

    def _compute_reward(self, env: EnvBase, completion_text: Optional[str]) -> float:
        if completion_text is None:
            return 0.0
        reward_value = env.get_reward(completion_text)
        return 0.0 if reward_value is None else float(reward_value)

    def _build_vllm_sampling_params(self) -> Any:
        if self._vllm_sampling_params_cls is None:
            raise RuntimeError("vLLM engine is not initialized.")
        if self._vllm_sampling_param_keys is None:
            raise RuntimeError("vLLM sampling parameters are unavailable.")
        return build_sampling_params(
            self._vllm_sampling_params_cls,
            self._vllm_sampling_param_keys,
            generation_args=self.generation_args,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.sampling_temperature,
        )

    def evaluate_with_model(self, model: torch.nn.Module, tokenizer: AutoTokenizer) -> dict[str, float]:
        """Evaluate an in-memory HF model without saving/loading checkpoints."""
        if self.engine != "hf":
            raise ValueError("evaluate_with_model is only supported for engine='hf'.")
        previous_model = self.model
        previous_tokenizer = self.tokenizer
        self.model = model
        self.tokenizer = tokenizer
        try:
            return self.evaluate()
        finally:
            self.model = previous_model
            self.tokenizer = previous_tokenizer

    def _sample_completions_batch_vllm(
        self, envs: List[EnvBase]
    ) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
        if self._vllm is None:
            raise RuntimeError("vLLM engine is not initialized.")

        formatted_prompts = []
        for env in envs:
            formatted_prompt = self.tokenizer.apply_chat_template(
                env.prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            formatted_prompts.append(formatted_prompt)

        sampling_params = self._build_vllm_sampling_params()
        outputs = self._vllm.generate(formatted_prompts, sampling_params)

        ret_texts = []
        completion_lengths = []
        prompt_lengths = []
        eos_token_id = self.tokenizer.eos_token_id

        for output in outputs:
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            prompt_lengths.append(0 if prompt_token_ids is None else len(prompt_token_ids))

            if not output.outputs:
                ret_texts.append("")
                completion_lengths.append(0)
                continue

            completion = output.outputs[0]
            token_ids = getattr(completion, "token_ids", None)
            if token_ids is None:
                token_ids = self.tokenizer.encode(completion.text, add_special_tokens=False)
            token_ids = list(token_ids)
            end_offset = len(token_ids)
            if eos_token_id is not None:
                for idx, token_id in enumerate(token_ids):
                    if token_id == eos_token_id:
                        end_offset = idx
                        break

            completion_token_ids = token_ids[:end_offset]
            text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=True)
            ret_texts.append(text)
            completion_lengths.append(len(completion_token_ids))

        sequences = torch.empty((len(envs), 0), dtype=torch.long)
        return sequences, ret_texts, prompt_lengths, completion_lengths

    def sample_completions_batch(
        self, envs: List[EnvBase]
    ) -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
        if self.engine == "vllm":
            return self._sample_completions_batch_vllm(envs)
        model = self.model
        return sample_completions_batch_hf(
            model=model,
            tokenizer=self.tokenizer,
            envs=envs,
            do_sample=self.do_sample,
            sampling_temperature=self.sampling_temperature,
            max_new_tokens=self.max_new_tokens,
            generation_args=self.generation_args,
        )

    def evaluate(self) -> dict[str, float]:
        dataset_len = len(self.dataset)
        if dataset_len == 0:
            raise ValueError("Eval dataset is empty.")

        num_batches = self.num_batches
        if num_batches is None:
            indices = list(range(dataset_len))
        else:
            max_samples = min(dataset_len, num_batches * self.batch_size)
            indices = list(range(max_samples))

        all_rewards = []
        all_completion_lengths = []
        first_env = None
        first_completions = None
        results_path = self.output_dir / "results.jsonl"

        was_training = None
        if self.engine == "hf":
            was_training = self.model.training
            self.model.eval()

        total_batches = (len(indices) + self.batch_size - 1) // self.batch_size
        progress = tqdm(
            range(0, len(indices), self.batch_size),
            total=total_batches,
            desc="Evaluating",
            disable=not self.show_progress or not sys.stderr.isatty(),
        )
        with results_path.open("w", encoding="utf-8") as results_file:
            for start in progress:
                batch_indices = indices[start : start + self.batch_size]
                env_batch = [self.dataset[i] for i in batch_indices]

                for env in env_batch:
                    if not isinstance(env, EnvBase):
                        raise TypeError("Dataset items must be EnvBase instances.")

                _, texts, _, completion_lens = self.sample_completions_batch(env_batch)

                rewards = [self._compute_reward(env, text) for env, text in zip(env_batch, texts)]
                all_rewards.extend(rewards)
                all_completion_lengths.extend(completion_lens)

                for env, text, reward in zip(env_batch, texts, rewards):
                    record = {
                        "prompt": env.prompt,
                        "answer": env.answer,
                        "generated_answer": text,
                        "reward": reward,
                    }
                    results_file.write(json.dumps(record, default=str) + "\n")

                if first_completions is None:
                    first_env = env_batch[0]
                    first_completions = texts[0]

        if self.engine == "hf" and was_training:
            self.model.train()

        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        reward_mean = float(rewards_tensor.mean().item())
        reward_std = float(rewards_tensor.std(unbiased=False).item())
        completion_tokens_mean = float(
            torch.tensor(all_completion_lengths, dtype=torch.float32).mean().item()
        )

        metrics = {
            "num_samples": float(len(all_rewards)),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "completion_tokens_mean": completion_tokens_mean,
        }

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
            logger.info("generated: %s", truncate_text(first_completions, self.max_log_chars))

        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if self._wandb is not None:
            self._wandb.log(metrics)
            artifact = self._wandb.Artifact("evaluation_results", type="evaluation")
            artifact.add_file(str(results_path))
            self._wandb.log_artifact(artifact)
            self._wandb.finish()

        return metrics
