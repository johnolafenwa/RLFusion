"""GRPO on johnolafenwa/reasoning-rl with Qwen/Qwen3-4B-Instruct-2507.

This model card describes Qwen3-4B-Instruct-2507 as a non-thinking model, so
the reward uses boxed-answer exact match instead of requiring `<think>` tags.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.trainers.grpo_trainer import GRPOTrainer
from rlfusion.utils import get_boxed_answer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BOXED_INSTRUCTION = "Please reason step by step in at most three short steps, and put your final answer within \\boxed{}."


@dataclass
class QwenReasoningEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        if prediction is None or self.answer is None:
            return 0.0

        boxed = get_boxed_answer(str(prediction))
        if boxed is None:
            return 0.0
        return 1.0 if boxed == str(self.answer) else 0.0


class QwenReasoningRLDataset(Dataset):
    """Adapter for johnolafenwa/reasoning-rl with a boxed-answer prompt."""

    def __init__(
        self,
        split: str,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for QwenReasoningRLDataset. Install with: uv pip install datasets"
            ) from exc

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        dataset = load_dataset("johnolafenwa/reasoning-rl", split=split)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> QwenReasoningEnv:
        row = self.dataset[index]
        if row.get("problem") is None:
            raise ValueError("Dataset row missing required field: problem.")
        if row.get("answer") is None:
            raise ValueError("Dataset row missing required field: answer.")

        prompt_text = f"{str(row['problem'])}\n\n{BOXED_INSTRUCTION}"
        answer = str(row["answer"])
        return QwenReasoningEnv(
            prompt=[{"role": "user", "content": prompt_text}],
            answer=answer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training on reasoning-rl with Qwen3-4B-Instruct-2507.")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_qwen3_4b_instruct_2507")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--saving-steps", type=int, default=50)
    parser.add_argument(
        "--save-final-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final model checkpoint.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-samples", type=int, default=5_000)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--use-accelerate", action="store_true", help="Enable Accelerate multi-process training")
    parser.add_argument("--max-error", type=float, default=100.0)
    parser.add_argument("--invalid-penalty", type=float, default=1.0)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--max-log-chars", type=int, default=320)
    parser.add_argument("--use-base-kl", action="store_true", help="Enable KL penalty with a reference model.")
    parser.add_argument("--kl-penalty", type=float, default=0.0)
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use colocated vLLM generation. Defaults to enabled on CUDA and disabled otherwise.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3, help="vLLM GPU memory utilization (0-1).")
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. Keep this at 1 when using --use-accelerate.",
    )
    parser.add_argument(
        "--vllm-enable-sleep",
        action="store_true",
        help="Enable vLLM sleep mode between generations to free GPU memory during training.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=2048,
        help=(
            "Cap the vLLM context length for colocated RL training. "
            "The model defaults to 262144 tokens, which wastes KV cache on this short-answer task."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dataset = QwenReasoningRLDataset(split="train", max_samples=args.train_max_samples, seed=args.seed)
    eval_dataset = QwenReasoningRLDataset(split="test", max_samples=args.test_max_samples, seed=args.seed)

    if args.num_epochs is not None:
        steps_for_checkpoint_interval = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs
    else:
        steps_for_checkpoint_interval = args.num_steps
    saving_steps = args.saving_steps
    if args.save_final_only:
        saving_steps = steps_for_checkpoint_interval + 1

    vllm_args = {
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "max_model_len": args.vllm_max_model_len,
    }

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=train_dataset,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        saving_steps=saving_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_dataset=eval_dataset,
        sampling_temperature=args.sampling_temperature,
        kl_penalty=args.kl_penalty if args.use_base_kl else 0.0,
        output_dir=args.output_dir,
        generation_args={
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        },
        optimizer_args={"lr": args.learning_rate},
        batch_size=args.batch_size,
        group_size=args.group_size,
        ppo_steps=args.ppo_steps,
        clip_eps=args.clip_eps,
        max_new_tokens=args.max_new_tokens,
        max_grad_norm=args.max_grad_norm,
        log_completions=args.log_completions,
        max_log_chars=args.max_log_chars,
        max_error=args.max_error,
        invalid_penalty=args.invalid_penalty,
        enable_wandb=False,
        seed=args.seed,
        use_accelerate=args.use_accelerate,
        use_vllm=args.use_vllm,
        vllm_args=vllm_args,
        vllm_enable_sleep=args.vllm_enable_sleep,
        log_level=args.log_level,
    )

    trainer.train()


if __name__ == "__main__":
    main()
