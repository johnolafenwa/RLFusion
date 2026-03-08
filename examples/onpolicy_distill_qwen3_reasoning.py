"""On-policy distillation on johnolafenwa/reasoning-rl with Qwen3.

Teacher:
  - Qwen/Qwen3-4B-Thinking-2507

Student:
  - Qwen/Qwen3-1.7B

This uses a short-answer math prompt so the student samples remain tractable in a
colocated training setup. Reward logging is only for monitoring; the training
objective is reverse KL from the teacher on student-generated completions.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer
from rlfusion.utils import get_boxed_answer

STUDENT_MODEL_ID = "Qwen/Qwen3-1.7B"
TEACHER_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
THINK_REWARD = 0.3
BOX_REWARD = 0.2
CORRECT_REWARD = 0.5
FORMAT_INSTRUCTION = (
    "Please reason step by step, keep your thinking very short, "
    "and put your final answer within \\boxed{}."
)


def _get_adamw8bit() -> object:
    try:
        from bitsandbytes.optim import AdamW8bit
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for this example. "
            "Install with: uv sync --extra gpu-train --extra vllm --extra dev --extra test"
        ) from exc
    return AdamW8bit


@dataclass
class ReasoningRLEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        if prediction is None or self.answer is None:
            return 0.0

        text = str(prediction)
        think_close = "</think>"
        reward = 0.0
        answer_text = text

        close_idx = text.find(think_close)
        if close_idx != -1:
            think_content = text[:close_idx]
            if think_content.replace("<think>", "").strip():
                reward += THINK_REWARD
                answer_text = text[close_idx + len(think_close) :]

        boxed = get_boxed_answer(answer_text)
        if boxed is None:
            return reward

        reward += BOX_REWARD
        if boxed == str(self.answer):
            reward += CORRECT_REWARD
        return reward


class ReasoningRLDataset(Dataset):
    """Adapter for johnolafenwa/reasoning-rl."""

    def __init__(
        self,
        split: str,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for ReasoningRLDataset. Install with: uv pip install datasets"
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

    def __getitem__(self, index: int) -> ReasoningRLEnv:
        row = self.dataset[index]
        if row.get("problem") is None:
            raise ValueError("Dataset row missing required field: problem.")
        if row.get("answer") is None:
            raise ValueError("Dataset row missing required field: answer.")

        return ReasoningRLEnv(
            prompt=[
                {
                    "role": "user",
                    "content": f"{row['problem']}\n\n{FORMAT_INSTRUCTION}",
                }
            ],
            answer=str(row["answer"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run on-policy distillation on reasoning-rl with Qwen/Qwen3-4B-Thinking-2507 "
            "as teacher and Qwen/Qwen3-1.7B as student."
        )
    )
    parser.add_argument("--model", type=str, default=STUDENT_MODEL_ID)
    parser.add_argument("--teacher-model", type=str, default=TEACHER_MODEL_ID)
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        default=STUDENT_MODEL_ID,
        help="Tokenizer to use for vLLM student rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/qwen3_1_7b_from_qwen3_4b_thinking_onpolicy_reasoning",
    )
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
    parser.add_argument("--ppo-steps", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sampling-temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--use-accelerate", action="store_true", help="Enable Accelerate multi-process training")
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--max-log-chars", type=int, default=320)
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use colocated vLLM generation. Defaults to enabled on CUDA and disabled otherwise.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.2,
        help="vLLM GPU memory utilization (0-1) for the colocated student rollout engine.",
    )
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
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM torch.compile/CUDA-graph startup for this colocated setup.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=2048,
        help="Cap the vLLM context length for this short-answer task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dataset = ReasoningRLDataset(split="train", max_samples=args.train_max_samples, seed=args.seed)
    eval_dataset = ReasoningRLDataset(split="test", max_samples=args.test_max_samples, seed=args.seed)

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
        "enforce_eager": args.vllm_enforce_eager,
        "tokenizer": args.base_tokenizer,
    }

    trainer = OnPolicyDistillationTrainer(
        model=args.model,
        teacher_model=args.teacher_model,
        train_dataset=train_dataset,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        saving_steps=saving_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        sampling_temperature=args.sampling_temperature,
        output_dir=args.output_dir,
        generation_args={
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
        },
        optimizer=_get_adamw8bit(),
        optimizer_args={"lr": args.learning_rate},
        batch_size=args.batch_size,
        ppo_steps=args.ppo_steps,
        clip_eps=args.clip_eps,
        max_new_tokens=args.max_new_tokens,
        max_grad_norm=args.max_grad_norm,
        log_completions=args.log_completions,
        max_log_chars=args.max_log_chars,
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
