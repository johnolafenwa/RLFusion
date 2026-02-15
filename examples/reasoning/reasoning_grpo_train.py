"""GRPO on johnolafenwa/reasoning-rl with strict reasoning-format reward.

Expected answer format:
  <think>...</think> \boxed{answer}

Both conditions are required:
1) a non-empty <think>...</think> block at the beginning (case-sensitive tags),
2) exact boxed-match to the dataset answer on the text after </think>.
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


@dataclass
class ReasoningRLEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        if prediction is None or self.answer is None:
            return 0.0

        text = str(prediction)
        think_open = "<think>"
        think_close = "</think>"

        if not text.startswith(think_open):
            return 0.0

        close_idx = text.find(think_close)
        if close_idx == -1:
            return 0.0

        think_content = text[len(think_open) : close_idx]
        if not think_content.strip():
            return 0.0

        answer_text = text[close_idx + len(think_close) :]
        if not answer_text.strip():
            return 0.0

        boxed = get_boxed_answer(answer_text)
        if boxed is None:
            return 0.0

        return 1.0 if boxed == str(self.answer) else 0.0


class ReasoningRLDataset(Dataset):
    """Adapter for johnolafenwa/reasoning-rl."""

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
        prompt_text = str(row["problem"])
        answer = str(row["answer"])

        return ReasoningRLEnv(
            prompt=[
                {"role": "user", "content": prompt_text}
            ],
            answer=answer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training on johnolafenwa/reasoning-rl.")
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default="./outputs/reasoning/reasoning_sft/final",
        help="Path to SFT checkpoint to use as GRPO base.",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/reasoning/reasoning_grpo")
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
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
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

    trainer = GRPOTrainer(
        model=args.sft_checkpoint,
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
        generation_args={"top_p": args.top_p},
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
        log_level=args.log_level,
    )

    trainer.train()


if __name__ == "__main__":
    main()
