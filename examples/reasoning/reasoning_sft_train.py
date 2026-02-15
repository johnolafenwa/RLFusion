"""SFT on johnolafenwa/reasoning-sft using native train/test splits."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.trainers import SFTTrainer


logger = logging.getLogger(__name__)


@dataclass
class ReasoningSFTEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        return 0.0


class ReasoningSFTDataset(Dataset):
    """Adapter for johnolafenwa/reasoning-sft input/output rows."""

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
                "datasets is required for ReasoningSFTDataset. Install with: uv pip install datasets"
            ) from exc

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        dataset = load_dataset("johnolafenwa/reasoning-sft", split=split)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, index: int) -> ReasoningSFTEnv:
        row = self.dataset[index]
        user_input = row.get("input")
        assistant_output = row.get("output")
        prompt = [{"role": "user", "content": user_input}]
        answer = assistant_output

        return ReasoningSFTEnv(prompt=prompt, answer=answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SFT on johnolafenwa/reasoning-sft."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--output-dir", type=str, default="./outputs/reasoning/reasoning_sft")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--test-max-samples", type=int, default=None)
    parser.add_argument("--saving-steps", type=int, default=100)
    parser.add_argument(
        "--save-final-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final model checkpoint.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument("--use-accelerate", action="store_true", help="Enable Accelerate multi-process training")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logger.setLevel(args.log_level)

    train_dataset = ReasoningSFTDataset(
        split="train",
        max_samples=args.train_max_samples,
        seed=args.seed,
    )
    eval_dataset = ReasoningSFTDataset(
        split="test",
        max_samples=args.test_max_samples,
        seed=args.seed,
    )

    if args.num_epochs is not None:
        steps_for_checkpoint_interval = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs
    else:
        steps_for_checkpoint_interval = args.num_steps
    saving_steps = args.num_steps + 1 if args.save_final_only else args.saving_steps
    if args.save_final_only and args.num_epochs is not None:
        saving_steps = steps_for_checkpoint_interval + 1

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        saving_steps=saving_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        max_seq_len=args.max_seq_len,
        optimizer_args={"lr": args.lr},
        output_dir=args.output_dir,
        seed=args.seed,
        use_accelerate=args.use_accelerate,
        log_level=args.log_level,
    )
    if args.gradient_checkpointing:
        checkpointing_enable = getattr(trainer.model, "gradient_checkpointing_enable", None)
        if callable(checkpointing_enable):
            checkpointing_enable()
        else:
            logging.getLogger(__name__).warning(
                "gradient checkpointing requested, but model does not expose gradient_checkpointing_enable()."
            )
    trainer.train()


if __name__ == "__main__":
    main()
