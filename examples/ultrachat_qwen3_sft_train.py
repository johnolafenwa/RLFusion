"""UltraChat SFT training script for Qwen3-class base models."""

from __future__ import annotations

import argparse

from rlfusion.datasets.ultrachat_sft import UltraChatSFTDataset
from rlfusion.trainers import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT on UltraChat with RLFusion.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./outputs/ultrachat_sft_qwen3_0_6b")
    parser.add_argument("--num-steps", type=int, default=1_000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=20_000)
    parser.add_argument("--eval-max-samples", type=int, default=2_000)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--saving-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Enable accelerate-based multi-process training if launched via accelerate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dataset = UltraChatSFTDataset(
        train=True,
        max_samples=args.train_max_samples,
        seed=args.seed,
    )
    eval_dataset = UltraChatSFTDataset(
        train=False,
        max_samples=args.eval_max_samples,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        saving_steps=args.saving_steps,
        logging_steps=args.logging_steps,
        max_seq_len=args.max_seq_len,
        optimizer_args={"lr": args.lr},
        output_dir=args.output_dir,
        seed=args.seed,
        use_accelerate=args.use_accelerate,
    )
    trainer.train()


if __name__ == "__main__":
    main()
