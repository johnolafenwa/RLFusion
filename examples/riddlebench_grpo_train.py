"""GRPO training on AI4Bharat RiddleBench.

A compact reasoning benchmark with short answers (sequences, seating, coding/decoding, blood relations).
"""

from __future__ import annotations

import argparse
import logging

from rlfusion.datasets import RiddleBenchDataset
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training on RiddleBench.")
    parser.add_argument("--model", type=str, required=True, help="Base model id or checkpoint path.")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo_riddlebench")
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--saving-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--kl-penalty", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-max-samples", type=int, default=1_200)
    parser.add_argument("--eval-max-samples", type=int, default=256)
    parser.add_argument("--train-split-ratio", type=float, default=0.9)
    parser.add_argument(
        "--task-types",
        type=str,
        default=None,
        help="Comma-separated RiddleBench task types, e.g. 'sequence tasks,seating task'.",
    )
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Enable accelerate-based multi-process training if launched via accelerate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_types = None
    if args.task_types:
        task_types = [item.strip() for item in args.task_types.split(",") if item.strip()]

    train_dataset = RiddleBenchDataset(
        train=True,
        max_samples=args.train_max_samples,
        seed=args.seed,
        train_split_ratio=args.train_split_ratio,
        task_types=task_types,
    )
    eval_dataset = RiddleBenchDataset(
        train=False,
        max_samples=args.eval_max_samples,
        seed=args.seed,
        train_split_ratio=args.train_split_ratio,
        task_types=task_types,
    )

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=train_dataset,
        num_steps=args.num_steps,
        saving_steps=args.saving_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        sampling_temperature=args.sampling_temperature,
        kl_penalty=args.kl_penalty,
        output_dir=args.output_dir,
        generation_args={"top_p": args.top_p},
        optimizer_args={"lr": args.lr},
        batch_size=args.batch_size,
        group_size=args.group_size,
        ppo_steps=args.ppo_steps,
        max_new_tokens=args.max_new_tokens,
        max_grad_norm=1.0,
        log_completions=False,
        log_level=logging.INFO,
        use_accelerate=args.use_accelerate,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()
