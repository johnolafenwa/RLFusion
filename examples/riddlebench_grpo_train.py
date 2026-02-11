"""GRPO training on AI4Bharat RiddleBench.

A compact reasoning benchmark with short answers (sequences, seating, coding/decoding, blood relations).
"""

import logging

from rlfusion.datasets import RiddleBenchDataset
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def main() -> None:
    train_dataset = RiddleBenchDataset(
        train=True,
        max_samples=1_200,
        seed=123,
        train_split_ratio=0.9,
    )
    eval_dataset = RiddleBenchDataset(
        train=False,
        max_samples=256,
        seed=123,
        train_split_ratio=0.9,
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=train_dataset,
        num_steps=400,
        saving_steps=100,
        logging_steps=10,
        eval_steps=100,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        sampling_temperature=0.7,
        kl_penalty=0.02,
        output_dir="./outputs/grpo_riddlebench",
        generation_args={"top_p": 0.9},
        optimizer_args={"lr": 1e-5},
        group_size=4,
        ppo_steps=2,
        max_new_tokens=64,
        max_grad_norm=1.0,
        log_completions=False,
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
