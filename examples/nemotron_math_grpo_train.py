"""GRPO training on NVIDIA Nemotron-RL-math-OpenMathReasoning.

Uses the built-in train/validation splits (~113k train, ~30k validation)
with boxed-answer exact-match reward.
"""

import logging

from rlfusion.datasets import NemotronMathDataset
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def main() -> None:
    train_dataset = NemotronMathDataset(
        train=True,
        max_samples=5_000,
        seed=123,
    )
    eval_dataset = NemotronMathDataset(
        train=False,
        max_samples=1_000,
        seed=123,
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=train_dataset,
        num_steps=500,
        num_epochs=1,
        saving_steps=100,
        logging_steps=10,
        eval_steps=100,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        sampling_temperature=0.7,
        kl_penalty=0.0,
        output_dir="./outputs/grpo_nemotron_math",
        generation_args={"top_p": 0.9},
        group_size=4,
        ppo_steps=1,
        max_new_tokens=512,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
