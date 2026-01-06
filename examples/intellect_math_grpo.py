import logging

from rlfusion.datasets import IntellectMathDataset
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def main() -> None:
    train_dataset = IntellectMathDataset(
        train=True,
        max_samples=64,
        seed=123,
    )
    eval_dataset = IntellectMathDataset(
        train=False,
        max_samples=32,
        seed=123,
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=5,
        saving_steps=5,
        logging_steps=1,
        enable_wandb=False,
        sampling_temperature=0.7,
        kl_penalty=0.0,
        output_dir="./outputs/grpo_intellect_math",
        group_size=2,
        ppo_steps=1,
        max_new_tokens=128,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
