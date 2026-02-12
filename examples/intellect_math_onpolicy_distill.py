"""On-policy distillation on IntellectMath with a held-out eval split.

Expected: reverse_kl and loss stats in logs.
"""

import logging

from rlfusion.datasets import IntellectMathDataset
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


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

    trainer = OnPolicyDistillationTrainer(
        model="Qwen/Qwen3-4B-Instruct-2507",
        teacher_model="Qwen/Qwen3-8B",
        train_dataset=train_dataset,
        num_steps=5,
        num_epochs=1,
        saving_steps=5,
        logging_steps=1,
        eval_steps=1,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        sampling_temperature=0.7,
        output_dir="./outputs/onpolicy_distill_intellect_math",
        generation_args={"top_p": 0.9},
        batch_size=2,
        ppo_steps=1,
        clip_eps=0.2,
        max_new_tokens=128,
        max_grad_norm=1.0,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
