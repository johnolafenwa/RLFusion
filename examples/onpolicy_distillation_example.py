"""Minimal on-policy distillation run on the toy math dataset.

Expected: logs show reverse_kl and loss values each step.
"""

import logging

from rlfusion.datasets.rlvr import MathDataset
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


def main() -> None:
    dataset = MathDataset(
        num_samples=64,
        min_val=0,
        max_val=50,
        operand="add",
    )

    trainer = OnPolicyDistillationTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
        train_dataset=dataset,
        num_steps=5,
        saving_steps=5,
        logging_steps=1,
        eval_steps=1,
        eval_dataset=dataset,
        enable_wandb=False,
        sampling_temperature=0.7,
        output_dir="./outputs/onpolicy_distill_math",
        batch_size=2,
        max_new_tokens=64,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    # Expected: reverse_kl and loss stats are logged each step.
    trainer.train()


if __name__ == "__main__":
    main()
