"""Minimal SFT run.

Expected:
- training logs print `loss=...` every step
- eval logs `ce_loss` and `perplexity` by default
- set `eval_sample_completions=True` to also log reward metrics (`reward_mean`, `reward_std`)
"""

import logging

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.datasets.rlvr import MathDataset
from rlfusion.trainers.sft_trainer import SFTTrainer


class ToySFTDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            EnvBase(
                prompt=[
                    {"role": "system", "content": "Answer briefly and politely."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                answer="The answer is 4.",
            ),
            EnvBase(
                prompt=[
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Name a primary color."},
                ],
                answer="Red.",
            ),
        ]

    def __getitem__(self, index: int) -> EnvBase:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def main() -> None:
    dataset = ToySFTDataset()
    eval_dataset = MathDataset(
        num_samples=64,
        min_val=0,
        max_val=50,
        operand="add",
    )

    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
        num_steps=100,
        num_epochs=1,
        batch_size=2,
        saving_steps=2,
        logging_steps=1,
        eval_steps=10,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        output_dir="./outputs/sft_example",
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
