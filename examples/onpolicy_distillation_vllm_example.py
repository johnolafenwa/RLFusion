import logging

from rlfusion.datasets import IntellectMathDataset
from rlfusion.datasets.aime import AIME2025
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


def main() -> None:
    train_dataset = IntellectMathDataset(
        train=True,
        max_samples=64,
        seed=123,
    )
    eval_dataset = AIME2025()

    trainer = OnPolicyDistillationTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=5,
        saving_steps=5,
        logging_steps=1,
        enable_wandb=False,
        sampling_temperature=0.7,
        output_dir="./outputs/onpolicy_distill_intellect_math_vllm",
        batch_size=2,
        max_new_tokens=64,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
        engine="vllm",
        vllm_args={
            "tensor_parallel_size": 1,
        },
    )

    trainer.train()


if __name__ == "__main__":
    main()
