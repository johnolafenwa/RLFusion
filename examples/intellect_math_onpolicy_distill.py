import logging

from rlfusion.datasets import IntellectMathDataset
from rlfusion.evaluation.evaluator import Evaluator
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
        saving_steps=5,
        logging_steps=1,
        eval_steps=1,
        evaluator=Evaluator(
            model="Qwen/Qwen3-4B-Instruct-2507",
            dataset=eval_dataset,
            output_dir="./outputs/onpolicy_distill_intellect_math/eval",
            num_batches=1,
            engine="hf",
            generation_args={"top_p": 0.9},
            max_new_tokens=128,
            batch_size=1,
        ),
        enable_wandb=False,
        sampling_temperature=0.7,
        output_dir="./outputs/onpolicy_distill_intellect_math",
        generation_args={"top_p": 0.9},
        batch_size=2,
        max_new_tokens=128,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
