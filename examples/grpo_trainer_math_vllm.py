import logging

from rlfusion.datasets import IntellectMathDataset
from rlfusion.datasets.aime import AIME2025
from rlfusion.evaluation.evaluator import Evaluator
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def main() -> None:
    train_dataset = IntellectMathDataset(
        train=True,
        max_samples=64,
        seed=123,
    )
    eval_dataset = AIME2025()
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=train_dataset,
        num_steps=5,
        saving_steps=5,
        logging_steps=1,
        eval_steps=1,
        eval_dataset=eval_dataset,
        enable_wandb=False,
        sampling_temperature=0.7,
        kl_penalty=0.0,
        output_dir="./outputs/grpo_intellect_math_vllm",
        group_size=2,
        ppo_steps=1,
        max_new_tokens=64,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
        use_accelerate=True,
    )

    trainer.train()
    if trainer._is_main_process():
        evaluator = Evaluator(
            model="./outputs/grpo_intellect_math_vllm/final",
            dataset=eval_dataset,
            output_dir="./outputs/grpo_intellect_math_vllm/eval",
            num_batches=1,
            engine="vllm",
            vllm_args={"tensor_parallel_size": 1},
            max_new_tokens=64,
            batch_size=1,
        )
        evaluator.evaluate()


if __name__ == "__main__":
    main()
