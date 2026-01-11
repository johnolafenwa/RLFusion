import logging

from rlfusion.datasets.rlvr import MathDataset
from rlfusion.evaluation.evaluator import Evaluator
from rlfusion.trainers.grpo_trainer import GRPOTrainer


def main() -> None:
    dataset = MathDataset(
        num_samples=64,
        min_val=0,
        max_val=50,
        operand="add",
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
        num_steps=5,
        saving_steps=5,
        logging_steps=1,
        eval_steps=1,
        evaluator=Evaluator(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dataset=dataset,
            output_dir="./outputs/grpo_math/eval",
            num_batches=1,
            engine="hf",
            max_new_tokens=64,
            batch_size=1,
        ),
        enable_wandb=False,
        sampling_temperature=0.7,
        kl_penalty=0.0,
        output_dir="./outputs/grpo_math",
        group_size=2,
        ppo_steps=1,
        max_new_tokens=64,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    trainer.train()


if __name__ == "__main__":
    main()
