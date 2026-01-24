"""Minimal AIME2025 evaluation run.

Expected: metrics.json and results.jsonl are written to the output directory.
"""

import logging

from rlfusion.datasets.aime import AIME2025
from rlfusion.evaluation.evaluator import Evaluator


def main() -> None:
    dataset = AIME2025()

    evaluator = Evaluator(
        model="Qwen/Qwen3-0.6B",
        dataset=dataset,
        output_dir="./outputs/aime2025_eval",
        num_batches=1,
        batch_size=3,
        max_new_tokens=2048,
        do_sample=False,
        enable_wandb=False,
        log_completions=True,
        max_log_chars=200,
        log_level=logging.INFO,
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()
