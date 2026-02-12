"""Evaluate a checkpoint on AIME2025 with the RLFusion evaluator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rlfusion.datasets.aime import AIME2025
from rlfusion.evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on AIME2025.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B-Base",
        help="HF model id or local checkpoint path (base/sft/grpo).",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/reasoning/aime2025_eval")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--log-completions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = AIME2025(
        cache_dir=args.cache_dir,
    )

    evaluator = Evaluator(
        model=args.model,
        dataset=dataset,
        output_dir=args.output_dir,
        enable_wandb=False,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        do_sample=args.do_sample,
        sampling_temperature=args.sampling_temperature,
        num_batches=args.num_batches,
        log_completions=args.log_completions,
        show_progress=args.show_progress,
        log_level=args.log_level,
        seed=args.seed,
    )

    # Decoder-only checkpoints generally require left padding during generation.
    evaluator.tokenizer.padding_side = "left"

    metrics = evaluator.evaluate()
    print(json.dumps(metrics, indent=2))
    print(f"metrics_path={Path(args.output_dir) / 'metrics.json'}")
    print(f"results_path={Path(args.output_dir) / 'results.jsonl'}")


if __name__ == "__main__":
    main()
