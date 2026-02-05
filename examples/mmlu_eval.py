"""Run MMLU evaluation with the built-in RLFusion Evaluator.

This script formats MMLU multiple-choice questions as chat prompts and
computes exact-match accuracy over answer letters (A/B/C/D).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.evaluation.evaluator import Evaluator


CHOICES = ("A", "B", "C", "D")
LETTER_PATTERN = re.compile(r"\b([ABCD])\b")
PAREN_PATTERN = re.compile(r"\(([ABCD])\)")


def extract_choice_letter(text: str | None) -> Optional[str]:
    if text is None:
        return None
    normalized = text.strip().upper()
    if not normalized:
        return None
    if normalized[0] in CHOICES:
        return normalized[0]

    match = LETTER_PATTERN.search(normalized)
    if match:
        return match.group(1)

    match = PAREN_PATTERN.search(normalized)
    if match:
        return match.group(1)

    return None


@dataclass
class MMLUEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        if self.answer is None:
            return 0.0
        pred = extract_choice_letter(prediction)
        if pred is None:
            return 0.0
        return 1.0 if pred == str(self.answer).strip().upper() else 0.0


class MMLUDataset(Dataset):
    def __init__(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
        seed: int = 42,
        system_prompt: str = (
            "You are a careful assistant. Choose the single best option for each multiple-choice "
            "question and respond with only one letter: A, B, C, or D."
        ),
    ) -> None:
        dataset = load_dataset("cais/mmlu", "all", split=split)
        dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.dataset = dataset
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> MMLUEnv:
        row = self.dataset[idx]
        question = row["question"]
        options = row["choices"]
        answer_index = int(row["answer"])

        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(CHOICES, options))
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Choices:\n{option_lines}\n\n"
            "Return only the letter of the correct answer."
        )
        answer_letter = CHOICES[answer_index]

        return MMLUEnv(
            prompt=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            answer=answer_letter,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on MMLU with RLFusion Evaluator.")
    parser.add_argument(
        "--model",
        type=str,
        default="./ultrachat_sft_qwen3_4b/final",
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test", "dev"],
        help="MMLU split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max generation tokens.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/mmlu_eval",
        help="Directory for metrics/results artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MMLUDataset(split=args.split, max_samples=args.max_samples, seed=args.seed)
    evaluator = Evaluator(
        model=args.model,
        dataset=dataset,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        do_sample=False,
        generation_args={"use_cache": True},
        log_completions=True,
        max_log_chars=160,
        log_level=logging.INFO,
    )
    # Decoder-only models are typically evaluated with left padding.
    evaluator.tokenizer.padding_side = "left"
    metrics = evaluator.evaluate()
    print(json.dumps(metrics, indent=2))
    print(f"metrics_path={Path(args.output_dir) / 'metrics.json'}")
    print(f"results_path={Path(args.output_dir) / 'results.jsonl'}")


if __name__ == "__main__":
    main()
