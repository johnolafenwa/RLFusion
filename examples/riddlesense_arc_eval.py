"""Evaluate a model on RiddleSense or ARC-Challenge with RLFusion Evaluator."""

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


CHOICES = ("A", "B", "C", "D", "E")
LETTER_PATTERN = re.compile(r"\b([ABCDE])\b")
PAREN_PATTERN = re.compile(r"\(([ABCDE])\)")


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
class MCQEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        if self.answer is None:
            return 0.0
        pred = extract_choice_letter(prediction)
        if pred is None:
            return 0.0
        return 1.0 if pred == str(self.answer).strip().upper() else 0.0


class RiddleSenseDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
        system_prompt: str = (
            "You are a careful assistant. Choose the single best option and respond with only one "
            "letter: A, B, C, D, or E."
        ),
    ) -> None:
        dataset = load_dataset("mlfoundations-dev/riddle_sense_converted", split=split)
        dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.dataset = dataset
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> MCQEnv:
        row = self.dataset[idx]
        question = str(row["question"])
        answer = str(row["answerKey"]).strip().upper()
        user_prompt = (
            f"{question}\n\n"
            "Return only the letter of the correct answer."
        )
        return MCQEnv(
            prompt=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            answer=answer,
        )


class ARCChallengeDataset(Dataset):
    def __init__(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
        seed: int = 42,
        system_prompt: str = (
            "You are a careful assistant. Choose the single best option and respond with only one "
            "letter."
        ),
    ) -> None:
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.dataset = dataset
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> MCQEnv:
        row = self.dataset[idx]
        question = str(row["question"])
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        option_lines = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Choices:\n{option_lines}\n\n"
            "Return only the letter of the correct answer."
        )
        answer = str(row["answerKey"]).strip().upper()
        return MCQEnv(
            prompt=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            answer=answer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on RiddleSense or ARC-Challenge with RLFusion Evaluator."
    )
    parser.add_argument("--model", type=str, required=True, help="HF model id or local checkpoint path.")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["riddlesense", "arc-challenge"],
        required=True,
        help="Benchmark to evaluate.",
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split override.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max generation tokens.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.benchmark == "riddlesense":
        split = args.split or "train"
        dataset = RiddleSenseDataset(split=split, max_samples=args.max_samples, seed=args.seed)
    else:
        split = args.split or "validation"
        dataset = ARCChallengeDataset(split=split, max_samples=args.max_samples, seed=args.seed)

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
    evaluator.tokenizer.padding_side = "left"
    metrics = evaluator.evaluate()
    print(json.dumps(metrics, indent=2))
    print(f"metrics_path={Path(args.output_dir) / 'metrics.json'}")
    print(f"results_path={Path(args.output_dir) / 'results.jsonl'}")


if __name__ == "__main__":
    main()
