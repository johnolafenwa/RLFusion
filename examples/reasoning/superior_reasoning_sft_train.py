"""SFT on Superior Reasoning SFT (stage1 only).

This script reads Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b (split `stage1`),
splits it 90/10 into local train/test shards, and runs the SFT trainer.
"""

from __future__ import annotations

import argparse
import math
import logging
from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.trainers import SFTTrainer


@dataclass
class SuperiorReasoningSFTEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        # SFT-only dataset; no task reward.
        return 0.0


class SuperiorReasoningSFTDataset(Dataset):
    """Adapter for Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b.

    The dataset is expected to use `stage1` and contain either a `messages` column
    (list of chat messages) or standard text columns (instruction/question + output).
    """

    def __init__(
        self,
        train: bool = True,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
        split_ratio: float = 0.9,
        system_prompt: Optional[str] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for SuperiorReasoningSFTDataset. Install with: uv pip install datasets"
            ) from exc

        if not 0.0 < split_ratio < 1.0:
            raise ValueError("split_ratio must be between 0 and 1.")

        dataset = load_dataset(
            "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", split="stage1"
        )

        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        total_len = len(dataset)
        split_idx = int(total_len * split_ratio)
        if total_len >= 2:
            if split_idx <= 0:
                split_idx = 1
            elif split_idx >= total_len:
                split_idx = total_len - 1
        else:
            split_idx = 1 if train else 0

        if split_idx < 0 or split_idx > total_len:
            raise ValueError("Invalid split generated from stage1 dataset length.")

        if train:
            dataset = dataset.select(range(0, split_idx))
        else:
            dataset = dataset.select(range(split_idx, total_len))

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _normalize_messages(messages: Any) -> list[dict[str, str]]:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list.")

        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict.")

            role = message.get("role")
            if role is None:
                role = message.get("from") or message.get("speaker")
            if role is None:
                raise ValueError("Message must include role (role/from/speaker).")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role in message: {role}")

            content = message.get("content")
            if content is None:
                content = message.get("value")
            if content is None:
                content = message.get("text")

            if content is None:
                content = ""

            normalized.append({"role": str(role), "content": str(content)})

        return normalized

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    def __getitem__(self, index: int) -> SuperiorReasoningSFTEnv:
        row = self.dataset[index]

        messages = row.get("messages")
        answer = None

        if messages is not None:
            messages = self._normalize_messages(messages)

            # If this appears to be turn-based data, use the final assistant turn as answer.
            if messages:
                last_message = messages[-1]
                if last_message["role"] == "assistant":
                    answer = last_message["content"]
                    messages = messages[:-1]
            prompt = messages
        else:
            # Fallback for non-conversation-shaped rows.
            user_parts: list[str] = []
            for key in ("instruction", "question", "prompt", "input", "query"):
                value = row.get(key)
                if value not in {None, ""}:
                    user_parts.append(self._as_text(value))

            user_content = "\n".join(part for part in user_parts if part)
            if not user_content:
                raise ValueError("Dataset row missing a usable prompt field (e.g., messages/instruction/question/prompt/input).")

            answer = row.get("output")
            if answer is None:
                answer = row.get("response")
            if answer is None:
                answer = row.get("completion")
            if answer is None:
                answer = row.get("answer")

            prompt = [{"role": "user", "content": user_content}]

        if self.system_prompt is not None:
            prompt = [{"role": "system", "content": self.system_prompt}] + prompt

        if answer is None:
            # No explicit response field. Keep answer unset for pure prompt completion datasets.
            return SuperiorReasoningSFTEnv(prompt=prompt)

        return SuperiorReasoningSFTEnv(prompt=prompt, answer=self._as_text(answer))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SFT on stage1 of Superior Reasoning SFT dataset."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--output-dir", type=str, default="./outputs/reasoning/superior_reasoning_sft")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--saving-steps", type=int, default=100)
    parser.add_argument(
        "--save-final-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only the final model checkpoint.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--use-accelerate", action="store_true", help="Enable Accelerate multi-process training")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dataset = SuperiorReasoningSFTDataset(
        train=True,
        max_samples=args.train_max_samples,
        seed=args.seed,
        split_ratio=args.split_ratio,
        system_prompt=args.system_prompt,
    )

    if args.num_epochs is not None:
        steps_for_checkpoint_interval = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs
    else:
        steps_for_checkpoint_interval = args.num_steps
    saving_steps = args.num_steps + 1 if args.save_final_only else args.saving_steps
    if args.save_final_only and args.num_epochs is not None:
        saving_steps = steps_for_checkpoint_interval + 1

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=train_dataset,
        eval_dataset=None,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        saving_steps=saving_steps,
        logging_steps=args.logging_steps,
        max_seq_len=args.max_seq_len,
        optimizer_args={"lr": args.lr},
        output_dir=args.output_dir,
        seed=args.seed,
        use_accelerate=args.use_accelerate,
        log_level=args.log_level,
    )
    trainer.train()


if __name__ == "__main__":
    main()
