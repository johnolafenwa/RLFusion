import re
from dataclasses import dataclass
from typing import Optional, Sequence

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer


def _normalize_answer(text: str) -> str:
    normalized = " ".join(text.strip().split())
    normalized = normalized.strip("`'\"")
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = normalized.rstrip(".,;:!?")
    return normalized.casefold()


_CHOICE_START_PATTERN = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|option|choice)?(?:\s+is)?\s*[:\-]?\s*\(?([A-Ea-e])\)?(?:[\s\.\):,\-]|$)",
    flags=re.IGNORECASE,
)
_VALUE_START_PATTERN = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|option|choice)?(?:\s+is)?\s*[:\-]?\s*(.+)$",
    flags=re.IGNORECASE,
)


def _is_choice_answer(answer: str) -> bool:
    return len(answer) == 1 and answer.upper() in {"A", "B", "C", "D", "E"}


def _extract_choice_letter(text: str) -> Optional[str]:
    match = _CHOICE_START_PATTERN.match(text)
    if match:
        return match.group(1).upper()
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in {"A", "B", "C", "D", "E"}:
        return stripped.upper()
    return None


def _extract_value_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    match = _VALUE_START_PATTERN.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _contains_standalone_text(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(needle)}(?![0-9A-Za-z])")
    return bool(pattern.search(haystack))


def _prediction_candidates(prediction: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Optional[str]) -> None:
        if candidate is None:
            return
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            return
        seen.add(stripped)
        candidates.append(stripped)

    _add(get_boxed_answer(prediction))
    _add(prediction)
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    if lines:
        _add(lines[0])
        _add(lines[-1])

    return candidates


@dataclass
class RiddleBenchEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        if prediction is None or self.answer is None:
            return 0.0

        normalized_answer = _normalize_answer(str(self.answer))
        if not normalized_answer:
            return 0.0

        candidates = _prediction_candidates(prediction)
        if _is_choice_answer(normalized_answer):
            answer_choice = normalized_answer.upper()
            for candidate in candidates:
                choice = _extract_choice_letter(candidate)
                if choice == answer_choice:
                    return 1.0
            return 0.0

        for candidate in candidates:
            normalized_candidate = _normalize_answer(_extract_value_text(candidate))
            if normalized_candidate == normalized_answer:
                return 1.0

        normalized_prediction = _normalize_answer(prediction)
        if _contains_standalone_text(normalized_prediction, normalized_answer):
            return 1.0
        return 0.0


class RiddleBenchDataset(Dataset):
    """AI4Bharat RiddleBench dataset with a local train/eval split.

    Dataset card:
    https://huggingface.co/datasets/ai4bharat/RiddleBench
    """

    def __init__(
        self,
        train: bool = True,
        system_prompt: str = (
            "Solve the puzzle. Return only the final answer in \\boxed{...} format."
        ),
        answer_instruction: str = (
            "Return exactly one boxed answer: \\boxed{D} for options or \\boxed{5435} for values. "
            "No explanation."
        ),
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
        train_split_ratio: float = 0.9,
        task_types: Optional[Sequence[str]] = None,
    ):
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be >= 1 or None.")
        if not 0.0 < train_split_ratio < 1.0:
            raise ValueError("train_split_ratio must be between 0 and 1.")

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for RiddleBenchDataset. Install with: uv pip install datasets"
            ) from exc

        dataset = load_dataset("ai4bharat/RiddleBench", split="train")

        if task_types is not None:
            allowed_types = {item.strip() for item in task_types if item.strip()}
            if not allowed_types:
                raise ValueError("task_types must include at least one non-empty type.")

            selected_indices = [
                idx
                for idx in range(len(dataset))
                if str(dataset[idx].get("type", "")).strip() in allowed_types
            ]
            if not selected_indices:
                raise ValueError("No rows matched task_types.")
            dataset = dataset.select(selected_indices)

        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        total_len = len(dataset)
        if total_len < 2:
            raise ValueError("Dataset must contain at least 2 rows after filtering.")

        split_idx = int(total_len * train_split_ratio)
        if split_idx <= 0 or split_idx >= total_len:
            raise ValueError("train_split_ratio produced an empty split.")

        if train:
            dataset = dataset.select(range(split_idx))
        else:
            dataset = dataset.select(range(split_idx, total_len))

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset
        self.system_prompt = system_prompt
        self.answer_instruction = answer_instruction

    def __getitem__(self, index: int) -> RiddleBenchEnv:
        row = self.dataset[index]
        question = row.get("question")
        answer = row.get("answer")

        if question is None:
            raise ValueError("Dataset row missing 'question' field.")

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n{self.answer_instruction}"
                    if self.answer_instruction
                    else question
                ),
            },
        ]
        return RiddleBenchEnv(prompt=prompt, answer=answer)

    def __len__(self) -> int:
        return len(self.dataset)
