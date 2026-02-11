from dataclasses import dataclass
from typing import Optional, Sequence

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer


def _normalize_answer(text: str) -> str:
    normalized = " ".join(text.strip().split()).casefold()
    return normalized.rstrip(".,;:!?")


@dataclass
class RiddleBenchEnv(EnvBase):
    def get_reward(self, prediction: str | None) -> float:
        if prediction is None or self.answer is None:
            return 0.0

        final_answer = get_boxed_answer(prediction)
        if final_answer is None:
            final_answer = prediction

        return 1.0 if _normalize_answer(final_answer) == _normalize_answer(str(self.answer)) else 0.0


class RiddleBenchDataset(Dataset):
    """AI4Bharat RiddleBench dataset with a local train/eval split.

    Dataset card:
    https://huggingface.co/datasets/ai4bharat/RiddleBench
    """

    def __init__(
        self,
        train: bool = True,
        system_prompt: str = (
            "Solve the puzzle. You may reason internally, but reply with only the final answer."
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

    def __getitem__(self, index: int) -> RiddleBenchEnv:
        row = self.dataset[index]
        question = row.get("question")
        answer = row.get("answer")

        if question is None:
            raise ValueError("Dataset row missing 'question' field.")

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        return RiddleBenchEnv(prompt=prompt, answer=answer)

    def __len__(self) -> int:
        return len(self.dataset)
