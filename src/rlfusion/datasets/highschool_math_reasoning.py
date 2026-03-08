from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase


@dataclass
class HighSchoolMathReasoningEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        # This dataset is used for SFT and does not define task rewards.
        _ = prediction
        return 0.0


class HighSchoolMathReasoningSFTDataset(Dataset):
    """SFT adapter for johnolafenwa/highschool-math-reasoning."""

    def __init__(
        self,
        train: bool = True,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for HighSchoolMathReasoningSFTDataset. Install with: uv pip install datasets"
            ) from exc

        selected_split = split or ("train" if train else "test")
        if selected_split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        dataset = load_dataset("johnolafenwa/highschool-math-reasoning", split=selected_split)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset

    def __getitem__(self, index: int) -> HighSchoolMathReasoningEnv:
        row = self.dataset[index]
        messages = row.get("messages")
        if messages is None:
            raise ValueError("Dataset row missing 'messages' field.")
        if not isinstance(messages, list):
            raise ValueError("Dataset row 'messages' must be a list.")

        prompt: list[dict[str, object]] = []
        has_assistant = False
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict with role/content.")
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role in dataset: {role}")
            content = message.get("content", "")
            prompt.append({"role": str(role), "content": str(content)})
            if role == "assistant":
                has_assistant = True

        if not has_assistant:
            raise ValueError("Dataset row must include at least one assistant message.")

        return HighSchoolMathReasoningEnv(prompt=prompt)

    def __len__(self) -> int:
        return len(self.dataset)
