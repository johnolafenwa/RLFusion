from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase


@dataclass
class UltraChatEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        # UltraChat is used as an SFT corpus and does not define task rewards.
        return 0.0


class UltraChatSFTDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        split: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for UltraChatSFTDataset. Install with: uv pip install datasets"
            ) from exc

        selected_split = split or ("train_sft" if train else "test_sft")
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=selected_split)

        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset
        self.system_prompt = system_prompt

    def __getitem__(self, index: int) -> UltraChatEnv:
        row = self.dataset[index]
        messages = row.get("messages")
        if messages is None:
            raise ValueError("Dataset row missing 'messages' field.")
        if not isinstance(messages, list):
            raise ValueError("Dataset row 'messages' must be a list.")

        prompt: list[dict[str, object]] = []
        if self.system_prompt is not None:
            prompt.append({"role": "system", "content": self.system_prompt})

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict with role/content.")
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role in dataset: {role}")
            content = message.get("content", "")
            prompt.append({"role": str(role), "content": str(content)})

        return UltraChatEnv(prompt=prompt)

    def __len__(self) -> int:
        return len(self.dataset)
