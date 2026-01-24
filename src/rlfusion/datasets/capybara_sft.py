from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer


@dataclass
class CapyBaraEnv(EnvBase):

    def get_reward(self, prediction: str) -> float:
        return None


class CapyBaraSFTDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        system_prompt: str = "You are a helpful assistant",
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for CapyBaraSFTDataset. Install with: uv pip install datasets"
            ) from exc

        dataset = load_dataset("trl-lib/Capybara", split="train" if train else "test")
        
        self.dataset = dataset
        self.system_prompt = system_prompt

    def __getitem__(self, index: int) -> CapyBaraEnv:
        row = self.dataset[index]
        messages  = row.get("messages")
        if messages is None:
            raise ValueError("Dataset row missing 'messages' field.")

        prompt = [
            {"role": "system", "content": self.system_prompt},
        ] + messages

        return CapyBaraEnv(prompt=prompt)

    def __len__(self) -> int:
        return len(self.dataset)
