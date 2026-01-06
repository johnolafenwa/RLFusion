from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer


@dataclass
class IntellectMathEnv(EnvBase):
    max_error: float = 1.0

    def get_reward(self, prediction: str) -> float:
        if self.answer is None:
            return 0.0
        boxed = get_boxed_answer(prediction)
        if boxed is None:
            return 0.0
        return 1.0 if boxed.strip() == str(self.answer).strip() else 0.0


class IntellectMathDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        system_prompt: str = "Solve the problem and put the final answer in [result].",
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for IntellectMathDataset. Install with: uv pip install datasets"
            ) from exc

        dataset = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        total_len = len(dataset)
        split_idx = int(total_len * 0.8)
        if train:
            dataset = dataset.select(range(split_idx))
        else:
            dataset = dataset.select(range(split_idx, total_len))

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset
        self.system_prompt = system_prompt

    def __getitem__(self, index: int) -> IntellectMathEnv:
        row = self.dataset[index]
        question = row.get("question")
        answer = row.get("answer")
        if question is None:
            raise ValueError("Dataset row missing 'question' field.")

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        return IntellectMathEnv(prompt=prompt, answer=answer)

    def __len__(self) -> int:
        return len(self.dataset)
