from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer


@dataclass
class NemotronMathEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        if self.answer is None:
            return 0.0
        boxed = get_boxed_answer(prediction)
        if boxed is None:
            return 0.0
        return 1.0 if boxed.strip() == str(self.answer).strip() else 0.0


class NemotronMathDataset(Dataset):
    """NVIDIA Nemotron-RL-math-OpenMathReasoning dataset.

    A collection of ~113k math problems sourced from AoPS forums,
    designed for reinforcement learning with verifiable rewards.

    See: https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning
    """

    def __init__(
        self,
        train: bool = True,
        system_prompt: str = "Solve the following math problem. Make sure to put the answer (and only the answer) inside \\boxed{}.",
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for NemotronMathDataset. Install with: uv pip install datasets"
            ) from exc

        split = "train" if train else "validation"
        dataset = load_dataset(
            "nvidia/Nemotron-RL-math-OpenMathReasoning", split=split
        )

        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset
        self.system_prompt = system_prompt

    def __getitem__(self, index: int) -> NemotronMathEnv:
        row = self.dataset[index]
        question = row.get("question")
        answer = row.get("expected_answer")
        if question is None:
            raise ValueError("Dataset row missing 'question' field.")

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        return NemotronMathEnv(prompt=prompt, answer=answer)

    def __len__(self) -> int:
        return len(self.dataset)
