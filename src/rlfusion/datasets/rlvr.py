import math
import random
from dataclasses import dataclass
from typing import Callable

from torch.utils.data import Dataset

from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer

class MathDataEnv(EnvBase):

    max_error = 1000000

    def get_reward(self, prediction) -> float:

        if prediction is not None:
           
            boxed_result = get_boxed_answer(prediction)
           
            numeric_value = None
            if boxed_result is not None:
                try:
                    numeric_value = float(boxed_result)
                except Exception:
                    numeric_value = None
            if numeric_value is None or not math.isfinite(numeric_value):
               
                return -self.max_error
            else:
                if not isinstance(self.answer, (int, float)):
                    return -self.max_error
                diff = abs(float(self.answer) - numeric_value)
                diff = min(diff, self.max_error)
                return -diff

        else:

            return -self.max_error

@dataclass(frozen=True)
class OperandConfig:
    symbol: str
    fn: Callable[[int, int], float]


class MathDataset(Dataset):
    def __init__(self, num_samples: int = 200, min_val: int = 0, max_val: int = 200, operand: str = "mult"):
        super().__init__()

        operand_to_value = {
            "mult": OperandConfig("*", lambda a, b: a * b),
            "add": OperandConfig("+", lambda a, b: a + b),
            "substraction": OperandConfig("-", lambda a, b: a - b),
            "division": OperandConfig("/", lambda a, b: a / b),
        }

        assert operand in operand_to_value.keys(), f"Operand can only be one of {operand_to_value.keys()}"

        self.dataset = []

        for i in range(num_samples):
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)

            operand_values = operand_to_value[operand]
            operand_value = operand_values.symbol
            fn = operand_values.fn

            prompt = [
                {"role": "system", "content": "Think step by step about the user question, Give the final answer as [result], for example, if the result is 5, then you should finish your response with [5]"},
                {"role": "user", "content": f"What is {a} {operand_value} {b} ?"}
            ]

            target_value = fn(a, b)

            self.dataset.append(
                MathDataEnv(
                    prompt=prompt,
                    answer=target_value
                )
            )

    def __getitem__(self, index: int) -> MathDataEnv:

        return self.dataset[index]
    
    def __len__(self) -> int:

        return len(self.dataset)


