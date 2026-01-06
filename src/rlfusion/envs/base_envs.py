""" Base Environments"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvBase:
    prompt: list[ dict[str, object]]
    answer: Optional[float | int | str] = None 

    def get_reward(self, prediction: str) -> float:
        raise NotImplementedError()
