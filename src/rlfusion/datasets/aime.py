from rlfusion.envs import EnvBase
from rlfusion.utils import get_boxed_answer
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass

@dataclass
class AimeEnv(EnvBase):

    def get_reward(self, prediction) -> float:

        final_answer = get_boxed_answer(prediction)

        if final_answer is None:
            return 0.0
        else:
            return 1.0 if str(prediction) == str(final_answer) else 0.0
    

class AIME2025(Dataset):
    def __init__(self, 
                 system_prompt: str = "You are a helpful assistant, solve this problem and return the final answer is boxed format.",
                 cache_dir: str | None = None):
        super().__init__()

        self.system_prompt = system_prompt

        self.dataset = load_dataset("MathArena/aime_2025", split="train", cache_dir=cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> EnvBase:

        return AimeEnv(
            prompt=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": self.dataset[index]["problem"]
                }
            ],
            answer=self.dataset[index]["answer"]
        )
        




