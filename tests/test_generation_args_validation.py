from types import SimpleNamespace

import pytest
import torch

from rlfusion.evaluation.evaluator import Evaluator
from rlfusion.envs import EnvBase
from rlfusion.trainers.grpo_trainer import GRPOTrainer
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer
from rlfusion.trainers.sft_trainer import SFTTrainer
from rlfusion.trainers.utils import normalize_generation_args


class DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 0.0


class DummyDataset:
    def __init__(self, items: list[DummyEnv]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> DummyEnv:
        return self.items[index]


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = SimpleNamespace(use_cache=True)


class FakeAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeModel()


class CompatibleTokenizer:
    def __init__(self, *, offset: int) -> None:
        self.offset = offset
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.padding_side = "right"
        self.chat_template = "{{ messages }}"

    def __len__(self) -> int:
        return 32

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [len(text) + self.offset]


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_id, **kwargs):
        _ = kwargs
        offset = 0 if "student" in str(model_id) else 1
        return CompatibleTokenizer(offset=offset)


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: GRPOTrainer(
            model="unused",
            train_dataset=DummyDataset([DummyEnv(prompt=[{"role": "user", "content": "q"}], answer=None)]),
            output_dir=str(tmp_path),
            generation_args={"temperature": 0.3},
        ),
        lambda tmp_path: OnPolicyDistillationTrainer(
            model="unused",
            teacher_model="teacher",
            train_dataset=DummyDataset([DummyEnv(prompt=[{"role": "user", "content": "q"}], answer=None)]),
            output_dir=str(tmp_path),
            generation_args={"temperature": 0.3},
        ),
        lambda tmp_path: SFTTrainer(
            model="unused",
            train_dataset=[{"prompt": [{"role": "user", "content": "q"}], "answer": "a"}],
            output_dir=str(tmp_path),
            generation_args={"temperature": 0.3},
        ),
        lambda tmp_path: Evaluator(
            model="unused",
            dataset=[DummyEnv(prompt=[], answer=None)],
            output_dir=str(tmp_path),
            generation_args={"temperature": 0.3},
        ),
    ],
)
def test_public_apis_reject_generation_temperature_override(factory, tmp_path):
    with pytest.raises(
        ValueError,
        match="Use sampling_temperature \\(and eval_temperature for trainer\\.test\\) instead of generation_args\\['temperature'\\]\\.",
    ):
        factory(tmp_path)


def test_normalize_generation_args_preserves_supported_keys():
    assert normalize_generation_args({"top_p": 0.9}) == {"top_p": 0.9}


def test_onpolicy_trainer_rejects_incompatible_teacher_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rlfusion.trainers.onpolicy_distillation_trainer.AutoModelForCausalLM",
        FakeAutoModelForCausalLM,
    )
    monkeypatch.setattr(
        "rlfusion.trainers.onpolicy_distillation_trainer.AutoTokenizer",
        FakeAutoTokenizer,
    )

    dataset = DummyDataset([DummyEnv(prompt=[{"role": "user", "content": "q"}], answer=None)])

    with pytest.raises(
        ValueError,
        match="teacher_model tokenizer must produce the same token IDs as the student tokenizer.",
    ):
        OnPolicyDistillationTrainer(
            model="student",
            teacher_model="teacher",
            train_dataset=dataset,
            output_dir=str(tmp_path),
        )
