import torch

from rlfusion.envs import EnvBase
from rlfusion.inference.hf_utils import sample_completions_batch_hf


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.padding_side_seen: str | None = None
        self.pad_token_id = 0
        self.eos_token_id = 2

    def apply_chat_template(
        self, prompt: list[dict[str, object]], add_generation_prompt: bool, tokenize: bool
    ) -> str:
        assert add_generation_prompt is True
        assert tokenize is False
        return str(prompt[0]["content"])

    def __call__(self, prompts, return_tensors: str, padding: bool):
        assert return_tensors == "pt"
        assert padding is True
        self.padding_side_seen = self.padding_side
        return {
            "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return "decoded"


class _DummyGenerateOutput:
    def __init__(self, sequences: torch.Tensor) -> None:
        self.sequences = sequences


class _DummyModel:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        completions = torch.full((batch_size, 2), 7, dtype=input_ids.dtype, device=input_ids.device)
        return _DummyGenerateOutput(torch.cat([input_ids, completions], dim=1))


class _DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.0


def test_sample_completions_batch_hf_uses_left_padding_and_restores_tokenizer():
    tokenizer = _DummyTokenizer()
    model = _DummyModel()
    envs = [
        _DummyEnv(prompt=[{"role": "user", "content": "short"}], answer="x"),
        _DummyEnv(prompt=[{"role": "user", "content": "longer prompt"}], answer="y"),
    ]

    _, texts, prompt_lengths, completion_lengths = sample_completions_batch_hf(
        model=model,
        tokenizer=tokenizer,
        envs=envs,
        do_sample=True,
        sampling_temperature=1.0,
        max_new_tokens=2,
        generation_args={},
        return_attention_mask=False,
    )

    assert tokenizer.padding_side_seen == "left"
    assert tokenizer.padding_side == "right"
    assert texts == ["decoded", "decoded"]
    assert prompt_lengths == [2, 3]
    assert completion_lengths == [2, 2]
