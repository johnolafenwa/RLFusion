import torch

from rlfusion.envs import EnvBase
from rlfusion.inference.hf_utils import sample_completions_batch_hf


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.padding_side_seen: str | None = None
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.im_end_token_id = 3

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
        if len(prompts) == 1:
            return {
                "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }
        return {
            "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long),
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|im_end|>":
            return self.im_end_token_id
        return -1

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return "decoded"


class _DummyGenerateOutput:
    def __init__(self, sequences: torch.Tensor) -> None:
        self.sequences = sequences


class _DummyModel:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.last_generate_kwargs = None

    def parameters(self):
        return iter([self._param])

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
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


def test_sample_completions_batch_hf_returns_full_attention_mask():
    tokenizer = _DummyTokenizer()
    model = _DummyModel()
    envs = [
        _DummyEnv(prompt=[{"role": "user", "content": "short"}], answer="x"),
        _DummyEnv(prompt=[{"role": "user", "content": "longer prompt"}], answer="y"),
    ]

    sequences, _, _, _, attention_mask = sample_completions_batch_hf(
        model=model,
        tokenizer=tokenizer,
        envs=envs,
        do_sample=True,
        sampling_temperature=1.0,
        max_new_tokens=2,
        generation_args={},
        return_attention_mask=True,
    )

    assert tuple(sequences.shape) == (2, 5)
    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(attention_mask, expected)


def test_sample_completions_batch_hf_allows_generation_temperature_override():
    tokenizer = _DummyTokenizer()
    model = _DummyModel()
    envs = [_DummyEnv(prompt=[{"role": "user", "content": "short"}], answer="x")]

    sample_completions_batch_hf(
        model=model,
        tokenizer=tokenizer,
        envs=envs,
        do_sample=True,
        sampling_temperature=0.8,
        max_new_tokens=2,
        generation_args={"temperature": 0.3},
        return_attention_mask=False,
    )

    assert model.last_generate_kwargs is not None
    assert model.last_generate_kwargs["temperature"] == 0.3
    assert model.last_generate_kwargs["use_cache"] is True


def test_sample_completions_batch_hf_stops_on_chat_end_token():
    tokenizer = _DummyTokenizer()
    model = _DummyModel()
    envs = [_DummyEnv(prompt=[{"role": "user", "content": "short"}], answer="x")]

    def _generate(**kwargs):
        model.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        completion = torch.tensor(
            [[7, tokenizer.im_end_token_id, 9]] * input_ids.shape[0],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return _DummyGenerateOutput(torch.cat([input_ids, completion], dim=1))

    model.generate = _generate

    _, texts, _, completion_lengths = sample_completions_batch_hf(
        model=model,
        tokenizer=tokenizer,
        envs=envs,
        do_sample=False,
        sampling_temperature=1.0,
        max_new_tokens=3,
        generation_args={},
        return_attention_mask=False,
    )

    assert texts == ["decoded"]
    assert completion_lengths == [1]
    assert model.last_generate_kwargs is not None
    assert model.last_generate_kwargs["eos_token_id"] == [2, 3]
