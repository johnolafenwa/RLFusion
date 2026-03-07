import torch

from rlfusion.envs import EnvBase
from rlfusion.inference.vllm_utils import sample_completions_batch_vllm


class _DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.0


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt is True
        assert tokenize is False
        return str(prompt)

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return ",".join(str(token_id) for token_id in token_ids)


class _FakeCompletion:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids


class _FakeRequestOutput:
    def __init__(self, prompt_token_ids: list[int], token_ids: list[int]) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [_FakeCompletion("completion", token_ids)]


class _FakeEngine:
    def generate(self, prompts, sampling_params):
        _ = (prompts, sampling_params)
        return [
            _FakeRequestOutput([11, 12, 13], [7, 2, 9]),
            _FakeRequestOutput([21, 22, 23, 24, 25], [8, 8, 2]),
        ]


def test_sample_completions_batch_vllm_returns_full_attention_mask():
    envs = [
        _DummyEnv(prompt=[{"role": "user", "content": "short"}], answer="x"),
        _DummyEnv(prompt=[{"role": "user", "content": "long"}], answer="y"),
    ]

    sequences, texts, prompt_lengths, completion_lengths, attention_mask = (
        sample_completions_batch_vllm(
            vllm_engine=_FakeEngine(),
            tokenizer=_FakeTokenizer(),
            envs=envs,
            sampling_params=object(),
            return_attention_mask=True,
        )
    )

    assert texts == ["7", "8,8"]
    assert prompt_lengths == [3, 5]
    assert completion_lengths == [1, 2]
    expected_sequences = torch.tensor(
        [
            [0, 0, 11, 12, 13, 7, 2, 9],
            [21, 22, 23, 24, 25, 8, 8, 2],
        ],
        dtype=torch.long,
    )
    expected_attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    assert torch.equal(sequences, expected_sequences)
    assert torch.equal(attention_mask, expected_attention_mask)
