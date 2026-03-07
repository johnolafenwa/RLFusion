import os

import pytest
import torch

from rlfusion.envs import EnvBase
from rlfusion.inference.vllm_utils import (
    ensure_vllm_env,
    prepare_vllm_runtime_args,
    resolve_vllm_training_config,
    sample_completions_batch_vllm,
    sync_model_weights_to_vllm,
)


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


def test_ensure_vllm_env_preserves_backend_autoselection(monkeypatch):
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)

    ensure_vllm_env()

    assert "VLLM_ATTENTION_BACKEND" not in os.environ
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"


def test_ensure_vllm_env_preserves_explicit_backend_override(monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")

    ensure_vllm_env()

    assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"


def test_prepare_vllm_runtime_args_enables_sleep_mode():
    resolved = prepare_vllm_runtime_args(
        {"tensor_parallel_size": 2},
        enable_sleep=True,
        use_accelerate=False,
    )

    assert resolved["tensor_parallel_size"] == 2
    assert resolved["enable_sleep_mode"] is True


def test_prepare_vllm_runtime_args_rejects_multi_gpu_vllm_with_accelerate():
    with pytest.raises(ValueError, match="per-process vLLM engines"):
        prepare_vllm_runtime_args(
            {"tensor_parallel_size": 2},
            enable_sleep=False,
            use_accelerate=True,
        )


def test_resolve_vllm_training_config_defaults_to_vllm_on_cuda():
    use_vllm, resolved_args, auto_selected = resolve_vllm_training_config(
        device="cuda",
        use_vllm=None,
        vllm_args=None,
        enable_sleep=False,
        use_accelerate=False,
    )

    assert use_vllm is True
    assert auto_selected is True
    assert resolved_args["gpu_memory_utilization"] == 0.5


def test_resolve_vllm_training_config_defaults_to_hf_on_cpu():
    use_vllm, resolved_args, auto_selected = resolve_vllm_training_config(
        device="cpu",
        use_vllm=None,
        vllm_args={"gpu_memory_utilization": 0.3},
        enable_sleep=False,
        use_accelerate=False,
    )

    assert use_vllm is False
    assert auto_selected is True
    assert resolved_args == {}


def test_resolve_vllm_training_config_rejects_non_cuda_vllm():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_vllm_training_config(
            device="cpu",
            use_vllm=True,
            vllm_args=None,
            enable_sleep=False,
            use_accelerate=False,
        )


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)


def test_sync_model_weights_to_vllm_uses_direct_load_weights():
    model = _TinyModel()

    class _Engine:
        def __init__(self) -> None:
            self.weight_pairs = None

        def load_weights(self, weight_pairs):
            self.weight_pairs = list(weight_pairs)

    engine = _Engine()
    sync_model_weights_to_vllm(model, engine)

    assert engine.weight_pairs is not None
    assert [name for name, _ in engine.weight_pairs] == ["proj.weight"]


def test_sync_model_weights_to_vllm_uses_apply_model_reload_weights():
    model = _TinyModel()

    class _Engine:
        def __init__(self) -> None:
            self.calls = []

        def apply_model(self, fn):
            class _WorkerModel:
                def __init__(self) -> None:
                    self.received = None

                def load_weights(self, weight_pairs):
                    self.received = list(weight_pairs)
                    return {"proj.weight"}

            worker_model = _WorkerModel()
            result = fn(worker_model)
            self.calls.append((worker_model.received, result))
            return [result]

    engine = _Engine()
    sync_model_weights_to_vllm(model, engine)

    assert len(engine.calls) == 1
    transferred, result = engine.calls[0]
    assert [name for name, _ in transferred] == ["proj.weight"]
    assert all(tensor.device.type == "cpu" for _name, tensor in transferred)
    assert result == 1
