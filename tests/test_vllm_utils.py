import os

import pytest
import torch

import rlfusion.inference.vllm_utils as vllm_utils
from rlfusion.envs import EnvBase
from rlfusion.inference.vllm_utils import (
    _build_ipc_weight_update_info,
    ensure_vllm_env,
    pin_vllm_to_local_cuda_device,
    prepare_vllm_runtime_args,
    resolve_local_vllm_visible_device,
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
    envs: list[EnvBase] = [
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


def test_resolve_local_vllm_visible_device_uses_process_local_gpu_index():
    resolved = resolve_local_vllm_visible_device(
        1,
        cuda_visible_devices=None,
        cuda_device_count=2,
    )

    assert resolved == "1"


def test_resolve_local_vllm_visible_device_respects_existing_cuda_visible_devices():
    resolved = resolve_local_vllm_visible_device(
        1,
        cuda_visible_devices="4,7",
        cuda_device_count=8,
    )

    assert resolved == "7"


def test_pin_vllm_to_local_cuda_device_restores_previous_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    with pin_vllm_to_local_cuda_device(1):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,5"


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


def test_build_ipc_weight_update_info_creates_per_tensor_handles():
    model = _TinyModel()

    def _reduce_tensor(tensor):
        return ("rebuild", (tuple(tensor.shape), str(tensor.dtype)))

    update_info, retained_weights = _build_ipc_weight_update_info(
        list(model.named_parameters()),
        gpu_uuid="GPU-123",
        reduce_tensor_fn=_reduce_tensor,
    )

    assert update_info["names"] == ["proj.weight"]
    assert update_info["dtype_names"] == ["float32"]
    assert update_info["shapes"] == [[2, 2]]
    assert update_info["ipc_handles"] == [{"GPU-123": ("rebuild", ((2, 2), "torch.float32"))}]
    assert len(retained_weights) == 1
    assert retained_weights[0].device.type == "cpu"


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)


class _TwoParamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
        self.second = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))


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


def test_sync_model_weights_to_vllm_streams_apply_model_payloads_in_chunks(monkeypatch):
    model = _TwoParamModel()

    class _Engine:
        def __init__(self) -> None:
            self.calls = []

        def apply_model(self, fn):
            class _WorkerModel:
                def __init__(self) -> None:
                    self.received = None

                def load_weights(self, weight_pairs):
                    self.received = list(weight_pairs)
                    return {name for name, _ in self.received}

            worker_model = _WorkerModel()
            result = fn(worker_model)
            self.calls.append((worker_model.received, result))
            return [result]

    monkeypatch.setattr(vllm_utils, "_APPLY_MODEL_SYNC_MAX_CHUNK_BYTES", 32)

    engine = _Engine()
    sync_model_weights_to_vllm(model, engine)

    assert len(engine.calls) == 2
    assert [name for name, _ in engine.calls[0][0]] == ["first"]
    assert [name for name, _ in engine.calls[1][0]] == ["second"]
    assert all(tensor.device.type == "cpu" for call, _ in engine.calls for _name, tensor in call)


def test_sync_model_weights_to_vllm_skips_ipc_for_multi_gpu_engines():
    model = _TinyModel()

    class _ParallelConfig:
        tensor_parallel_size = 2
        pipeline_parallel_size = 1
        data_parallel_size = 1

    class _Engine:
        def __init__(self) -> None:
            self.calls = []
            self.ipc_attempted = False
            self.llm_engine = type(
                "_LLMEngine",
                (),
                {
                    "vllm_config": type(
                        "_Config",
                        (),
                        {"parallel_config": _ParallelConfig()},
                    )()
                },
            )()

        def init_weight_transfer_engine(self, _request):
            self.ipc_attempted = True

        def update_weights(self, _request):
            self.ipc_attempted = True

        def apply_model(self, fn):
            class _WorkerModel:
                def __init__(self) -> None:
                    self.received = None

                def load_weights(self, weight_pairs):
                    self.received = list(weight_pairs)
                    return {name for name, _ in self.received}

            worker_model = _WorkerModel()
            result = fn(worker_model)
            self.calls.append((worker_model.received, result))
            return [result]

    engine = _Engine()
    sync_model_weights_to_vllm(model, engine)

    assert engine.ipc_attempted is False
    assert len(engine.calls) == 1
    transferred, result = engine.calls[0]
    assert [name for name, _ in transferred] == ["proj.weight"]
    assert all(tensor.device.type == "cpu" for _name, tensor in transferred)
    assert result == 1
