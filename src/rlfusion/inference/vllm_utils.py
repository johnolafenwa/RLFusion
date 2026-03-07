"""vLLM loader utilities and sampling parameter translation."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from collections.abc import Iterator
from typing import Any, Literal, overload

import torch

from rlfusion.envs import EnvBase

logger = logging.getLogger(__name__)
_APPLY_MODEL_SYNC_LOGGED = False
_APPLY_MODEL_SYNC_MAX_CHUNK_BYTES = 1 << 30
DEFAULT_COLOCATED_VLLM_ARGS: dict[str, Any] = {
    "gpu_memory_utilization": 0.5,
}

CompletionBatch = tuple[torch.Tensor, list[str], list[int], list[int]]
CompletionBatchWithMask = tuple[torch.Tensor, list[str], list[int], list[int], torch.Tensor]


class _ApplyModelReloadWeights:
    def __init__(self, weight_pairs: list[tuple[str, torch.Tensor]]) -> None:
        self.weight_pairs = weight_pairs

    def __call__(self, worker_model: Any) -> int:
        loaded = worker_model.load_weights(self.weight_pairs)
        return len(self.weight_pairs) if loaded is None else len(loaded)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _iter_cpu_weight_chunks(
    weight_pairs: list[tuple[str, torch.Tensor]],
    *,
    max_chunk_bytes: int | None = None,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    if max_chunk_bytes is None:
        max_chunk_bytes = _APPLY_MODEL_SYNC_MAX_CHUNK_BYTES

    current_chunk: list[tuple[str, torch.Tensor]] = []
    current_chunk_bytes = 0

    for name, tensor in weight_pairs:
        cpu_tensor = tensor.to("cpu")
        tensor_bytes = _tensor_nbytes(cpu_tensor)

        if current_chunk and current_chunk_bytes + tensor_bytes > max_chunk_bytes:
            yield current_chunk
            current_chunk = []
            current_chunk_bytes = 0

        current_chunk.append((name, cpu_tensor))
        current_chunk_bytes += tensor_bytes

        if current_chunk_bytes >= max_chunk_bytes:
            yield current_chunk
            current_chunk = []
            current_chunk_bytes = 0

    if current_chunk:
        yield current_chunk


def ensure_vllm_env() -> None:
    attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if attention_backend is None:
        logger.debug("VLLM_ATTENTION_BACKEND is unset; letting vLLM auto-select an attention backend.")
    else:
        logger.info("Using VLLM_ATTENTION_BACKEND=%s for vLLM.", attention_backend)

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("Set VLLM_WORKER_MULTIPROC_METHOD=spawn for vLLM.")

    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") is None:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        logger.info(
            "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 for vLLM worker callable "
            "serialization used by trainer weight sync."
        )


def load_vllm_engine(model_path_or_id: str, vllm_args: dict[str, Any]) -> tuple[Any, type, set[str]]:
    ensure_vllm_env()
    try:
        vllm_mod = importlib.import_module("vllm")
        llm_cls = getattr(vllm_mod, "LLM")
        sampling_params_cls = getattr(vllm_mod, "SamplingParams")
    except Exception as exc:
        raise ImportError("vllm is required for engine='vllm'.") from exc

    llm = llm_cls(model=model_path_or_id, **vllm_args)
    param_keys = set(inspect.signature(sampling_params_cls).parameters.keys())
    return llm, sampling_params_cls, param_keys


def prepare_vllm_runtime_args(
    vllm_args: dict[str, Any] | None,
    *,
    enable_sleep: bool,
    use_accelerate: bool,
) -> dict[str, Any]:
    resolved = {} if vllm_args is None else dict(vllm_args)

    tensor_parallel_size = int(resolved.get("tensor_parallel_size", 1))
    pipeline_parallel_size = int(resolved.get("pipeline_parallel_size", 1))
    if tensor_parallel_size <= 0:
        raise ValueError("vllm_args['tensor_parallel_size'] must be >= 1.")
    if pipeline_parallel_size <= 0:
        raise ValueError("vllm_args['pipeline_parallel_size'] must be >= 1.")

    if use_accelerate and (tensor_parallel_size != 1 or pipeline_parallel_size != 1):
        raise ValueError(
            "use_vllm with use_accelerate only supports per-process vLLM engines "
            "(tensor_parallel_size=1 and pipeline_parallel_size=1)."
        )

    if enable_sleep:
        resolved["enable_sleep_mode"] = True

    return resolved


def resolve_vllm_training_config(
    *,
    device: str,
    use_vllm: bool | None,
    vllm_args: dict[str, Any] | None,
    enable_sleep: bool,
    use_accelerate: bool,
) -> tuple[bool, dict[str, Any], bool]:
    auto_selected = use_vllm is None
    resolved_use_vllm = device == "cuda" if auto_selected else use_vllm

    if not resolved_use_vllm:
        return False, {}, auto_selected

    if device != "cuda":
        raise ValueError("use_vllm requires a CUDA device. Pass use_vllm=False on non-GPU devices.")

    merged_args = dict(DEFAULT_COLOCATED_VLLM_ARGS)
    if vllm_args is not None:
        merged_args.update(vllm_args)

    return (
        True,
        prepare_vllm_runtime_args(
            merged_args,
            enable_sleep=enable_sleep,
            use_accelerate=use_accelerate,
        ),
        auto_selected,
    )


def build_sampling_params(
    sampling_params_cls: type,
    param_keys: set[str],
    *,
    generation_args: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Any:
    max_tokens = generation_args.get("max_tokens", max_new_tokens)
    if "max_new_tokens" in generation_args and "max_tokens" not in generation_args:
        max_tokens = generation_args["max_new_tokens"]

    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature if do_sample else 0.0,
    }

    for key, value in generation_args.items():
        if key == "max_new_tokens":
            continue
        if key in param_keys:
            sampling_kwargs[key] = value

    return sampling_params_cls(**sampling_kwargs)


# ---------------------------------------------------------------------------
# vLLM generation with padded sequence tensor reconstruction
# ---------------------------------------------------------------------------


@overload
def sample_completions_batch_vllm(
    *,
    vllm_engine: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    sampling_params: Any,
    return_attention_mask: Literal[False] = False,
) -> CompletionBatch: ...


@overload
def sample_completions_batch_vllm(
    *,
    vllm_engine: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    sampling_params: Any,
    return_attention_mask: Literal[True],
) -> CompletionBatchWithMask: ...


def sample_completions_batch_vllm(
    *,
    vllm_engine: Any,
    tokenizer: Any,
    envs: list[EnvBase],
    sampling_params: Any,
    return_attention_mask: bool = False,
) -> CompletionBatch | CompletionBatchWithMask:
    """Generate completions using a vLLM engine and return padded tensors matching the HF format.

    The returned sequence tensors are left-padded to a common length. When
    ``return_attention_mask=True``, the attention mask matches the full padded
    sequence and marks only prompt plus valid completion tokens as attendable.
    """
    formatted_prompts = [
        tokenizer.apply_chat_template(
            env.prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        for env in envs
    ]

    outputs = vllm_engine.generate(formatted_prompts, sampling_params)

    ret_texts: list[str] = []
    completion_lengths: list[int] = []
    prompt_lengths: list[int] = []
    all_token_ids: list[list[int]] = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    for output in outputs:
        prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
        prompt_lengths.append(len(prompt_token_ids))

        if not output.outputs:
            ret_texts.append("")
            completion_lengths.append(0)
            all_token_ids.append(prompt_token_ids)
            continue

        completion = output.outputs[0]
        token_ids = getattr(completion, "token_ids", None)
        if token_ids is None:
            token_ids = tokenizer.encode(completion.text, add_special_tokens=False)
        token_ids = list(token_ids)

        # Trim at eos
        end_offset = len(token_ids)
        if eos_token_id is not None:
            for idx, tid in enumerate(token_ids):
                if tid == eos_token_id:
                    end_offset = idx
                    break

        completion_token_ids = token_ids[:end_offset]
        text = tokenizer.decode(completion_token_ids, skip_special_tokens=True)
        ret_texts.append(text)
        completion_lengths.append(len(completion_token_ids))

        # Full sequence = prompt + all generated tokens (including past eos for padding alignment)
        all_token_ids.append(prompt_token_ids + token_ids)

    # Left-pad all sequences to the same length (matching HF behaviour)
    if pad_token_id is None:
        pad_token_id = 0
    max_len = max(len(ids) for ids in all_token_ids) if all_token_ids else 0

    padded_sequences: list[list[int]] = []
    full_attention_masks: list[list[int]] = []

    for i, ids in enumerate(all_token_ids):
        pad_len = max_len - len(ids)
        padded_sequences.append([pad_token_id] * pad_len + ids)
        active_len = prompt_lengths[i] + completion_lengths[i]
        inactive_generated_len = max(len(ids) - active_len, 0)
        full_attention_masks.append(
            ([0] * pad_len)
            + ([1] * active_len)
            + ([0] * inactive_generated_len)
        )

    sequences = torch.tensor(padded_sequences, dtype=torch.long)

    if return_attention_mask:
        full_attention_mask = torch.tensor(full_attention_masks, dtype=torch.long)
        return sequences, ret_texts, prompt_lengths, completion_lengths, full_attention_mask

    return sequences, ret_texts, prompt_lengths, completion_lengths


# ---------------------------------------------------------------------------
# Weight sync: training model -> colocated vLLM engine
# ---------------------------------------------------------------------------


def sync_model_weights_to_vllm(model: Any, vllm_engine: Any) -> None:
    """Copy training model weights into a colocated vLLM engine.

    Handles DDP / Accelerate wrappers by unwrapping via ``.module``.
    Prefers the public ``llm.load_weights()`` API, falling back to the
    legacy internal model executor path, then the current ``apply_model``
    worker hook used by the standardized `vllm 0.17.x` path.
    """
    global _APPLY_MODEL_SYNC_LOGGED
    unwrapped = model.module if hasattr(model, "module") else model

    # Collect (name, tensor) pairs
    weight_pairs = [(name, param.detach()) for name, param in unwrapped.named_parameters()]

    # Current vLLM releases expose load_weights() directly on the LLM object.
    if hasattr(vllm_engine, "load_weights"):
        vllm_engine.load_weights(weight_pairs)
        logger.info("Synced %d parameter tensors to vLLM via load_weights().", len(weight_pairs))
        return

    # Fallback for older vLLM versions
    try:
        model_runner = (
            vllm_engine.llm_engine
            .model_executor
            .driver_worker
            .model_runner
            .model
        )
        model_runner.load_weights(weight_pairs)
        logger.info("Synced %d parameter tensors to vLLM via internal model runner.", len(weight_pairs))
        return
    except AttributeError as exc:
        if hasattr(vllm_engine, "apply_model"):
            total_tensors_loaded = 0
            chunk_count = 0
            loaded_per_worker_total = 0

            for chunk_index, transfer_pairs in enumerate(_iter_cpu_weight_chunks(weight_pairs), start=1):
                chunk_count = chunk_index
                loaded_per_worker = vllm_engine.apply_model(_ApplyModelReloadWeights(transfer_pairs))
                total_tensors_loaded += len(transfer_pairs)
                loaded_per_worker_total += sum(int(result) for result in loaded_per_worker)
                logger.debug(
                    "Synced vLLM weight chunk %d (%d tensors).",
                    chunk_index,
                    len(transfer_pairs),
                )

            if not _APPLY_MODEL_SYNC_LOGGED:
                logger.info(
                    "Using vLLM apply_model(load_weights) for trainer-to-engine "
                    "weight sync in streamed chunks; this is compatible with current "
                    "vLLM releases but can be slower than dedicated weight-transfer backends."
                )
                _APPLY_MODEL_SYNC_LOGGED = True
            logger.info(
                "Synced %d parameter tensors to vLLM via apply_model reload "
                "(chunks=%d, loaded=%d).",
                total_tensors_loaded,
                chunk_count,
                loaded_per_worker_total,
            )
            return

        raise RuntimeError(
            "Unable to sync weights to vLLM engine. "
            "Ensure you are using a compatible vLLM version."
        ) from exc


# ---------------------------------------------------------------------------
# Sleep / wake helpers for colocated vLLM (GPU memory management)
# ---------------------------------------------------------------------------


def vllm_sleep(vllm_engine: Any, level: int = 2) -> None:
    """Put the vLLM engine to sleep to free GPU memory for training.

    Requires the engine to be created with sleep mode enabled. No-op if the
    API is unavailable.
    """
    if hasattr(vllm_engine, "sleep"):
        vllm_engine.sleep(level=level)
        logger.debug("vLLM engine put to sleep (level=%d).", level)
    else:
        logger.debug("vLLM engine does not support sleep(); skipping.")


def vllm_wake_up(vllm_engine: Any, tags: list[str] | None = None) -> None:
    """Wake up a sleeping vLLM engine before generation.

    No-op if the API is unavailable.
    """
    if hasattr(vllm_engine, "wake_up"):
        if tags is None:
            vllm_engine.wake_up()
        else:
            try:
                vllm_engine.wake_up(tags=tags)
            except TypeError:
                vllm_engine.wake_up()
        logger.debug("vLLM engine woken up (tags=%s).", tags)
    else:
        logger.debug("vLLM engine does not support wake_up(); skipping.")
