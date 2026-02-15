"""vLLM loader utilities and sampling parameter translation."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any, Literal, overload

import torch

from rlfusion.envs import EnvBase

logger = logging.getLogger(__name__)

CompletionBatch = tuple[torch.Tensor, list[str], list[int], list[int]]
CompletionBatchWithMask = tuple[torch.Tensor, list[str], list[int], list[int], torch.Tensor]


def ensure_vllm_env() -> None:
    if os.environ.get("VLLM_ATTENTION_BACKEND") is None:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        logger.info("Set VLLM_ATTENTION_BACKEND=FLASH_ATTN for vLLM.")

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("Set VLLM_WORKER_MULTIPROC_METHOD=spawn for vLLM.")


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

    The returned sequence tensors are left-padded so that ``get_log_probs()``,
    ``build_full_attention_mask()``, and ``generate_mask()`` work unchanged.
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
    attention_masks: list[list[int]] = []
    # The input_attention_mask covers only the prompt portion (before generation).
    # We need to track the padded prompt length for each sequence.
    padded_prompt_lengths: list[int] = []

    for i, ids in enumerate(all_token_ids):
        pad_len = max_len - len(ids)
        padded_sequences.append([pad_token_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
        padded_prompt_lengths.append(pad_len + prompt_lengths[i])

    sequences = torch.tensor(padded_sequences, dtype=torch.long)

    # Return padded prompt lengths so downstream code (generate_mask, get_log_probs)
    # gets the correct generation boundary in the padded tensor.
    # Note: prompt_lengths here are the *padded* prompt lengths (pad + true prompt).

    if return_attention_mask:
        # Build input_attention_mask: covers only the prompt portion of the padded tensor.
        # Shape: (batch, padded_prompt_len) where padded_prompt_len = max padded prompt length.
        max_padded_prompt = max(padded_prompt_lengths) if padded_prompt_lengths else 0
        input_mask_list: list[list[int]] = []
        for i in range(len(envs)):
            pad_len = max_len - len(all_token_ids[i])
            # The input mask has length = max_padded_prompt
            # First pad_len positions are 0 (padding), then prompt_lengths[i] positions are 1
            mask = [0] * pad_len + [1] * prompt_lengths[i]
            # Pad or truncate to max_padded_prompt
            if len(mask) < max_padded_prompt:
                mask = mask + [0] * (max_padded_prompt - len(mask))
            else:
                mask = mask[:max_padded_prompt]
            input_mask_list.append(mask)
        input_attention_mask = torch.tensor(input_mask_list, dtype=torch.long)
        return sequences, ret_texts, padded_prompt_lengths, completion_lengths, input_attention_mask

    return sequences, ret_texts, padded_prompt_lengths, completion_lengths


# ---------------------------------------------------------------------------
# Weight sync: training model -> colocated vLLM engine
# ---------------------------------------------------------------------------


def sync_model_weights_to_vllm(model: Any, vllm_engine: Any) -> None:
    """Copy training model weights into a colocated vLLM engine.

    Handles DDP / Accelerate wrappers by unwrapping via ``.module``.
    Prefers ``llm.load_weights()`` (vLLM >= 0.7), falling back to the
    internal model executor path for older versions.
    """
    unwrapped = model.module if hasattr(model, "module") else model

    # Collect (name, tensor) pairs
    weight_pairs = [(name, param.data) for name, param in unwrapped.named_parameters()]

    # vLLM >= 0.7 exposes load_weights() directly on the LLM object
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
    except AttributeError as exc:
        raise RuntimeError(
            "Unable to sync weights to vLLM engine. "
            "Ensure you are using vLLM >= 0.7 or a compatible version."
        ) from exc


# ---------------------------------------------------------------------------
# Sleep / wake helpers for colocated vLLM (GPU memory management)
# ---------------------------------------------------------------------------


def vllm_sleep(vllm_engine: Any, level: int = 2) -> None:
    """Put the vLLM engine to sleep to free GPU memory for training.

    Requires vLLM >= 0.7 with ``--enable-sleep-mode``.  No-op if the API
    is unavailable.
    """
    if hasattr(vllm_engine, "sleep"):
        vllm_engine.sleep(level=level)
        logger.debug("vLLM engine put to sleep (level=%d).", level)
    else:
        logger.debug("vLLM engine does not support sleep(); skipping.")


def vllm_wake_up(vllm_engine: Any) -> None:
    """Wake up a sleeping vLLM engine before generation.

    No-op if the API is unavailable.
    """
    if hasattr(vllm_engine, "wake_up"):
        vllm_engine.wake_up()
        logger.debug("vLLM engine woken up.")
    else:
        logger.debug("vLLM engine does not support wake_up(); skipping.")
