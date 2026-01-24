"""vLLM loader utilities and sampling parameter translation."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


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
