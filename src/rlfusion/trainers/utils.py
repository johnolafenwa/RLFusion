"""Trainer utilities: device selection, seeding, formatting, mask helpers."""

import importlib.util
import json
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Any

import numpy as np
import torch

from rlfusion.trainers.types import AttentionMask, TokenIds

def get_device():

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_torch_backends():
    """Configure torch backends for optimal performance."""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN benchmark for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True
        # Ensure deterministic behavior is off for speed (set True if reproducibility is critical)
        torch.backends.cudnn.deterministic = False


def resolve_attention_implementation(device_map: object) -> str:
    forced_attn_impl = os.getenv("RLFUSION_ATTN_IMPLEMENTATION")
    if forced_attn_impl:
        return forced_attn_impl
    if device_map != "auto":
        return "sdpa"
    if not torch.cuda.is_available():
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is None:
        return "sdpa"
    return "flash_attention_2"


def get_tokenizer_compat_kwargs(model_id_or_path: str) -> dict[str, Any]:
    """Return tokenizer kwargs that smooth over local checkpoint format drift.

    Newer transformers versions expect `extra_special_tokens` as a dict. Some
    saved checkpoints store it as a plain list, which raises at tokenizer load.
    """
    model_path = Path(model_id_or_path)
    if not model_path.is_dir():
        return {}

    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return {}

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
    except Exception:
        return {}

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return {}

    normalized_tokens: dict[str, str] = {}
    for idx, token in enumerate(extra_special_tokens):
        token_value = token if isinstance(token, str) else str(token)
        normalized_tokens[f"extra_special_token_{idx}"] = token_value
    return {"extra_special_tokens": normalized_tokens}


def normalize_generation_args(generation_args: Optional[dict[str, Any]]) -> dict[str, Any]:
    normalized = {} if generation_args is None else dict(generation_args)
    if "temperature" in normalized:
        raise ValueError(
            "Use sampling_temperature (and eval_temperature for trainer.test) instead of "
            "generation_args['temperature']."
        )
    return normalized


def truncate_text(text: Optional[str], max_chars: Optional[int]) -> str:
    if text is None:
        return "<none>"
    if max_chars is None:
        return text
    if len(text) <= max_chars:
        return text
    return "...<truncated>" + text[-max_chars:]


def format_prompt(prompt: list[dict]) -> str:
    parts = []
    for msg in prompt:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return " | ".join(parts)


def build_full_attention_mask(
    input_attention_mask: AttentionMask,
    completion_lengths: Sequence[int],
    sequence_ids: TokenIds,
) -> AttentionMask:
    if input_attention_mask.ndim == 1:
        input_attention_mask = input_attention_mask.unsqueeze(0)
    if input_attention_mask.shape[0] != sequence_ids.shape[0]:
        raise ValueError("input_attention_mask must match batch size.")
    if len(completion_lengths) != sequence_ids.shape[0]:
        raise ValueError("completion_lengths must match batch size.")
    if input_attention_mask.shape[1] > sequence_ids.shape[1]:
        raise ValueError("input_attention_mask exceeds sequence length.")

    # Some generation backends already return a full-sequence attention mask.
    if input_attention_mask.shape == sequence_ids.shape:
        return input_attention_mask.to(sequence_ids.device).long()

    # Preserve prompt padding holes while marking only generated tokens as attendable.
    input_attention_mask = input_attention_mask.to(sequence_ids.device)
    full_mask = torch.zeros_like(sequence_ids, dtype=torch.long)
    input_len = int(input_attention_mask.shape[1])
    full_mask[:, :input_len] = input_attention_mask.long()

    for idx, completion_len in enumerate(completion_lengths):
        end = min(input_len + int(completion_len), sequence_ids.shape[1])
        if end > input_len:
            full_mask[idx, input_len:end] = 1

    return full_mask


def build_completion_mask_from_attention(
    full_attention_mask: AttentionMask,
    completion_lengths: Sequence[int],
    sequence_ids: TokenIds,
    *,
    keep_first_token_when_zero: bool = False,
) -> AttentionMask:
    if full_attention_mask.ndim == 1:
        full_attention_mask = full_attention_mask.unsqueeze(0)
    if full_attention_mask.shape != sequence_ids.shape:
        raise ValueError("full_attention_mask must match sequence_ids shape.")
    if len(completion_lengths) != sequence_ids.shape[0]:
        raise ValueError("completion_lengths must match batch size.")

    full_attention_mask = full_attention_mask.to(sequence_ids.device).long()
    seq_len = sequence_ids.shape[1]
    if seq_len <= 1:
        return torch.zeros((sequence_ids.shape[0], 0), device=sequence_ids.device, dtype=torch.float32)

    batch_masks = torch.zeros(
        (sequence_ids.shape[0], seq_len - 1),
        device=sequence_ids.device,
        dtype=torch.float32,
    )

    for idx, completion_len in enumerate(completion_lengths):
        attended_positions = full_attention_mask[idx].nonzero(as_tuple=False).flatten()
        if attended_positions.numel() == 0:
            continue

        last_attended = int(attended_positions[-1].item())
        completion_len = int(completion_len)

        if completion_len <= 0:
            if keep_first_token_when_zero and last_attended < seq_len - 1:
                batch_masks[idx, last_attended] = 1.0
            continue

        first_completion_token = max(last_attended - completion_len + 1, 0)
        start = max(first_completion_token - 1, 0)
        end = min(last_attended, seq_len - 1)
        if end > start:
            batch_masks[idx, start:end] = 1.0

    return batch_masks
