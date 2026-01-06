import importlib.util
import random
from typing import Optional

import numpy as np
import torch

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


def resolve_attention_implementation(device_map: str) -> str:
    if device_map != "auto":
        return "sdpa"
    if not torch.cuda.is_available():
        return "sdpa"
    if importlib.util.find_spec("flash_attn") is None:
        return "sdpa"
    return "flash_attention_2"


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
