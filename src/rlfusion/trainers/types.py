"""Shared type aliases and protocols for trainer/inference code."""

from typing import Protocol, TypeAlias

import torch

TokenIds: TypeAlias = torch.Tensor
AttentionMask: TypeAlias = torch.Tensor
BatchMask: TypeAlias = torch.Tensor
LogProbs: TypeAlias = torch.Tensor


class GenerateOutput(Protocol):
    sequences: torch.Tensor
