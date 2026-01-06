from typing import Protocol

import torch


class GenerateOutput(Protocol):
    sequences: torch.Tensor
