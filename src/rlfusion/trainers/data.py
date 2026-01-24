"""Data containers used by trainer loops."""

from dataclasses import dataclass
from rlfusion.envs import EnvBase
import torch
from typing import Optional

@dataclass
class Trajectory:
    env: EnvBase
    sequence_ids: Optional[torch.Tensor] = None
    completion_text: Optional[str] = None 
    prompt_len: Optional[int] = None
    completion_len: Optional[int] = None
    old_log_probs: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None 
    reward: Optional[float] = None
    advantage: Optional[float] = None
    ref_log_probs: Optional[torch.Tensor] = None 
   
    
