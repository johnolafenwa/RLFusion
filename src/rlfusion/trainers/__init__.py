"""Trainer entry points with lazy imports."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .onpolicy_distillation_trainer import OnPolicyDistillationTrainer
    from .sft_trainer import SFTTrainer

__all__ = ["SFTTrainer", "OnPolicyDistillationTrainer"]


def __getattr__(name: str) -> Any:
    if name == "SFTTrainer":
        from .sft_trainer import SFTTrainer

        return SFTTrainer
    if name == "OnPolicyDistillationTrainer":
        from .onpolicy_distillation_trainer import OnPolicyDistillationTrainer

        return OnPolicyDistillationTrainer
    raise AttributeError(name)
