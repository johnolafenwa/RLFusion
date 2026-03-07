from types import SimpleNamespace

import pytest
import torch

from rlfusion.trainers.grpo_trainer import GRPOTrainer
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        _ = (attention_mask, use_cache)
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32, device=input_ids.device)
        logits[..., 0] = 2.0
        logits[..., 1] = -1.0
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("trainer_cls", [GRPOTrainer, OnPolicyDistillationTrainer])
def test_get_log_probs_is_independent_of_sampling_temperature(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.model = _ConstantLogitModel()
    trainer.tokenizer = SimpleNamespace(pad_token_id=99, eos_token_id=98)
    sequence_ids = torch.tensor([[1, 0, 1]], dtype=torch.long)

    trainer.sampling_temperature = 0.3
    low_temp = trainer.get_log_probs(sequence_ids)

    trainer.sampling_temperature = 2.0
    high_temp = trainer.get_log_probs(sequence_ids)

    assert torch.allclose(low_temp, high_temp)
