import torch

from rlfusion.trainers.grpo_trainer import GRPOTrainer
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer
from rlfusion.trainers.utils import build_completion_mask_from_attention, build_full_attention_mask


def test_grpo_full_attention_mask_ignores_padding():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    input_attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    completion_lengths = [2, 1]
    sequence_ids = torch.zeros((2, 7), dtype=torch.long)

    full_mask = trainer._build_full_attention_mask(
        input_attention_mask, completion_lengths, sequence_ids
    )

    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    assert torch.equal(full_mask, expected)


def test_onpolicy_full_attention_mask_ignores_padding():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    input_attention_mask = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    completion_lengths = [1]
    sequence_ids = torch.zeros((1, 6), dtype=torch.long)

    full_mask = trainer._build_full_attention_mask(
        input_attention_mask, completion_lengths, sequence_ids
    )

    expected = torch.tensor([[1, 1, 0, 0, 1, 0]], dtype=torch.long)
    assert torch.equal(full_mask, expected)


def test_build_full_attention_mask_accepts_full_sequence_masks():
    full_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    sequence_ids = torch.zeros((2, 8), dtype=torch.long)

    rebuilt = build_full_attention_mask(full_mask, [1, 2], sequence_ids)

    assert torch.equal(rebuilt, full_mask)


def test_build_completion_mask_from_attention_handles_variable_boundaries():
    full_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    sequence_ids = torch.zeros((2, 8), dtype=torch.long)

    completion_mask = build_completion_mask_from_attention(
        full_mask,
        [1, 2],
        sequence_ids,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(completion_mask, expected)
