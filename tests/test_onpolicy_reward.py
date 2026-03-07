import pytest
import torch

from rlfusion.envs import EnvBase
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


class DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.25


class NaNRewardEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return float("nan")


class DummyModeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))


def test_onpolicy_reward_returns_none_without_answer():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    env = DummyEnv(prompt=[], answer=None)
    assert trainer._compute_reward(env, "text") is None


def test_onpolicy_reward_uses_env_when_answer_present():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    env = DummyEnv(prompt=[], answer="ok")
    assert trainer._compute_reward(env, "text") == 1.25


def test_onpolicy_reward_returns_none_for_missing_completion():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    env = DummyEnv(prompt=[], answer="ok")
    assert trainer._compute_reward(env, None) is None


def test_onpolicy_reward_returns_none_for_non_finite_values():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    env = NaNRewardEnv(prompt=[], answer="ok")
    assert trainer._compute_reward(env, "text") is None


def test_onpolicy_test_restores_model_modes():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    trainer.batch_size = 1
    trainer.log_completions = False
    trainer.max_log_chars = 200
    trainer.sampling_temperature = 1.0
    trainer._wandb = None
    trainer.model = DummyModeModel()
    trainer.teacher_model = DummyModeModel()
    trainer.model.eval()
    trainer.teacher_model.train()

    def _sample_completions_batch_with_mask(envs):
        _ = envs
        return (
            torch.tensor([[5, 0]], dtype=torch.long),
            ["ok"],
            [1],
            [1],
            torch.tensor([[1, 1]], dtype=torch.long),
        )

    trainer._sample_completions_batch_with_mask = _sample_completions_batch_with_mask

    def _get_log_probs(sequence_ids, model=None, attention_mask=None):
        _ = (sequence_ids, attention_mask)
        target_model = trainer.model if model is None else model
        if target_model is trainer.teacher_model:
            return torch.tensor([[0.1]], dtype=torch.float32)
        return torch.tensor([[0.2]], dtype=torch.float32)

    trainer.get_log_probs = _get_log_probs

    dataset = [DummyEnv(prompt=[{"role": "user", "content": "q"}], answer="4")]
    trainer.test(dataset=dataset, num_batches=1)

    assert trainer.model.training is False
    assert trainer.teacher_model.training is True


def test_onpolicy_test_rejects_non_positive_num_batches():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    dataset = [DummyEnv(prompt=[{"role": "user", "content": "q"}], answer="4")]

    with pytest.raises(ValueError, match="num_batches must be >= 1 or None."):
        trainer.test(dataset=dataset, num_batches=0)


def test_onpolicy_test_rejects_non_positive_eval_temperature():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)
    dataset = [DummyEnv(prompt=[{"role": "user", "content": "q"}], answer="4")]

    with pytest.raises(ValueError, match="eval_temperature must be > 0 or None."):
        trainer.test(dataset=dataset, eval_temperature=0.0)


def test_onpolicy_build_masks_keeps_first_token_when_completion_len_zero():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)

    sequence_ids = torch.zeros((1, 6), dtype=torch.long)
    masks = trainer._build_masks(
        prompt_lengths=[4],
        completion_lengths=[0],
        sequence_ids=sequence_ids,
    )

    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    assert torch.equal(masks, expected)


def test_onpolicy_build_masks_full_attention_keeps_first_token_when_completion_len_zero():
    trainer = OnPolicyDistillationTrainer.__new__(OnPolicyDistillationTrainer)

    sequence_ids = torch.zeros((1, 6), dtype=torch.long)
    full_attention_mask = torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.long)
    masks = trainer._build_masks(
        prompt_lengths=[3],
        completion_lengths=[0],
        sequence_ids=sequence_ids,
        full_attention_mask=full_attention_mask,
    )

    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    assert torch.equal(masks, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_steps", 0, "num_steps must be >= 1."),
        ("num_epochs", 0, "num_epochs must be >= 1 or None."),
        ("saving_steps", 0, "saving_steps must be >= 1."),
        ("logging_steps", 0, "logging_steps must be >= 1."),
        ("sampling_temperature", 0.0, "sampling_temperature must be > 0."),
        ("max_new_tokens", 0, "max_new_tokens must be >= 1."),
        ("batch_size", 0, "batch_size must be >= 1."),
        ("ppo_steps", 0, "ppo_steps must be >= 1."),
        ("clip_eps", -0.1, "clip_eps must be >= 0."),
        ("max_grad_norm", 0.0, "max_grad_norm must be > 0 or None."),
    ],
)
def test_onpolicy_validate_init_args_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        if field == "num_steps":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=value,
                num_epochs=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "num_epochs":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                num_epochs=value,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "saving_steps":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=value,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "logging_steps":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=value,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "sampling_temperature":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=value,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "max_new_tokens":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=value,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "batch_size":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=value,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "ppo_steps":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=value,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "clip_eps":
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=value,
                max_grad_norm=None,
            )
        else:
            OnPolicyDistillationTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=value,
            )
