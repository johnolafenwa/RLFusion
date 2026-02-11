import pytest

from rlfusion.envs import EnvBase
from rlfusion.trainers.data import Trajectory
from rlfusion.trainers.grpo_trainer import GRPOTrainer


class DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.0 if prediction.strip() == "4" else 0.0


class DummyModel:
    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None


def test_grpo_test_reports_reward_and_tokens():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.batch_size = 2
    trainer.log_completions = False
    trainer.max_log_chars = 200
    trainer.model = DummyModel()
    trainer.max_error = 1.0
    trainer.invalid_penalty = 1.0
    trainer.sampling_temperature = 1.0
    trainer._wandb = None

    def sample_completions_batch(envs):
        texts = ["4", "0"]
        completion_lens = [1, 1]
        return None, texts, [0, 0], completion_lens

    trainer.sample_completions_batch = sample_completions_batch

    dataset = [
        DummyEnv(prompt=[{"role": "user", "content": "2 + 2?"}], answer="4"),
        DummyEnv(prompt=[{"role": "user", "content": "2 + 3?"}], answer="5"),
    ]

    results = trainer.test(dataset=dataset, num_batches=1)

    assert results["reward_mean"] == 0.5
    assert results["reward_std"] == 0.5
    assert results["completion_tokens_mean"] == 1.0


def test_grpo_test_rejects_non_positive_num_batches():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.batch_size = 1
    trainer.log_completions = False
    trainer.max_log_chars = 200
    trainer.model = DummyModel()
    trainer.max_error = 1.0
    trainer.invalid_penalty = 1.0
    trainer.sampling_temperature = 1.0
    trainer._wandb = None

    dataset = [DummyEnv(prompt=[{"role": "user", "content": "q"}], answer="4")]
    with pytest.raises(ValueError, match="num_batches must be >= 1 or None."):
        trainer.test(dataset=dataset, num_batches=0)


def test_grpo_advantage_normalizes_within_group():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    env = DummyEnv(prompt=[{"role": "user", "content": "q"}], answer="4")
    trajectories = [
        Trajectory(env=env, reward=0.0),
        Trajectory(env=env, reward=1.0),
        Trajectory(env=env, reward=100.0),
        Trajectory(env=env, reward=101.0),
    ]

    trainer.compute_advantage(trajectories, group_size=2)

    assert trajectories[0].advantage == pytest.approx(-1.0, abs=1e-6)
    assert trajectories[1].advantage == pytest.approx(1.0, abs=1e-6)
    assert trajectories[2].advantage == pytest.approx(-1.0, abs=1e-6)
    assert trajectories[3].advantage == pytest.approx(1.0, abs=1e-6)


def test_grpo_compute_reward_does_not_require_answer():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.max_error = 100.0
    trainer.invalid_penalty = 1.0

    env = DummyEnv(prompt=[{"role": "user", "content": "q"}], answer=None)
    assert trainer._compute_reward(env, "4") == 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_steps", 0, "num_steps must be >= 1."),
        ("saving_steps", 0, "saving_steps must be >= 1."),
        ("logging_steps", 0, "logging_steps must be >= 1."),
        ("sampling_temperature", 0.0, "sampling_temperature must be > 0."),
        ("max_new_tokens", 0, "max_new_tokens must be >= 1."),
        ("batch_size", 0, "batch_size must be >= 1."),
        ("group_size", 1, "group_size must be >= 2."),
        ("ppo_steps", 0, "ppo_steps must be >= 1."),
    ],
)
def test_grpo_validate_init_args_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        if field == "num_steps":
            GRPOTrainer._validate_init_args(
                num_steps=value,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                group_size=2,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "saving_steps":
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=value,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                group_size=2,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "logging_steps":
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=value,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                group_size=2,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "sampling_temperature":
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=value,
                max_new_tokens=1,
                batch_size=1,
                group_size=2,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "max_new_tokens":
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=value,
                batch_size=1,
                group_size=2,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "batch_size":
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=value,
                group_size=2,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        elif field == "group_size":
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                group_size=value,
                ppo_steps=1,
                clip_eps=0.2,
                max_grad_norm=None,
            )
        else:
            GRPOTrainer._validate_init_args(
                num_steps=1,
                saving_steps=1,
                logging_steps=1,
                eval_steps=None,
                sampling_temperature=1.0,
                max_new_tokens=1,
                batch_size=1,
                group_size=2,
                ppo_steps=value,
                clip_eps=0.2,
                max_grad_norm=None,
            )
