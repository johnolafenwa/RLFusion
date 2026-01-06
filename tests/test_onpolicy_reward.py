from rlfusion.envs import EnvBase
from rlfusion.trainers.onpolicy_distillation_trainer import OnPolicyDistillationTrainer


class DummyEnv(EnvBase):
    def get_reward(self, prediction: str) -> float:
        return 1.25


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
