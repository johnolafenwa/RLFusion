from rlfusion.envs import EnvBase
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
