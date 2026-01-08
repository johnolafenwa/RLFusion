from rlfusion.datasets.aime import AIME2025, AimeEnv
from rlfusion.envs import EnvBase


class _DummyDataset:
    def __init__(self, items: list[dict[str, str]]) -> None:
        self._items = items

    def __getitem__(self, index: int) -> dict[str, str]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


def test_aime2025_dataset_builds_prompt(monkeypatch) -> None:
    dummy_dataset = _DummyDataset(
        [
            {"problem": "problem 1", "answer": "1"},
            {"problem": "problem 2", "answer": "2"},
        ]
    )

    def fake_load_dataset(name: str, split: str, cache_dir: str | None = None):
        assert name == "MathArena/aime_2025"
        assert split == "train"
        assert cache_dir == "cache"
        return dummy_dataset

    monkeypatch.setattr("rlfusion.datasets.aime.load_dataset", fake_load_dataset)

    dataset = AIME2025(system_prompt="Solve it.", cache_dir="cache")
    sample = dataset[0]

    assert isinstance(sample, EnvBase)
    assert sample.prompt[0]["role"] == "system"
    assert sample.prompt[0]["content"] == "Solve it."
    assert sample.prompt[1]["role"] == "user"
    assert sample.prompt[1]["content"] == "problem 1"


def test_aime_env_reward_matches_boxed_answer() -> None:
    env = AimeEnv(prompt=[{"role": "user", "content": "Question"}], answer="42")

    assert env.get_reward("Solution: \\boxed{42}") == 1.0


def test_aime_env_reward_zero_when_missing_or_wrong() -> None:
    env = AimeEnv(prompt=[{"role": "user", "content": "Question"}], answer="42")

    assert env.get_reward("No boxed answer here.") == 0.0
    assert env.get_reward("Solution: \\boxed{41}") == 0.0
    assert env.get_reward(None) == 0.0
