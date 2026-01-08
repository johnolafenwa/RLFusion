from rlfusion.datasets.aime import AIME2025
from rlfusion.envs import EnvBase


class _DummyDataset:
    def __init__(self, items: list[str]) -> None:
        self._items = items

    def __getitem__(self, index: int) -> str:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


def test_aime2025_dataset_builds_prompt(monkeypatch) -> None:
    dummy_dataset = _DummyDataset(["problem 1", "problem 2"])

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
