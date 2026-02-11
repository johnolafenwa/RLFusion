import pytest

from rlfusion.datasets.riddlebench import RiddleBenchDataset, RiddleBenchEnv
from rlfusion.envs import EnvBase


class _DummyDataset:
    def __init__(self, items: list[dict[str, object]]) -> None:
        self._items = items

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def select(self, indices: range | list[int]) -> "_DummyDataset":
        return _DummyDataset([self._items[i] for i in indices])

    def shuffle(self, seed: int) -> "_DummyDataset":
        _ = seed
        return _DummyDataset(list(reversed(self._items)))


def test_riddlebench_dataset_builds_prompt(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_load_dataset(name: str, split: str):
        captured["name"] = name
        captured["split"] = split
        return _DummyDataset(
            [
                {"type": "sequence tasks", "question": "1, 2, ?, 4", "answer": "3"},
                {"type": "blood relations", "question": "A is B's brother", "answer": "A"},
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = RiddleBenchDataset(train=True, train_split_ratio=0.5)
    sample = dataset[0]

    assert captured["name"] == "ai4bharat/RiddleBench"
    assert captured["split"] == "train"
    assert isinstance(sample, EnvBase)
    assert sample.prompt[0]["role"] == "system"
    assert sample.prompt[1] == {"role": "user", "content": "1, 2, ?, 4"}
    assert sample.answer == "3"


def test_riddlebench_dataset_filters_task_types_and_eval_split(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset(
            [
                {"type": "sequence tasks", "question": "q1", "answer": "1"},
                {"type": "seating task", "question": "q2", "answer": "B"},
                {"type": "sequence tasks", "question": "q3", "answer": "3"},
                {"type": "blood relations", "question": "q4", "answer": "A"},
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = RiddleBenchDataset(
        train=False,
        task_types=["sequence tasks"],
        train_split_ratio=0.5,
    )

    assert len(dataset) == 1
    assert dataset[0].prompt[1]["content"] == "q3"


def test_riddlebench_dataset_invalid_args_raise(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset(
            [
                {"type": "sequence tasks", "question": "q1", "answer": "1"},
                {"type": "sequence tasks", "question": "q2", "answer": "2"},
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    with pytest.raises(ValueError, match="train_split_ratio must be between 0 and 1."):
        RiddleBenchDataset(train_split_ratio=1.0)

    with pytest.raises(ValueError, match="No rows matched task_types."):
        RiddleBenchDataset(task_types=["unknown type"])


def test_riddlebench_env_reward_handles_boxed_and_casefold() -> None:
    env = RiddleBenchEnv(
        prompt=[{"role": "user", "content": "Puzzle"}],
        answer="D",
    )

    assert env.get_reward("\\boxed{d}") == 1.0
    assert env.get_reward(" d ") == 1.0
    assert env.get_reward("C") == 0.0
    assert env.get_reward(None) == 0.0
