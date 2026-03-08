import pytest

from rlfusion.datasets import HighSchoolMathReasoningSFTDataset
from rlfusion.envs import EnvBase


class _DummyDataset:
    def __init__(self, items: list[dict[str, object]]) -> None:
        self._items = items

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def select(self, indices: range) -> "_DummyDataset":
        return _DummyDataset([self._items[i] for i in indices])

    def shuffle(self, seed: int) -> "_DummyDataset":
        _ = seed
        return _DummyDataset(list(reversed(self._items)))


def test_highschool_math_reasoning_dataset_uses_train_split(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_load_dataset(name: str, split: str):
        captured["name"] = name
        captured["split"] = split
        return _DummyDataset(
            [
                {
                    "answer": "4",
                    "messages": [
                        {"role": "user", "content": "2 + 2"},
                        {"role": "assistant", "content": "<think>easy</think>\n\\boxed{4}"},
                    ],
                }
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningSFTDataset(train=True)
    sample = dataset[0]

    assert captured["name"] == "johnolafenwa/highschool-math-reasoning"
    assert captured["split"] == "train"
    assert isinstance(sample, EnvBase)
    assert sample.prompt == [
        {"role": "user", "content": "2 + 2"},
        {"role": "assistant", "content": "<think>easy</think>\n\\boxed{4}"},
    ]
    assert sample.answer is None


def test_highschool_math_reasoning_dataset_supports_test_split(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_load_dataset(name: str, split: str):
        captured["name"] = name
        captured["split"] = split
        return _DummyDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Answer"},
                    ]
                }
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningSFTDataset(train=False)
    sample = dataset[0]

    assert captured["name"] == "johnolafenwa/highschool-math-reasoning"
    assert captured["split"] == "test"
    assert sample.prompt == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_highschool_math_reasoning_dataset_respects_seed_and_max_samples(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset(
            [
                {"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"}]},
                {"messages": [{"role": "user", "content": "second"}, {"role": "assistant", "content": "b"}]},
                {"messages": [{"role": "user", "content": "third"}, {"role": "assistant", "content": "c"}]},
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningSFTDataset(train=True, seed=7, max_samples=2)
    assert len(dataset) == 2
    assert dataset[0].prompt[0]["content"] == "third"
    assert dataset[1].prompt[0]["content"] == "second"


def test_highschool_math_reasoning_dataset_missing_messages_raises(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"answer": "missing"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningSFTDataset(train=True)
    with pytest.raises(ValueError, match="Dataset row missing 'messages' field."):
        _ = dataset[0]


def test_highschool_math_reasoning_dataset_requires_assistant(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"messages": [{"role": "user", "content": "Question"}]}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningSFTDataset(train=True)
    with pytest.raises(ValueError, match="Dataset row must include at least one assistant message."):
        _ = dataset[0]


def test_highschool_math_reasoning_dataset_invalid_role_raises(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"messages": [{"role": "tool", "content": "x"}]}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningSFTDataset(train=True)
    with pytest.raises(ValueError, match="Unsupported role in dataset"):
        _ = dataset[0]
