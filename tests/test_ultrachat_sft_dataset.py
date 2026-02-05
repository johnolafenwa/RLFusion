import pytest

from rlfusion.datasets.ultrachat_sft import UltraChatSFTDataset
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
        # Deterministic behavior for tests; we only need to verify ordering changes.
        _ = seed
        return _DummyDataset(list(reversed(self._items)))


def test_ultrachat_dataset_uses_train_sft_split(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_load_dataset(name: str, split: str):
        captured["name"] = name
        captured["split"] = split
        return _DummyDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                }
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = UltraChatSFTDataset(train=True)
    sample = dataset[0]

    assert captured["name"] == "HuggingFaceH4/ultrachat_200k"
    assert captured["split"] == "train_sft"
    assert isinstance(sample, EnvBase)
    assert sample.prompt == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_ultrachat_dataset_supports_system_prompt_and_test_split(monkeypatch) -> None:
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

    dataset = UltraChatSFTDataset(train=False, system_prompt="Be concise.")
    sample = dataset[0]

    assert captured["name"] == "HuggingFaceH4/ultrachat_200k"
    assert captured["split"] == "test_sft"
    assert sample.prompt[0] == {"role": "system", "content": "Be concise."}
    assert sample.prompt[1:] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_ultrachat_dataset_respects_seed_and_max_samples(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset(
            [
                {"messages": [{"role": "user", "content": "first"}]},
                {"messages": [{"role": "user", "content": "second"}]},
                {"messages": [{"role": "user", "content": "third"}]},
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = UltraChatSFTDataset(train=True, seed=7, max_samples=2)
    assert len(dataset) == 2
    assert dataset[0].prompt[0]["content"] == "third"
    assert dataset[1].prompt[0]["content"] == "second"


def test_ultrachat_dataset_missing_messages_raises(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"prompt": "missing"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = UltraChatSFTDataset(train=True)
    with pytest.raises(ValueError, match="Dataset row missing 'messages' field."):
        _ = dataset[0]


def test_ultrachat_dataset_invalid_role_raises(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"messages": [{"role": "tool", "content": "x"}]}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = UltraChatSFTDataset(train=True)
    with pytest.raises(ValueError, match="Unsupported role in dataset"):
        _ = dataset[0]
