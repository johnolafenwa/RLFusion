import pytest

from rlfusion.datasets import HighSchoolMathReasoningRLDataset
from rlfusion.datasets.highschool_math_reasoning_rl import (
    FORMAT_INSTRUCTION,
    STRICT_FORMAT_REWARD,
    HighSchoolMathReasoningRLEnv,
    _parse_gold_answer,
)
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


def _build_env(answer: str) -> HighSchoolMathReasoningRLEnv:
    return HighSchoolMathReasoningRLEnv(
        prompt=[{"role": "user", "content": "Question"}],
        answer=answer,
        parsed_answer=_parse_gold_answer(answer),
    )


def test_highschool_math_reasoning_rl_dataset_uses_train_split(monkeypatch) -> None:
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

    dataset = HighSchoolMathReasoningRLDataset(train=True)
    sample = dataset[0]

    assert captured["name"] == "johnolafenwa/highschool-math-reasoning-rl"
    assert captured["split"] == "train"
    assert isinstance(sample, EnvBase)
    assert sample.prompt == [
        {"role": "user", "content": f"2 + 2\n\n{FORMAT_INSTRUCTION}"},
    ]
    assert sample.answer == "4"
    assert sample.get_reward("<think>add</think>\n\\boxed{4}") == 1.0


def test_highschool_math_reasoning_rl_dataset_supports_test_split(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_load_dataset(name: str, split: str):
        captured["name"] = name
        captured["split"] = split
        return _DummyDataset(
            [
                {
                    "answer": "7",
                    "messages": [
                        {"role": "system", "content": "Solve carefully."},
                        {"role": "user", "content": "3 + 4"},
                        {"role": "assistant", "content": "<think>sum</think>\n\\boxed{7}"},
                    ],
                }
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningRLDataset(train=False)
    sample = dataset[0]

    assert captured["name"] == "johnolafenwa/highschool-math-reasoning-rl"
    assert captured["split"] == "test"
    assert sample.prompt == [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": f"3 + 4\n\n{FORMAT_INSTRUCTION}"},
    ]


def test_highschool_math_reasoning_rl_dataset_respects_seed_and_max_samples(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset(
            [
                {
                    "answer": "1",
                    "messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"}],
                },
                {
                    "answer": "2",
                    "messages": [{"role": "user", "content": "second"}, {"role": "assistant", "content": "b"}],
                },
                {
                    "answer": "3",
                    "messages": [{"role": "user", "content": "third"}, {"role": "assistant", "content": "c"}],
                },
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningRLDataset(train=True, seed=7, max_samples=2)
    assert len(dataset) == 2
    assert dataset[0].prompt[0]["content"] == f"third\n\n{FORMAT_INSTRUCTION}"
    assert dataset[1].prompt[0]["content"] == f"second\n\n{FORMAT_INSTRUCTION}"


def test_highschool_math_reasoning_rl_reward_uses_math_verify_for_numeric_equivalence() -> None:
    env = _build_env("50196")

    assert env.get_reward("<think>multiply</think>\nFinal answer: \\boxed{50,196}") == 1.0
    assert env.get_reward("<think>multiply</think>\n\\boxed{50195}") == STRICT_FORMAT_REWARD


def test_highschool_math_reasoning_rl_reward_uses_math_verify_for_matrix_equivalence() -> None:
    answer = "\\begin{pmatrix}6 & -7\\\\ -2 & 4\\end{pmatrix}"
    env = _build_env(answer)

    assert env.get_reward(
        "<think>swap diagonal and negate off-diagonal</think>\n"
        "\\boxed{\\begin{pmatrix}6 & -7\\\\ -2 & 4\\end{pmatrix}}"
    ) == 1.0


@pytest.mark.parametrize(
    ("prediction", "expected_reward"),
    [
        ("\\boxed{4}", 0.0),
        ("<think></think>\n\\boxed{4}", 0.0),
        ("<think>reasoning \\boxed{4}</think>\n\\boxed{4}", 0.0),
        ("<think>reasoning</think>\n\\boxed{4}\nThanks", 0.0),
        ("<think>reasoning</think>\n\\boxed{1} \\boxed{4}", 0.0),
        ("<think>reasoning</think>\nAnswer: 4", 0.0),
        ("<think>reasoning</think>\n\\boxed{5}", STRICT_FORMAT_REWARD),
    ],
)
def test_highschool_math_reasoning_rl_reward_enforces_strict_format(
    prediction: str,
    expected_reward: float,
) -> None:
    env = _build_env("4")

    assert env.get_reward(prediction) == expected_reward


def test_highschool_math_reasoning_rl_dataset_missing_messages_raises(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"answer": "4"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningRLDataset(train=True)
    with pytest.raises(ValueError, match="Dataset row missing 'messages' field."):
        _ = dataset[0]


def test_highschool_math_reasoning_rl_dataset_requires_assistant_final_turn(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset([{"answer": "4", "messages": [{"role": "user", "content": "Question"}]}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningRLDataset(train=True)
    with pytest.raises(ValueError, match="Dataset row must end with an assistant message."):
        _ = dataset[0]


def test_highschool_math_reasoning_rl_dataset_requires_user_before_assistant(monkeypatch) -> None:
    def fake_load_dataset(name: str, split: str):
        _ = (name, split)
        return _DummyDataset(
            [
                {
                    "answer": "4",
                    "messages": [
                        {"role": "system", "content": "Only system"},
                        {"role": "assistant", "content": "<think>x</think>\n\\boxed{4}"},
                    ],
                }
            ]
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    dataset = HighSchoolMathReasoningRLDataset(train=True)
    with pytest.raises(ValueError, match="Dataset row must include at least one user message"):
        _ = dataset[0]
