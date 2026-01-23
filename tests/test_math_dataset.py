import math
import random

import pytest

from rlfusion.datasets.rlvr import MathDataset
from rlfusion.envs import EnvBase


@pytest.mark.parametrize(
    "operand, symbol, fn",
    [
        ("mult", "*", lambda a, b: a * b),
        ("add", "+", lambda a, b: a + b),
        ("substraction", "-", lambda a, b: a - b),
        ("division", "/", lambda a, b: a / b),
    ],
)
def test_math_dataset_samples_match_expected(operand, symbol, fn):
    random.seed(123)
    dataset = MathDataset(num_samples=3, min_val=1, max_val=5, operand=operand)

    random.seed(123)
    expected_pairs = [(random.randint(1, 5), random.randint(1, 5)) for _ in range(3)]

    assert len(dataset) == 3

    for idx, (a, b) in enumerate(expected_pairs):
        sample = dataset[idx]
        assert isinstance(sample, EnvBase)
        assert sample.prompt[0]["role"] == "system"
        assert sample.prompt[1]["role"] == "user"
        assert sample.prompt[1]["content"] == f"What is {a} {symbol} {b} ?"
        expected_value = fn(a, b)
        assert isinstance(sample.answer, (int, float))
        assert math.isclose(float(sample.answer), expected_value, rel_tol=0, abs_tol=0)


def test_math_dataset_invalid_operand_raises():
    with pytest.raises(ValueError):
        MathDataset(operand="pow")
