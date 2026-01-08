from rlfusion.utils import get_boxed_answer


def test_get_boxed_answer_none():
    assert get_boxed_answer("no boxed answer here") is None


def test_get_boxed_answer_simple():
    assert get_boxed_answer("Final: \\boxed{42}") == "42"


def test_get_boxed_answer_multiple_returns_last():
    text = "First \\boxed{1} then \\boxed{2}"
    assert get_boxed_answer(text) == "2"


def test_get_boxed_answer_nested_braces():
    text = "Answer: \\boxed{\\frac{1}{2}}"
    assert get_boxed_answer(text) == "\\frac{1}{2}"
