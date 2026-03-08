from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import regex as re
from torch.utils.data import Dataset

from rlfusion.envs import EnvBase

THINK_BLOCK_REWARD = 0.05
BOXED_ANSWER_REWARD = 0.05
STRICT_TERMINAL_BOX_REWARD = 0.10
STRICT_FORMAT_REWARD = THINK_BLOCK_REWARD + BOXED_ANSWER_REWARD + STRICT_TERMINAL_BOX_REWARD
CORRECT_ANSWER_REWARD = 0.80
FORMAT_INSTRUCTION = (
    "Solve the problem and respond in exactly this format:\n"
    "<think>your reasoning</think>\n"
    "\\boxed{final answer}\n"
    "Use exactly one top-level <think>...</think> block and end the response with a single boxed answer."
)

_STRICT_THINK_PATTERN = re.compile(r"\A<think>(?P<think>[\s\S]+?)</think>(?P<answer>[\s\S]+)\Z")
_BOXED_PATTERN = re.compile(
    r"(?(DEFINE)(?P<BRACE>\{(?:[^{}]+|(?&BRACE))*\}))"
    r"\\boxed\{(?P<content>(?:[^{}]+|(?&BRACE))*)\}"
)


@lru_cache(maxsize=1)
def _get_math_verify_components():
    try:
        from math_verify import parse, verify
        from math_verify.parser import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            NormalizationConfig,
            StringExtractionConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "math-verify is required for HighSchoolMathReasoningRLDataset. "
            "Install project dependencies with: uv sync --extra dev --extra test"
        ) from exc

    return parse, verify, ExprExtractionConfig, LatexExtractionConfig, NormalizationConfig, StringExtractionConfig


def _normalize_answer_text(text: str) -> str:
    return " ".join(text.split())


def _build_extraction_config(reference_answer: str) -> list[object]:
    _, _, ExprExtractionConfig, LatexExtractionConfig, NormalizationConfig, StringExtractionConfig = (
        _get_math_verify_components()
    )
    return [
        LatexExtractionConfig(
            try_extract_without_anchor=True,
            normalization_config=NormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=True,
                nits=True,
                boxed="all",
                equations=False,
            ),
        ),
        ExprExtractionConfig(try_extract_without_anchor=True),
        StringExtractionConfig(
            strings=(reference_answer,),
            try_extract_without_anchor=True,
            lowercase=False,
        ),
    ]


def _parse_wrapped_answer(text: str, reference_answer: str) -> tuple[object, ...]:
    parse, _, _, _, _, _ = _get_math_verify_components()
    parsed = parse(
        text,
        extraction_config=_build_extraction_config(reference_answer),
        extraction_mode="first_match",
        fallback_mode="first_match",
        raise_on_error=False,
    )
    return tuple(parsed)


@lru_cache(maxsize=2048)
def _parse_gold_answer(answer: str) -> tuple[object, ...]:
    normalized_answer = answer.strip()
    return _parse_wrapped_answer(f"\\boxed{{{normalized_answer}}}", normalized_answer)


def _extract_reasoning_sections(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return None

    think_start = len("<think>")
    think_end = stripped.find("</think>", think_start)
    if think_end < 0:
        return None

    think_content = stripped[think_start:think_end]
    answer_text = stripped[think_end + len("</think>") :].strip()
    if not think_content.strip() or not answer_text:
        return None
    if "\\boxed{" in think_content:
        return None
    return think_content, answer_text


def _extract_strict_reasoning_sections(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    match = _STRICT_THINK_PATTERN.fullmatch(stripped)
    if match is None:
        return None

    think_content = match.group("think")
    answer_text = match.group("answer").strip()
    if not think_content.strip() or not answer_text:
        return None
    if "<think>" in think_content or "</think>" in think_content:
        return None
    if "<think>" in answer_text or "</think>" in answer_text:
        return None
    if "\\boxed{" in think_content:
        return None
    return think_content, answer_text


def _extract_boxed_answers(answer_text: str) -> list[str]:
    boxed_answers: list[str] = []
    for match in _BOXED_PATTERN.finditer(answer_text):
        boxed_answer = match.group("content").strip()
        if boxed_answer:
            boxed_answers.append(boxed_answer)
    return boxed_answers


def _extract_terminal_boxed_answer(answer_text: str) -> str | None:
    stripped = answer_text.strip()
    matches = list(_BOXED_PATTERN.finditer(stripped))
    if len(matches) != 1:
        return None

    match = matches[0]
    if stripped[match.end() :].strip():
        return None

    boxed_answer = match.group("content").strip()
    return boxed_answer or None


def _answers_match(candidate_answer: str, gold_answer: str, parsed_gold_answer: tuple[object, ...]) -> bool:
    normalized_candidate = candidate_answer.strip()
    normalized_gold = gold_answer.strip()
    if not normalized_candidate or not normalized_gold:
        return False

    parsed_candidate = _parse_wrapped_answer(
        f"\\boxed{{{normalized_candidate}}}",
        normalized_gold,
    )
    if parsed_gold_answer and parsed_candidate:
        _, verify, _, _, _, _ = _get_math_verify_components()
        if verify(
            list(parsed_gold_answer),
            list(parsed_candidate),
            strict=True,
            raise_on_error=False,
        ):
            return True

    return _normalize_answer_text(normalized_candidate) == _normalize_answer_text(normalized_gold)


@dataclass
class HighSchoolMathReasoningRLEnv(EnvBase):
    parsed_answer: tuple[object, ...] = field(default_factory=tuple, repr=False)

    def get_reward(self, prediction: str | None) -> float:
        if prediction is None or self.answer is None:
            return 0.0

        prediction_text = str(prediction)
        sections = _extract_reasoning_sections(prediction_text)
        if sections is None:
            return 0.0

        _, answer_text = sections
        reward = THINK_BLOCK_REWARD

        boxed_answers = _extract_boxed_answers(answer_text)
        if not boxed_answers:
            return reward

        reward += BOXED_ANSWER_REWARD
        if any(_answers_match(boxed_answer, str(self.answer), self.parsed_answer) for boxed_answer in boxed_answers):
            reward += CORRECT_ANSWER_REWARD

        strict_sections = _extract_strict_reasoning_sections(prediction_text)
        if strict_sections is None:
            return min(reward, 1.0)

        _, strict_answer_text = strict_sections
        if _extract_terminal_boxed_answer(strict_answer_text) is not None:
            reward += STRICT_TERMINAL_BOX_REWARD

        return min(reward, 1.0)


class HighSchoolMathReasoningRLDataset(Dataset):
    """GRPO adapter for johnolafenwa/highschool-math-reasoning-rl."""

    def __init__(
        self,
        train: bool = True,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for HighSchoolMathReasoningRLDataset. Install with: uv pip install datasets"
            ) from exc

        selected_split = split or ("train" if train else "test")
        if selected_split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        dataset = load_dataset("johnolafenwa/highschool-math-reasoning-rl", split=selected_split)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.dataset = dataset

    def __getitem__(self, index: int) -> HighSchoolMathReasoningRLEnv:
        row = self.dataset[index]
        messages = row.get("messages")
        if messages is None:
            raise ValueError("Dataset row missing 'messages' field.")
        if not isinstance(messages, list):
            raise ValueError("Dataset row 'messages' must be a list.")
        if not messages:
            raise ValueError("Dataset row 'messages' must not be empty.")
        if row.get("answer") is None:
            raise ValueError("Dataset row missing 'answer' field.")

        prompt_messages: list[dict[str, object]] = []
        last_user_idx: int | None = None
        for idx, message in enumerate(messages[:-1]):
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dict with role/content.")
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role in dataset: {role}")
            content = str(message.get("content", ""))
            prompt_messages.append({"role": str(role), "content": content})
            if role == "user":
                last_user_idx = idx

        final_message = messages[-1]
        if not isinstance(final_message, dict):
            raise ValueError("Each message must be a dict with role/content.")
        if final_message.get("role") != "assistant":
            raise ValueError("Dataset row must end with an assistant message.")
        if last_user_idx is None:
            raise ValueError("Dataset row must include at least one user message before the assistant answer.")

        prompt_messages[last_user_idx]["content"] = (
            f"{prompt_messages[last_user_idx]['content']}\n\n{FORMAT_INSTRUCTION}"
        )

        answer = str(row["answer"])
        return HighSchoolMathReasoningRLEnv(
            prompt=prompt_messages,
            answer=answer,
            parsed_answer=_parse_gold_answer(answer),
        )

    def __len__(self) -> int:
        return len(self.dataset)
