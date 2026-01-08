from typing import Optional

import regex as re

_BOXED_PATTERN = re.compile(
    r"(?(DEFINE)(?P<BRACE>\{(?:[^{}]+|(?&BRACE))*\}))"
    r"\\boxed\{(?P<content>(?:[^{}]+|(?&BRACE))*)\}"
)


def get_boxed_answer(text: str) -> Optional[str]:
    last_match = None
    for match in _BOXED_PATTERN.finditer(text):
        last_match = match.group("content")
    return last_match
    
