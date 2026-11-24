"""Finds a spoken or typed phone number after a dial command in transcribed text."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9",
}
MULTIPLIERS = {"double": 2, "triple": 3, "treble": 3}
FILLERS = {"number", "to", "the", "a", "please", "on", "at", "this", "mobile", "phone"}
MAX_FILLERS = 3
_TOKEN = re.compile(r"\+|[^\W\d_]+|\d+", re.UNICODE)


def _digit_value(token: str) -> str | None:
    if token in DIGIT_WORDS:
        return DIGIT_WORDS[token]
    if token.isdigit():
        return "".join(str(unicodedata.digit(ch)) for ch in token)
    return None


@dataclass
class DialParser:
    triggers: list[str] = field(default_factory=lambda: ["dial", "call", "phone"])
    require_trigger: bool = True
    min_digits: int = 3
    max_digits: int = 15
    allow_plus: bool = True
    number_pattern: str | None = None

    def __post_init__(self) -> None:
        self._triggers = {t.lower() for t in self.triggers}
        self._pattern = re.compile(self.number_pattern) if self.number_pattern else None

    def find_numbers(self, text: str) -> list[str]:
        """Return valid numbers in order of appearance (duplicates in one text are kept once)."""
        tokens = _TOKEN.findall(text.lower())
        found: list[str] = []
        i = 0
        while i < len(tokens):
            start = i
            if self.require_trigger:
                if tokens[i] not in self._triggers:
                    i += 1
                    continue
                start = i + 1
                skipped = 0
                while start < len(tokens) and tokens[start] in FILLERS and skipped < MAX_FILLERS:
                    start += 1
                    skipped += 1
            number, end = self._read_number(tokens, start)
            if number and self._valid(number) and number not in found:
                found.append(number)
            i = max(end, i + 1)
        return found

    def _read_number(self, tokens: list[str], start: int) -> tuple[str, int]:
        k = start
        plus = False
        if k < len(tokens) and tokens[k] in ("+", "plus"):
            if self.allow_plus:
                plus = True
            k += 1
        digits: list[str] = []
        repeat = 1
        end = k
        while k < len(tokens):
            token = tokens[k]
            if token in MULTIPLIERS:
                repeat = MULTIPLIERS[token]
                k += 1
                continue
            value = _digit_value(token)
            if value is None:
                break
            digits.append(value * repeat if len(value) == 1 else value)
            repeat = 1
            k += 1
            end = k
        if not digits:
            return "", max(end, start)
        return ("+" if plus else "") + "".join(digits), end

    def _valid(self, number: str) -> bool:
        count = sum(ch.isdigit() for ch in number)
        if not self.min_digits <= count <= self.max_digits:
            return False
        return self._pattern.match(number) is not None if self._pattern else True
