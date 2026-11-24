from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One piece of transcribed speech with times in seconds."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    language: str | None = None


@dataclass
class Utterance:
    """A phrase of live audio (float32, mono) cut out by the VAD."""

    audio: np.ndarray
    start: float
    end: float
