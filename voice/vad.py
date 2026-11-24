"""Streaming voice activity detection for live audio (Silero VAD)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from voice.config import Config


class StreamingVAD(Protocol):
    def process(self, frame: np.ndarray) -> str | None:
        """Feed one frame. Returns "start" when speech begins, "end" when it ends, else None."""

    def reset(self) -> None: ...


class SileroStreamingVAD:
    """Wraps silero_vad.VADIterator. The Silero model ships inside the pip package (works offline)."""

    def __init__(self, threshold: float, min_silence_ms: int, speech_pad_ms: int, sample_rate: int):
        import torch  # lazy: only needed for live mode
        from silero_vad import VADIterator, load_silero_vad

        self._torch = torch
        self._iterator = VADIterator(
            load_silero_vad(),
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, frame: np.ndarray) -> str | None:
        result = self._iterator(self._torch.from_numpy(frame), return_seconds=False)
        if not result:
            return None
        if "start" in result:
            return "start"
        if "end" in result:
            return "end"
        return None

    def reset(self) -> None:
        self._iterator.reset_states()


def build_streaming_vad(cfg: Config) -> StreamingVAD:
    return SileroStreamingVAD(
        threshold=float(cfg.get("live.vad.threshold", 0.5)),
        min_silence_ms=int(cfg.get("live.vad.min_silence_ms", 500)),
        speech_pad_ms=int(cfg.get("live.vad.speech_pad_ms", 100)),
        sample_rate=int(cfg.get("live.sample_rate", 16000)),
    )
