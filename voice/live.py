"""Live pipeline: audio frames -> VAD -> phrases -> a handler running on its own thread."""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Callable

import numpy as np

from voice.audio_sources import AudioSource, build_source
from voice.config import Config
from voice.models import Utterance
from voice.vad import StreamingVAD, build_streaming_vad

log = logging.getLogger(__name__)


class LiveSegmenter:
    """Turns a stream of frames into phrases using VAD start/end events."""

    def __init__(
        self,
        vad: StreamingVAD,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        pre_roll_ms: int = 300,
        min_speech_ms: int = 250,
        max_utterance_sec: float = 20.0,
        trailing_silence_ms: int = 0,
    ):
        self._vad = vad
        self._rate = sample_rate
        self._frame_sec = frame_samples / sample_rate
        self._min_frames = max(1, int(min_speech_ms / 1000 / self._frame_sec))
        self._max_frames = max(1, int(max_utterance_sec / self._frame_sec))
        # Silence the VAD waits for before it reports "end"; not counted as speech.
        self._trailing_frames = int(trailing_silence_ms / 1000 / self._frame_sec)
        self._preroll_used = 0
        self._preroll: deque[np.ndarray] = deque(maxlen=max(0, int(pre_roll_ms / 1000 / self._frame_sec)))
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._start_time = 0.0
        self._index = 0

    def feed(self, frame: np.ndarray) -> list[Utterance]:
        event = self._vad.process(frame)
        now = self._index * self._frame_sec
        self._index += 1

        if not self._speaking:
            if event == "start":
                self._speaking = True
                self._preroll_used = len(self._preroll)
                self._start_time = now - self._preroll_used * self._frame_sec
                self._buffer = list(self._preroll) + [frame]
                self._preroll.clear()
            else:
                self._preroll.append(frame)
            return []

        self._buffer.append(frame)
        if event == "end":
            return self._finish(now + self._frame_sec, self._trailing_frames)
        if len(self._buffer) >= self._max_frames:
            self._vad.reset()
            return self._finish(now + self._frame_sec, 0)
        return []

    def flush(self) -> list[Utterance]:
        """Call when the stream ends so a phrase in progress is not lost."""
        if self._speaking:
            return self._finish(self._index * self._frame_sec, 0)
        return []

    def _finish(self, end_time: float, trailing_frames: int) -> list[Utterance]:
        frames, self._buffer = self._buffer, []
        self._speaking = False
        speech_frames = len(frames) - self._preroll_used - trailing_frames
        if speech_frames < self._min_frames:
            return []
        return [Utterance(audio=np.concatenate(frames), start=self._start_time, end=end_time)]


def build_segmenter(cfg: Config, vad: StreamingVAD | None = None) -> LiveSegmenter:
    return LiveSegmenter(
        vad=vad or build_streaming_vad(cfg),
        sample_rate=int(cfg.get("live.sample_rate", 16000)),
        frame_samples=int(cfg.get("live.frame_samples", 512)),
        pre_roll_ms=int(cfg.get("live.pre_roll_ms", 300)),
        min_speech_ms=int(cfg.get("live.min_speech_ms", 250)),
        max_utterance_sec=float(cfg.get("live.max_utterance_sec", 20)),
        trailing_silence_ms=int(cfg.get("live.vad.min_silence_ms", 500)),
    )


def run_live(
    source: AudioSource,
    segmenter: LiveSegmenter,
    handler: Callable[[Utterance], None],
    stop: threading.Event,
) -> None:
    """Capture on this thread, run the handler (ASR etc.) on a worker so audio is never missed."""
    pending: queue.Queue[Utterance | None] = queue.Queue()

    def worker() -> None:
        while True:
            item = pending.get()
            if item is None:
                return
            try:
                handler(item)
            except Exception:
                log.exception("Failed to process a live phrase")

    thread = threading.Thread(target=worker, name="live-handler", daemon=True)
    thread.start()
    try:
        for frame in source.frames():
            for utterance in segmenter.feed(frame):
                pending.put(utterance)
            if stop.is_set():
                break
        for utterance in segmenter.flush():
            pending.put(utterance)
    finally:
        source.close()
        pending.put(None)
        thread.join()


def run_live_from_config(cfg: Config, handler: Callable[[Utterance], None], stop: threading.Event) -> None:
    run_live(build_source(cfg), build_segmenter(cfg), handler, stop)
