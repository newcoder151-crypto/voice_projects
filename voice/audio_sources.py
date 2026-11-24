"""Live audio sources. Each yields float32 mono frames of live.frame_samples samples."""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from typing import Iterator, Protocol

import numpy as np

from voice.config import Config

log = logging.getLogger(__name__)


class AudioSource(Protocol):
    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class MicrophoneSource:
    def __init__(self, sample_rate: int, frame_samples: int, device):
        self._rate = sample_rate
        self._frame_samples = frame_samples
        self._device = device
        self._stop = threading.Event()

    def frames(self) -> Iterator[np.ndarray]:
        import sounddevice as sd  # lazy: needs PortAudio, only for live microphone mode

        buffer: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frame_count, time_info, status):
            if status:
                log.warning("Microphone status: %s", status)
            buffer.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=self._rate,
            channels=1,
            dtype="float32",
            blocksize=self._frame_samples,
            device=self._device,
            callback=callback,
        ):
            log.info("Listening on the microphone")
            while not self._stop.is_set():
                try:
                    yield buffer.get(timeout=0.5)
                except queue.Empty:
                    continue

    def close(self) -> None:
        self._stop.set()


class StreamSource:
    """Reads any network or device stream through ffmpeg (RTP, RTSP, HTTP, UDP, ...)."""

    def __init__(self, url: str, ffmpeg_path: str, input_args: list[str], sample_rate: int, frame_samples: int):
        if not url:
            raise ValueError("live.stream_url is empty; set it in config/settings.yaml")
        self._cmd = [
            ffmpeg_path, "-hide_banner", "-loglevel", "error",
            *[str(a) for a in input_args],
            "-i", url,
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
        ]
        self._frame_bytes = frame_samples * 4
        self._proc: subprocess.Popen | None = None

    def frames(self) -> Iterator[np.ndarray]:
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE)
        log.info("Reading audio stream through ffmpeg")
        assert self._proc.stdout is not None
        try:
            while True:
                data = self._proc.stdout.read(self._frame_bytes)
                if len(data) < self._frame_bytes:
                    log.info("Audio stream ended")
                    return
                yield np.frombuffer(data, dtype=np.float32).copy()
        finally:
            self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def build_source(cfg: Config) -> AudioSource:
    rate = int(cfg.get("live.sample_rate", 16000))
    frame = int(cfg.get("live.frame_samples", 512))
    kind = str(cfg.get("live.source", "microphone")).lower()
    if kind == "microphone":
        return MicrophoneSource(rate, frame, cfg.get("live.microphone_device", None))
    if kind == "stream":
        return StreamSource(
            url=str(cfg.get("live.stream_url", "")),
            ffmpeg_path=str(cfg.get("live.ffmpeg_path", "ffmpeg")),
            input_args=list(cfg.get("live.ffmpeg_input_args", []) or []),
            sample_rate=rate,
            frame_samples=frame,
        )
    raise ValueError(f"live.source must be 'microphone' or 'stream', got: {kind}")
