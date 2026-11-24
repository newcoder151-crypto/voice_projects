"""faster-whisper wrapper. Loads the model from a local folder only (offline)."""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from voice.config import Config
from voice.models import Segment

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            model_path: Path = self._cfg.path("asr.model_path")
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"ASR model folder not found: {model_path}. Copy the model files from the "
                    "offline bundle into this folder (see README)."
                )
            from faster_whisper import WhisperModel  # imported lazily so tests need no GPU stack

            log.info("Loading ASR model from %s", model_path)
            self._model = WhisperModel(
                str(model_path),
                device=self._cfg.get("asr.device", "cuda"),
                device_index=self._cfg.get("asr.device_index", 0),
                compute_type=self._cfg.get("asr.compute_type", "float16"),
                local_files_only=True,
            )
            device = getattr(getattr(self._model, "model", None), "device", "unknown")
            log.info("ASR model ready on device: %s", device)

    def transcribe_file(
        self, path: str | Path, language: str | None = None, initial_prompt: str | None = None
    ) -> list[Segment]:
        """Transcribe a whole audio file. faster-whisper decodes it and applies Silero VAD."""
        vad = dict(self._cfg.get("asr.file_vad", {}) or {})
        return self._run(str(path), 0.0, language, initial_prompt, vad_filter=True, vad_parameters=vad)

    def transcribe_array(
        self,
        audio: np.ndarray,
        offset: float = 0.0,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe an already-cut phrase (float32, 16 kHz, mono). Times are shifted by offset."""
        return self._run(audio, offset, language, initial_prompt, vad_filter=False, vad_parameters=None)

    def _run(
        self,
        audio: Any,
        offset: float,
        language: str | None,
        initial_prompt: str | None,
        vad_filter: bool,
        vad_parameters: dict | None,
    ) -> list[Segment]:
        self.load()
        language = language or self._cfg.get("asr.language", None)
        with self._lock:
            raw_segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=int(self._cfg.get("asr.beam_size", 5)),
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters or None,
                condition_on_previous_text=False,
            )
            segments = []
            for seg in raw_segments:  # the generator does the real work, so keep it under the lock
                text = seg.text.strip()
                if not text:
                    continue
                segments.append(
                    Segment(
                        start=offset + float(seg.start),
                        end=offset + float(seg.end),
                        text=text,
                        confidence=math.exp(seg.avg_logprob) if seg.avg_logprob is not None else None,
                        language=info.language,
                    )
                )
        return segments
