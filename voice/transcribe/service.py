"""Project 2: transcribe audio files and live audio into text files."""
from __future__ import annotations

import logging
from pathlib import Path

from voice.config import Config
from voice.models import Utterance
from voice.transcribe.writer import LiveSession, write_file_transcript

log = logging.getLogger(__name__)


class TranscribeService:
    def __init__(self, cfg: Config, transcriber):
        self._transcriber = transcriber
        self._out_dir = cfg.path("transcribe.output_dir")
        self._language = cfg.get("transcribe.language", None)
        self._prompt = cfg.get("transcribe.initial_prompt", None)
        self._with_times = bool(cfg.get("transcribe.timestamps_in_txt", True))
        self._session: LiveSession | None = None

    def process_file(self, path: Path) -> tuple[Path, Path]:
        segments = self._transcriber.transcribe_file(path, language=self._language, initial_prompt=self._prompt)
        txt_path, json_path = write_file_transcript(self._out_dir, Path(path), segments, self._with_times)
        log.info("%s -> %s (%d segments)", Path(path).name, txt_path.name, len(segments))
        return txt_path, json_path

    def handle_utterance(self, utterance: Utterance) -> None:
        segments = self._transcriber.transcribe_array(
            utterance.audio, offset=utterance.start, language=self._language, initial_prompt=self._prompt
        )
        if not segments:
            return
        if self._session is None:
            self._session = LiveSession(self._out_dir, self._with_times)
            log.info("Live transcript: %s", self._session.txt_path)
        self._session.add(segments)
        for seg in segments:
            log.info("Heard: %s", seg.text)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
