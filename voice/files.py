"""Audio file discovery, processed-file bookkeeping and a folder watcher."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def list_audio_files(directory: Path, extensions: list[str]) -> list[Path]:
    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in wanted)


class ProcessedState:
    """Remembers finished files (name + size + modified time) across restarts."""

    def __init__(self, state_file: Path):
        self._file = state_file
        self._done: set[str] = set()
        if state_file.is_file():
            try:
                self._done = set(json.loads(state_file.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                log.warning("Could not read state file %s; starting empty", state_file)

    @staticmethod
    def key(path: Path) -> str:
        st = path.stat()
        return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"

    def is_done(self, path: Path) -> bool:
        return self.key(path) in self._done

    def mark_done(self, path: Path) -> None:
        self._done.add(self.key(path))
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(self._file.suffix + ".tmp")
        tmp.write_text(json.dumps(sorted(self._done)), encoding="utf-8")
        os.replace(tmp, self._file)


def watch_directory(
    directory: Path,
    extensions: list[str],
    state: ProcessedState,
    handler: Callable[[Path], None],
    poll_seconds: float,
    settle_seconds: float,
    max_retries: int,
    stop: threading.Event,
) -> None:
    """Process new audio files as they appear. Failed files are retried, then skipped."""
    directory.mkdir(parents=True, exist_ok=True)
    failures: dict[str, int] = {}
    log.info("Watching %s for audio files", directory)
    while not stop.is_set():
        for path in list_audio_files(directory, extensions):
            if stop.is_set():
                break
            try:
                if state.is_done(path):
                    continue
                if time.time() - path.stat().st_mtime < settle_seconds:
                    continue  # probably still being copied
                key = ProcessedState.key(path)
                if failures.get(key, 0) >= max_retries:
                    continue
                try:
                    handler(path)
                except Exception:
                    failures[key] = failures.get(key, 0) + 1
                    log.exception("Failed to process %s (attempt %d of %d)", path.name, failures[key], max_retries)
                    continue
                state.mark_done(path)
            except FileNotFoundError:
                continue  # file removed while scanning
        stop.wait(poll_seconds)
