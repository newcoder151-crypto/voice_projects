"""Writes transcripts to files. Audio files get .txt + .json; live sessions get .txt + .jsonl."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from voice.models import Segment


def format_time(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _segment_dict(seg: Segment) -> dict:
    return {
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text,
        "confidence": None if seg.confidence is None else round(seg.confidence, 4),
        "language": seg.language,
    }


def _txt_line(seg: Segment, with_times: bool) -> str:
    if with_times:
        return f"[{format_time(seg.start)} --> {format_time(seg.end)}] {seg.text}"
    return seg.text


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_file_transcript(
    out_dir: Path, source: Path, segments: Iterable[Segment], with_times: bool = True
) -> tuple[Path, Path]:
    """Write <name>.txt and <name>.json for one audio file. Returns both paths."""
    segments = list(segments)
    stem = Path(source).stem
    txt_path, json_path = out_dir / f"{stem}.txt", out_dir / f"{stem}.json"
    _atomic_write(txt_path, "\n".join(_txt_line(s, with_times) for s in segments) + ("\n" if segments else ""))
    document = {
        "source": Path(source).name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "language": segments[0].language if segments else None,
        "text": " ".join(s.text for s in segments),
        "segments": [_segment_dict(s) for s in segments],
    }
    _atomic_write(json_path, json.dumps(document, ensure_ascii=False, indent=2))
    return txt_path, json_path


class LiveSession:
    """Appends live phrases to a .txt and a .jsonl file, flushing after every phrase."""

    def __init__(self, out_dir: Path, with_times: bool = True, now: datetime | None = None):
        out_dir.mkdir(parents=True, exist_ok=True)
        started = now or datetime.now()
        stem = f"live_{started:%Y%m%d_%H%M%S}"
        self.txt_path = out_dir / f"{stem}.txt"
        self.jsonl_path = out_dir / f"{stem}.jsonl"
        self._with_times = with_times
        self._txt = self.txt_path.open("a", encoding="utf-8")
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")

    def add(self, segments: Iterable[Segment]) -> None:
        for seg in segments:
            self._txt.write(_txt_line(seg, self._with_times) + "\n")
            self._jsonl.write(json.dumps(_segment_dict(seg), ensure_ascii=False) + "\n")
        self._txt.flush()
        self._jsonl.flush()

    def close(self) -> None:
        self._txt.close()
        self._jsonl.close()
