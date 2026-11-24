"""Command line entry point.

  python -m voice dial file <audio file or folder>
  python -m voice dial watch
  python -m voice dial live
  python -m voice transcribe file|watch|live      (same modes)
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from pathlib import Path

from voice.config import Config
from voice.files import ProcessedState, list_audio_files, watch_directory
from voice.logging_setup import setup_logging

log = logging.getLogger("voice")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voice", description="Offline voice dial and transcription")
    parser.add_argument("--config", default="config/settings.yaml", help="path to settings.yaml")
    projects = parser.add_subparsers(dest="project", required=True)
    for name, helptext in (("dial", "voice dial: send the spoken number over UDP"),
                           ("transcribe", "speech to text files")):
        sub = projects.add_parser(name, help=helptext).add_subparsers(dest="mode", required=True)
        file_cmd = sub.add_parser("file", help="process one audio file or every audio file in a folder")
        file_cmd.add_argument("path", type=Path)
        sub.add_parser("watch", help="process new files in the input folder as they appear")
        sub.add_parser("live", help="listen to live audio")
    return parser.parse_args(argv)


def build_service(project: str, cfg: Config, transcriber):
    if project == "dial":
        from voice.dial.service import DialService
        return DialService(cfg, transcriber)
    from voice.transcribe.service import TranscribeService
    return TranscribeService(cfg, transcriber)


def main(argv: list[str] | None = None) -> int:
    # Make sure nothing tries to reach the internet to download a model.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args = parse_args(argv)
    cfg = Config.load(args.config)
    setup_logging(cfg)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    from voice.asr import Transcriber
    transcriber = Transcriber(cfg)
    transcriber.load()
    service = build_service(args.project, cfg, transcriber)
    extensions = list(cfg.get("files.extensions"))

    try:
        if args.mode == "file":
            targets = list_audio_files(args.path, extensions) if args.path.is_dir() else [args.path]
            if not targets:
                log.error("No audio files found at %s", args.path)
                return 1
            for target in targets:
                service.process_file(target)
        elif args.mode == "watch":
            state = ProcessedState(cfg.path("paths.state_dir") / f"{args.project}_processed.json")
            watch_directory(
                directory=cfg.path(f"{args.project}.input_dir"),
                extensions=extensions,
                state=state,
                handler=service.process_file,
                poll_seconds=float(cfg.get("files.poll_seconds", 5)),
                settle_seconds=float(cfg.get("files.settle_seconds", 2)),
                max_retries=int(cfg.get("files.max_retries", 3)),
                stop=stop,
            )
        else:
            from voice.live import run_live_from_config
            run_live_from_config(cfg, service.handle_utterance, stop)
    finally:
        service.close()
    return 0
