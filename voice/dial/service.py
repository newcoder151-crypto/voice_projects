"""Project 1: voice dial. Transcribe, find the number after a dial command, send it over UDP."""
from __future__ import annotations

import logging
from pathlib import Path

from voice.config import Config
from voice.dial.parser import DialParser
from voice.dial.sender import UdpSender
from voice.models import Utterance

log = logging.getLogger(__name__)


def build_parser(cfg: Config) -> DialParser:
    return DialParser(
        triggers=list(cfg.get("dial.triggers", ["dial", "call", "phone"])),
        require_trigger=bool(cfg.get("dial.require_trigger", True)),
        min_digits=int(cfg.get("dial.min_digits", 3)),
        max_digits=int(cfg.get("dial.max_digits", 15)),
        allow_plus=bool(cfg.get("dial.allow_plus", True)),
        number_pattern=cfg.get("dial.number_pattern", None),
    )


def build_sender(cfg: Config) -> UdpSender:
    return UdpSender(
        host=str(cfg.get("dial.udp.host")),
        port=int(cfg.get("dial.udp.port")),
        payload_template=str(cfg.get("dial.udp.payload_template", "{number}")),
        encoding=str(cfg.get("dial.udp.encoding", "utf-8")),
        dry_run=bool(cfg.get("dial.udp.dry_run", False)),
        dedupe_seconds=float(cfg.get("dial.dedupe_seconds", 5)),
    )


class DialService:
    def __init__(self, cfg: Config, transcriber, parser: DialParser | None = None, sender: UdpSender | None = None):
        self._transcriber = transcriber
        self._parser = parser or build_parser(cfg)
        self._sender = sender or build_sender(cfg)
        self._language = cfg.get("dial.language", None)
        self._prompt = cfg.get("dial.initial_prompt", None)

    def handle_text(self, text: str) -> list[str]:
        sent = []
        for number in self._parser.find_numbers(text):
            if self._sender.send(number):
                sent.append(number)
        return sent

    def process_file(self, path: Path) -> list[str]:
        segments = self._transcriber.transcribe_file(path, language=self._language, initial_prompt=self._prompt)
        text = " ".join(seg.text for seg in segments)
        log.info("%s: %s", Path(path).name, text)
        sent = self.handle_text(text)
        if not sent:
            log.info("%s: no dial command found", Path(path).name)
        return sent

    def handle_utterance(self, utterance: Utterance) -> list[str]:
        segments = self._transcriber.transcribe_array(
            utterance.audio, offset=utterance.start, language=self._language, initial_prompt=self._prompt
        )
        text = " ".join(seg.text for seg in segments)
        if not text:
            return []
        log.info("Heard: %s", text)
        return self.handle_text(text)

    def close(self) -> None:
        self._sender.close()
