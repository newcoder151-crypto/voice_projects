"""Sends the dialled number as a UDP message to the host and port from config."""
from __future__ import annotations

import logging
import socket
import time
from typing import Callable

log = logging.getLogger(__name__)


class UdpSender:
    def __init__(
        self,
        host: str,
        port: int,
        payload_template: str = "{number}",
        encoding: str = "utf-8",
        dry_run: bool = False,
        dedupe_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._template = payload_template
        self._encoding = encoding
        self._dry_run = dry_run
        self._dedupe = dedupe_seconds
        self._clock = clock
        self._last_sent: dict[str, float] = {}
        self._target = f"{host}:{port}"
        family, socktype, proto, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
        self._address = address
        self._sock = None if dry_run else socket.socket(family, socktype, proto)

    def send(self, number: str) -> bool:
        """Send the number. Returns False if it was skipped as a recent duplicate."""
        now = self._clock()
        last = self._last_sent.get(number)
        if last is not None and now - last < self._dedupe:
            log.info("Skipping duplicate number %s (sent %.1fs ago)", number, now - last)
            return False
        payload = self._template.format(number=number).encode(self._encoding)
        if self._dry_run:
            log.info("DRY RUN: would send %r to %s", payload, self._target)
        else:
            self._sock.sendto(payload, self._address)
            log.info("Sent %r to %s", payload, self._target)
        self._last_sent[number] = now
        return True

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
