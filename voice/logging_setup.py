from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from voice.config import Config


def setup_logging(cfg: Config) -> None:
    level = getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO)
    log_dir = cfg.path("paths.log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "voice.log",
        maxBytes=int(cfg.get("logging.max_bytes", 5_000_000)),
        backupCount=int(cfg.get("logging.backup_count", 5)),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
