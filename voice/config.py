"""YAML configuration with dotted-key access and project-relative paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


class ConfigError(Exception):
    """Raised when the configuration file or a required key is missing."""


class Config:
    def __init__(self, data: dict[str, Any], base_dir: Path):
        self._data = data
        self.base_dir = Path(base_dir)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config file must contain a mapping: {path}")
        # Project root is the folder that holds the config/ directory.
        return cls(data, path.parent.parent)

    def get(self, key: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise ConfigError(f"Missing config key: {key}")
                return default
        return node

    def path(self, key: str) -> Path:
        value = self.get(key)
        if value is None:
            raise ConfigError(f"Config key is empty: {key}")
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else self.base_dir / candidate
