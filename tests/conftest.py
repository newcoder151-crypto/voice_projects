import socket
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice.config import Config  # noqa: E402
from voice.models import Segment  # noqa: E402


@pytest.fixture
def test_host() -> str:
    return yaml.safe_load((ROOT / "tests" / "test_settings.yaml").read_text())["udp_test_host"]


@pytest.fixture
def receiver(test_host):
    """A UDP socket on an OS-assigned port; yields (socket, host, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((test_host, 0))
    sock.settimeout(2)
    yield sock, test_host, sock.getsockname()[1]
    sock.close()


@pytest.fixture
def make_cfg(tmp_path):
    def build(overrides=None):
        data = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
        data["paths"]["state_dir"] = str(tmp_path / "state")
        data["paths"]["log_dir"] = str(tmp_path / "logs")
        data["transcribe"]["output_dir"] = str(tmp_path / "out")
        for dotted, value in (overrides or {}).items():
            node = data
            *parents, last = dotted.split(".")
            for part in parents:
                node = node[part]
            node[last] = value
        return Config(data, ROOT)
    return build


class FakeTranscriber:
    """Stands in for faster-whisper so tests need no model or GPU."""

    def __init__(self, text="call nine eight seven six five four three two one zero"):
        self.text = text
        self.calls = []

    def transcribe_file(self, path, language=None, initial_prompt=None):
        self.calls.append(("file", str(path), language))
        return [Segment(0.0, 2.5, self.text, 0.9, "en")]

    def transcribe_array(self, audio, offset=0.0, language=None, initial_prompt=None):
        self.calls.append(("array", len(audio), language))
        return [Segment(offset, offset + len(audio) / 16000, self.text, 0.9, "en")]
