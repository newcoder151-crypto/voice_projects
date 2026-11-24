import pytest

from voice.cli import parse_args
from voice.config import Config, ConfigError


def test_parse_modes():
    a = parse_args(["dial", "file", "x.wav"])
    assert (a.project, a.mode, str(a.path)) == ("dial", "file", "x.wav")
    assert parse_args(["transcribe", "live"]).mode == "live"
    assert parse_args(["--config", "c.yaml", "dial", "watch"]).config == "c.yaml"


def test_requires_project():
    with pytest.raises(SystemExit):
        parse_args([])


def test_config_paths_and_missing_keys(make_cfg):
    cfg = make_cfg()
    assert cfg.path("asr.model_path").is_absolute()
    assert cfg.get("nope.nothing", "fallback") == "fallback"
    with pytest.raises(ConfigError):
        cfg.get("nope.nothing")


def test_config_load_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "missing.yaml")
