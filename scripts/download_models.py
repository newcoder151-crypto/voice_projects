"""Run on the STAGING machine (with internet). Downloads the Whisper model into asr.model_path.

    python scripts/download_models.py [--config config/settings.yaml]

The Silero VAD model needs no download: it ships inside the silero-vad pip package.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    name = cfg.get("asr.model_name")
    target = cfg.path("asr.model_path")
    target.mkdir(parents=True, exist_ok=True)

    from faster_whisper.utils import download_model

    print(f"Downloading '{name}' to {target} ...")
    download_model(name, output_dir=str(target))
    files = sorted(p.name for p in target.iterdir())
    print("Done. Files:", ", ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
