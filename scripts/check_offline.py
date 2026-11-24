"""Pre-flight check. Run on the TARGET machine before starting (works with the network unplugged).

    python scripts/check_offline.py [--config config/settings.yaml]
"""
import argparse
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.config import Config  # noqa: E402

failures = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"[{'OK' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    for module in ("faster_whisper", "silero_vad", "torch", "numpy", "yaml"):
        try:
            importlib.import_module(module)
            report(True, f"import {module}")
        except Exception as exc:  # noqa: BLE001
            report(False, f"import {module}", str(exc))

    model_path = cfg.path("asr.model_path")
    report(model_path.is_dir() and any(model_path.iterdir()), "ASR model folder", str(model_path))

    device = str(cfg.get("asr.device"))
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        if device == "cuda":
            report(count > 0, "CUDA devices visible to CTranslate2", str(count))
        elif device == "auto":
            report(True, "GPU detection (asr.device is auto)",
                   f"{count} CUDA device(s)" + ("" if count else ", will run on CPU: use a smaller model"))
    except Exception as exc:  # noqa: BLE001
        report(False, "GPU detection", str(exc))

    source = str(cfg.get("live.source"))
    if source == "stream":
        ffmpeg = str(cfg.get("live.ffmpeg_path"))
        report(shutil.which(ffmpeg) is not None, "ffmpeg available", ffmpeg)
        report(bool(cfg.get("live.stream_url")), "live.stream_url is set")
    else:
        try:
            import sounddevice as sd

            inputs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
            report(bool(inputs), "microphone input device", f"{len(inputs)} found")
        except Exception as exc:  # noqa: BLE001
            report(False, "microphone input device", str(exc))

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
