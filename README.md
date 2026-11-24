# Offline Voice Projects

Two tools on one shared offline speech pipeline (ffmpeg/PyAV decoding, Silero VAD, faster-whisper on the GPU):

| Project | What it does |
|---|---|
| **dial** | Listens (live or from audio files) for a command like "call nine eight seven six ...", extracts the number and sends it as a **UDP message** to the receiver set in `config/settings.yaml`. The call section itself is not touched. |
| **transcribe** | Transcribes audio files and live audio to text files: `<name>.txt` + `<name>.json` per audio file, `live_<time>.txt` + `.jsonl` per live session. |

Everything (hosts, ports, paths, model names, limits) is in `config/settings.yaml`. Nothing is hardcoded.

## Layout

```
config/settings.yaml      all settings        config/bundle.conf   offline-bundle settings
voice/                    asr.py vad.py live.py files.py audio_sources.py cli.py
voice/dial/               parser.py (number extraction)  sender.py (UDP)  service.py
voice/transcribe/         writer.py (txt/json/jsonl)     service.py
scripts/                  download_models.py build_bundle.sh install_offline.sh check_offline.py
tests/                    37 tests, no GPU or model needed
start.sh                  launcher (activates venv, forces offline mode, finds pip CUDA libs)
```

## 1. Build the bundle (staging machine, internet, same OS / Python / CPU arch as the target)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-gpu.txt -r requirements-dev.txt
python -m pytest                       # sanity check
./scripts/build_bundle.sh              # wheels + Whisper model + SHA256SUMS -> ./bundle
```

Skip `requirements-gpu.txt` (and set `INCLUDE_GPU_LIBS=0` in `config/bundle.conf`) if the target already has CUDA 12 and cuDNN 9.

## 2. Install (target machine, no internet)

Copy the project folder and `bundle/` over, then:

```bash
./scripts/install_offline.sh           # verifies checksums, creates .venv, installs from wheelhouse
python scripts/check_offline.py        # imports, model folder, GPU, microphone/ffmpeg
```

## 3. Run

```bash
./start.sh dial file call.wav          # one file or a whole folder
./start.sh dial watch                  # new files in dial.input_dir
./start.sh dial live                   # microphone or network stream

./start.sh transcribe file talk.mp3
./start.sh transcribe watch            # transcribe.input_dir -> transcribe.output_dir
./start.sh transcribe live
```

Set `dial.udp.host` / `dial.udp.port` first. Use `dial.udp.dry_run: true` to log the message instead of sending it. The UDP message is the bare number (for example `9876543210` or `+919876543210`); change `dial.udp.payload_template` if the receiver needs another format. Live audio comes from the microphone by default (`live.source: microphone`); set `live.source: stream` and `live.stream_url` to read a network stream instead.

## How the dial command is read

Words after a trigger (`dial`, `call`, `phone`) are turned into digits: "nine eight 7" gives 987, "double nine" gives 99, "plus nine one" gives +91, and Hindi/Arabic numerals are accepted. The result must have `min_digits` to `max_digits` digits and match the optional `number_pattern`; the same number is not sent twice within `dedupe_seconds`. Spoken tens ("ninety eight") are not supported, digits are read one by one.

## GPU, CPU and operating system

`asr.device: auto` uses a GPU when one is found and falls back to the CPU otherwise. `large-v3` is only practical on a GPU; on a CPU switch `asr.model_name` / `asr.model_path` to a smaller model such as `small.en` and rebuild the bundle. The Python code runs on Linux and Windows, but the shell scripts (`start.sh`, `scripts/*.sh`) are for Linux; on Windows use `python -m voice ...` and `pip download` / `pip install --no-index` by hand.

## Notes

- Audio files are decoded by faster-whisper's bundled PyAV, so ffmpeg is only needed for network streams.
- Live phrases are cut by Silero VAD (`live.vad.*`); a phrase is sent to Whisper as soon as the speaker pauses.
- Live transcripts are written as `.txt` plus `.jsonl` (append-only, safe for long sessions). File transcripts get a full `.json`.
- Tested here: parser, UDP sender, live phrase cutting, file watcher, file/live output. Not testable without the model and GPU: the faster-whisper and Silero calls, so run `check_offline.py` and one sample file on the real machine first.
