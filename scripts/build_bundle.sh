#!/usr/bin/env bash
# Run on the STAGING machine (internet, same OS / Python / CPU architecture as the target).
# Produces ./<BUNDLE_DIR> with wheels, model files and SHA-256 checksums to carry to the target.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=../config/bundle.conf
source config/bundle.conf

WHEELS="$BUNDLE_DIR/wheelhouse"
rm -rf "$BUNDLE_DIR"
mkdir -p "$WHEELS"

echo "== Python wheels (runtime) =="
"$PYTHON_BIN" -m pip download -r requirements.txt --extra-index-url "$TORCH_INDEX_URL" -d "$WHEELS"

if [ "${INCLUDE_GPU_LIBS:-0}" = "1" ]; then
  echo "== CUDA 12 runtime wheels =="
  "$PYTHON_BIN" -m pip download -r requirements-gpu.txt -d "$WHEELS"
fi

echo "== Whisper model =="
"$PYTHON_BIN" scripts/download_models.py
mkdir -p "$BUNDLE_DIR/models"
cp -r models/. "$BUNDLE_DIR/models/"

echo "== Bundle info and checksums =="
{
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python: $("$PYTHON_BIN" --version 2>&1)"
  echo "platform: $(uname -sm)"
} > "$BUNDLE_DIR/BUNDLE_INFO.txt"
(cd "$BUNDLE_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "Bundle ready in $BUNDLE_DIR ($(du -sh "$BUNDLE_DIR" | cut -f1)). Copy it, with the project, to the target."
