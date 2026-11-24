#!/usr/bin/env bash
# Run on the TARGET machine (no internet). Expects ./<BUNDLE_DIR> next to the project files.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=../config/bundle.conf
source config/bundle.conf

[ -d "$BUNDLE_DIR" ] || { echo "Bundle folder '$BUNDLE_DIR' not found" >&2; exit 1; }

echo "== Verifying checksums =="
(cd "$BUNDLE_DIR" && sha256sum -c SHA256SUMS --quiet)
cat "$BUNDLE_DIR/BUNDLE_INFO.txt"

echo "== Creating virtual environment =="
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

echo "== Installing from the wheelhouse (no network) =="
pip install --no-index --find-links "$BUNDLE_DIR/wheelhouse" -r requirements.txt
if [ "${INCLUDE_GPU_LIBS:-0}" = "1" ]; then
  pip install --no-index --find-links "$BUNDLE_DIR/wheelhouse" -r requirements-gpu.txt
fi

echo "== Copying model files =="
mkdir -p models
cp -rn "$BUNDLE_DIR/models/." models/

echo "Installed. Next: python scripts/check_offline.py, then ./start.sh --help"
