#!/usr/bin/env bash
# Usage: ./start.sh dial live | ./start.sh transcribe watch | ./start.sh --help
# Host, port and every other setting come from config/settings.yaml.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=config/bundle.conf
source config/bundle.conf

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$VENV_DIR/bin/activate"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# If the CUDA libraries came from pip wheels (requirements-gpu.txt), make them visible.
CUDA_LIBS="$(python - <<'PY'
try:
    import nvidia.cublas.lib, nvidia.cudnn.lib
    print(":".join(list(m.__path__)[0] for m in (nvidia.cublas.lib, nvidia.cudnn.lib)))
except Exception:
    pass
PY
)"
if [ -n "$CUDA_LIBS" ]; then
  export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python -m voice "$@"
