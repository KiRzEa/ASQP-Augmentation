#!/usr/bin/env bash
# One-time setup on the GPU server (no Internet needed). Run from the package root, with the Python environment
# that already has torch + CUDA active (e.g. `conda activate LLM`):
#   bash llm/setup.sh
# Creates .venv-asqp that SEES the existing packages (torch, numpy, ...) but installs the pinned transformers / peft /
# accelerate from llm/wheels inside it only, so the existing environment is not changed. Then runs the preflight.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" - <<'PY'
import sys
if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
    sys.exit(f"Python {sys.version.split()[0]}: use Python 3.10, 3.11 or 3.12 (activate the conda env with torch)")
try:
    import torch
except ImportError:
    sys.exit("torch is not installed in this Python: activate the environment that has torch + CUDA")
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
PY
if [ ! -d .venv-asqp ]; then
  "$PY" -m venv --system-site-packages .venv-asqp
fi
.venv-asqp/bin/python -m pip install --no-index --find-links llm/wheels -r llm/requirements.txt
echo
echo "== installed; preflight:"
.venv-asqp/bin/python llm/inlg.py preflight "$@"
