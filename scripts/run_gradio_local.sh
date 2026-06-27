#!/usr/bin/env bash
# Local Gradio demo — use Python 3.11/3.12 (Gradio 5.x + pydantic wheels; Python 3.14 not supported yet).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv_gradio"
PY="${PYTHON:-python3.11}"

if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3.12
fi
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: Need python3.11 or python3.12 (Python 3.14 cannot install Gradio 5.x deps yet)."
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV"
fi

"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -r "$ROOT/proto/requirements-web.txt"

if [[ -f "$ROOT/hfb.png" ]]; then
  cp "$ROOT/hfb.png" "$ROOT/proto/hfb.png"
fi

cd "$ROOT/proto"
exec "$VENV/bin/python" gradio_demo.py