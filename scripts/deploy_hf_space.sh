#!/usr/bin/env bash
# Deploy space/orbital-braille to Hugging Face Spaces.
# Requires: pip install huggingface_hub && HF_TOKEN (write access)
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./scripts/deploy_hf_space.sh
#   ./scripts/deploy_hf_space.sh kinaar111/orbital-braille-vqc

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_ID="${1:-kinaar111/orbital-braille-vqc}"
SPACE_DIR="$ROOT/space/orbital-braille"

"$ROOT/scripts/sync_hf_space.sh"

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "ERROR: Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN with write access."
  echo "Create the Space at https://huggingface.co/new-space then re-run."
  exit 1
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "${ROOT}/.venv"
  "${ROOT}/.venv/bin/pip" install -q huggingface_hub
elif ! "$PYTHON" -c "import huggingface_hub" 2>/dev/null; then
  "${ROOT}/.venv/bin/pip" install -q huggingface_hub
fi

"$PYTHON" - <<PY
from huggingface_hub import HfApi

api = HfApi()
repo_id = "${SPACE_ID}"
try:
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
except Exception as e:
    print(f"create_repo note: {e}")

api.upload_folder(
    folder_path="${SPACE_DIR}",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Deploy Orbital Braille Gradio demo",
)
print(f"Uploaded → https://huggingface.co/spaces/{repo_id}")

try:
    api.restart_space(repo_id, factory_reboot=True)
    print("Factory reboot triggered")
except Exception as e:
    print(f"restart_space note: {e}")
PY