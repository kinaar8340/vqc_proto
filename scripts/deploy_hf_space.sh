#!/usr/bin/env bash
# Deploy space/orbital-braille to Hugging Face Spaces.
# Requires: pip install huggingface_hub && HF_TOKEN (write access)
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./scripts/deploy_hf_space.sh
#   ./scripts/deploy_hf_space.sh kinaar8340/orbital-braille-vqc

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_ID="${1:-kinaar8340/orbital-braille-vqc}"
SPACE_DIR="$ROOT/space/orbital-braille"

"$ROOT/scripts/sync_hf_space.sh"

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "ERROR: Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN with write access."
  echo "Create the Space at https://huggingface.co/new-space then re-run."
  exit 1
fi

python3 -m pip install -q huggingface_hub

python3 - <<PY
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
print(f"Deployed → https://huggingface.co/spaces/{repo_id}")
PY