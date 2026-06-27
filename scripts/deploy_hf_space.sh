#!/usr/bin/env bash
# Sync, commit, push GitHub, and deploy to HF Space (SSH git).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Sync HF space bundle ==="
bash scripts/sync_hf_space.sh

echo "=== 2. Git commit (vqc_proto) ==="
git add -A
git status --short
if git diff --cached --quiet; then
  echo "No staged changes"
  GH_SHA="$(git rev-parse HEAD)"
else
  git commit -m "fix(hf-space): patch gradio_client bool schema crash, upgrade to 5.27"
  GH_SHA="$(git rev-parse HEAD)"
fi
echo "GitHub SHA: $GH_SHA"

echo "=== 3. Git push origin main ==="
git push origin main

echo "=== 4. Deploy to HF Space ==="
HF_DIR="/tmp/hf-orbital-braille"
rm -rf "$HF_DIR"
git clone git@hf.co:spaces/kinaar111/orbital-braille-vqc "$HF_DIR"
rsync -av --delete \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "$ROOT/space/orbital-braille/" "$HF_DIR/"
cd "$HF_DIR"
git add -A
git status --short
if git diff --cached --quiet; then
  echo "No HF changes to commit"
  HF_SHA="$(git rev-parse HEAD)"
  HF_PUSH="no changes"
else
  git commit -m "fix(hf-space): patch gradio_client bool schema crash, upgrade to 5.27"
  HF_SHA="$(git rev-parse HEAD)"
  git push origin main
  HF_PUSH="OK"
fi

echo ""
echo "=== RESULTS ==="
echo "GITHUB_SHA=$GH_SHA"
echo "HF_SHA=$HF_SHA"
echo "HF_PUSH=$HF_PUSH"
echo ""
echo "=== Deployed requirements.txt (first 10 lines) ==="
head -10 "$HF_DIR/requirements.txt"