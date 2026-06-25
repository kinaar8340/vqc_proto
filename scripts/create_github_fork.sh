#!/usr/bin/env bash
# Create kinaar8340/vqc_proto fork from vqc_sims_public (requires: gh auth login)
set -euo pipefail

if ! gh auth status -h github.com &>/dev/null; then
  echo "Run: gh auth login -h github.com -p ssh -s repo"
  exit 1
fi

gh repo fork kinaar8340/vqc_sims_public \
  --fork-name vqc_proto \
  --clone=false \
  --remote=false

echo "Fork created: https://github.com/kinaar8340/vqc_proto"
echo "Push vqc_proto branch:"
echo "  git push git@github.com:kinaar8340/vqc_proto.git vqc_proto:main"