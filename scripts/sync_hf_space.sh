#!/usr/bin/env bash
# Sync proto sources into the self-contained Hugging Face Space folder.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/proto"
DST="$ROOT/space/orbital-braille"

mkdir -p "$DST"
rm -rf "$DST/orbital_braille"
cp -r "$SRC/orbital_braille" "$DST/"
cp "$SRC/demo_core.py" "$DST/"
cp "$SRC/gradio_demo.py" "$DST/app.py"

# HF Spaces expect requirements.txt at space root (not requirements-web.txt)
cat > "$DST/requirements.txt" <<'EOF'
numpy>=1.24.0,<3.0.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
gradio==5.12.0
requests>=2.31.0
EOF
# Note: Python 3.12 Docker image includes audioop; no pyaudioop needed

# Docker SDK for HF (Python 3.12 pin)
if [[ ! -f "$DST/Dockerfile" ]]; then
  echo "WARN: space/orbital-braille/Dockerfile missing — create manually"
fi

echo "Synced → $DST"
ls -la "$DST"