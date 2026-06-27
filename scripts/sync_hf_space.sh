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

# HF Spaces requirements — Gradio 5.12 + Python 3.12 (see space README frontmatter).
# We stay on 5.12 (not 4.44.1): 4.44 hits HfFolder/huggingface_hub breakage and the
# const/bool API schema bug. audioop-lts covers Python 3.13 if HF bumps the runtime.
cat > "$DST/requirements.txt" <<'EOF'
numpy>=1.24.0,<3.0.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
gradio==5.12.0
requests>=2.31.0
huggingface_hub>=0.23.0
audioop-lts>=0.2.1; python_version >= "3.13"
EOF

echo "Synced → $DST"
ls -la "$DST"