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

# HF Spaces requirements — do NOT pin gradio here; HF installs gradio[oauth] from
# README sdk_version automatically. Pinning gradio in requirements.txt causes
# "Cannot install gradio==5.12.0 and gradio==5.27.0" build failures.
# pydantic==2.10.6 fixes gradio_client bool-schema crash; app.py also patches it.
cat > "$DST/requirements.txt" <<'EOF'
numpy>=1.24.0,<3.0.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
pydantic==2.10.6
requests>=2.31.0
audioop-lts>=0.2.1; python_version >= "3.13"
EOF

cat > "$DST/README.md" <<'EOF'
---
title: Orbital Braille VQC Typehead
emoji: 🔤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
python_version: 3.12
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Orbital Braille VQC Typehead — browser demo
---

# Orbital Braille — VQC Typehead

Browser demo of the **Orbital Braille** prototype: *N* PWM-gated point sources whose interference imprints **pyramidal spectral shards** on an **OAM/quaternion carrier**.

## Try it

1. Enter a payload (default: `"I live in Oregon"`)
2. Set orb count (2–6; **4** is the validated prototype sweet spot)
3. Choose **Quick** resolution for sub-second runs; **Full** for publication-quality figures
4. Adjust **γ₁** (p-wave BMGL strength) if desired
5. Click **Run demo** — metrics + 6-panel figure
6. **Download SLM package** — `manifest.json` + `phase_stack.npy` (optional PNG frames)

Use **Load example from paper** for patent Figure 1 (`"I live in Oregon"`, 4 orbs).

## Example payloads

| Payload | Orbs | Notes |
|---------|------|-------|
| `I live in Oregon` | 4 | Patent Figure 1 reference |
| `VQC prototype` | 4 | General ASCII shard test |
| `Hello OAM` | 2 | Fastest decode, smaller alphabet |

## Source & license

- Live demo: [kinaar111/orbital-braille-vqc](https://huggingface.co/spaces/kinaar111/orbital-braille-vqc)
- GitHub: [kinaar8340/vqc_proto](https://github.com/kinaar8340/vqc_proto)
- **CC-BY-NC-SA-4.0** + patent restrictions — non-commercial research only
- US Provisional Patent 63/913,110

Synced from `proto/gradio_demo.py` via `scripts/sync_hf_space.sh`.
EOF

echo "Synced → $DST"
ls -la "$DST"