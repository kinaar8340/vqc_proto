# SLM Quickstart — 4-Orb Virtual Typehead on Real Hardware

Load Orbital Braille phase holograms onto a **phase-only SLM** — no mechanical rotation required. Software-defined virtual orbits replace the spinning laser array.

**Prerequisites:** `pip install -r requirements.txt` · laser + SLM + Fourier lens + camera (see bench layout below)

---

## 1. Generate hologram package

```bash
cd proto   # or repo root if using integrated layout: cd vqc_proto/proto

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Default 512×512 validation package
.venv/bin/python generate_slm_holograms.py --payload "I live in Oregon" --num-orbs 4

# Holoeye PLUTO-2 class (1920×1080)
.venv/bin/python generate_slm_holograms.py --device holoeye_pluto_2 --frames 64

# Sharper far-field (slower): Gerchberg-Saxton refinement
.venv/bin/python generate_slm_holograms.py --device generic_512 --gerchberg-saxton --gs-iter 32
```

### Output bundle (`outputs/slm/<device>_4orb/`)

| File | Purpose |
|------|---------|
| `frames/phase_0000.png` … | 8-bit (or 16-bit TIFF) phase maps — **upload to SLM** |
| `frames/phase_0000.raw` | Raw gray levels for custom drivers |
| `manifest.json` | Orb radii, ℓ charges, PWM duties, quaternion, frame timing |
| `preview_montage.png` | Visual check before bench |
| `LUT_calibration.txt` | Linear phase→gray mapping + calibration steps |
| `phase_stack.npy` | Full stack for offline analysis |

---

## 2. Device presets

| Preset | Resolution | Pitch | Bits | Typical hardware |
|--------|------------|-------|------|------------------|
| `generic_512` | 512×512 | 8 µm | 8 | Algorithm validation |
| `holoeye_pluto_2` | 1920×1080 | 8 µm | 8 | Holoeye PLUTO-2 |
| `meadowlark_512` | 512×512 | 15 µm | 16 | Meadowlark LCOS |
| `thorlabs_1080p` | 1920×1080 | 6.4 µm | 8 | Thorlabs Exulus / similar |

Override wavelength: `--wavelength-nm 1550` (default matches VQC `configs/params.yaml`).

---

## 3. Bench optical layout

```
Laser (1550 nm or 633 nm HeNe for visible demo)
    |
    v
Beam expander (optional)
    |
    v
Phase-only SLM  <-- load frames/phase_XXXX.png
    |
    v
Fourier lens (f ~ 100–300 mm)
    |
    v
Far field / camera sensor
    |
    +-- Optional: helical grating or ℓ-sorter for OAM demux
```

**What to look for:**
- **Donut ring** (ℓ = 1 LG carrier) with **2–4 bright lobes** (orbital Braille dots)
- **Rotation/evolution** when playing frame sequence
- **Spectral chirp** if you tap photodiode + OSA on transmitted pulse train

---

## 4. Upload workflow

### Holoeye PLUTO-2 (HoloVision / Load Hologram)

1. Generate with `--device holoeye_pluto_2`
2. Upload `frames/phase_0000.bmp` or PNG (8-bit grayscale, **no gamma**)
3. Set **phase-only mode**; disable amplitude coupling if available
4. Play sequence at `manifest.json` → `t_max_ns / frames` per frame

**Common pitfalls (Holoeye):**
- **Gamma-corrected PNGs** flatten phase contrast — export raw 8-bit linear grayscale.
- **Wrong SLM window size** — PLUTO-2 is 1920×1080; using `generic_512` stretches the hologram.
- **LC temperature drift** — re-run LUT calibration after warm-up (~15 min).
- **USB bandwidth** — reduce `--frames` or use Holoeye's on-board sequence player for >30 fps.

### Meadowlark LCOS (512 / 1920 variants)

1. Generate with `--device meadowlark_512` (16-bit TIFF output when supported)
2. Use Meadowlark Blink SDK or `.tiff` loader; map 0–65535 → 0–2π
3. Enable **phase-only** addressing; disable dithering (destroys OAM structure)
4. 15 µm pitch → adjust beam expander so fill factor matches your aperture

**Common pitfalls (Meadowlark):**
- **Dithering enabled** — produces speckle instead of clean orbital lobes.
- **8-bit down-conversion** — prefer native 16-bit path; re-quantize only after LUT fit.
- **Back-reflection** — use non-polarizing beam splitter or angled pickoff; LCOS is reflective.

### Thorlabs Exulus / similar 1080p LCOS

1. Generate with `--device thorlabs_1080p`
2. 6.4 µm pitch — verify Fourier lens focal length matches your λ and aperture
3. Exulus software accepts 8-bit BMP; disable any auto-scaling

**Common pitfalls (Thorlabs):**
- **Auto-scaling in Exulus GUI** — lock min/max gray to 0–255 before upload.
- **Polarization** — LCOS efficiency is polarization-sensitive; align laser to manufacturer spec.
- **Thermal blanking** — long sequences may throttle refresh; match `manifest.json` frame period.

### Generic / custom driver

1. Read `frames/phase_XXXX.raw` (little-endian uint8)
2. Map: `gray = (phase_rad / 2π) × 255`
3. Apply LUT correction per `LUT_calibration.txt`
4. Stream frames via SDK refresh loop

---

## 5. LUT calibration (recommended)

Linear 0→2π mapping is a starting point. For <λ/10 wavefront error:

1. Display uniform gray levels {0, 64, 128, 192, 255}
2. Measure phase shift with interferometer or shearing interferometry
3. Fit polynomial LUT; update SLM driver
4. Re-export with `--gerchberg-saxton` after LUT is loaded

---

## 6. Decode / verify

| Check | Method |
|-------|--------|
| OAM donut present | Camera far-field; annular intensity |
| Orb lobes | Compare to `preview_montage.png` and `run_demo.py` intensity panel |
| Glyph recovery | Camera → ICA on lobe intensities; match `manifest.json` glyph_duties |
| Shard spectrum | Photodiode + OSA during frame playback; compare Welch PSD in demo |

Simulation decode: `run_demo.py` (92.9% shard FID reference at 4 orbs).

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Uniform blob, no donut | Check Fourier lens distance; verify phase-only mode |
| Low contrast | Run `--gerchberg-saxton`; calibrate LUT |
| Wrong size | Match `--device` preset to your SLM resolution |
| Flickering glyphs | Increase `--frames`; match SLM refresh to `t_max_ns/frames` |

---

## 8. Patent / enablement note

This package is **reduction-to-practice** for the claim element: *PWM-gated point sources on distinct orbital trajectories generating pyramidal spectral shards on an OAM carrier*. The `manifest.json` sidecar documents reproducible orb geometry tied to emergent font constants (350/π, κ = 0.85).

See [`IP_NOTICE.md`](IP_NOTICE.md) for license and patent restrictions.

**Contact:** kinaar0@protonmail.com · Repo: https://github.com/kinaar8340/vqc_proto