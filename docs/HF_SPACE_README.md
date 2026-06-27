# Orbital Braille — VQC Typehead

<p align="center">
  <img src="https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png" alt="Orbital Braille VQC — OAM phase background" width="100%" style="max-width: 720px; border-radius: 12px;" />
</p>

**Beginner-friendly browser demo** of the Orbital Braille prototype: several virtual laser “dots” orbit like a typeball, their timed flashes (PWM) imprint data as **spectral shards** on an **OAM (orbital angular momentum)** light beam.

No install required — use the **App** tab above. This page explains what you are looking at.

> **Simulation demo** — browser-based encode → turbulence → decode. Outputs model OAM, PWM orbs, BMGL noise, and FastICA decode; they are not live SLM hardware. Use the **SLM package** zip for bench upload after a run.

---

## In 60 seconds: what is this?

Think of an **IBM Selectric typeball**, but optical:

1. **Characters** are not metal letters — they are **PWM duty patterns** on *N* orbiting spots (like Braille dots on a spinning ball).
2. The spots **interfere** on a shared **Laguerre-Gaussian (LG) donut beam** carrying OAM (helical phase, topological charge ℓ).
3. A **pyramidal FM pulse** (triangular envelope + chirp) turns the payload into **discrete spectral shards** — a barcode in frequency.
4. A **quaternion** compresses the byte stream onto the carrier rotation.
5. **p-wave BMGL** (γ₁ slider) models how the decoder fights phase noise — like typing on a vibrating desk.

The demo runs the full encode → turbulence → decode loop and shows a **6-panel figure**, an **interactive 3D orb plot**, optional **typehead animation**, and an **SLM hardware export zip**.

---

## Watch the typehead process

<p align="center">
  <video src="https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/docs/typehead_screencast.mp4" controls width="100%" style="max-width: 900px; border-radius: 8px;"></video>
</p>

*Recorded walkthrough: preset or **Run demo** → **Animate typehead** — helical phase, OAM intensity, pyramidal pulse, PWM orbs (payload: `"I live in Oregon"`).*

<p align="center">
  <img src="https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/typehead_demo.gif" alt="Animated typehead preview" width="100%" style="max-width: 820px; border-radius: 8px;" />
</p>

---

## Typeball analogy (mechanical → optical)

```mermaid
flowchart LR
    subgraph mechanical["Classic Selectric"]
        A[Spinning typeball] --> B[Angular position picks glyph]
        B --> C[Impact timing / force]
        C --> D[Raised dots hit paper]
    end
    subgraph optical["Orbital Braille / VQC"]
        E[N virtual orbs on rings] --> F[PWM duty picks glyph shard]
        F --> G[Pyramidal FM pulse envelope]
        G --> H[OAM donut + spectral shards]
    end
    mechanical -. analog .-> optical
```

---

## Pipeline (what happens when you click Run demo)

```mermaid
flowchart TB
    P[Your text payload] --> E[Encode: quaternion + stable font + orbs]
    E --> T[Propagate: BMGL phase noise × channel noise level]
    T --> D[Decode: OAM projection + ICA + glyph match]
    D --> M[Metrics + 6-panel figure + 3D Plotly]
    E --> S[SLM package: manifest.json + phase_stack.npy]
    M --> A[Animate typehead: MP4 + GIF]
```

| Step | Plain English |
|------|----------------|
| **Encode** | Turn text into orb positions, PWM duties, and a unit quaternion on the LG carrier. |
| **Propagate** | Add channel phase noise scaled by the **Channel noise** slider; γ₁ controls BMGL inhibition during decode. |
| **Decode** | Recover glyph and shard fidelity from the noisy field. |
| **3D view** | Plotly helices — drag to rotate, scroll to zoom, hover for time and PWM state. |
| **SLM export** | Zip for phase-only SLM benches (see accordion in the app). |

---

## How to use this Space (step by step)

1. Open the **App** tab (top of this page).
2. Expand **New here? 60-second guide** for the typeball analogy and metrics glossary (no GitHub required).
3. Click an **example preset** — each button loads payload, orb count, γ₁, noise level, and **runs** the demo automatically:
   - **Patent Fig. 1** — `"I live in Oregon"`, 4 orbs
   - **VQC prototype** — short label test, 4 orbs
   - **Hello OAM** — minimal 2-orb intro (gentler noise)
   - **6-orb stress** — max orbs + higher γ₁ and noise
4. Or enter your own **payload** and adjust **Number of orbs** (2–6). **4** is the validated sweet spot.
5. Choose **Quick** (fast preview; recommended on HF) or **Full** (publication-quality grid).
6. Tune **Channel noise** (0–1) — turbulence strength; **γ₁** — BMGL inhibition vs. phase noise.
7. Change **Random seed** for a different noise realization at the same channel noise level.
8. Click **Run demo** → read **Metrics**, view the **6-panel output** and **interactive 3D** plot.
9. Click **Animate typehead** → per-run MP4 player + GIF (phase · intensity · pulse · orb trails).
10. Expand **SLM package download** → `slm_package.zip` (`manifest.json`, `phase_stack.npy`, `README.txt`, …).
11. Expand **How this maps to VQC claims** for patent-element ↔ demo-panel alignment.

**HF performance notes:** Animation frames are capped on Spaces for responsiveness. SLM PNG frame export is disabled on HF — use the local Gradio demo for full `frames/` sequences.

---

## App controls reference

| Control | Range / options | Effect |
|---------|-----------------|--------|
| **Payload** | text (≤32 chars ideal) | Message encoded into spectral shards + quaternion |
| **Number of orbs** | 2–6 | PWM-gated sources on orbital rings |
| **Resolution** | Quick / Full | Grid size and time steps (Quick ≈ sub-second on HF) |
| **Random seed** | 0–9999 | RNG seed for turbulence draws |
| **γ₁** | 1.0–2.0 | p-wave BMGL inhibition strength (default 1.5) |
| **Channel noise** | 0.0–1.0 | Phase turbulence multiplier (0.35 ≈ legacy default) |
| **SLM PNG frames** | checkbox | Adds `frames/` to zip (local demo only on HF) |

---

## Reading the outputs

### 6-panel figure

| Panel | What it shows |
|-------|----------------|
| **Top-left** | Clean encoded phase (helical OAM structure). |
| **Top-middle** | Phase after BMGL + turbulence (compare noise + γ₁ sliders). |
| **Top-right** | Intensity — OAM donut with Braille-like lobes. |
| **Bottom-left** | Pyramidal FM pulse in time (triangular chirp). |
| **Bottom-middle** | Welch PSD — discrete **spectral shards** (barcode). |
| **Bottom-right** | Typehead layout — orb rings, ℓ charges, PWM duties. |

### Interactive 3D (Plotly)

- Each orb traces a **helix** in (x, y, time ns).
- **Drag** to rotate · **scroll** to zoom · **shift-drag** to pan.
- **Hover** for ℓ, radius, time, PWM on/off, position.
- Dotted rings at t = 0 show each orb’s orbital radius.

### Metrics block

| Metric | Plain English |
|--------|----------------|
| **Font separation** | Fisher-Rao distance between glyphs (higher → easier decode). |
| **Shard fidelity** | Agreement between encoded and recovered spectral subcarriers. |
| **Glyph fidelity** | Confidence in the decoded Braille glyph index. |
| **Quaternion** | Compressed payload orientation from encode. |
| **Channel noise** | Slider value and internal phase-noise scale. |

### Typehead animation

Four synchronized panels over time: helical phase · OAM intensity · pyramidal pulse cursor · PWM-gated orb trails (orange = ON).

---

## Key terms (mini glossary)

| Term | One-line meaning |
|------|------------------|
| **OAM** | Twisted light — beams with helical phase; ℓ is the “twist number.” |
| **LG mode** | Laguerre-Gaussian beam — standard math for OAM donuts. |
| **PWM** | Pulse-width modulation — each orb is ON/OFF over time to encode a duty vector. |
| **Spectral shards** | Sharp peaks in the pulse spectrum — subcarrier barcode for the payload. |
| **Quaternion** | 4-number rotation code compressing payload bytes onto the carrier. |
| **BMGL / γ₁** | Beam-Motion-Gated Learning — error inhibition; γ₁ tunes suppression strength. |
| **Channel noise** | UI slider scaling phase turbulence in the propagate step (0 = clean, 1 = harsh). |
| **Fisher-Rao separation** | How distinct glyph codewords are in PWM space (higher = safer decode). |
| **SLM** | Spatial light modulator — chip that displays phase holograms for real optics. |

Full glossary: [GLOSSARY.md](https://github.com/kinaar8340/vqc_proto/blob/main/GLOSSARY.md)

---

## How this compares to existing OAM work

| Approach | Typical goal | What Orbital Braille adds |
|----------|--------------|---------------------------|
| **Allen et al. — OAM modes** (1992) | ℓ modes as orthogonal channels | Time-multiplexed **virtual typehead** orbs + PWM glyph font on one carrier |
| **OAM + DWDM multiplexing** (Willner group, etc.) | Many spatial modes per wavelength | **Pyramidal FM spectral shards** as an extra barcode layer + quaternion compression |
| **SLM OAM holograms** | Static ℓ hologram on phase SLM | **Animated phase stack** + `manifest.json` bench sidecar (Holoeye / Meadowlark / Thorlabs notes) |
| **OAM mode sorters** (log-polar, etc.) | Hardware demultiplex by ℓ | Simulation path: OAM projection + **FastICA** + Fisher-Rao nearest glyph |
| **Orbital-angular-momentum communications demos** | Bessel/LG basis channels | **BMGL turbulence proxy**, live noise slider, shard fidelity metric, patent Figure 1 payload |
| **Classical coherent ISI/OFDM** | Frequency subcarriers | Chirped **pyramidal** envelope tying shards to physical typehead timing |

Orbital Braille is a **research prototype** for the [VQC (Vortex Quaternion Conduit)](https://github.com/kinaar8340/vqc_proto) architecture — not a drop-in replacement for commercial OAM links, but a distinct embodiment combining typeball-like coding, OAM carriers, and exportable SLM artifacts.

---

## Example payloads

| Preset | Payload | Orbs | Noise | Notes |
|--------|---------|------|-------|-------|
| Patent Fig. 1 | `I live in Oregon` | 4 | 0.35 | Paper / Figure 1 reference |
| VQC prototype | `VQC prototype` | 4 | 0.35 | General ASCII shard test |
| Hello OAM | `Hello OAM` | 2 | 0.25 | Fastest run; gentler channel |
| 6-orb stress | `noise test` | 6 | 0.75 | Max orbs + strong γ₁ and noise |

Validated metrics (4 orbs, full mode, seed 42, default noise): Fisher-Rao **0.989 rad**, shard fidelity **0.929**.

---

## Going further

| Resource | Link |
|----------|------|
| Full prototype docs | [proto/README.md](https://github.com/kinaar8340/vqc_proto/blob/main/proto/README.md) |
| SLM bench quickstart | [proto/SLM_QUICKSTART.md](https://github.com/kinaar8340/vqc_proto/blob/main/proto/SLM_QUICKSTART.md) |
| Source code | [github.com/kinaar8340/vqc_proto](https://github.com/kinaar8340/vqc_proto) |
| Local Gradio (same UI) | `cd proto && pip install -r requirements-web.txt && python gradio_demo.py` |
| Patent / IP | [IP_NOTICE.md](https://github.com/kinaar8340/vqc_proto/blob/main/IP_NOTICE.md) · US Provisional 63/913,110 |

---

## License

**CC-BY-NC-SA-4.0** + patent restrictions — **non-commercial research only**.

Synced from [`proto/gradio_demo.py`](https://github.com/kinaar8340/vqc_proto/blob/main/proto/gradio_demo.py) via `scripts/sync_hf_space.sh`.