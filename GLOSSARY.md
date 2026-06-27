# VQC / Orbital Braille Glossary

Short definitions for terms used across this repository. See [`proto/README.md`](proto/README.md) for the encoding pipeline and [`IP_NOTICE.md`](IP_NOTICE.md) for licensing.

| Term | Definition |
|------|------------|
| **VQC** | Vortex Quaternion Conduit — hybrid optical architecture multiplexing OAM modes with quaternion-compressed payload shards. |
| **OAM** | Orbital Angular Momentum — helical phase structure of a light beam; topological charge ℓ labels distinct orthogonal modes. |
| **LG mode** | Laguerre-Gaussian beam — standard analytic basis for OAM modes (`lg_modes.py`). |
| **DWDM** | Dense Wavelength Division Multiplexing — parallel wavelength channels; VQC stacks OAM modes within each channel. |
| **Orbital Braille** | Prototype embodiment: *N* PWM-gated point sources on circular orbits whose interference imprints shard geometry on an OAM carrier (Selectric typeball analog). |
| **Typehead** | The multi-orb modulator (`OrbitalTypehead`) that places virtual laser spots on orbits and superposes them onto an LG carrier. |
| **Pyramidal FM pulse** | Triangular time envelope with linear frequency chirp; Welch PSD shows discrete **spectral shards**. |
| **Spectral shards** | Discrete sub-bands in the pulse power spectrum used as a barcode layer for payload recovery. |
| **Quaternion shard** | Unit quaternion (w, x, y, z) encoding payload bytes via hypercomplex compression (`quaternion_codec.py`). |
| **BMGL** | Beam-Motion-Gated Learning — turbulence/error gating protocol tied to OAM rotation rates. |
| **p-wave BMGL** | Odd-parity altermagnetic variant with SOC λ and splitting p; γ₁ controls inhibition boost (`altermagnetic.py`). |
| **Channel noise** | Gradio demo slider (0–1) scaling phase turbulence amplitude in `propagate_with_turbulence()`; 0.35 ≈ unit scale (`noise_level_to_scale`). |
| **Fisher-Rao distance** | Geodesic distance on the PWM duty probability simplex; measures glyph separability in the stable font. |
| **Stable font** | Codeword table of PWM duty vectors locked to emergent constants W_g = 350/π, κ = 0.85, braiding 0.084. |
| **W_g** | Emergent angular frequency constant 350/π used in the stable phase ladder. |
| **ICA** | Independent Component Analysis (FastICA) — demixes overlapping orb intensity channels at decode time. |
| **QEC** | Quantum Error Correction — 16-qubit canonical mode in the full pipeline; repetition proxy in the proto decoder. |
| **L_max** | Maximum OAM topological charge horizon in the full VQC pipeline (`configs/params.yaml`). |
| **Kolmogorov turbulence** | Phase-screen model for free-space scintillation (`turbulence.py`). |
| **SLM** | Spatial Light Modulator — phase-only display for virtual typehead holograms (`slm_typehead.py`). |
| **Gerchberg-Saxton** | Iterative phase-retrieval algorithm for sharper far-field holograms. |
| **Isomap** | Nonlinear manifold embedding used in the full pipeline for stress-guarded geometry checks. |
| **Stevedore 8₃ knot** | Topological protection backbone in the full VQC simulation (`knots.py`). |