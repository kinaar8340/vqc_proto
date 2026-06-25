# vqc_proto — Orbital Braille Typehead Prototype

Fork of [vqc_sims_public](https://github.com/kinaar8340/vqc_sims_public) implementing the **Orbital Braille / VQC Typehead** embodiment: multi-orb PWM-gated point sources whose interference generates pyramidal spectral shards on an OAM/quaternion carrier.

## Concept

| Typeball | VQC |
|----------|-----|
| Spinning ball selects character | Orbital phases + PWM duties select glyph |
| Braille dots | N orbiting laser spots |
| Impact timing | Pyramidal FM pulse |
| Font | Stable codewords (350/π, κ=0.85, braiding 0.084) |
| Paper impression | LG OAM donut + spectral shards |

## Quick Start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python run_demo.py --payload "I live in Oregon" --num-orbs 4
.venv/bin/python sweep_orbs.py
.venv/bin/python meta_optimize_orbital.py
.venv/bin/python generate_slm_holograms.py
```

## Modules

- `orbital_braille/typehead.py` — Multi-orb encoder (Selectric typeball analog)
- `orbital_braille/decoder.py` — OAM projection + ICA glyph recovery
- `orbital_braille/lg_modes.py` — Laguerre-Gaussian mode generation
- `orbital_braille/altermagnetic.py` — p-wave BMGL (γ₁=1.5)
- `orbital_braille/slm_typehead.py` — Phase-only SLM virtual typehead
- `orbital_braille/font_optimizer.py` — Fisher-Rao glyph separation
- `orbital_braille/turbulence.py` — Kolmogorov + pointing jitter (LEO)

## Orb Sweep (prototype sweet spot)

| Orbs | Fisher-Rao sep | Shard FID | Glyph FID |
|------|----------------|-----------|-----------|
| 2 | 0.787 | 0.937 | 0.999 |
| **4** | **0.989** | **0.929** | **0.868** |
| 6 | 1.027 | 0.920 | 0.804 |

**4 orbs** is the recommended prototype configuration.

## Parent Repository

This fork extends the VQC simulation suite:

- https://github.com/kinaar8340/vqc_sims_public

## License

Same as parent repository (see vqc_sims_public).