#!/usr/bin/env python3
"""Generate the canonical typehead_demo.gif for README / HF docs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto"
sys.path.insert(0, str(PROTO))

from demo_core import render_typehead_animation  # noqa: E402
from orbital_braille import (  # noqa: E402
    EmergentConstants,
    OrbitalTypehead,
    PWaveBMGL,
    TypeheadConfig,
)

PAYLOAD = "I live in Oregon"
OUT_PATH = ROOT / "typehead_demo.gif"


def main() -> None:
    cfg = TypeheadConfig(
        num_orbs=4,
        grid_size=48,
        num_times=24,
        bmgl=PWaveBMGL(gamma_1=1.5),
        constants=EmergentConstants(),
    )
    th = OrbitalTypehead(cfg, seed=42)
    enc = th.encode(PAYLOAD)
    noisy = th.propagate_with_turbulence(enc)
    render_typehead_animation(enc, noisy, PAYLOAD, OUT_PATH, max_frames=24)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()