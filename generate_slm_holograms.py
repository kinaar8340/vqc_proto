#!/usr/bin/env python3
"""Generate SLM phase hologram sequence for 4-orb virtual typehead."""

from __future__ import annotations

import argparse
from pathlib import Path

from orbital_braille import OrbitalTypehead, TypeheadConfig, PWaveBMGL, EmergentConstants
from orbital_braille.font_optimizer import optimize_font
from orbital_braille.slm_typehead import SLMConfig, slm_phase_sequence, save_phase_hologram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="I live in Oregon")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "outputs" / "slm")
    args = parser.parse_args()

    cfg = TypeheadConfig(num_orbs=4, bmgl=PWaveBMGL(gamma_1=1.5), constants=EmergentConstants())
    th = OrbitalTypehead(cfg)
    font = optimize_font(num_orbs=4, num_glyphs=16)
    th.font = font.font

    enc = th.encode(args.payload)
    slm_cfg = SLMConfig(resolution=512)
    stack = slm_phase_sequence(enc.orbs, args.frames, enc.t[-1], cfg=slm_cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(8, args.frames)):
        save_phase_hologram(stack[i], str(args.out_dir / f"slm_frame_{i:03d}.png"))

    print(f"SLM holograms: {args.out_dir}  ({args.frames} frames, 4-orb virtual typehead)")
    print(f"Font separation: {font.mean_separation:.4f} rad")


if __name__ == "__main__":
    main()