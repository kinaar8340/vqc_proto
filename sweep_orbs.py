#!/usr/bin/env python3
"""Sweep num_orbs (2–6) and report shard/glyph fidelity + font separation."""

from __future__ import annotations

import argparse

from orbital_braille import (
    OrbitalTypehead,
    TypeheadConfig,
    PWaveBMGL,
    EmergentConstants,
    decode_field,
    build_stable_font,
    font_separation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="I live in Oregon")
    parser.add_argument("--min-orbs", type=int, default=2)
    parser.add_argument("--max-orbs", type=int, default=6)
    args = parser.parse_args()

    print(f"{'orbs':>5}  {'font_sep':>10}  {'shard_fid':>10}  {'glyph_fid':>10}  {'dom_ell':>8}")
    print("-" * 50)

    for n in range(args.min_orbs, args.max_orbs + 1):
        cfg = TypeheadConfig(num_orbs=n, bmgl=PWaveBMGL(gamma_1=1.5), constants=EmergentConstants())
        th = OrbitalTypehead(cfg, seed=42)
        enc = th.encode(args.payload)
        noisy = th.propagate_with_turbulence(enc)
        dec = decode_field(
            noisy,
            enc.intensity_time,
            th.font,
            [o.ell for o in enc.orbs],
            bmgl=cfg.bmgl,
            rho=enc.rho,
            phi=enc.phi,
        )
        sep = font_separation(build_stable_font(n, num_glyphs=16))
        print(
            f"{n:>5}  {sep:>10.4f}  {dec.shard_fidelity:>10.4f}  "
            f"{dec.glyph_fidelity:>10.4f}  {dec.recovered_ells[0]:>8}"
        )


if __name__ == "__main__":
    main()