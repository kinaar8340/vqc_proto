#!/usr/bin/env python3
"""
Meta-optimizer for orbital typehead parameters.

Ties emergent TOE invariants (wg_base=350, kappa=0.85, braiding=0.084)
to orb geometry: radii, omega, PWM duties, and Fisher-Rao font separation.

Inspired by toe/scripts/meta_optimize_invariants.py and Rhythm 3-body emergence bake.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from orbital_braille import (
    OrbitalTypehead,
    TypeheadConfig,
    PWaveBMGL,
    EmergentConstants,
    decode_field,
)
from orbital_braille.font_optimizer import optimize_font
from orbital_braille.turbulence import apply_free_space_channel


TARGETS = {
    "wg_base": 350.0,
    "kappa": 0.85,
    "braiding_linking": 0.084,
    "shard_fidelity_min": 0.92,
    "font_separation_min": 0.95,
}


def evaluate_config(
    num_orbs: int,
    gamma_1: float,
    r0_m: float,
    payload: str,
    seed: int = 42,
) -> dict:
    constants = EmergentConstants()
    font_result = optimize_font(num_orbs=num_orbs, num_glyphs=16, constants=constants)

    cfg = TypeheadConfig(
        num_orbs=num_orbs,
        bmgl=PWaveBMGL(gamma_1=gamma_1),
        constants=constants,
    )
    th = OrbitalTypehead(cfg, seed=seed)
    th.font = font_result.font

    enc = th.encode(payload)
    noisy = apply_free_space_channel(
        enc.field_time,
        cfg.bmgl,
        phi=enc.phi,
        r0_m=r0_m,
        rng=th.rng,
    )
    dec = decode_field(
        noisy,
        enc.intensity_time,
        th.font,
        [o.ell for o in enc.orbs],
        bmgl=cfg.bmgl,
        rho=enc.rho,
        phi=enc.phi,
    )

    hopf_match = abs(constants.Wg - TARGETS["wg_base"] / np.pi)
    braid_match = abs(constants.braiding_linking - TARGETS["braiding_linking"])

    loss = (
        max(0, TARGETS["shard_fidelity_min"] - dec.shard_fidelity) * 10
        + max(0, TARGETS["font_separation_min"] - font_result.mean_separation) * 5
        + hopf_match * 0.1
        + braid_match * 2
        + (1 - dec.glyph_fidelity)
    )

    return {
        "num_orbs": num_orbs,
        "gamma_1": gamma_1,
        "r0_m": r0_m,
        "shard_fidelity": dec.shard_fidelity,
        "glyph_fidelity": dec.glyph_fidelity,
        "font_separation": font_result.mean_separation,
        "font_min_separation": font_result.min_separation,
        "hopf_match": hopf_match,
        "braiding_match": braid_match,
        "loss": loss,
        "dominant_ell": dec.recovered_ells[0] if dec.recovered_ells else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-optimize orbital typehead")
    parser.add_argument("--payload", default="I live in Oregon")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "outputs" / "meta")
    args = parser.parse_args()

    trials = []
    for num_orbs in [2, 3, 4, 5, 6]:
        for gamma_1 in [1.3, 1.5, 1.7]:
            for r0_m in [0.05, 0.1, 0.2]:
                trials.append(evaluate_config(num_orbs, gamma_1, r0_m, args.payload))

    trials.sort(key=lambda t: t["loss"])
    best = trials[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_dir / f"meta_orbital_{ts}.json"
    out_json.write_text(json.dumps({"targets": TARGETS, "trials": trials, "best": best}, indent=2))

    print("=" * 60)
    print("ORBITAL TYPEHEAD META-OPTIMIZATION")
    print("=" * 60)
    print(f"Best: {best['num_orbs']} orbs | γ₁={best['gamma_1']} | r0={best['r0_m']}m")
    print(f"  shard_fid={best['shard_fidelity']:.4f}  glyph_fid={best['glyph_fidelity']:.4f}")
    print(f"  font_sep={best['font_separation']:.4f}  loss={best['loss']:.4f}")
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()