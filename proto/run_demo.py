#!/usr/bin/env python3
"""
Orbital Braille prototype demo — multi-orb typehead for VQC pyramidal shards.

Usage:
    python run_demo.py
    python run_demo.py --payload "I live in Oregon" --num-orbs 4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from demo_core import build_config, plot_results, run_pipeline
from orbital_braille import build_stable_font, font_separation

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbital Braille VQC prototype")
    parser.add_argument("--payload", default="I live in Oregon", help="Text to encode")
    parser.add_argument("--num-orbs", type=int, default=4, help="Number of orbiting sources")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Low-resolution mode (~seconds). Use run_demo_quick.py for the same behavior.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    t0 = time.perf_counter()
    cfg, encoded, noisy, decoded, _, _ = run_pipeline(
        args.payload, args.num_orbs, quick=args.quick, seed=args.seed
    )
    font_sep = font_separation(build_stable_font(args.num_orbs, num_glyphs=16))
    mode = "QUICK" if args.quick else "FULL"

    print("=" * 60)
    print("ORBITAL BRAILLE — VQC TYPEHEAD PROTOTYPE")
    print("=" * 60)
    print(f"Mode: {mode}  (grid={cfg.grid_size}, times={cfg.num_times})")
    print(f"Emergent W_g = {cfg.constants.Wg:.4f}  (350/π)")
    print(f"Braiding linking target = {cfg.constants.braiding_linking}")
    print(f"p-wave BMGL γ₁ = {cfg.bmgl.gamma_1}  boost = {cfg.bmgl.inhibition_boost:.3f}")
    print(f"Stable font mean Fisher-Rao separation = {font_sep:.4f} rad")
    print(f"Orbs: {args.num_orbs}  |  Payload: {args.payload!r}")
    print()
    print(f"Encoded quaternion: w={encoded.quaternion.w:.3f} "
          f"x={encoded.quaternion.x:.3f} y={encoded.quaternion.y:.3f} "
          f"z={encoded.quaternion.z:.3f}")
    print(f"Glyph duties: {encoded.glyph_duties.round(3)}")
    print(
        f"OAM weights: "
        f"{', '.join(f'ℓ={k}:{abs(v):.3f}' for k, v in sorted(decoded.oam_weights.items()))}"
    )
    print(f"Dominant ℓ recovered: {decoded.recovered_ells}")
    print(f"Shard fidelity (Pearson): {decoded.shard_fidelity:.4f}")
    print(f"Glyph match: index={decoded.glyph_index}  fidelity={decoded.glyph_fidelity:.4f}")
    print(f"Recovered bytes (1st 4): {list(decoded.recovered_bytes)}")
    print()

    path = plot_results(encoded, noisy, decoded, args.out_dir, args.payload)
    elapsed = time.perf_counter() - t0
    print(f"Saved → {path}")
    print(f"Demo complete in {elapsed:.1f}s.")
    if args.quick:
        print("Tip: omit --quick (or use run_demo.py) for publication-quality figures.")


if __name__ == "__main__":
    main()