#!/usr/bin/env python3
"""Profile LG mode generation, OAM projection, ICA demix, and full encode/decode."""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import FastICA

from orbital_braille import (
    OrbitalTypehead,
    decode_field,
    lg_mode,
    project_oam_spectrum,
)
from run_demo import build_config

logger = logging.getLogger(__name__)


def _grid(size: int, extent: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-extent, extent, size)
    y = np.linspace(-extent, extent, size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    return rho, phi, X


def bench_lg_modes(grid_size: int, repeats: int) -> float:
    rho, phi, _ = _grid(grid_size)
    t0 = time.perf_counter()
    for ell in range(-6, 7):
        for _ in range(repeats):
            lg_mode(ell, rho, phi, w0=1.0)
    return time.perf_counter() - t0


def bench_oam_projection(grid_size: int, repeats: int) -> float:
    rho, phi, _ = _grid(grid_size)
    field = lg_mode(1, rho, phi) + 0.5 * lg_mode(2, rho, phi)
    ell_range = list(range(-6, 7))
    t0 = time.perf_counter()
    for _ in range(repeats):
        project_oam_spectrum(field, rho, phi, ell_range)
    return time.perf_counter() - t0


def bench_ica(n_t: int, grid_size: int, n_orbs: int, repeats: int) -> float:
    flat = np.random.default_rng(42).random((grid_size * grid_size, n_t))
    t0 = time.perf_counter()
    for _ in range(repeats):
        ica = FastICA(n_components=min(n_orbs, n_t), random_state=42, max_iter=2000, tol=1e-4)
        ica.fit_transform(flat)
    return time.perf_counter() - t0


def bench_encode_decode(quick: bool, num_orbs: int) -> tuple[float, float, float]:
    cfg = build_config(num_orbs, quick=quick)
    typehead = OrbitalTypehead(cfg, seed=42)
    payload = "I live in Oregon"

    t0 = time.perf_counter()
    encoded = typehead.encode(payload)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    noisy = typehead.propagate_with_turbulence(encoded)
    t_turb = time.perf_counter() - t0

    t0 = time.perf_counter()
    decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=cfg.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        pulse_ref=encoded.pulse,
        t=encoded.t,
    )
    t_decode = time.perf_counter() - t0
    return t_encode, t_turb, t_decode


def run_cprofile(quick: bool, num_orbs: int, out_path: Path) -> None:
    cfg = build_config(num_orbs, quick=quick)

    def _pipeline() -> None:
        typehead = OrbitalTypehead(cfg, seed=42)
        encoded = typehead.encode("profile")
        noisy = typehead.propagate_with_turbulence(encoded)
        decode_field(
            noisy,
            encoded.intensity_time,
            typehead.font,
            [o.ell for o in encoded.orbs],
            bmgl=cfg.bmgl,
            rho=encoded.rho,
            phi=encoded.phi,
            pulse_ref=encoded.pulse,
            t=encoded.t,
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _pipeline()
    profiler.disable()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        stats = pstats.Stats(profiler, stream=fh)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
    logger.info("Wrote cProfile report → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Orbital Braille hot paths")
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--num-orbs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Profile quick-res encode/decode")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "profile",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    grid = 32 if args.quick else args.grid_size
    n_t = 16 if args.quick else 64

    print("=" * 60)
    print("ORBITAL BRAILLE — HOT PATH PROFILE")
    print("=" * 60)
    print(f"grid={grid}  times={n_t}  orbs={args.num_orbs}  repeats={args.repeats}")
    print()

    lg_s = bench_lg_modes(grid, args.repeats)
    oam_s = bench_oam_projection(grid, args.repeats)
    ica_s = bench_ica(n_t, grid, args.num_orbs, args.repeats)
    t_enc, t_turb, t_dec = bench_encode_decode(args.quick, args.num_orbs)

    rows = [
        ("LG mode generation (13 ℓ × repeats)", lg_s),
        ("OAM projection (13 ℓ × repeats)", oam_s),
        ("FastICA demix (repeats)", ica_s),
        ("encode() single pass", t_enc),
        ("propagate_with_turbulence()", t_turb),
        ("decode_field() single pass", t_dec),
    ]
    print(f"{'Stage':<42} {'seconds':>10}")
    print("-" * 54)
    for name, secs in rows:
        print(f"{name:<42} {secs:>10.3f}")
    print("-" * 54)
    total = t_enc + t_turb + t_dec
    print(f"{'encode + turbulence + decode':<42} {total:>10.3f}")
    print()

    report = args.out_dir / ("profile_quick.txt" if args.quick else "profile_full.txt")
    run_cprofile(args.quick, args.num_orbs, report)
    print(f"cProfile top-40 → {report}")


if __name__ == "__main__":
    main()