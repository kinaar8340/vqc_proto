#!/usr/bin/env python3
"""
Orbital Braille prototype demo — multi-orb typehead for VQC pyramidal shards.

Concept: N orbiting laser spots (Selectric typeball analog) whose PWM-gated
interference creates pyramidal FM pulses, spectral shards, and OAM helical
content. Stable codeword fonts lock to emergent TOE constants (350/π, κ=0.85).

Usage:
    python run_demo.py
    python run_demo.py --payload "I live in Oregon" --num-orbs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbital_braille import (
    OrbitalTypehead,
    TypeheadConfig,
    PWaveBMGL,
    EmergentConstants,
    decode_field,
    build_stable_font,
    font_separation,
)


def plot_results(
    encoded,
    noisy,
    decoded,
    out_dir: Path,
    payload: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = encoded.field_time.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Orbital Braille — VQC Typehead Prototype\n"
        f'Payload: "{payload[:32]}" | Glyph {decoded.glyph_index} | '
        f"Shard FID {decoded.shard_fidelity:.3f}",
        fontsize=11,
    )

    extent = [-2.5, 2.5, -2.5, 2.5]

    im0 = axes[0, 0].imshow(
        np.angle(encoded.field_time[mid]),
        cmap="hsv",
        extent=extent,
        origin="lower",
    )
    axes[0, 0].set_title("Encoded phase (clean)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(
        np.angle(noisy[mid]),
        cmap="hsv",
        extent=extent,
        origin="lower",
    )
    axes[0, 1].set_title("Phase after p-wave BMGL turbulence")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(
        encoded.intensity_time[mid],
        cmap="inferno",
        extent=extent,
        origin="lower",
    )
    axes[0, 2].set_title("Intensity — OAM donut + orbital Braille")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].plot(encoded.t * 1e9, encoded.pulse, "b-", lw=1.5)
    axes[1, 0].fill_between(encoded.t * 1e9, 0, encoded.pulse, alpha=0.3)
    axes[1, 0].set_title("Pyramidal FM pulse (time)")
    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(encoded.freqs / 1e9, encoded.spectral_shards + 1e-20, "r-")
    axes[1, 1].set_title("Spectral shards (Welch PSD)")
    axes[1, 1].set_xlabel("Frequency (GHz)")
    axes[1, 1].set_ylabel("PSD")
    axes[1, 1].grid(True, alpha=0.3)

    orb_x = [o.radius * np.cos(o.phase0) for o in encoded.orbs]
    orb_y = [o.radius * np.sin(o.phase0) for o in encoded.orbs]
    axes[1, 2].scatter(orb_x, orb_y, s=120, c=range(len(encoded.orbs)), cmap="tab10")
    for i, o in enumerate(encoded.orbs):
        theta = np.linspace(0, 2 * np.pi, 64)
        axes[1, 2].plot(
            o.radius * np.cos(theta),
            o.radius * np.sin(theta),
            "--",
            alpha=0.4,
            color=plt.cm.tab10(i / max(len(encoded.orbs), 1)),
        )
        axes[1, 2].annotate(
            f"ℓ={o.ell}\nd={o.pwm_duty:.2f}",
            (orb_x[i], orb_y[i]),
            fontsize=7,
            ha="center",
        )
    axes[1, 2].set_title("Typehead orb layout (Braille dots)")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    axes[1, 2].set_aspect("equal")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "orbital_braille_demo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbital Braille VQC prototype")
    parser.add_argument("--payload", default="I live in Oregon", help="Text to encode")
    parser.add_argument("--num-orbs", type=int, default=4, help="Number of orbiting sources")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()

    cfg = TypeheadConfig(
        num_orbs=args.num_orbs,
        bmgl=PWaveBMGL(gamma_1=1.5),
        constants=EmergentConstants(),
    )
    typehead = OrbitalTypehead(cfg, seed=args.seed)

    font_sep = font_separation(build_stable_font(args.num_orbs, num_glyphs=16))
    print("=" * 60)
    print("ORBITAL BRAILLE — VQC TYPEHEAD PROTOTYPE")
    print("=" * 60)
    print(f"Emergent W_g = {cfg.constants.Wg:.4f}  (350/π)")
    print(f"Braiding linking target = {cfg.constants.braiding_linking}")
    print(f"p-wave BMGL γ₁ = {cfg.bmgl.gamma_1}  boost = {cfg.bmgl.inhibition_boost:.3f}")
    print(f"Stable font mean Fisher-Rao separation = {font_sep:.4f} rad")
    print(f"Orbs: {args.num_orbs}  |  Payload: {args.payload!r}")
    print()

    encoded = typehead.encode(args.payload)
    noisy = typehead.propagate_with_turbulence(encoded)

    decoded = decode_field(
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

    print(f"Encoded quaternion: w={encoded.quaternion.w:.3f} "
          f"x={encoded.quaternion.x:.3f} y={encoded.quaternion.y:.3f} "
          f"z={encoded.quaternion.z:.3f}")
    print(f"Glyph duties: {encoded.glyph_duties.round(3)}")
    print(f"OAM weights: {', '.join(f'ℓ={k}:{abs(v):.3f}' for k, v in sorted(decoded.oam_weights.items()))}")
    print(f"Dominant ℓ recovered: {decoded.recovered_ells}")
    print(f"Shard fidelity (Pearson): {decoded.shard_fidelity:.4f}")
    print(f"Glyph match: index={decoded.glyph_index}  fidelity={decoded.glyph_fidelity:.4f}")
    print(f"Recovered bytes (1st 4): {list(decoded.recovered_bytes)}")
    print()

    plot_results(encoded, noisy, decoded, args.out_dir, args.payload)
    print("Demo complete.")


if __name__ == "__main__":
    main()