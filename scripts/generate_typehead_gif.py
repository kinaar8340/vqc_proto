#!/usr/bin/env python3
"""Generate an animated GIF of the Orbital Braille typehead process."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto"
sys.path.insert(0, str(PROTO))

from orbital_braille import (  # noqa: E402
    EmergentConstants,
    OrbitalTypehead,
    PWaveBMGL,
    TypeheadConfig,
)

PAYLOAD = "I live in Oregon"
OUT_PATH = ROOT / "typehead_demo.gif"
FRAMES = 24
DURATION_MS = 130


def _render_frame(encoded, frame_idx: int) -> Image.Image:
    t_idx = min(frame_idx, encoded.intensity_time.shape[0] - 1)
    t = float(encoded.t[t_idx])
    t_max = float(encoded.t[-1])
    extent = [-2.5, 2.5, -2.5, 2.5]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), facecolor="#0a0818")
    fig.suptitle(
        f'Orbital Braille typehead — "{PAYLOAD}"  |  frame {frame_idx + 1}/{FRAMES}',
        color="white",
        fontsize=10,
    )

    ax0, ax1 = axes
    ax0.set_facecolor("#0a0818")
    ax1.set_facecolor("#0a0818")

    im = ax0.imshow(
        encoded.intensity_time[t_idx],
        cmap="inferno",
        extent=extent,
        origin="lower",
    )
    ax0.set_title("OAM intensity (donut + Braille lobes)", color="#ddd", fontsize=9)
    ax0.set_xlabel("x", color="#aaa")
    ax0.set_ylabel("y", color="#aaa")
    plt.colorbar(im, ax=ax0, fraction=0.046)

    for i, orb in enumerate(encoded.orbs):
        theta = orb.phase0 + orb.omega * t
        x0 = orb.radius * np.cos(theta)
        y0 = orb.radius * np.sin(theta)
        pwm_on = (np.sin(2 * np.pi * orb.omega * t / t_max) + 1) / 2 < orb.pwm_duty
        color = "#ff8c42" if pwm_on else "#4a5568"
        ax1.scatter(x0, y0, s=140, c=color, edgecolors="white", linewidths=0.6, zorder=3)
        ring = plt.Circle((0, 0), orb.radius, fill=False, linestyle="--", alpha=0.35, color="#888")
        ax1.add_patch(ring)
        ax1.annotate(
            f"ℓ={orb.ell}",
            (x0, y0),
            fontsize=7,
            ha="center",
            va="bottom",
            color="#eee",
        )

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect("equal")
    ax1.set_title("PWM-gated orbs on typehead tracks", color="#ddd", fontsize=9)
    ax1.grid(True, alpha=0.2, color="#666")

    pulse = encoded.pulse[t_idx]
    ax1.text(
        0.02,
        0.98,
        f"t = {t * 1e9:.1f} ns  |  pulse = {pulse:.2f}",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        color="#ccc",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1530", alpha=0.8),
    )

    for ax in axes:
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#444")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    cfg = TypeheadConfig(
        num_orbs=4,
        grid_size=48,
        num_times=FRAMES,
        bmgl=PWaveBMGL(gamma_1=1.5),
        constants=EmergentConstants(),
    )
    typehead = OrbitalTypehead(cfg, seed=42)
    encoded = typehead.encode(PAYLOAD)

    frames = [_render_frame(encoded, i) for i in range(FRAMES)]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT_PATH} ({len(frames)} frames, {OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()