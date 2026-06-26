#!/usr/bin/env python3
"""Lightweight Gradio web demo for the Orbital Braille prototype."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from orbital_braille import (
    OrbitalTypehead,
    PWaveBMGL,
    EmergentConstants,
    decode_field,
    build_stable_font,
    font_separation,
)
from run_demo import build_config, plot_results

logger = logging.getLogger(__name__)
DEFAULT_PAYLOAD = "I live in Oregon"


def run_demo(
    payload: str,
    num_orbs: int,
    quick: bool,
    seed: int,
) -> tuple[str, str | None]:
    """Encode → turbulence → decode; return metrics text and figure path."""
    if not payload.strip():
        payload = DEFAULT_PAYLOAD

    cfg = build_config(num_orbs, quick=quick)
    typehead = OrbitalTypehead(cfg, seed=seed)
    font_sep = font_separation(build_stable_font(num_orbs, num_glyphs=16))

    encoded = typehead.encode(payload)
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

    mode = "QUICK" if quick else "FULL"
    metrics = "\n".join(
        [
            f"Mode: {mode} (grid={cfg.grid_size}, times={cfg.num_times})",
            f"Payload: {payload!r}",
            f"Orbs: {num_orbs}",
            f"Font separation: {font_sep:.4f} rad",
            f"Shard fidelity: {decoded.shard_fidelity:.4f}",
            f"Glyph: index={decoded.glyph_index}  fidelity={decoded.glyph_fidelity:.4f}",
            f"Quaternion: w={encoded.quaternion.w:.3f} "
            f"x={encoded.quaternion.x:.3f} y={encoded.quaternion.y:.3f} "
            f"z={encoded.quaternion.z:.3f}",
            f"Dominant ℓ: {decoded.recovered_ells}",
        ]
    )

    out_dir = Path(tempfile.mkdtemp(prefix="vqc_gradio_"))
    plot_results(encoded, noisy, decoded, out_dir, payload)
    fig_path = str(out_dir / "orbital_braille_demo.png")
    return metrics, fig_path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Orbital Braille — VQC Typehead") as demo:
        gr.Markdown(
            "# Orbital Braille — VQC Typehead Prototype\n"
            "Multi-orb PWM-gated sources → pyramidal spectral shards on an OAM carrier. "
            "Enable **Quick mode** for sub-second runs."
        )
        with gr.Row():
            payload = gr.Textbox(label="Payload", value=DEFAULT_PAYLOAD)
            num_orbs = gr.Slider(2, 6, value=4, step=1, label="Number of orbs")
        with gr.Row():
            quick = gr.Checkbox(label="Quick mode (low resolution)", value=True)
            seed = gr.Number(label="Random seed", value=42, precision=0)
        run_btn = gr.Button("Run demo", variant="primary")
        with gr.Row():
            metrics = gr.Textbox(label="Metrics", lines=10)
            figure = gr.Image(label="6-panel output", type="filepath")
        run_btn.click(run_demo, [payload, num_orbs, quick, seed], [metrics, figure])
        gr.Markdown(
            "Repo: [kinaar8340/vqc_proto](https://github.com/kinaar8340/vqc_proto) · "
            "Non-commercial research only — see IP_NOTICE.md"
        )
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()