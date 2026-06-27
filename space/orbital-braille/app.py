#!/usr/bin/env python3
"""Lightweight Gradio web demo for the Orbital Braille prototype."""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from demo_core import plot_results, run_pipeline

logger = logging.getLogger(__name__)
DEFAULT_PAYLOAD = "I live in Oregon"
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar111/orbital-braille-vqc"
GITHUB_URL = "https://github.com/kinaar8340/vqc_proto"


def run_demo(
    payload: str,
    num_orbs: float,
    resolution: str,
    seed: float,
) -> tuple[str, str | None]:
    if not payload.strip():
        payload = DEFAULT_PAYLOAD

    try:
        quick = resolution.strip().lower() == "quick"
        _, encoded, noisy, decoded, metrics, _ = run_pipeline(
            payload, int(num_orbs), quick=quick, seed=int(seed)
        )

        out_dir = Path(tempfile.mkdtemp(prefix="vqc_gradio_"))
        fig_path = str(plot_results(encoded, noisy, decoded, out_dir, payload))
        return metrics, fig_path
    except Exception as exc:
        logger.exception("run_demo failed for payload=%r", payload)
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return err, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Orbital Braille — VQC Typehead", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Orbital Braille — VQC Typehead Prototype\n"
            "Multi-orb PWM-gated sources → pyramidal spectral shards on an OAM carrier. "
            "Use **Quick** resolution for sub-second runs.\n\n"
            f"Source: [{GITHUB_URL}]({GITHUB_URL}) · "
            f"[Live demo]({HF_SPACE_URL})"
        )
        with gr.Row():
            payload = gr.Textbox(label="Payload", value=DEFAULT_PAYLOAD)
            num_orbs = gr.Slider(2, 6, value=4, step=1, label="Number of orbs")
        with gr.Row():
            resolution = gr.Radio(
                choices=["Quick", "Full"],
                value="Quick",
                label="Resolution",
                info="Quick = low grid (fast); Full = publication quality",
            )
            seed = gr.Slider(0, 9999, value=42, step=1, label="Random seed")
        run_btn = gr.Button("Run demo", variant="primary")
        gr.Markdown(
            "**Example payloads:** `I live in Oregon` (4 orbs) · `VQC prototype` (4 orbs) · `Hello OAM` (2 orbs)"
        )
        with gr.Row():
            metrics = gr.Textbox(label="Metrics", lines=10)
            figure = gr.Image(label="6-panel output", type="filepath")
        run_btn.click(run_demo, [payload, num_orbs, resolution, seed], [metrics, figure])
        gr.Markdown(
            "Non-commercial research only · CC-BY-NC-SA-4.0 + patent restrictions · "
            f"[IP notice]({GITHUB_URL}/blob/main/IP_NOTICE.md)"
        )
    return demo


demo = build_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    on_hf = bool(os.environ.get("SPACE_ID"))
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    launch_kwargs: dict = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
        "show_api": False,
        "ssr": False,
        "inbrowser": False,
        "share": False if on_hf else True,
    }

    demo.queue(default_concurrency_limit=2).launch(**launch_kwargs)


if __name__ == "__main__":
    main()