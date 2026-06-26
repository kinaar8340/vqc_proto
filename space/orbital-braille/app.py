#!/usr/bin/env python3
"""Lightweight Gradio web demo for the Orbital Braille prototype."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import gradio as gr

from demo_core import plot_results, run_pipeline

logger = logging.getLogger(__name__)
DEFAULT_PAYLOAD = "I live in Oregon"
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar8340/orbital-braille-vqc"
GITHUB_URL = "https://github.com/kinaar8340/vqc_proto"


def run_demo(
    payload: str,
    num_orbs: int,
    quick: bool,
    seed: int,
) -> tuple[str, str | None]:
    if not payload.strip():
        payload = DEFAULT_PAYLOAD

    _, encoded, noisy, decoded, metrics, _ = run_pipeline(
        payload, int(num_orbs), quick=quick, seed=int(seed)
    )

    out_dir = Path(tempfile.mkdtemp(prefix="vqc_gradio_"))
    fig_path = str(plot_results(encoded, noisy, decoded, out_dir, payload))
    return metrics, fig_path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Orbital Braille — VQC Typehead") as demo:
        gr.Markdown(
            "# Orbital Braille — VQC Typehead Prototype\n"
            "Multi-orb PWM-gated sources → pyramidal spectral shards on an OAM carrier. "
            "Enable **Quick mode** for sub-second runs.\n\n"
            f"Source: [{GITHUB_URL}]({GITHUB_URL})"
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
        gr.Examples(
            examples=[
                ["I live in Oregon", 4, True, 42],
                ["VQC prototype", 4, True, 42],
                ["Hello OAM", 2, True, 7],
            ],
            inputs=[payload, num_orbs, quick, seed],
            outputs=[metrics, figure],
            fn=run_demo,
            cache_examples=False,
        )
        run_btn.click(run_demo, [payload, num_orbs, quick, seed], [metrics, figure])
        gr.Markdown(
            "Non-commercial research only · CC-BY-NC-SA-4.0 + patent restrictions · "
            f"[IP notice]({GITHUB_URL}/blob/main/IP_NOTICE.md)"
        )
    return demo


demo = build_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    on_hf = os.getenv("SPACE_ID") is not None
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        **({} if on_hf else {}),
    )


if __name__ == "__main__":
    main()