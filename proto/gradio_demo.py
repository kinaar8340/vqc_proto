#!/usr/bin/env python3
"""Lightweight Gradio web demo for the Orbital Braille prototype."""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from demo_core import (
    PATENT_FIGURE1_PAYLOAD,
    VQC_CLAIMS_MD,
    export_slm_bundle,
    get_build_label,
    plot_results,
    run_pipeline,
)

logger = logging.getLogger(__name__)


def _patch_gradio_client_bool_schema() -> None:
    """Avoid gradio_client crash when JSON schema contains bare bool nodes."""
    try:
        from gradio_client import utils as client_utils

        if getattr(client_utils, "_vqc_bool_patch", False):
            return

        orig_get_type = client_utils.get_type

        def get_type(schema):  # noqa: ANN001
            if isinstance(schema, bool):
                return "boolean"
            return orig_get_type(schema)

        client_utils.get_type = get_type
        client_utils._vqc_bool_patch = True
        logger.info("Patched gradio_client bool JSON-schema handling")
    except Exception:
        logger.warning("Could not patch gradio_client", exc_info=True)


_patch_gradio_client_bool_schema()
DEFAULT_PAYLOAD = PATENT_FIGURE1_PAYLOAD
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar111/orbital-braille-vqc"
GITHUB_URL = "https://github.com/kinaar8340/vqc_proto"
HFB_RAW_URL = "https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png"

SLM_PACKAGE_IDLE = (
    "**Package files** (generated after **Run demo**):\n"
    "- `manifest.json` — orb geometry, PWM duties, quaternion, timing\n"
    "- `phase_stack.npy` — phase sequence array `[frames, H, W]`\n"
    "- `preview_montage.png` — visual sanity check\n"
    "- `LUT_calibration.txt` — gray→phase mapping notes\n"
    "- `README.txt` — driver quick-start (Holoeye, Meadowlark, Thorlabs)\n"
    "- `frames/` — optional PNG sequence (enable checkbox above)"
)

# Panels/fields: 10% opacity. Buttons and sliders stay solid for usability.
HFB_CSS = f"""
.gradio-container {{
    background-image: url('{HFB_RAW_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
    background-attachment: fixed !important;
}}
.gradio-container .contain,
.gradio-container .main,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .form,
.gradio-container .column,
.gradio-container .row,
.gradio-container .block,
.gradio-container .panel,
.gradio-container .gr-panel,
.gradio-container .gr-box,
.gradio-container .input-container,
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container [data-testid="textbox"],
.gradio-container .accordion,
.gradio-container details,
.gradio-container .file-preview,
.gradio-container .gr-file,
.gradio-container .image-container,
.gradio-container .gr-image {{
    background-color: rgba(10, 8, 24, 0.1) !important;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border-radius: 10px;
}}
.gradio-container label.wrap {{
    background: transparent !important;
}}
.gradio-container button,
.gradio-container .gr-button {{
    background-color: rgba(28, 22, 48, 0.92) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    opacity: 1 !important;
}}
.gradio-container button.primary,
.gradio-container .primary {{
    background-color: rgba(234, 88, 12, 0.95) !important;
    border-color: rgba(255, 180, 80, 0.5) !important;
}}
.gradio-container .gr-slider,
.gradio-container .gr-slider * {{
    background-color: rgba(28, 22, 48, 0.9) !important;
    opacity: 1 !important;
}}
.gradio-container input[type="range"] {{
    opacity: 1 !important;
}}
footer {{ visibility: hidden; }}
"""


def load_patent_example() -> tuple[str, float, float]:
    """Auto-fill patent Figure 1 payload and recommended orb / BMGL settings."""
    return PATENT_FIGURE1_PAYLOAD, 4, 1.5


def run_demo(
    payload: str,
    num_orbs: float,
    resolution: str,
    seed: float,
    gamma_1: float,
    export_slm_frames: bool,
) -> tuple[str, str | None, str, str | None]:
    if not payload.strip():
        payload = DEFAULT_PAYLOAD

    try:
        quick = resolution.strip().lower() == "quick"
        _, encoded, noisy, decoded, metrics, font_sep = run_pipeline(
            payload,
            int(num_orbs),
            quick=quick,
            seed=int(seed),
            gamma_1=float(gamma_1),
        )

        out_dir = Path(tempfile.mkdtemp(prefix="vqc_gradio_"))
        fig_path = str(plot_results(encoded, noisy, decoded, out_dir, payload))

        slm_dir = Path(tempfile.mkdtemp(prefix="vqc_slm_"))
        zip_path, slm_summary = export_slm_bundle(
            encoded,
            payload=payload,
            num_orbs=int(num_orbs),
            font_sep=font_sep,
            quick=quick,
            include_frames=export_slm_frames,
            out_dir=slm_dir / "slm",
        )
        slm_info = (
            f"**Ready:** `slm_package.zip`\n\n{slm_summary}\n\n"
            "**Files in zip:**\n"
            "- manifest.json\n"
            "- phase_stack.npy\n"
            "- preview_montage.png\n"
            "- LUT_calibration.txt\n"
            "- README.txt"
            + ("\n- frames/ (PNG sequence)" if export_slm_frames else "")
        )

        return metrics, fig_path, slm_info, str(zip_path)
    except Exception as exc:
        logger.exception("run_demo failed for payload=%r", payload)
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return err, None, SLM_PACKAGE_IDLE, None


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Orbital Braille — VQC Typehead",
        analytics_enabled=False,
        css=HFB_CSS,
    ) as demo:
        gr.Markdown(
            "# Orbital Braille — VQC Typehead Prototype\n"
            "Multi-orb PWM-gated sources → pyramidal spectral shards on an OAM carrier. "
            "Use **Quick** resolution for sub-second runs.\n\n"
            f"Source: [{GITHUB_URL}]({GITHUB_URL}) · "
            f"[Live demo]({HF_SPACE_URL}) · "
            f"[SLM quickstart]({GITHUB_URL}/blob/main/proto/SLM_QUICKSTART.md)\n\n"
            f"*{get_build_label()}*"
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
            gamma_1 = gr.Slider(
                1.0,
                2.0,
                value=1.5,
                step=0.1,
                label="p-wave BMGL strength (γ₁)",
                info="Higher γ₁ → stronger inhibition vs. phase noise (default 1.5)",
            )
        with gr.Row():
            export_slm_frames = gr.Checkbox(
                label="Include SLM-ready phase frames (PNG)",
                value=False,
                info="Adds frames/ to zip; slower. Core zip always has manifest + phase_stack.npy",
            )
            load_paper_btn = gr.Button("Load example from paper", variant="secondary")
        with gr.Accordion("How this maps to VQC claims", open=False):
            gr.Markdown(VQC_CLAIMS_MD)
        run_btn = gr.Button("Run demo", variant="primary")
        gr.Markdown(
            "**Example payloads:** `I live in Oregon` (4 orbs, patent Fig. 1) · "
            "`VQC prototype` (4 orbs) · `Hello OAM` (2 orbs)"
        )
        with gr.Row():
            metrics = gr.Textbox(label="Metrics", lines=12)
            figure = gr.Image(label="6-panel output", type="filepath")
        with gr.Accordion("SLM package download", open=False):
            slm_info = gr.Markdown(SLM_PACKAGE_IDLE)
            slm_file = gr.File(
                label="slm_package.zip",
                interactive=False,
                file_count="single",
                type="filepath",
            )
        run_btn.click(
            run_demo,
            [payload, num_orbs, resolution, seed, gamma_1, export_slm_frames],
            [metrics, figure, slm_info, slm_file],
        )
        load_paper_btn.click(load_patent_example, outputs=[payload, num_orbs, gamma_1])
        gr.Markdown(
            "Non-commercial research only · CC-BY-NC-SA-4.0 + patent restrictions · "
            f"[IP notice]({GITHUB_URL}/blob/main/IP_NOTICE.md)"
        )
    return demo


demo = build_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    try:
        demo.get_api_info()
        logger.info("Gradio API info check passed")
    except Exception:
        logger.exception("Gradio API info check failed")

    on_hf = bool(os.environ.get("SPACE_ID"))
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    launch_kwargs: dict = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
        "show_api": False,
        "inbrowser": False,
        "share": False if on_hf else True,
    }

    demo.queue(default_concurrency_limit=2).launch(**launch_kwargs)


if __name__ == "__main__":
    main()