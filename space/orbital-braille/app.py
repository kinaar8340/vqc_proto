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
    DEFAULT_NOISE_LEVEL,
    EXAMPLE_PRESETS,
    ONBOARDING_MD,
    PATENT_FIGURE1_PAYLOAD,
    SIMULATION_BANNER_MD,
    VQC_CLAIMS_MD,
    export_slm_bundle,
    get_animation_max_frames,
    get_build_label,
    is_hf_space,
    load_example_preset,
    build_orb_trajectory_3d_plotly,
    plot_results,
    render_typehead_animation_bundle,
    run_pipeline,
)

DEMO_SCREENCAST_URL = (
    "https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/docs/typehead_screencast.mp4"
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

# Background: full hfb.png at 5% opacity (pseudo-layer). Panels stay lightly frosted.
HFB_CSS = f"""
.gradio-container {{
    position: relative !important;
    background-color: #0a0818 !important;
}}
.gradio-container::before {{
    content: "" !important;
    position: fixed !important;
    inset: 0 !important;
    z-index: 0 !important;
    pointer-events: none !important;
    background-image: url('{HFB_RAW_URL}') !important;
    background-size: contain !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    opacity: 0.05 !important;
}}
.gradio-container > .main,
.gradio-container > .wrap,
.gradio-container > .contain {{
    position: relative !important;
    z-index: 1 !important;
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
.gradio-container .vqc-full-width {{
    width: 100% !important;
}}
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {{
    max-width: 100% !important;
}}
.gradio-container .vqc-animation-panel {{
    width: 100% !important;
    min-height: 480px;
}}
.gradio-container .vqc-animation-panel video,
.gradio-container .vqc-animation-panel .image-container,
.gradio-container .vqc-animation-panel img {{
    width: 100% !important;
    max-width: 100% !important;
    object-fit: contain;
}}
.gradio-container .vqc-figure-panel .image-container,
.gradio-container .vqc-figure-panel img {{
    width: 100% !important;
    object-fit: contain;
}}
.gradio-container .vqc-plot3d-panel,
.gradio-container .vqc-plot3d-panel > div,
.gradio-container .vqc-plot3d-panel .plot-container {{
    width: 100% !important;
    min-height: 480px;
}}
footer {{ visibility: hidden; }}
"""


def load_patent_example() -> tuple[str, float, float, float]:
    """Auto-fill patent Figure 1 payload and recommended orb / BMGL settings."""
    return load_example_preset("patent")


def run_demo(
    payload: str,
    num_orbs: float,
    resolution: str,
    seed: float,
    gamma_1: float,
    noise_level: float,
    export_slm_frames: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str | None, object | None, str, str | None, tuple | None]:
    if not payload.strip():
        payload = DEFAULT_PAYLOAD

    try:
        quick = resolution.strip().lower() == "quick"
        if is_hf_space() and export_slm_frames:
            export_slm_frames = False

        progress(0.05, desc="Encoding payload…")
        _, encoded, noisy, decoded, metrics, font_sep = run_pipeline(
            payload,
            int(num_orbs),
            quick=quick,
            seed=int(seed),
            gamma_1=float(gamma_1),
            noise_level=float(noise_level),
        )

        out_dir = Path(tempfile.mkdtemp(prefix="vqc_gradio_"))
        progress(0.45, desc="Rendering 6-panel figure…")
        fig_path = str(plot_results(encoded, noisy, decoded, out_dir, payload))

        progress(0.65, desc="Building interactive 3D orb trajectories…")
        fig_3d = build_orb_trajectory_3d_plotly(encoded, payload)

        progress(0.8, desc="Packaging SLM export…")
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

        run_cache = (encoded, noisy, payload, quick)
        progress(1.0, desc="Done")
        return metrics, fig_path, fig_3d, slm_info, str(zip_path), run_cache
    except Exception as exc:
        logger.exception("run_demo failed for payload=%r", payload)
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return err, None, None, SLM_PACKAGE_IDLE, None, None


def animate_typehead(
    run_cache: tuple | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str | None, str | None, str]:
    if run_cache is None:
        return None, None, "*Run **Run demo** first, then click **Animate typehead**.*"

    encoded, noisy, payload, quick = run_cache
    try:
        max_frames = get_animation_max_frames(quick=quick)
        n_total = encoded.intensity_time.shape[0]
        n_render = min(n_total, max_frames) if max_frames else n_total

        progress(0.1, desc=f"Rendering {n_render} animation frames…")
        out_dir = Path(tempfile.mkdtemp(prefix="vqc_anim_"))
        gif_path, mp4_path = render_typehead_animation_bundle(
            encoded,
            noisy,
            payload,
            out_dir,
            max_frames=max_frames,
        )
        progress(1.0, desc="Animation ready")
        fmt = "MP4 + GIF" if mp4_path else "GIF"
        cap_note = f" (capped to {n_render} on HF)" if max_frames and n_render < n_total else ""
        msg = (
            f"**Animation ready** ({fmt}) — {n_render} frames{cap_note} · payload `{payload[:40]}`\n\n"
            "Four panels: helical phase · OAM intensity · pyramidal pulse · PWM orbs with trails."
        )
        return (
            str(mp4_path) if mp4_path else None,
            str(gif_path),
            msg,
        )
    except Exception as exc:
        logger.exception("animate_typehead failed")
        return None, None, f"Animation error: {exc}\n\n{traceback.format_exc()}"


def build_app() -> gr.Blocks:
    on_hf = is_hf_space()
    slm_frames_info = (
        "Disabled on Hugging Face for speed — use local demo for PNG frame export"
        if on_hf
        else "Adds frames/ to zip; slower. Core zip always has manifest + phase_stack.npy"
    )

    with gr.Blocks(
        title="Orbital Braille — VQC Typehead",
        analytics_enabled=False,
        css=HFB_CSS,
        fill_width=True,
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
        gr.Markdown(SIMULATION_BANNER_MD)
        with gr.Accordion("New here? 60-second guide (Selectric typeball → OAM)", open=False):
            gr.Markdown(ONBOARDING_MD)
        with gr.Row():
            payload = gr.Textbox(label="Payload", value=DEFAULT_PAYLOAD)
            num_orbs = gr.Slider(2, 6, value=4, step=1, label="Number of orbs")
        with gr.Row():
            resolution = gr.Radio(
                choices=["Quick", "Full"],
                value="Quick",
                label="Resolution",
                info="Quick = low grid (fast); Full = publication quality"
                + (" — Full is slower on HF" if on_hf else ""),
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
            noise_level = gr.Slider(
                0.0,
                1.0,
                value=DEFAULT_NOISE_LEVEL,
                step=0.05,
                label="Channel noise",
                info="0 = clean link · 0.35 = default turbulence · 1 = harsh",
            )
        gr.Markdown("**Example presets** — one click loads settings and **runs** the demo:")
        with gr.Row():
            preset_buttons: dict[str, gr.Button] = {}
            for key, preset in EXAMPLE_PRESETS.items():
                preset_buttons[key] = gr.Button(preset["label"], variant="secondary", size="sm")
        with gr.Row():
            export_slm_frames = gr.Checkbox(
                label="Include SLM-ready phase frames (PNG)",
                value=False,
                interactive=not on_hf,
                info=slm_frames_info,
            )
        with gr.Accordion("How this maps to VQC claims", open=False):
            gr.Markdown(VQC_CLAIMS_MD)
        with gr.Accordion("Example walkthrough (recorded demo)", open=False):
            gr.HTML(
                f'<video src="{DEMO_SCREENCAST_URL}" controls playsinline '
                f'style="width:100%;max-width:100%;border-radius:8px;" '
                f'poster="https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png">'
                f"Your browser does not support video.</video>"
            )
            gr.Markdown(
                f"[Direct MP4 link]({DEMO_SCREENCAST_URL}) · "
                "same flow as **Run demo** → **Animate typehead**"
            )
        run_btn = gr.Button("Run demo", variant="primary", elem_classes=["vqc-full-width"])
        run_cache = gr.State(value=None)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                metrics = gr.Textbox(label="Metrics", lines=14)
            with gr.Column(scale=2):
                figure = gr.Image(
                    label="6-panel output",
                    type="filepath",
                    elem_classes=["vqc-figure-panel"],
                )
        figure_3d = gr.Plot(
            label="3D orb trajectories — drag to rotate · scroll to zoom",
            elem_classes=["vqc-plot3d-panel"],
        )
        animate_btn = gr.Button(
            "Animate typehead",
            variant="secondary",
            elem_classes=["vqc-full-width"],
        )
        animation_video = gr.Video(
            label="Typehead animation (MP4)",
            height=540,
            elem_classes=["vqc-animation-panel"],
        )
        animation_gif = gr.Image(
            label="Typehead animation (GIF download)",
            type="filepath",
            height=360,
            elem_classes=["vqc-animation-panel"],
        )
        animation_info = gr.Markdown(
            "*After **Run demo**, click **Animate typehead** for a full-width per-run animation "
            "(MP4 player + GIF — phase · intensity · pulse · orb trails).*"
        )
        with gr.Accordion("SLM package download", open=False):
            slm_info = gr.Markdown(SLM_PACKAGE_IDLE)
            slm_file = gr.File(
                label="slm_package.zip",
                interactive=False,
                file_count="single",
                type="filepath",
            )
        run_inputs = [
            payload,
            num_orbs,
            resolution,
            seed,
            gamma_1,
            noise_level,
            export_slm_frames,
        ]
        run_outputs = [metrics, figure, figure_3d, slm_info, slm_file, run_cache]

        run_btn.click(run_demo, run_inputs, run_outputs)
        animate_btn.click(
            animate_typehead,
            inputs=[run_cache],
            outputs=[animation_video, animation_gif, animation_info],
        )
        for key, btn in preset_buttons.items():
            btn.click(
                lambda k=key: load_example_preset(k),
                outputs=[payload, num_orbs, gamma_1, noise_level],
            ).then(run_demo, inputs=run_inputs, outputs=run_outputs)
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