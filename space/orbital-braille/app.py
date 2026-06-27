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
    plot_orb_trajectory_3d_static,
    plot_results,
    render_typehead_animation_bundle,
    run_pipeline,
)

DEMO_SCREENCAST_BASE = "https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/docs"
DEMO_SCREENCAST_URLS = (
    f"{DEMO_SCREENCAST_BASE}/typehead_screencast_1.mp4",
    f"{DEMO_SCREENCAST_BASE}/typehead_screencast_2.mp4",
)


def _screencast_dual_html() -> str:
    """Side-by-side recorded demo clips on the Animations page."""
    clips = "".join(
        f'<video class="vqc-screencast-video" src="{url}" controls playsinline '
        f'poster="{HFB_RAW_URL}">Your browser does not support video.</video>'
        for url in DEMO_SCREENCAST_URLS
    )
    return f'<div class="vqc-screencast-wrap">{clips}</div>'

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

WEBGL_3D_NOTE = (
    "Interactive rotate/zoom needs **WebGL**. Firefox enables it by default; "
    "**Brave** may block it via Shields (try lowering Shields for this Space, or "
    "enable WebGL in `brave://settings` → System). "
    "The **static 3D view** above always works without WebGL."
)

SLM_PACKAGE_IDLE = (
    "**Package files** (generated after **Run demo**):\n"
    "- `manifest.json` — orb geometry, PWM duties, quaternion, timing\n"
    "- `phase_stack.npy` — phase sequence array `[frames, H, W]`\n"
    "- `preview_montage.png` — visual sanity check\n"
    "- `LUT_calibration.txt` — gray→phase mapping notes\n"
    "- `README.txt` — driver quick-start (Holoeye, Meadowlark, Thorlabs)\n"
    "- `frames/` — optional PNG sequence (enable checkbox above)"
)

_VQC_ACCENT = "#ea580c"  # matches slider / primary button orange
_VQC_HF_RUNNING = "#1ed760"  # Hugging Face "Running" status green
_VQC_FIELD_FILL = "rgba(10, 8, 24, 0.50)"
_VQC_TAB_GREEN_BG = "#14532d"
_VQC_TAB_GREEN_BG_HOVER = "#166534"
_VQC_TAB_GREEN_BORDER = "#1ed760"
_VQC_TAB_GREEN_TEXT = "#86efac"
_VQC_TAB_GREEN_TEXT_HOVER = "#bbf7d0"
_VQC_TAB_ORANGE_BG = "#7c2d12"
_VQC_TAB_ORANGE_BORDER = "#ea580c"
_VQC_TAB_ORANGE_TEXT = "#fdba74"

ANIMATIONS_INTRO_MD = (
    "Recorded end-to-end flow: pick a preset or **Run demo**, then **Animate typehead** — "
    "helical phase, OAM intensity, pyramidal pulse, and PWM orbs (payload: "
    f'`"{DEFAULT_PAYLOAD}"`).'
)


def _external_tab_html(label: str, url: str, tab_id: str) -> str:
    """External Source bookmark — opens in a new tab."""
    return (
        f'<a href="{url}" class="vqc-source-tab" data-tab="{tab_id}" '
        f'target="_blank" rel="noopener noreferrer">{label}</a>'
    )


def _source_tab_btn_update(*, active: bool) -> gr.Update:
    """Animations tab — orange only when that page is open; otherwise green."""
    if active:
        return gr.update(interactive=False, elem_classes=["vqc-source-tab", "active"])
    return gr.update(interactive=True, elem_classes=["vqc-source-tab"], variant="secondary")


def _home_tab_update(*, on_demo_page: bool) -> gr.Update:
    """Live demo tab: orange on demo page, green link back from Animations."""
    if on_demo_page:
        return gr.update(interactive=False, elem_classes=["vqc-source-tab", "active"], variant="secondary")
    return gr.update(interactive=True, elem_classes=["vqc-source-tab"], variant="secondary")


def _nav_to_page(page: str) -> tuple:
    """Switch between demo and animations screens; refresh Source tab highlights."""
    on_demo = page == "demo"
    return (
        gr.update(visible=on_demo),
        gr.update(visible=not on_demo),
        _home_tab_update(on_demo_page=on_demo),
        _source_tab_btn_update(active=not on_demo),
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
        page,
    )


def _toggle_newhere(is_open: bool) -> tuple:
    """Expand/collapse the beginner guide panel below the Links bar."""
    show = not is_open
    return (
        gr.update(visible=show),
        _source_tab_btn_update(active=show),
        show,
    )


def _build_vqc_theme() -> gr.themes.Base:
    """Dark transparent theme — works in light and dark OS modes (HF-safe)."""
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.orange,
            secondary_hue=gr.themes.colors.zinc,
            neutral_hue=gr.themes.colors.zinc,
        )
        .set(
            body_background_fill="transparent",
            body_background_fill_dark="transparent",
            background_fill_primary="transparent",
            background_fill_primary_dark="transparent",
            background_fill_secondary="transparent",
            background_fill_secondary_dark="transparent",
            block_background_fill=_VQC_FIELD_FILL,
            block_background_fill_dark=_VQC_FIELD_FILL,
            panel_background_fill=_VQC_FIELD_FILL,
            panel_background_fill_dark=_VQC_FIELD_FILL,
            input_background_fill=_VQC_FIELD_FILL,
            input_background_fill_dark=_VQC_FIELD_FILL,
            body_text_color="#e8e0f8",
            body_text_color_dark="#e8e0f8",
            block_label_text_color="#c9b8ff",
            block_label_text_color_dark="#c9b8ff",
            block_title_text_color="#f0e6ff",
            block_title_text_color_dark="#f0e6ff",
            border_color_primary="rgba(255, 255, 255, 0.12)",
            border_color_primary_dark="rgba(255, 255, 255, 0.12)",
            button_primary_background_fill="#ea580c",
            button_primary_background_fill_dark="#ea580c",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(28, 22, 48, 0.92)",
            button_secondary_background_fill_dark="rgba(28, 22, 48, 0.92)",
            button_secondary_text_color="#e8e0f8",
            button_secondary_text_color_dark="#e8e0f8",
            checkbox_label_background_fill="transparent",
            checkbox_label_background_fill_dark="transparent",
            checkbox_label_background_fill_hover="transparent",
            checkbox_label_background_fill_hover_dark="transparent",
            slider_color=_VQC_ACCENT,
            slider_color_dark=_VQC_ACCENT,
            link_text_color=_VQC_ACCENT,
            link_text_color_dark=_VQC_ACCENT,
            link_text_color_hover="#f97316",
            link_text_color_hover_dark="#f97316",
            link_text_color_active=_VQC_ACCENT,
            link_text_color_active_dark=_VQC_ACCENT,
            link_text_color_visited=_VQC_ACCENT,
            link_text_color_visited_dark=_VQC_ACCENT,
        )
    )


# Wallpaper: #vqc-wallpaper (body child) + body::before fallback — cover, fixed to viewport.
WALLPAPER_HEAD = f"""
<style id="vqc-wallpaper-style">
#vqc-wallpaper {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -9999 !important;
    pointer-events: none !important;
    background-color: #0a0818 !important;
    background-image: url('{HFB_RAW_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
}}
</style>
<script>
(function() {{
    function mountWallpaper() {{
        if (document.getElementById('vqc-wallpaper')) return;
        var wp = document.createElement('div');
        wp.id = 'vqc-wallpaper';
        wp.setAttribute('aria-hidden', 'true');
        document.body.insertBefore(wp, document.body.firstChild);
    }}
    if (document.body) mountWallpaper();
    document.addEventListener('DOMContentLoaded', mountWallpaper);
    window.addEventListener('load', mountWallpaper);
}})();
</script>
"""

HFB_CSS = f"""
:root, :root .dark {{
    --body-background-fill: transparent !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: {_VQC_FIELD_FILL} !important;
    --panel-background-fill: {_VQC_FIELD_FILL} !important;
    --input-background-fill: {_VQC_FIELD_FILL} !important;
    --body-text-color: #e8e0f8 !important;
    --block-label-text-color: #c9b8ff !important;
    --block-title-text-color: #f0e6ff !important;
    --border-color-primary: rgba(255, 255, 255, 0.12) !important;
    --link-text-color: {_VQC_ACCENT} !important;
    --link-text-color-hover: #f97316 !important;
    --link-text-color-active: {_VQC_ACCENT} !important;
    --link-text-color-visited: {_VQC_ACCENT} !important;
    color-scheme: dark;
}}
html {{
    background-color: #0a0818 !important;
    min-height: 100% !important;
}}
body {{
    background: transparent !important;
    background-color: transparent !important;
    color: #e8e0f8 !important;
    min-height: 100vh !important;
    width: 100% !important;
    overflow-x: hidden !important;
    position: relative !important;
}}
body::before {{
    content: "" !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -9998 !important;
    pointer-events: none !important;
    background-color: #0a0818 !important;
    background-image: url('{HFB_RAW_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
}}
#root, .app {{
    background: transparent !important;
    background-color: transparent !important;
    min-height: 0 !important;
    height: auto !important;
    width: 100% !important;
}}
.gradio-container {{
    position: relative !important;
    width: 100% !important;
    max-width: 100% !important;
    min-height: 0 !important;
    height: auto !important;
    background: transparent !important;
    background-color: transparent !important;
}}
footer {{
    background: transparent !important;
}}
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .form,
.gradio-container .column,
.gradio-container .row,
.gradio-container .gr-group,
.gradio-container label.wrap,
.gradio-container .label-wrap {{
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}
.gradio-container .block {{
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    background-color: {_VQC_FIELD_FILL} !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
}}
.gradio-container .markdown,
.gradio-container .prose,
.gradio-container .markdown p,
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown li {{
    color: #e8e0f8 !important;
}}
.gradio-container .vqc-source-tabs-row {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.45rem 0.65rem !important;
    margin: 0.35rem 0 0.1rem 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
.gradio-container .vqc-source-nav-row {{
    margin: 0 0 0.1rem 0 !important;
}}
.gradio-container .vqc-newhere-panel {{
    margin: 0 0 0.35rem 0 !important;
    padding: 0.65rem 0.85rem !important;
}}
.gradio-container .vqc-newhere-panel .markdown h3 {{
    margin: 0 0 0.35rem 0 !important;
    font-size: 1rem !important;
    color: #f0e6ff !important;
}}
.gradio-container .vqc-source-tabs-row > .block,
.gradio-container .vqc-source-tabs-row > .form,
.gradio-container .vqc-source-tabs-row .block,
.gradio-container .vqc-source-tabs-row .form {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
}}
.gradio-container .vqc-source-tabs-row .html-container {{
    padding: 0 !important;
    margin: 0 !important;
}}
.gradio-container .vqc-source-label {{
    color: #e8e0f8 !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin-right: 0.15rem !important;
    line-height: 1.2 !important;
}}
.gradio-container .vqc-source-tab,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.4rem !important;
    padding: 0.3rem 0.85rem !important;
    border-radius: 999px !important;
    border: 1px solid {_VQC_TAB_GREEN_BORDER} !important;
    background: {_VQC_TAB_GREEN_BG} !important;
    background-color: {_VQC_TAB_GREEN_BG} !important;
    color: {_VQC_TAB_GREEN_TEXT} !important;
    -webkit-text-fill-color: {_VQC_TAB_GREEN_TEXT} !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    line-height: 1.2 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    min-height: unset !important;
    height: auto !important;
    width: auto !important;
    margin: 0 !important;
    opacity: 0.8 !important;
    transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease, opacity 0.15s ease;
}}
.gradio-container a.vqc-source-tab:hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab:not(.active):hover {{
    color: {_VQC_TAB_GREEN_TEXT_HOVER} !important;
    -webkit-text-fill-color: {_VQC_TAB_GREEN_TEXT_HOVER} !important;
    border-color: {_VQC_TAB_GREEN_BORDER} !important;
    background: {_VQC_TAB_GREEN_BG_HOVER} !important;
    background-color: {_VQC_TAB_GREEN_BG_HOVER} !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab {{
    cursor: pointer !important;
    font-family: inherit !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab:disabled:not(.active),
.gradio-container .vqc-source-tabs-row button.vqc-source-tab[disabled]:not(.active),
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.secondary:disabled:not(.active) {{
    opacity: 0.8 !important;
    cursor: default !important;
    color: {_VQC_TAB_GREEN_TEXT} !important;
    -webkit-text-fill-color: {_VQC_TAB_GREEN_TEXT} !important;
    border-color: {_VQC_TAB_GREEN_BORDER} !important;
    background: {_VQC_TAB_GREEN_BG} !important;
    background-color: {_VQC_TAB_GREEN_BG} !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active:disabled,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active[disabled] {{
    opacity: 0.8 !important;
    cursor: default !important;
    color: {_VQC_TAB_ORANGE_TEXT} !important;
    -webkit-text-fill-color: {_VQC_TAB_ORANGE_TEXT} !important;
    border-color: {_VQC_TAB_ORANGE_BORDER} !important;
    background: {_VQC_TAB_ORANGE_BG} !important;
    background-color: {_VQC_TAB_ORANGE_BG} !important;
}}
.gradio-container .vqc-source-tab.active,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active,
.gradio-container .vqc-source-tab.active:hover,
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active:hover {{
    color: {_VQC_TAB_ORANGE_TEXT} !important;
    -webkit-text-fill-color: {_VQC_TAB_ORANGE_TEXT} !important;
    border-color: {_VQC_TAB_ORANGE_BORDER} !important;
    background: {_VQC_TAB_ORANGE_BG} !important;
    background-color: {_VQC_TAB_ORANGE_BG} !important;
    cursor: default !important;
    opacity: 0.8 !important;
}}
.gradio-container .vqc-source-tabs-row button.vqc-source-tab.active::before {{
    content: "" !important;
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: {_VQC_TAB_ORANGE_TEXT} !important;
    flex-shrink: 0 !important;
    box-shadow: 0 0 6px rgba(253, 186, 116, 0.65) !important;
}}
.gradio-container a:hover,
.gradio-container .markdown a:hover,
.gradio-container .prose a:hover {{
    color: #f97316 !important;
    -webkit-text-fill-color: #f97316 !important;
}}
.gradio-container .vqc-build-label {{
    color: #a89ec8 !important;
    font-size: 0.9rem;
    margin: 0 0 0.5rem 0;
}}
.gradio-container .vqc-animations-page .markdown h2 {{
    font-size: 1.35rem !important;
    margin: 0.15rem 0 0.35rem 0 !important;
}}
.gradio-container .vqc-animations-page .markdown p {{
    font-size: 0.92rem !important;
    margin: 0.15rem 0 0.35rem 0 !important;
    line-height: 1.45 !important;
}}
.gradio-container .vqc-screencast-wrap {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    align-items: stretch !important;
    gap: 0.65rem !important;
    width: 100% !important;
    max-width: min(1040px, 92vw) !important;
    margin: 0.2rem auto 0.45rem !important;
}}
.gradio-container .vqc-screencast-video {{
    flex: 1 1 280px !important;
    min-width: 0 !important;
    width: calc(50% - 0.35rem) !important;
    height: auto !important;
    max-height: min(42vh, 400px) !important;
    object-fit: contain !important;
    border-radius: 8px !important;
    display: block !important;
    background: rgba(10, 8, 24, 0.35) !important;
}}
.gradio-container .markdown blockquote {{
    border-left-color: rgba(255, 180, 80, 0.5) !important;
    background: transparent !important;
}}
.gradio-container .accordion,
.gradio-container details,
.gradio-container summary {{
    background-color: {_VQC_FIELD_FILL} !important;
    color: #e8e0f8 !important;
    border-radius: 10px;
}}
.gradio-container .image-container img,
.gradio-container .gr-image img,
.gradio-container video,
.gradio-container .plot-container {{
    background-color: transparent !important;
    opacity: 1 !important;
}}
.gradio-container button,
.gradio-container .gr-button {{
    opacity: 1 !important;
}}
.gradio-container input[type="range"] {{
    opacity: 1 !important;
}}
.gradio-container .vqc-full-width {{
    width: 100% !important;
}}
.gradio-container .main,
.gradio-container .wrap {{
    width: 100% !important;
    max-width: 100% !important;
    min-height: 0 !important;
    height: auto !important;
}}
.gradio-container .contain {{
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    min-height: 0 !important;
    height: auto !important;
}}
.gradio-container .vqc-animation-panel,
.gradio-container .vqc-figure-panel,
.gradio-container .vqc-plot3d-panel {{
    width: 100% !important;
    max-width: 100% !important;
    min-height: 0 !important;
}}
.gradio-container .vqc-animation-panel video,
.gradio-container .vqc-animation-panel .image-container,
.gradio-container .vqc-animation-panel img,
.gradio-container .vqc-figure-panel .image-container,
.gradio-container .vqc-figure-panel img {{
    width: 100% !important;
    max-width: 100% !important;
    object-fit: contain;
}}
.gradio-container .vqc-plot3d-panel .plot-container {{
    width: 100% !important;
    min-height: 360px;
}}
.gradio-container .gr-video .empty,
.gradio-container .gr-image .empty,
.gradio-container .gr-image .icon-wrap {{
    min-height: 80px !important;
    background: transparent !important;
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
) -> tuple[str, str | None, str | None, object | None, str, str | None, tuple | None]:
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

        progress(0.6, desc="Rendering static 3D orb trajectories…")
        fig_3d_static = str(plot_orb_trajectory_3d_static(encoded, out_dir, payload))

        progress(0.7, desc="Building interactive 3D (WebGL)…")
        fig_3d_plotly = build_orb_trajectory_3d_plotly(encoded, payload)

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
        return metrics, fig_path, fig_3d_static, fig_3d_plotly, slm_info, str(zip_path), run_cache
    except Exception as exc:
        logger.exception("run_demo failed for payload=%r", payload)
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return err, None, None, None, SLM_PACKAGE_IDLE, None, None


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
        theme=_build_vqc_theme(),
        head=WALLPAPER_HEAD,
        css=HFB_CSS,
        fill_width=True,
    ) as demo:
        gr.Markdown(
            "# Orbital Braille — VQC Typehead Prototype\n"
            "Multi-orb PWM-gated sources → pyramidal spectral shards on an OAM carrier. "
            "Use **Quick** resolution for sub-second runs."
        )
        current_page = gr.State("demo")
        newhere_open = gr.State(False)
        with gr.Row(elem_classes=["vqc-source-tabs-row"]):
            gr.HTML('<span class="vqc-source-label">Source:</span>')
            tab_demo_btn = gr.Button(
                "Live Demo",
                elem_classes=["vqc-source-tab", "active"],
                interactive=False,
                scale=0,
                variant="secondary",
            )
            tab_anim_btn = gr.Button(
                "Animations",
                elem_classes=["vqc-source-tab"],
                scale=0,
                variant="secondary",
            )
        with gr.Row(elem_classes=["vqc-source-tabs-row", "vqc-source-nav-row"]):
            gr.HTML('<span class="vqc-source-label">Links:</span>')
            gr.HTML(_external_tab_html("GitHub", GITHUB_URL, "github"))
            gr.HTML(
                _external_tab_html(
                    "SLM Quickstart",
                    f"{GITHUB_URL}/blob/main/proto/SLM_QUICKSTART.md",
                    "slm",
                )
            )
            tab_newhere_btn = gr.Button(
                "New here?",
                elem_classes=["vqc-source-tab"],
                scale=0,
                variant="secondary",
            )
        with gr.Column(visible=False, elem_classes=["vqc-newhere-panel"]) as panel_newhere:
            gr.Markdown("### New here? 60-second guide (Selectric typeball → OAM)")
            gr.Markdown(ONBOARDING_MD)
        gr.HTML(f'<p class="vqc-build-label"><em>{get_build_label()}</em></p>')

        with gr.Column(visible=True) as page_demo:
            gr.Markdown(SIMULATION_BANNER_MD)
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
            figure_3d_static = gr.Image(
                label="3D orb trajectories (static — all browsers)",
                type="filepath",
                elem_classes=["vqc-figure-panel"],
            )
            with gr.Accordion("Interactive 3D — drag to rotate (WebGL)", open=False):
                gr.Markdown(WEBGL_3D_NOTE)
                figure_3d_plotly = gr.Plot(
                    label="Plotly 3D",
                    elem_classes=["vqc-plot3d-panel"],
                )
            animate_btn = gr.Button(
                "Animate typehead",
                variant="secondary",
                elem_classes=["vqc-full-width"],
            )
            animation_video = gr.Video(
                label="Typehead animation (MP4)",
                elem_classes=["vqc-animation-panel"],
            )
            animation_gif = gr.Image(
                label="Typehead animation (GIF download)",
                type="filepath",
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
            run_outputs = [
                metrics,
                figure,
                figure_3d_static,
                figure_3d_plotly,
                slm_info,
                slm_file,
                run_cache,
            ]

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

        with gr.Column(visible=False, elem_classes=["vqc-animations-page"]) as page_animations:
            gr.Markdown("## Animations")
            gr.Markdown(ANIMATIONS_INTRO_MD)
            gr.HTML(_screencast_dual_html())
            gr.Markdown(
                f"[Screencast 1]({DEMO_SCREENCAST_URLS[0]}) · "
                f"[Screencast 2]({DEMO_SCREENCAST_URLS[1]}) · "
                "same flow as **Run demo** → **Animate typehead** on the **Live Demo** tab."
            )

        nav_outputs = [
            page_demo,
            page_animations,
            tab_demo_btn,
            tab_anim_btn,
            panel_newhere,
            tab_newhere_btn,
            newhere_open,
            current_page,
        ]
        tab_demo_btn.click(lambda: _nav_to_page("demo"), outputs=nav_outputs)
        tab_anim_btn.click(lambda: _nav_to_page("animations"), outputs=nav_outputs)
        tab_newhere_btn.click(_toggle_newhere, inputs=[newhere_open], outputs=nav_outputs[4:7])

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