"""Tests for shared Gradio / HF demo_core helpers."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

PROTO_ROOT = Path(__file__).resolve().parents[1] / "proto"
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))

from demo_core import (  # noqa: E402
    EXAMPLE_PRESETS,
    PATENT_FIGURE1_PAYLOAD,
    export_slm_bundle,
    get_animation_max_frames,
    get_build_label,
    load_example_preset,
    plot_orb_trajectory_3d,
    render_typehead_animation,
    run_pipeline,
)


def test_patent_figure1_payload():
    assert PATENT_FIGURE1_PAYLOAD == "I live in Oregon"


def test_run_pipeline_respects_gamma_1():
    _, _, _, _, metrics, _ = run_pipeline("Hi", 2, quick=True, seed=0, gamma_1=1.7)
    assert "γ₁ = 1.70" in metrics


def test_export_slm_bundle_zip_contents():
    _, encoded, _, _, _, font_sep = run_pipeline(PATENT_FIGURE1_PAYLOAD, 4, quick=True, seed=42)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "slm"
        zip_path, summary = export_slm_bundle(
            encoded,
            payload=PATENT_FIGURE1_PAYLOAD,
            num_orbs=4,
            font_sep=font_sep,
            quick=True,
            include_frames=False,
            out_dir=out_dir,
        )
        assert zip_path.is_file()
        assert "manifest.json" in summary
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "phase_stack.npy" in names
            assert "README.txt" in names
            readme = zf.read("README.txt").decode()
            assert not any(n.startswith("frames/") for n in names)
        assert "Holoeye PLUTO-2" in readme
        assert "phase_stack.npy" in readme


def test_render_typehead_animation(tmp_path):
    _, encoded, noisy, _, _, _ = run_pipeline("Hi", 2, quick=True, seed=1)
    out = tmp_path / "anim.gif"
    render_typehead_animation(encoded, noisy, "Hi", out, max_frames=8)
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_get_build_label():
    label = get_build_label()
    assert label
    assert "commit" in label.lower() or "development" in label.lower()


def test_example_presets_load():
    assert len(EXAMPLE_PRESETS) >= 4
    payload, orbs, gamma = load_example_preset("hello")
    assert payload == "Hello OAM"
    assert orbs == 2
    assert gamma == 1.5


def test_plot_orb_trajectory_3d(tmp_path):
    _, encoded, _, _, _, _ = run_pipeline("Hi", 2, quick=True, seed=0)
    path = plot_orb_trajectory_3d(encoded, tmp_path, "Hi")
    assert path.is_file()
    assert path.stat().st_size > 500


def test_get_animation_max_frames_local_uncapped(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    assert get_animation_max_frames(quick=True) is None


def test_get_animation_max_frames_hf_capped(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "user/space")
    assert get_animation_max_frames(quick=True) == 12
    assert get_animation_max_frames(quick=False) == 20