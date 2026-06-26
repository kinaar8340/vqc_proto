"""Tests for analysis/proto_loader.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.proto_loader import discover_proto_outputs, latest_proto_demo, load_manifest_summary


def test_discover_finds_demo_png():
    bundle = discover_proto_outputs()
    assert bundle.demo_png is not None
    assert Path(bundle.demo_png).is_file()


def test_latest_proto_demo_returns_path():
    path = latest_proto_demo()
    assert path is not None
    assert path.endswith("orbital_braille_demo.png")


def test_load_manifest_summary_keys(tmp_path: Path):
    manifest = {
        "device": "generic_512",
        "num_orbs": 4,
        "payload": "test",
        "frames": 16,
        "wavelength_nm": 1550,
        "glyph_duties": [0.5, 0.5, 0.1, 0.1],
        "extra": "ignored",
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    summary = load_manifest_summary(str(p))
    assert summary["device"] == "generic_512"
    assert "extra" not in summary