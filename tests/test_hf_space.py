"""Smoke test for the Hugging Face Space app bundle."""

from __future__ import annotations

import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parents[1] / "space" / "orbital-braille"


def test_hf_space_app_imports():
    assert SPACE_DIR.is_dir(), "Run scripts/sync_hf_space.sh first"
    sys.path.insert(0, str(SPACE_DIR))
    import app  # noqa: WPS433

    assert app.demo is not None
    assert hasattr(app, "run_demo")