"""Tests for transmit-side [[3,1,3]] repetition encoding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROTO_ROOT = Path(__file__).resolve().parents[1] / "proto"
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))

from orbital_braille.encode_redundancy import (  # noqa: E402
    effective_num_times,
    repeat_triplets_1d,
    repeat_triplets_along_time,
    triplet_codeword_check,
)


def test_effective_num_times_padding():
    assert effective_num_times(16, 3) == 18
    assert effective_num_times(64, 3) == 66
    assert effective_num_times(18, 3) == 18


def test_repeat_triplets_along_time():
    raw = np.arange(18 * 4).reshape(18, 2, 2).astype(float)
    field, intensity = repeat_triplets_along_time(raw, raw + 0.5)
    assert field.shape[0] == 18
    assert triplet_codeword_check(intensity)


def test_repeat_triplets_1d():
    pulse = np.sin(np.linspace(0, 1, 16))
    out = repeat_triplets_1d(pulse)
    assert out.size == 18
    for i in range(0, 18, 3):
        assert out[i] == out[i + 1] == out[i + 2]