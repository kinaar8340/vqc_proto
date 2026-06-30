"""Tests for [[3,1,3]] bit-flip repetition QEC stub."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROTO_ROOT = Path(__file__).resolve().parents[1] / "proto"
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))

from orbital_braille import PWaveBMGL  # noqa: E402
from orbital_braille.qec_stub import (  # noqa: E402
    BitFlipRepetitionCode,
    bitflip_repetition_qec,
    measure_qec_threshold,
    physical_error_rate,
    simulate_code_memory,
)


def test_hamilton_syndrome_single_flip_q0():
    code = BitFlipRepetitionCode()
    encoded = code.encode(0)
    corrupted = encoded.copy()
    corrupted[0] ^= 1
    assert code.syndrome(corrupted) == (1, 0)
    assert code.decode_logical(corrupted) == 0


def test_hamilton_syndrome_single_flip_q1():
    code = BitFlipRepetitionCode()
    encoded = code.encode(1)
    corrupted = encoded.copy()
    corrupted[1] ^= 1
    assert code.syndrome(corrupted) == (1, 1)
    assert code.decode_logical(corrupted) == 1


def test_double_flip_uncorrectable():
    code = BitFlipRepetitionCode()
    corrupted = np.array([1, 0, 1], dtype=np.int8)
    assert code.decode_logical(corrupted) != 0


def test_simulate_code_memory_reduces_logical_errors():
    p = 0.05
    stats = simulate_code_memory(p, n_trials=20_000, seed=1)
    assert stats.physical_error_rate == pytest.approx(p, rel=0.15)
    assert stats.logical_error_rate < stats.physical_error_rate
    assert stats.logical_error_rate < p * p * 10


def test_logical_error_scales_with_gamma_1():
    rows = measure_qec_threshold(
        gamma_1_values=[1.0, 2.0],
        noise_scales=[1.0],
        n_trials=8_000,
        seed=0,
    )
    low = next(r for r in rows if r["gamma_1"] == 1.0)
    high = next(r for r in rows if r["gamma_1"] == 2.0)
    assert high["p_physical"] < low["p_physical"]
    assert high["logical_error_rate"] <= low["logical_error_rate"]


def test_bitflip_qec_with_reference_improves_logical_rate():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.2, 0.8, size=(12, 8, 8))
    noisy = ref + rng.normal(0, 0.08, ref.shape)
    _, stats = bitflip_repetition_qec(noisy, reference=ref, bmgl=PWaveBMGL(gamma_1=1.5))
    assert stats.n_groups > 0
    assert stats.logical_error_rate <= stats.logical_error_rate_pre + 1e-12


def test_physical_error_rate_bmgl_mapping():
    bmgl = PWaveBMGL(gamma_1=1.5)
    p = physical_error_rate(bmgl, noise_scale=1.0)
    expected = bmgl.alpha_chemical / bmgl.effective_inhibition
    assert p == pytest.approx(expected, rel=1e-6)