"""Tests for orb interference metrics and manifold optimizer stats."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROTO_ROOT = Path(__file__).resolve().parents[1] / "proto"
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))

from orbital_braille import (  # noqa: E402
    OrbitalTypehead,
    TypeheadConfig,
    EmergentConstants,
    PWaveBMGL,
)
from orbital_braille.orb_interference import (  # noqa: E402
    effective_oam_modes,
    mean_pairwise_intensity_correlation,
)
from orbital_braille.quaternion_oam import (  # noqa: E402
    recover_quaternion_manifold,
    predict_oam_weights_from_quaternion,
)


@pytest.fixture
def quick_config() -> TypeheadConfig:
    return TypeheadConfig(
        num_orbs=4,
        grid_size=32,
        num_times=16,
        bmgl=PWaveBMGL(gamma_1=1.5),
        constants=EmergentConstants(),
    )


def test_effective_oam_modes_single_peak():
    weights = {0: 1.0 + 0j, 1: 0.0, -1: 0.0, 2: 0.0}
    assert effective_oam_modes(weights) == pytest.approx(1.0, rel=1e-6)


def test_pairwise_correlation_increases_with_overlap():
    base = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    overlapping = np.stack([base, base * 0.9 + 0.05])
    disjoint = np.zeros((2, 4, 4))
    disjoint[0, :, :2] = base[:, :2]
    disjoint[1, :, 2:] = base[:, 2:]
    overlap_corr = mean_pairwise_intensity_correlation(overlapping)
    disjoint_corr = mean_pairwise_intensity_correlation(disjoint)
    assert overlap_corr > 0.9
    assert disjoint_corr < overlap_corr


def test_manifold_result_exposes_optimizer_stats(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=1)
    encoded = typehead.encode("opt")
    q = encoded.quaternion
    weights = predict_oam_weights_from_quaternion(q, encoded.rho, encoded.phi)
    result = recover_quaternion_manifold(
        weights,
        encoded.rho,
        encoded.phi,
        warm_start=q,
        loss_tol=1e-6,
    )
    assert result.optimizer_nfev is not None
    assert result.optimizer_nfev > 0
    assert result.optimizer_nit is not None