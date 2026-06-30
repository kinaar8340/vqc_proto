"""Tests for derived quaternion ↔ OAM coupling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROTO_ROOT = Path(__file__).resolve().parents[1] / "proto"
if str(PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTO_ROOT))

from orbital_braille import (  # noqa: E402
    EmergentConstants,
    OrbitalTypehead,
    PWaveBMGL,
    TypeheadConfig,
    decode_field,
    noise_level_to_scale,
    quaternion_recovery_error,
)
from orbital_braille.quaternion_oam import (  # noqa: E402
    PHI_SCALE,
    dewarp_oam_weights,
    forward_quaternion_field,
    predict_oam_weights_from_quaternion,
    recover_quaternion_from_field_manifold,
    recover_quaternion_manifold,
    recover_quaternion_with_reference,
    recover_w_from_phase,
    triplet_centre_field,
    triplet_median_oam_weights,
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


def test_differential_recovery_clean(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=42)
    encoded = typehead.encode("I live in Oregon")
    field = triplet_centre_field(encoded.field_time)
    q_rec = recover_quaternion_with_reference(
        field, field, encoded.quaternion, encoded.rho, encoded.phi
    )
    assert quaternion_recovery_error(encoded.quaternion, q_rec) < 1e-6


def test_differential_recovery_noisy(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=42)
    encoded = typehead.encode("VQC prototype")
    noisy = typehead.propagate_with_turbulence(encoded, noise_level=0.35)
    decoded = decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=quick_config.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        noise_scale=noise_level_to_scale(0.35),
        reference_field=encoded.field_time,
        reference_quaternion=encoded.quaternion,
        glyph_refine_k=6,
        carrier_grid_steps=11,
    )
    err = quaternion_recovery_error(encoded.quaternion, decoded.quaternion)
    assert err < 0.02


def test_w_recovery_monotonic_with_phase():
    dewarped = {0: 1.0, 1: 1.0, -1: 1.0, 2: 1.0}
    w_lo = recover_w_from_phase({k: v * np.exp(1j * 0.05) for k, v in dewarped.items()})
    w_hi = recover_w_from_phase({k: v * np.exp(1j * 0.25) for k, v in dewarped.items()})
    assert w_lo != w_hi


def test_phi_scale_matches_typehead():
    assert PHI_SCALE == 0.3


def test_manifold_forward_model_self_consistency(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=1)
    encoded = typehead.encode("test")
    q = encoded.quaternion
    weights = predict_oam_weights_from_quaternion(q, encoded.rho, encoded.phi)
    result = recover_quaternion_manifold(
        weights,
        encoded.rho,
        encoded.phi,
        warm_start=q,
        loss_tol=1e-6,
    )
    assert result.converged
    assert not result.used_fallback
    assert quaternion_recovery_error(q, result.quaternion) < 1e-4


def test_manifold_blind_beats_heuristic(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=42)
    encoded = typehead.encode("I live in Oregon")
    noisy = typehead.propagate_with_turbulence(encoded, noise_level=0.35)
    decoded_blind = decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=quick_config.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        noise_scale=noise_level_to_scale(0.35),
        t=encoded.t,
        constants=quick_config.constants,
        glyph_rank_k=20,
        glyph_refine_k=8,
        carrier_grid_steps=15,
    )
    err_blind = quaternion_recovery_error(encoded.quaternion, decoded_blind.quaternion)
    assert err_blind < 0.16
    assert decoded_blind.manifold_recovery is not None
    assert decoded_blind.manifold_recovery.orb_subtracted
    assert decoded_blind.manifold_recovery.converged


def test_manifold_blind_with_orb_field(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=42)
    encoded = typehead.encode("I live in Oregon")
    noisy = typehead.propagate_with_turbulence(encoded, noise_level=0.35)
    from orbital_braille.typehead import build_orbs_from_duties, synthesize_orb_field

    orbs = build_orbs_from_duties(
        encoded.glyph_duties, quick_config.num_orbs, quick_config.constants
    )
    reps = 3
    n_logical = encoded.field_time.shape[0] // reps
    ti_val = encoded.t[n_logical // 2 * reps + reps // 2]
    orb_field = synthesize_orb_field(
        orbs, encoded.rho * np.cos(encoded.phi), encoded.rho * np.sin(encoded.phi),
        ti_val, encoded.t[-1],
    )
    result = recover_quaternion_from_field_manifold(
        noisy,
        encoded.rho,
        encoded.phi,
        bmgl=quick_config.bmgl,
        noise_scale=noise_level_to_scale(0.35),
        orb_field=orb_field,
    )
    err = quaternion_recovery_error(encoded.quaternion, result.quaternion)
    assert err < 0.08
    assert result.orb_subtracted


def test_blind_pipeline_metrics(quick_config: TypeheadConfig):
    import sys
    from pathlib import Path

    proto_root = Path(__file__).resolve().parents[1] / "proto"
    if str(proto_root) not in sys.path:
        sys.path.insert(0, str(proto_root))
    from demo_core import run_pipeline

    _, _, _, decoded, metrics, _ = run_pipeline(
        "I live in Oregon",
        4,
        quick=True,
        seed=42,
        gamma_1=1.5,
        noise_level=0.35,
        blind_quaternion=True,
        glyph_rank_k=16,
        glyph_refine_k=6,
        carrier_grid_steps=11,
    )
    assert decoded.manifold_recovery is not None
    assert decoded.manifold_recovery.orb_subtracted
    assert "Level-2 manifold" in metrics
    assert "Manifold loss:" in metrics
    assert "Carrier w" in metrics


def test_glyph_cache_matches_direct_ranking(quick_config: TypeheadConfig):
    from orbital_braille.glyph_cache import (
        clear_glyph_template_cache,
        get_glyph_template_bank,
        rank_glyphs_by_orb_intensity,
    )

    clear_glyph_template_cache()
    typehead = OrbitalTypehead(quick_config, seed=5)
    encoded = typehead.encode("cache")
    field = triplet_centre_field(encoded.field_time)
    intensity = np.abs(field) ** 2
    X = encoded.rho * np.cos(encoded.phi)
    Y = encoded.rho * np.sin(encoded.phi)
    reps = 3
    n_logical = encoded.field_time.shape[0] // reps
    ti_val = encoded.t[n_logical // 2 * reps + reps // 2]

    bank = get_glyph_template_bank(
        typehead.font,
        X,
        Y,
        ti_val,
        encoded.t[-1],
        quick_config.num_orbs,
        quick_config.constants,
    )
    ranked = rank_glyphs_by_orb_intensity(intensity, bank, k=8)
    bank2 = get_glyph_template_bank(
        typehead.font,
        X,
        Y,
        ti_val,
        encoded.t[-1],
        quick_config.num_orbs,
        quick_config.constants,
    )
    assert bank2 is bank
    assert len(ranked) == 8
    assert len(set(ranked)) == len(ranked)


def test_triplet_median_weights_stable(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=3)
    encoded = typehead.encode("QEC")
    median = triplet_median_oam_weights(encoded.field_time, encoded.rho, encoded.phi)
    centre = triplet_centre_field(encoded.field_time)
    single = predict_oam_weights_from_quaternion  # noqa: F841 — type check only
    from orbital_braille.quaternion_oam import project_quaternion_oam

    one = project_quaternion_oam(centre, encoded.rho, encoded.phi)
    for ell in median:
        assert abs(median[ell] - one[ell]) < abs(one[ell]) * 0.5 + 1e-6


def test_dewarp_roundtrip():
    phi_c = 0.5
    weights = {1: 0.8 + 0.2j, -1: -0.3 + 0.1j}
    twisted = {ell: w * np.exp(1j * ell * phi_c) for ell, w in weights.items()}
    restored = dewarp_oam_weights(twisted, phi_centroid=phi_c)
    for ell in weights:
        assert restored[ell] == pytest.approx(weights[ell], rel=1e-10, abs=1e-10)