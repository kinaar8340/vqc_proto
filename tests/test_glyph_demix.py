"""Tests for geometry-aware glyph demix and complex-field glyph ranking."""

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
    PWaveBMGL,
    TypeheadConfig,
    build_stable_font,
    decode_field,
    noise_level_to_scale,
)
from orbital_braille.glyph_cache import (  # noqa: E402
    clear_glyph_template_cache,
    get_glyph_template_bank,
    glyph_field_coherence,
    rank_glyphs_by_orb_field,
    rank_glyphs_by_orb_intensity,
)
from orbital_braille.glyph_demix import recover_pwm_duties, nearest_glyph  # noqa: E402
from orbital_braille.quaternion_oam import triplet_centre_field  # noqa: E402
from orbital_braille.stable_fonts import fisher_rao_distance  # noqa: E402


@pytest.fixture
def typehead_8() -> OrbitalTypehead:
    cfg = TypeheadConfig(num_orbs=8, grid_size=32, num_times=16, bmgl=PWaveBMGL(gamma_1=1.5))
    return OrbitalTypehead(cfg, seed=42)


def test_recover_pwm_duties_better_than_uniform(typehead_8: OrbitalTypehead):
    encoded = typehead_8.encode(b"I")
    duties = recover_pwm_duties(
        encoded.intensity_time,
        8,
        typehead_8.X,
        typehead_8.Y,
        encoded.t,
        float(encoded.t[-1]),
        typehead_8.config.constants,
    )
    assert duties.shape == (8,)
    assert duties.sum() == pytest.approx(1.0, abs=1e-6)
    assert float(duties.max()) > float(duties.min())
    fr = fisher_rao_distance(duties, encoded.glyph_duties)
    uniform = np.full(8, 1.0 / 8.0)
    assert fr < fisher_rao_distance(uniform, encoded.glyph_duties)


def test_glyph_field_coherence_drops_with_noise(typehead_8: OrbitalTypehead):
    encoded = typehead_8.encode(b"I")
    clean = triplet_centre_field(encoded.field_time)
    noisy_low = typehead_8.propagate_with_turbulence(encoded, noise_level=0.2)
    noisy_high = typehead_8.propagate_with_turbulence(encoded, noise_level=0.7)
    coh_clean = glyph_field_coherence(triplet_centre_field(noisy_low), clean)
    coh_noisy = glyph_field_coherence(triplet_centre_field(noisy_high), clean)
    assert coh_clean > coh_noisy


def test_rank_glyphs_by_orb_field_includes_true_glyph(typehead_8: OrbitalTypehead):
    clear_glyph_template_cache()
    encoded = typehead_8.encode(b"I")
    noisy = typehead_8.propagate_with_turbulence(encoded, noise_level=0.35)
    field_mid = triplet_centre_field(noisy)
    mid = encoded.field_time.shape[0] // 2
    bank = get_glyph_template_bank(
        typehead_8.font,
        typehead_8.X,
        typehead_8.Y,
        float(encoded.t[mid]),
        float(encoded.t[-1]),
        8,
        typehead_8.config.constants,
    )
    true_idx, _ = nearest_glyph(encoded.glyph_duties, typehead_8.font)
    ranked_int = rank_glyphs_by_orb_intensity(np.abs(field_mid) ** 2, bank, k=32)
    ranked_blend = rank_glyphs_by_orb_field(field_mid, bank, k=32)
    assert true_idx in ranked_blend


def test_decode_uses_reference_glyph_duties(typehead_8: OrbitalTypehead):
    encoded = typehead_8.encode(b"I")
    noisy = typehead_8.propagate_with_turbulence(encoded, noise_level=0.5)
    true_idx, _ = nearest_glyph(encoded.glyph_duties, typehead_8.font)
    decoded = decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead_8.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=typehead_8.config.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        pulse_ref=encoded.pulse,
        t=encoded.t,
        noise_scale=noise_level_to_scale(0.5),
        constants=typehead_8.config.constants,
        reference_field=encoded.field_time,
        reference_quaternion=encoded.quaternion,
        reference_glyph_duties=encoded.glyph_duties,
    )
    assert decoded.glyph_index == true_idx
    assert decoded.glyph_field_coherence > 0.0
    assert decoded.glyph_fidelity > 0.5