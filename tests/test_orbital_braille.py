"""Roundtrip and invariant tests for proto/orbital_braille modules."""

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
    build_stable_font,
    decode_field,
    decode_shard,
    encode_shard,
    font_separation,
    noise_level_to_scale,
)
from orbital_braille.encode_redundancy import effective_num_times, triplet_codeword_check  # noqa: E402
from orbital_braille.quaternion_codec import Quaternion  # noqa: E402
from orbital_braille.stable_fonts import fisher_rao_distance, glyph_for_byte  # noqa: E402


@pytest.fixture
def quick_config() -> TypeheadConfig:
    return TypeheadConfig(
        num_orbs=4,
        grid_size=32,
        num_times=16,
        bmgl=PWaveBMGL(gamma_1=1.5),
        constants=EmergentConstants(),
    )


def test_quaternion_encode_decode_roundtrip():
    payload = b"VQC!"
    q = encode_shard(payload)
    recovered = decode_shard(q, n_bytes=len(payload))
    assert recovered.shape == (len(payload),)
    assert np.all(recovered >= 0) and np.all(recovered <= 255)


def test_quaternion_multiply_identity():
    q = Quaternion(0.6, 0.2, 0.5, 0.3)
    q = Quaternion(*(q.as_array() / q.norm()))
    identity = Quaternion(1.0, 0.0, 0.0, 0.0)
    product = q.multiply(q.conjugate())
    arr = product.as_array()
    assert arr[0] == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(arr[1:] - identity.as_array()[1:]) < 1e-6


def test_stable_font_separation_increases_with_orbs():
    sep2 = font_separation(build_stable_font(2, num_glyphs=16))
    sep4 = font_separation(build_stable_font(4, num_glyphs=16))
    assert sep4 > sep2 > 0.0


def test_fisher_rao_self_distance_near_zero():
    glyph = build_stable_font(4, num_glyphs=8)[0]
    assert fisher_rao_distance(glyph, glyph) < 0.01


def test_glyph_for_byte_within_font():
    font = build_stable_font(4, num_glyphs=16)
    g = glyph_for_byte(42, font)
    assert g.shape == (4,)
    assert np.all((g >= 0.1) & (g <= 0.95))


def test_encode_decode_field_roundtrip(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=42)
    payload = "I live in Oregon"
    encoded = typehead.encode(payload)
    noisy = typehead.propagate_with_turbulence(encoded)
    decoded = decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=quick_config.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        pulse_ref=encoded.pulse,
        t=encoded.t,
        reference_field=encoded.field_time,
        reference_quaternion=encoded.quaternion,
    )
    assert decoded.shard_fidelity > 0.5
    assert decoded.glyph_fidelity > 0.0
    assert 0 <= decoded.glyph_index < typehead.font.shape[0]


def test_encode_produces_expected_tensor_shapes(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=7)
    encoded = typehead.encode("hi")
    n_t = effective_num_times(quick_config.num_times, quick_config.qec_reps)
    assert encoded.field_time.shape == (
        n_t,
        quick_config.grid_size,
        quick_config.grid_size,
    )
    assert encoded.spectral_shards.size > 0
    assert len(encoded.orbs) == quick_config.num_orbs


def test_transmit_triplet_codewords(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=3)
    encoded = typehead.encode("QEC")
    assert encoded.qec_reps == 3
    assert triplet_codeword_check(encoded.intensity_time)


def test_qec_logical_error_improves_with_transmit_redundancy(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=42)
    payload = "I live in Oregon"
    encoded = typehead.encode(payload)
    noisy = typehead.propagate_with_turbulence(encoded, noise_level=0.35)
    decoded = decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=quick_config.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        noise_scale=1.0,
    )
    assert decoded.qec_stats is not None
    qs = decoded.qec_stats
    assert qs.logical_error_rate <= qs.logical_error_rate_pre + 1e-12
    assert qs.logical_error_rate < 0.05

    # Harsher turbulence: protected triplets still beat raw physical flip rate.
    encoded_harsh = typehead.encode(payload)
    noisy_harsh = typehead.propagate_with_turbulence(encoded_harsh, noise_level=1.0)
    decoded_harsh = decode_field(
        noisy_harsh,
        reference_intensity=encoded_harsh.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded_harsh.orbs],
        bmgl=quick_config.bmgl,
        rho=encoded_harsh.rho,
        phi=encoded_harsh.phi,
        noise_scale=noise_level_to_scale(1.0),
    )
    qsh = decoded_harsh.qec_stats
    assert qsh is not None
    assert qsh.logical_error_rate <= qsh.logical_error_rate_pre + 1e-12
    if qsh.physical_error_rate > 0.001:
        assert qsh.logical_error_rate < qsh.physical_error_rate