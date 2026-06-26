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
)
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
    )
    assert decoded.shard_fidelity > 0.5
    assert decoded.glyph_fidelity > 0.0
    assert 0 <= decoded.glyph_index < typehead.font.shape[0]


def test_encode_produces_expected_tensor_shapes(quick_config: TypeheadConfig):
    typehead = OrbitalTypehead(quick_config, seed=7)
    encoded = typehead.encode("hi")
    assert encoded.field_time.shape == (
        quick_config.num_times,
        quick_config.grid_size,
        quick_config.grid_size,
    )
    assert encoded.spectral_shards.size > 0
    assert len(encoded.orbs) == quick_config.num_orbs