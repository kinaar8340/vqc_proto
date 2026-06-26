"""Decode orbital Braille fields: OAM projection + spectral shard recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

from .altermagnetic import PWaveBMGL, apply_turbulence, repetition_qec
from .lg_modes import dominant_ell, project_oam_spectrum
from .quaternion_codec import Quaternion, decode_shard
from .stable_fonts import fisher_rao_distance


@dataclass
class DecodeResult:
    recovered_ells: list[int]
    shard_fidelity: float
    glyph_index: int
    glyph_fidelity: float
    quaternion: Quaternion
    recovered_bytes: np.ndarray
    oam_weights: dict[int, complex]


def _nearest_glyph(recovered_duties: np.ndarray, font: np.ndarray) -> tuple[int, float]:
    """Match recovered PWM duties to closest font glyph via Fisher-Rao distance."""
    best_idx, best_dist = 0, float("inf")
    for g in range(font.shape[0]):
        d = fisher_rao_distance(recovered_duties, font[g])
        if d < best_dist:
            best_dist, best_idx = d, g
    fidelity = 1.0 - best_dist / np.pi
    return best_idx, max(0.0, fidelity)


def decode_field(
    field_time: np.ndarray,
    reference_intensity: np.ndarray,
    font: np.ndarray,
    orbs_ells: list[int],
    bmgl: PWaveBMGL | None = None,
    rho: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    pulse_ref: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> DecodeResult:
    """
    Decode received field:
    1. BMGL denoise (if noisy)
    2. OAM mode projection
    3. ICA demix of intensity channels
    4. Spectral shard correlation
    5. Glyph + quaternion recovery
    """
    n_t, ny, nx = field_time.shape
    mid = n_t // 2
    field_mid = field_time[mid]

    if rho is None or phi is None:
        x = np.linspace(-2.5, 2.5, nx)
        y = np.linspace(-2.5, 2.5, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)

    ell_range = list(set(orbs_ells + [0, 1, -1, 2, -2]))
    oam_weights = project_oam_spectrum(field_mid, rho, phi, ell_range)
    recovered_ells = [dominant_ell(oam_weights)]

    intensity = np.abs(field_time) ** 2
    intensity = repetition_qec(intensity, reps=16, error_rate=bmgl.alpha_chemical if bmgl else 0.015)

    n_orbs = len(orbs_ells)
    flat = intensity.reshape(n_t, -1).T
    ica = FastICA(n_components=min(n_orbs, n_t), random_state=42, max_iter=2000, tol=1e-4)
    try:
        S = ica.fit_transform(flat)
        recovered_duties = np.clip(np.mean(np.abs(S), axis=0)[:n_orbs], 0, 1)
        recovered_duties = recovered_duties / (recovered_duties.sum() + 1e-12)
    except Exception:
        recovered_duties = np.mean(intensity, axis=(1, 2))
        recovered_duties = recovered_duties[:n_orbs]
        recovered_duties = recovered_duties / (recovered_duties.sum() + 1e-12)

    glyph_idx, glyph_fid = _nearest_glyph(recovered_duties, font)

    ref_flat = reference_intensity[mid].flatten()
    rec_flat = np.abs(field_mid).flatten()
    if np.std(ref_flat) < 1e-10:
        ref_flat = ref_flat + np.random.normal(0, 1e-10, ref_flat.shape)
    if np.std(rec_flat) < 1e-10:
        rec_flat = rec_flat + np.random.normal(0, 1e-10, rec_flat.shape)
    shard_fidelity = float(pearsonr(ref_flat, rec_flat)[0])

    q = Quaternion(
        w=float(np.real(oam_weights.get(1, 0.5))),
        x=float(np.real(oam_weights.get(0, 0.0))),
        y=float(np.imag(oam_weights.get(-1, 0.0))),
        z=float(np.abs(oam_weights.get(2, 0.0))),
    )
    q_arr = q.as_array()
    q_arr = q_arr / (np.linalg.norm(q_arr) + 1e-12)
    recovered_bytes = decode_shard(Quaternion(*q_arr))

    return DecodeResult(
        recovered_ells=recovered_ells,
        shard_fidelity=shard_fidelity,
        glyph_index=glyph_idx,
        glyph_fidelity=glyph_fid,
        quaternion=Quaternion(*q_arr),
        recovered_bytes=recovered_bytes,
        oam_weights=oam_weights,
    )