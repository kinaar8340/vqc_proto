"""Decode orbital Braille fields: OAM projection + spectral shard recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

from .altermagnetic import PWaveBMGL, apply_turbulence, repetition_qec
from .qec_stub import QECStats, bitflip_repetition_qec
from .lg_modes import dominant_ell, project_oam_spectrum
from .quaternion_codec import Quaternion, decode_shard
from .encode_redundancy import QEC_REPS, effective_num_times
from .quaternion_oam import (
    ManifoldRecoveryResult,
    OAM_QUAT_ELLS,
    _full_forward_oam_loss,
    oam_weights_to_quaternion,
    project_quaternion_oam,
    recover_quaternion_from_field_manifold,
    recover_quaternion_with_reference,
    triplet_centre_field,
)
from .glyph_cache import get_glyph_template_bank, rank_glyphs_by_orb_intensity
from .stable_fonts import EmergentConstants, fisher_rao_distance
from .typehead import build_orbs_from_duties, synthesize_orb_field


@dataclass
class DecodeResult:
    recovered_ells: list[int]
    shard_fidelity: float
    glyph_index: int
    glyph_fidelity: float
    quaternion: Quaternion
    recovered_bytes: np.ndarray
    oam_weights: dict[int, complex]
    qec_stats: QECStats | None = None
    manifold_recovery: ManifoldRecoveryResult | None = None


def _nearest_glyph(recovered_duties: np.ndarray, font: np.ndarray) -> tuple[int, float]:
    """Match recovered PWM duties to closest font glyph via Fisher-Rao distance."""
    best_idx, best_dist = 0, float("inf")
    for g in range(font.shape[0]):
        d = fisher_rao_distance(recovered_duties, font[g])
        if d < best_dist:
            best_dist, best_idx = d, g
    fidelity = 1.0 - best_dist / np.pi
    return best_idx, max(0.0, fidelity)


def _top_glyph_candidates(
    recovered_duties: np.ndarray,
    font: np.ndarray,
    *,
    k: int = 8,
) -> list[int]:
    """Return ``k`` font indices with smallest Fisher-Rao distance to ICA duties."""
    dists = [
        (fisher_rao_distance(recovered_duties, font[g]), g) for g in range(font.shape[0])
    ]
    dists.sort(key=lambda item: item[0])
    return [g for _, g in dists[:k]]


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
    *,
    reference_field: np.ndarray | None = None,
    reference_quaternion: Quaternion | None = None,
    noise_scale: float = 1.0,
    use_legacy_qec: bool = False,
    constants: EmergentConstants | None = None,
    pulse_duration_ns: float = 1.0,
    glyph_rank_k: int = 24,
    glyph_refine_k: int = 12,
    carrier_grid_steps: int = 15,
    carrier_refine_top_k: int = 5,
    slsqp_maxiter: int = 120,
    pin_carrier_w: float | None = None,
    force_carrier_top_k: int | None = None,
    auto_retry_early_exit: bool = True,
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
    field_mid = triplet_centre_field(field_time)
    mid = n_t // 2

    if rho is None or phi is None:
        x = np.linspace(-2.5, 2.5, nx)
        y = np.linspace(-2.5, 2.5, ny)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
    else:
        X = rho * np.cos(phi)
        Y = rho * np.sin(phi)

    intensity = np.abs(field_time) ** 2
    qec_stats: QECStats | None = None
    if use_legacy_qec:
        intensity = repetition_qec(
            intensity, reps=16, error_rate=bmgl.alpha_chemical if bmgl else 0.015
        )
    else:
        intensity, qec_stats = bitflip_repetition_qec(
            intensity,
            reference=reference_intensity,
            bmgl=bmgl,
            noise_scale=noise_scale,
        )

    n_orbs = len(orbs_ells)
    flat = intensity.reshape(n_t, -1).T
    ica = FastICA(
        n_components=min(n_orbs, n_t),
        random_state=42,
        max_iter=5000 if n_orbs >= 8 else 2000,
        tol=5e-4 if n_orbs >= 8 else 1e-4,
    )
    try:
        S = ica.fit_transform(flat)
        recovered_duties = np.clip(np.mean(np.abs(S), axis=0)[:n_orbs], 0, 1)
        recovered_duties = recovered_duties / (recovered_duties.sum() + 1e-12)
    except Exception:
        recovered_duties = np.mean(intensity, axis=(1, 2))
        recovered_duties = recovered_duties[:n_orbs]
        recovered_duties = recovered_duties / (recovered_duties.sum() + 1e-12)

    glyph_idx, glyph_fid = _nearest_glyph(recovered_duties, font)

    ell_range = list(set(orbs_ells + list(OAM_QUAT_ELLS) + [-2]))
    oam_weights = project_oam_spectrum(field_mid, rho, phi, ell_range)
    quat_weights = project_quaternion_oam(field_mid, rho, phi, w0=1.0)
    recovered_ells = [dominant_ell(oam_weights)]

    n_eff = effective_num_times(n_t, QEC_REPS)
    n_logical = n_eff // QEC_REPS
    mid_logical = n_logical // 2
    centre_idx = mid_logical * QEC_REPS + QEC_REPS // 2
    if t is not None and centre_idx < t.shape[0]:
        t_val = float(t[centre_idx])
        t_max = float(t[-1]) if t.size else pulse_duration_ns * 1e-9
    else:
        t_max = pulse_duration_ns * 1e-9
        t_val = (
            float(centre_idx) / max(n_eff - 1, 1) * t_max
            if n_eff > 1
            else 0.5 * t_max
        )
    intensity_mid = np.abs(field_mid) ** 2
    glyph_constants = constants or EmergentConstants()
    glyph_bank = get_glyph_template_bank(
        font,
        X,
        Y,
        t_val,
        t_max,
        len(orbs_ells),
        glyph_constants,
    )
    glyph_candidates = rank_glyphs_by_orb_intensity(
        intensity_mid,
        glyph_bank,
        k=glyph_rank_k,
    )[:glyph_refine_k]

    ref_flat = reference_intensity[mid].flatten()
    rec_flat = np.abs(field_mid).flatten()
    if np.std(ref_flat) < 1e-10:
        ref_flat = ref_flat + np.random.normal(0, 1e-10, ref_flat.shape)
    if np.std(rec_flat) < 1e-10:
        rec_flat = rec_flat + np.random.normal(0, 1e-10, rec_flat.shape)
    shard_fidelity = float(pearsonr(ref_flat, rec_flat)[0])

    manifold_recovery: ManifoldRecoveryResult | None = None
    if reference_field is not None and reference_quaternion is not None:
        ref_mid = triplet_centre_field(reference_field)
        q = recover_quaternion_with_reference(
            field_mid,
            ref_mid,
            reference_quaternion,
            rho,
            phi,
        )
    else:
        warm = oam_weights_to_quaternion(quat_weights, field=field_mid, phi=phi)
        observed_centre = project_quaternion_oam(field_mid, rho, phi, w0=1.0)
        best_ff = float("inf")
        manifold_recovery: ManifoldRecoveryResult | None = None
        for gidx in glyph_candidates:
            candidate_orb = glyph_bank.orb_fields[gidx]
            candidate = recover_quaternion_from_field_manifold(
                field_time,
                rho,
                phi,
                bmgl=bmgl,
                noise_scale=noise_scale,
                warm_start=warm,
                orb_field=candidate_orb,
                carrier_w=pin_carrier_w,
                carrier_grid_steps=carrier_grid_steps,
                carrier_refine_top_k=carrier_refine_top_k,
                slsqp_maxiter=slsqp_maxiter,
                force_carrier_top_k=force_carrier_top_k,
                auto_retry_early_exit=auto_retry_early_exit,
            )
            ff_loss = _full_forward_oam_loss(
                candidate.quaternion,
                candidate_orb,
                observed_centre,
                rho,
                phi,
                1.0,
            )
            if ff_loss < best_ff:
                best_ff = ff_loss
                manifold_recovery = candidate
                glyph_idx = gidx
                glyph_fid = max(
                    0.0,
                    1.0
                    - fisher_rao_distance(recovered_duties, font[gidx]) / np.pi,
                )
        if manifold_recovery is None:
            manifold_recovery = recover_quaternion_from_field_manifold(
                field_time,
                rho,
                phi,
                bmgl=bmgl,
                noise_scale=noise_scale,
                warm_start=warm,
            )
        q = manifold_recovery.quaternion
    q_arr = q.as_array()
    recovered_bytes = decode_shard(q)

    return DecodeResult(
        recovered_ells=recovered_ells,
        shard_fidelity=shard_fidelity,
        glyph_index=glyph_idx,
        glyph_fidelity=glyph_fid,
        quaternion=Quaternion(*q_arr),
        recovered_bytes=recovered_bytes,
        oam_weights=oam_weights,
        qec_stats=qec_stats,
        manifold_recovery=manifold_recovery,
    )