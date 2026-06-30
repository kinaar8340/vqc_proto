"""Quaternion ↔ OAM coupling derived from typehead encode geometry.

Forward model (typehead.encode)
-------------------------------
1. ``encode_shard`` maps payload → unit quaternion q = (w, x, y, z).
2. Orb superposition is multiplied by LG_{ℓ=1} carrier and a Rodrigues phase:
     φ_q = PHI_SCALE · cos(q.w · π/2)   applied as  exp(i φ_q)  on the whole field.
3. Quaternion components imprint on low-order OAM modes (orthogonal LG basis):
     ΔE ⊃  IMPRINT_SCALE · ( q.x·LG₀ + i·q.y·LG_{-1} + q.z·LG₂ )

Inverse (decode)
----------------
1. Dewarp helical phases  exp(i ℓ φ)  per mode.
2. Recover q.w from mean global phase via Rodrigues inversion.
3. Recover (q.x, q.y, q.z) by projecting onto {0, -1, 2} and dividing by
   ``IMPRINT_SCALE`` (ℓ=1 handled by w-channel).
4. Normalize to S³.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .altermagnetic import PWaveBMGL
from .encode_redundancy import QEC_REPS, effective_num_times
from .lg_modes import project_oam_spectrum
from .quaternion_codec import Quaternion

# Must match typehead.encode
PHI_SCALE = 0.3
IMPRINT_SCALE = 0.12
CARRIER_ELL = 1
OAM_QUAT_ELLS = (0, 1, -1, 2)  # x, w-carrier, y, z


def _phi_centroid(field: np.ndarray, phi: np.ndarray) -> float:
    intensity = np.abs(field) ** 2
    w = intensity / (intensity.sum() + 1e-12)
    return float(np.angle(np.sum(w * np.exp(1j * phi))))


def dewarp_oam_weights(
    weights: dict[int, complex],
    phi_centroid: float,
) -> dict[int, complex]:
    """Remove azimuthal helical phase exp(i ℓ φ_c) from each mode coefficient."""
    return {ell: w * np.exp(-1j * ell * phi_centroid) for ell, w in weights.items()}


def recover_w_from_phase(dewarped: dict[int, complex]) -> float:
    """Invert Rodrigues global phase imprint φ_q = PHI_SCALE · cos(q.w · π/2)."""
    if not dewarped:
        return 0.5
    phi_est = float(np.angle(np.mean(list(dewarped.values()))))
    cos_arg = float(np.clip(phi_est / PHI_SCALE, -1.0, 1.0))
    return float((2.0 / np.pi) * np.arccos(cos_arg))


def recover_xyz_from_projections(
    dewarped: dict[int, complex],
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> tuple[float, float, float]:
    """Linear inversion on orthogonal LG subspace {0, -1, 2}."""
    scale = imprint_scale + 1e-12
    x = float(np.real(dewarped.get(0, 0.0))) / scale
    y = float(np.imag(dewarped.get(-1, 0.0))) / scale
    z = float(np.real(dewarped.get(2, 0.0))) / scale
    return x, y, z


def oam_weights_to_quaternion(
    weights: dict[int, complex],
    *,
    phi_centroid: float | None = None,
    field: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    imprint_scale: float = IMPRINT_SCALE,
) -> Quaternion:
    """
    Level-1 OAM → quaternion recovery from LG projections.

    Parameters
    ----------
    weights
        Mode coefficients from ``project_oam_spectrum``.
    phi_centroid
        Azimuthal reference angle; estimated from ``field`` if omitted.
    field, phi
        Used to estimate ``phi_centroid`` when not supplied.
    """
    if phi_centroid is None:
        if field is not None and phi is not None:
            phi_centroid = _phi_centroid(field, phi)
        else:
            phi_centroid = 0.0

    dewarped = dewarp_oam_weights(weights, phi_centroid)
    w = recover_w_from_phase(dewarped)
    x, y, z = recover_xyz_from_projections(dewarped, imprint_scale=imprint_scale)
    arr = np.array([w, x, y, z], dtype=float)
    arr /= np.linalg.norm(arr) + 1e-12
    return Quaternion(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def triplet_centre_field(
    field_time: np.ndarray,
    *,
    reps: int = QEC_REPS,
) -> np.ndarray:
    """Return centre frame of the middle logical triplet (post-QEC snapshot)."""
    n_t = field_time.shape[0]
    n_eff = effective_num_times(n_t, reps)
    if n_eff > n_t:
        pad = n_eff - n_t
        field_time = np.concatenate(
            [field_time, np.tile(field_time[-1:], (pad, 1, 1))], axis=0
        )
    n_logical = n_eff // reps
    mid_logical = n_logical // 2
    centre = mid_logical * reps + reps // 2
    return field_time[centre]


def project_quaternion_oam(
    field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    ell_range: list[int] | None = None,
    w0: float = 1.0,
) -> dict[int, complex]:
    """Project field onto the quaternion-OAM subspace."""
    ells = ell_range or list(OAM_QUAT_ELLS)
    return project_oam_spectrum(field, rho, phi, ells, w0=w0)


def quaternion_recovery_error(q_ref: Quaternion, q_rec: Quaternion) -> float:
    """Chordal distance on S³ (identifies q with -q for rotation equivalence)."""
    a = q_ref.as_array()
    b = q_rec.as_array()
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def encode_imprint_field(
    q: Quaternion,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> np.ndarray:
    """Forward quaternion OAM imprint (matches typehead additive term)."""
    from .lg_modes import lg_mode

    return imprint_scale * (
        q.x * lg_mode(0, rho, phi, w0=w0)
        + 1j * q.y * lg_mode(-1, rho, phi, w0=w0)
        + q.z * lg_mode(2, rho, phi, w0=w0)
    )


def _carrier_phase_factor(q_w: float) -> complex:
    """Rodrigues global phase exp(i · PHI_SCALE · cos(q.w · π/2))."""
    axis_x = float(np.cos(q_w * np.pi / 2.0))
    return complex(np.cos(PHI_SCALE * axis_x), np.sin(PHI_SCALE * axis_x))


def recover_quaternion_with_reference(
    field_noisy: np.ndarray,
    field_clean: np.ndarray,
    q_ref: Quaternion,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> Quaternion:
    """
    Differential OAM → quaternion recovery using the clean encoded field.

    Forms per-mode complex ratios ``w_noisy / w_clean``, dewarps helical phase,
    then applies:
      - w  from Rodrigues phase on the ℓ=1 ratio (carrier channel)
      - x,y,z from imprint-mode magnitude ratios scaled against ``q_ref``
    """
    w_clean = project_quaternion_oam(field_clean, rho, phi, w0=w0)
    w_noisy = project_quaternion_oam(field_noisy, rho, phi, w0=w0)

    phi_delta = _phi_centroid(field_noisy, phi) - _phi_centroid(field_clean, phi)
    ratio: dict[int, complex] = {}
    for ell in OAM_QUAT_ELLS:
        denom = w_clean.get(ell, 0.0)
        if abs(denom) < 1e-12:
            denom = 1e-12 + 1e-12j
        ratio[ell] = w_noisy.get(ell, 0.0) / denom

    ratio_dewarp = dewarp_oam_weights(ratio, phi_delta)

    # w: differential Rodrigues phase on carrier (ℓ=1) ratio vs reference
    cos_ref = float(np.cos(q_ref.w * np.pi / 2.0))
    phi_1 = float(np.angle(ratio_dewarp.get(1, 1.0 + 0.0j)))
    cos_est = float(np.clip(cos_ref + phi_1 / PHI_SCALE, -1.0, 1.0))
    w = float((2.0 / np.pi) * np.arccos(cos_est))

    # x, y, z: multiplicative correction on reference imprint components
    ref = q_ref.as_array().copy()
    ref /= np.linalg.norm(ref) + 1e-12
    x_scale = float(np.clip(np.abs(ratio_dewarp.get(0, 1.0)), 0.2, 5.0))
    y_scale = float(np.clip(np.abs(ratio_dewarp.get(-1, 1.0)), 0.2, 5.0))
    z_scale = float(np.clip(np.abs(ratio_dewarp.get(2, 1.0)), 0.2, 5.0))
    arr = np.array(
        [
            w,
            ref[1] * x_scale,
            ref[2] * y_scale,
            ref[3] * z_scale,
        ],
        dtype=float,
    )
    arr /= np.linalg.norm(arr) + 1e-12
    return Quaternion(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


@dataclass(frozen=True)
class ManifoldRecoveryResult:
    """Outcome of reference-free S³ manifold optimization."""

    quaternion: Quaternion
    loss: float
    converged: bool
    used_fallback: bool
    carrier_w: float | None = None
    orb_subtracted: bool = False
    optimizer_nfev: int | None = None
    optimizer_nit: int | None = None
    optimizer_nfev_total: int | None = None
    carrier_grid_evals: int | None = None
    carrier_search_retried: bool = False


# Adaptive carrier retry when warm scoring lands in a shallow early-exit basin.
AUTO_RETRY_NFEV_THRESHOLD = 300
AUTO_RETRY_LOSS_THRESHOLD = 1e-6
AUTO_RETRY_FORCE_TOP_K = 5


def _should_retry_carrier_search(
    result: ManifoldRecoveryResult,
    *,
    nfev_threshold: int,
    loss_threshold: float,
) -> bool:
    """
    True when carrier SLSQP likely settled in a shallow basin.

    Requires elevated residual loss and nfev below threshold (default 300) so
    moderate-cost passes (e.g. nfev≈256) still retry while full searches skip.
    """
    if result.used_fallback:
        return False
    nfev = result.optimizer_nfev_total
    if nfev is None or nfev >= nfev_threshold:
        return False
    if result.loss <= loss_threshold:
        return False
    return True


def _normalize_quaternion_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr / (np.linalg.norm(arr) + 1e-12)


def forward_quaternion_field(
    q: Quaternion,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> np.ndarray:
    """Predict quaternion-only field: imprint × LG₁ × Rodrigues phase."""
    from .lg_modes import lg_mode

    imprint = encode_imprint_field(q, rho, phi, w0=w0, imprint_scale=imprint_scale)
    carrier = lg_mode(CARRIER_ELL, rho, phi, w0=w0) * _carrier_phase_factor(q.w)
    return imprint * carrier


def predict_oam_weights_from_quaternion(
    q: Quaternion,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> dict[int, complex]:
    """Forward operator: unit quaternion → expected LG coefficients."""
    field_q = forward_quaternion_field(
        q, rho, phi, w0=w0, imprint_scale=imprint_scale
    )
    return project_quaternion_oam(field_q, rho, phi, w0=w0)


def mode_weights_from_bmgl(
    bmgl: PWaveBMGL | None,
    *,
    noise_scale: float = 1.0,
) -> dict[int, float]:
    """BMGL-derived per-mode weights (higher inhibition → trust projections more)."""
    if bmgl is None:
        return {ell: 1.0 for ell in OAM_QUAT_ELLS}
    trust = bmgl.effective_inhibition / max(noise_scale, 1e-6)
    return {ell: float(np.clip(trust, 0.25, 4.0)) for ell in OAM_QUAT_ELLS}


def _triplet_centre_indices(n_t: int, reps: int = QEC_REPS) -> list[int]:
    n_eff = effective_num_times(n_t, reps)
    n_logical = n_eff // reps
    return [li * reps + reps // 2 for li in range(n_logical)]


def triplet_median_oam_weights(
    field_time: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    *,
    reps: int = QEC_REPS,
    w0: float = 1.0,
) -> dict[int, complex]:
    """Median LG projections across triplet-centre frames."""
    stacks: list[dict[int, complex]] = []
    for idx in _triplet_centre_indices(field_time.shape[0], reps):
        if idx < field_time.shape[0]:
            stacks.append(project_quaternion_oam(field_time[idx], rho, phi, w0=w0))
    if not stacks:
        return {ell: 0.0 + 0.0j for ell in OAM_QUAT_ELLS}
    median: dict[int, complex] = {}
    for ell in OAM_QUAT_ELLS:
        vals = [s.get(ell, 0.0 + 0.0j) for s in stacks]
        median[ell] = complex(
            float(np.median(np.real(vals))),
            float(np.median(np.imag(vals))),
        )
    return median


def orb_oam_background_weights(
    orb_field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    w_carrier: float,
    w0: float = 1.0,
) -> dict[int, complex]:
    """LG projections of synthesized orb field under Rodrigues carrier ``w_carrier``."""
    from .lg_modes import lg_mode

    carrier = lg_mode(CARRIER_ELL, rho, phi, w0=w0) * _carrier_phase_factor(w_carrier)
    return project_quaternion_oam(orb_field * carrier, rho, phi, w0=w0)


def residual_oam_weights_after_orb(
    observed: dict[int, complex],
    orb_field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    w_carrier: float,
    w0: float = 1.0,
) -> dict[int, complex]:
    """Subtract estimated orb+carrier background from observed quaternion-OAM weights."""
    bg = orb_oam_background_weights(orb_field, rho, phi, w_carrier, w0=w0)
    return {ell: observed.get(ell, 0.0) - bg.get(ell, 0.0) for ell in OAM_QUAT_ELLS}


def _full_forward_oam_loss(
    q: Quaternion,
    orb_field: np.ndarray,
    observed: dict[int, complex],
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float,
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> float:
    """Match observed weights against orb + imprint forward model (carrier included)."""
    from .lg_modes import lg_mode

    imprint = encode_imprint_field(q, rho, phi, w0=w0, imprint_scale=imprint_scale)
    carrier = lg_mode(CARRIER_ELL, rho, phi, w0=w0) * _carrier_phase_factor(q.w)
    pred = project_quaternion_oam(
        (orb_field + imprint) * carrier, rho, phi, w0=w0
    )
    return sum(
        float(np.abs(pred.get(ell, 0.0) - observed.get(ell, 0.0)) ** 2)
        for ell in OAM_QUAT_ELLS
    )


def _warm_carrier_forward_loss(
    w_carrier: float,
    observed: dict[int, complex],
    orb_field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float,
    *,
    imprint_scale: float = IMPRINT_SCALE,
) -> float:
    """Fast carrier candidate score using warm quaternion (no SLSQP)."""
    warm_arr = np.array([w_carrier, 0.25, 0.45, 0.45], dtype=float)
    warm_arr /= np.linalg.norm(warm_arr) + 1e-12
    return _full_forward_oam_loss(
        Quaternion(*warm_arr),
        orb_field,
        observed,
        rho,
        phi,
        w0,
        imprint_scale=imprint_scale,
    )


def _slsqp_s3_from_observed(
    observed: dict[int, complex],
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float,
    warm_start: Quaternion,
    mode_weights: dict[int, float],
    *,
    imprint_scale: float = IMPRINT_SCALE,
    maxiter: int = 200,
    ftol: float = 1e-8,
) -> tuple[Quaternion, float, bool, int, int]:
    """Single SLSQP fit on S³ (no orb branch)."""
    x0 = _normalize_quaternion_array(warm_start.as_array())

    def objective(v: np.ndarray) -> float:
        return _weighted_projection_loss(
            v,
            observed,
            rho,
            phi,
            w0,
            mode_weights,
            imprint_scale=imprint_scale,
        )

    constraints = {"type": "eq", "fun": lambda v: float(np.dot(v, v) - 1.0)}
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    q_opt = Quaternion(*_normalize_quaternion_array(result.x))
    return (
        q_opt,
        float(result.fun),
        bool(result.success),
        int(getattr(result, "nfev", 0)),
        int(getattr(result, "nit", 0)),
    )


def search_carrier_w_for_orb_residual(
    observed: dict[int, complex],
    orb_field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    mode_weights: dict[int, float] | None = None,
    imprint_scale: float = IMPRINT_SCALE,
    coarse_steps: int = 41,
) -> float:
    """Grid search for carrier ``w`` using full orb+imprint forward verification."""
    result = _recover_manifold_with_orb_background(
        observed,
        orb_field,
        rho,
        phi,
        w0=w0,
        mode_weights=mode_weights,
        imprint_scale=imprint_scale,
        coarse_steps=coarse_steps,
    )
    return float(result.carrier_w or 0.5)


# Clean/low-noise carrier basins always unioned when force_carrier_top_k is active.
CARRIER_FORCE_ANCHORS: tuple[float, ...] = (0.429,)


def _nearest_grid_carrier(anchor: float, grid_ws: list[float]) -> float:
    return min(grid_ws, key=lambda w: abs(w - anchor))


def _select_refine_carriers(
    coarse_scores: list[tuple[float, float]],
    *,
    carrier_refine_top_k: int,
    force_carrier_top_k: int | None,
) -> list[float]:
    """Pick carrier weights for SLSQP refinement (warm top-k, optionally forced + anchored)."""
    grid_ws = [w for _, w in coarse_scores]
    if force_carrier_top_k is None:
        top_k = max(1, min(carrier_refine_top_k, len(coarse_scores)))
        return [w for _, w in coarse_scores[:top_k]]

    refine_k = max(carrier_refine_top_k, force_carrier_top_k)
    refine_k = min(refine_k, len(coarse_scores))
    warm_top = [w for _, w in coarse_scores[:refine_k]]
    anchors = [_nearest_grid_carrier(a, grid_ws) for a in CARRIER_FORCE_ANCHORS]
    return sorted(set(warm_top + anchors))


def _recover_manifold_with_orb_background(
    observed: dict[int, complex],
    orb_field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    mode_weights: dict[int, float] | None = None,
    imprint_scale: float = IMPRINT_SCALE,
    coarse_steps: int = 41,
    warm_start: Quaternion | None = None,
    carrier_refine_top_k: int = 5,
    slsqp_maxiter: int = 120,
    pin_carrier_w: float | None = None,
    force_carrier_top_k: int | None = None,
    auto_retry_early_exit: bool = True,
    auto_retry_nfev_threshold: int = AUTO_RETRY_NFEV_THRESHOLD,
    auto_retry_loss_threshold: float = AUTO_RETRY_LOSS_THRESHOLD,
    auto_retry_force_top_k: int = AUTO_RETRY_FORCE_TOP_K,
) -> ManifoldRecoveryResult:
    """
    Orb-assisted Level-2: grid carrier ``w``, subtract background, S³ fit residual,
    pick candidate with lowest full forward-model loss vs. observed weights.

    Uses a two-phase carrier search: coarse warm forward scoring on the full grid,
    then SLSQP only on the top ``carrier_refine_top_k`` candidates.

    When ``pin_carrier_w`` is set, skips grid scoring and runs SLSQP only at that
    carrier weight (debug / robustness override).

    When ``force_carrier_top_k`` is set, always refines at least that many warm
    candidates and unions clean-basin anchor carriers (see ``CARRIER_FORCE_ANCHORS``).

    When ``auto_retry_early_exit`` is enabled (default), a fast low-nfev pass with
    elevated residual loss triggers one forced top-k retry.
    """
    mode_weights = mode_weights or {ell: 1.0 for ell in OAM_QUAT_ELLS}
    fallback = warm_start or oam_weights_to_quaternion(observed, imprint_scale=imprint_scale)

    if pin_carrier_w is not None:
        refine_carriers = [float(np.clip(pin_carrier_w, 0.0, 1.0))]
        grid_evals = 0
    else:
        grid_evals = 0
        coarse_scores: list[tuple[float, float]] = []
        for w_carrier in np.linspace(0.0, 1.0, coarse_steps):
            w_carrier = float(w_carrier)
            grid_evals += 1
            warm_ff = _warm_carrier_forward_loss(
                w_carrier,
                observed,
                orb_field,
                rho,
                phi,
                w0,
                imprint_scale=imprint_scale,
            )
            coarse_scores.append((warm_ff, w_carrier))
        coarse_scores.sort(key=lambda item: item[0])
        refine_carriers = _select_refine_carriers(
            coarse_scores,
            carrier_refine_top_k=carrier_refine_top_k,
            force_carrier_top_k=force_carrier_top_k,
        )

    slsqp_ftol = 1e-10 if force_carrier_top_k is not None else 1e-8
    best_ff = float("inf")
    best: tuple[Quaternion, float, float, bool, int, int] | None = None
    total_nfev = 0

    for w_carrier in refine_carriers:
        residual = residual_oam_weights_after_orb(
            observed, orb_field, rho, phi, w_carrier, w0=w0
        )
        warm_arr = np.array([w_carrier, 0.25, 0.45, 0.45], dtype=float)
        warm_arr /= np.linalg.norm(warm_arr) + 1e-12
        q_cand, res_loss, ok, nfev, nit = _slsqp_s3_from_observed(
            residual,
            rho,
            phi,
            w0,
            Quaternion(*warm_arr),
            mode_weights,
            imprint_scale=imprint_scale,
            maxiter=slsqp_maxiter,
            ftol=slsqp_ftol,
        )
        total_nfev += nfev
        if force_carrier_top_k is not None and ok:
            x_refined, loss_refined = _riemannian_s3_refine(
                q_cand.as_array(),
                residual,
                rho,
                phi,
                w0,
                mode_weights,
                imprint_scale=imprint_scale,
            )
            if loss_refined < res_loss:
                q_cand = Quaternion(*x_refined)
                res_loss = loss_refined
        ff_loss = _full_forward_oam_loss(
            q_cand,
            orb_field,
            observed,
            rho,
            phi,
            w0,
            imprint_scale=imprint_scale,
        )
        if ff_loss < best_ff:
            best_ff = ff_loss
            best = (q_cand, res_loss, w_carrier, ok, nfev, nit)

    if best is None:
        return ManifoldRecoveryResult(
            quaternion=fallback,
            loss=float("inf"),
            converged=False,
            used_fallback=True,
            carrier_w=None,
            orb_subtracted=True,
            carrier_grid_evals=grid_evals,
            optimizer_nfev_total=total_nfev,
        )

    q_opt, res_loss, carrier_w, ok, win_nfev, win_nit = best
    result = ManifoldRecoveryResult(
        quaternion=q_opt,
        loss=res_loss,
        converged=ok,
        used_fallback=False,
        carrier_w=carrier_w,
        orb_subtracted=True,
        optimizer_nfev=win_nfev,
        optimizer_nit=win_nit,
        optimizer_nfev_total=total_nfev,
        carrier_grid_evals=grid_evals,
    )

    if (
        auto_retry_early_exit
        and pin_carrier_w is None
        and force_carrier_top_k is None
        and _should_retry_carrier_search(
            result,
            nfev_threshold=auto_retry_nfev_threshold,
            loss_threshold=auto_retry_loss_threshold,
        )
    ):
        retry = _recover_manifold_with_orb_background(
            observed,
            orb_field,
            rho,
            phi,
            w0=w0,
            mode_weights=mode_weights,
            imprint_scale=imprint_scale,
            coarse_steps=coarse_steps,
            warm_start=warm_start,
            carrier_refine_top_k=carrier_refine_top_k,
            slsqp_maxiter=slsqp_maxiter,
            force_carrier_top_k=auto_retry_force_top_k,
            auto_retry_early_exit=False,
        )
        return replace(retry, carrier_search_retried=True)

    return result


def _riemannian_s3_refine(
    q_vec: np.ndarray,
    observed: dict[int, complex],
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float,
    mode_weights: dict[int, float],
    *,
    imprint_scale: float,
    steps: int = 40,
    step_size: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Riemannian gradient descent on S³ for local polish after SLSQP."""
    q_arr = _normalize_quaternion_array(q_vec)

    def loss_and_grad(v: np.ndarray) -> tuple[float, np.ndarray]:
        eps = 1e-5
        base = _weighted_projection_loss(
            v, observed, rho, phi, w0, mode_weights, imprint_scale=imprint_scale
        )
        grad = np.zeros(4, dtype=float)
        for i in range(4):
            vp = v.copy()
            vp[i] += eps
            vp = _normalize_quaternion_array(vp)
            grad[i] = (
                _weighted_projection_loss(
                    vp, observed, rho, phi, w0, mode_weights, imprint_scale=imprint_scale
                )
                - base
            ) / eps
        return base, grad

    loss = _weighted_projection_loss(
        q_arr, observed, rho, phi, w0, mode_weights, imprint_scale=imprint_scale
    )
    for _ in range(steps):
        _, grad = loss_and_grad(q_arr)
        grad_tan = grad - float(np.dot(grad, q_arr)) * q_arr
        norm = float(np.linalg.norm(grad_tan))
        if norm < 1e-10:
            break
        q_arr = _normalize_quaternion_array(q_arr - step_size * grad_tan / norm)
        loss = _weighted_projection_loss(
            q_arr, observed, rho, phi, w0, mode_weights, imprint_scale=imprint_scale
        )
    return q_arr, loss


def _weighted_projection_loss(
    q_vec: np.ndarray,
    observed: dict[int, complex],
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float,
    mode_weights: dict[int, float],
    *,
    imprint_scale: float,
) -> float:
    q_arr = _normalize_quaternion_array(q_vec)
    q = Quaternion(*q_arr)
    pred = predict_oam_weights_from_quaternion(
        q, rho, phi, w0=w0, imprint_scale=imprint_scale
    )
    loss = 0.0
    for ell in OAM_QUAT_ELLS:
        w = mode_weights.get(ell, 1.0)
        diff = pred.get(ell, 0.0) - observed.get(ell, 0.0)
        loss += w * float(np.abs(diff) ** 2)
    return loss


def recover_quaternion_manifold(
    observed: dict[int, complex],
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    warm_start: Quaternion | None = None,
    field: np.ndarray | None = None,
    phi_grid: np.ndarray | None = None,
    mode_weights: dict[int, float] | None = None,
    imprint_scale: float = IMPRINT_SCALE,
    loss_tol: float = 1e-2,
    orb_field: np.ndarray | None = None,
    carrier_w: float | None = None,
    use_riemannian_refine: bool = True,
    carrier_grid_steps: int = 41,
    carrier_refine_top_k: int = 5,
    slsqp_maxiter: int = 120,
    force_carrier_top_k: int | None = None,
    auto_retry_early_exit: bool = True,
) -> ManifoldRecoveryResult:
    """
    Level-2 reference-free recovery: optimize q ∈ S³ to match observed OAM weights.

    When ``orb_field`` is supplied, uses orb-background subtraction with carrier
    grid search verified against the full orb+imprint forward model.
    """
    mode_weights = mode_weights or {ell: 1.0 for ell in OAM_QUAT_ELLS}
    phi_use = phi_grid if phi_grid is not None else phi

    if orb_field is not None:
        return _recover_manifold_with_orb_background(
            observed,
            orb_field,
            rho,
            phi_use,
            w0=w0,
            mode_weights=mode_weights,
            imprint_scale=imprint_scale,
            warm_start=warm_start,
            coarse_steps=carrier_grid_steps,
            carrier_refine_top_k=carrier_refine_top_k,
            slsqp_maxiter=slsqp_maxiter,
            pin_carrier_w=carrier_w,
            force_carrier_top_k=force_carrier_top_k,
            auto_retry_early_exit=auto_retry_early_exit,
        )

    if warm_start is None:
        warm_start = oam_weights_to_quaternion(
            observed,
            field=field,
            phi=phi_use,
            imprint_scale=imprint_scale,
        )

    q_opt, loss, ok, nfev, nit = _slsqp_s3_from_observed(
        observed,
        rho,
        phi_use,
        w0,
        warm_start,
        mode_weights,
        imprint_scale=imprint_scale,
        maxiter=slsqp_maxiter,
    )
    x_opt = _normalize_quaternion_array(q_opt.as_array())
    if use_riemannian_refine and ok:
        x_refined, loss_refined = _riemannian_s3_refine(
            x_opt,
            observed,
            rho,
            phi_use,
            w0,
            mode_weights,
            imprint_scale=imprint_scale,
        )
        if loss_refined < loss:
            x_opt, loss = x_refined, loss_refined

    warm_loss = _weighted_projection_loss(
        warm_start.as_array(),
        observed,
        rho,
        phi_use,
        w0,
        mode_weights,
        imprint_scale=imprint_scale,
    )
    obs_scale = sum(
        float(np.abs(observed.get(ell, 0.0)) ** 2) for ell in OAM_QUAT_ELLS
    )
    adaptive_tol = max(loss_tol, obs_scale * 0.05)

    converged = ok
    q_opt = Quaternion(*x_opt)

    improved = loss < warm_loss * 0.999

    if not converged or (not improved and loss > adaptive_tol):
        return ManifoldRecoveryResult(
            quaternion=warm_start,
            loss=loss,
            converged=False,
            used_fallback=True,
            carrier_w=None,
            orb_subtracted=False,
            optimizer_nfev=nfev,
            optimizer_nit=nit,
            optimizer_nfev_total=nfev,
            carrier_grid_evals=1,
        )

    return ManifoldRecoveryResult(
        quaternion=q_opt,
        loss=loss,
        converged=True,
        used_fallback=False,
        carrier_w=None,
        orb_subtracted=False,
        optimizer_nfev=nfev,
        optimizer_nit=nit,
        optimizer_nfev_total=nfev,
        carrier_grid_evals=1,
    )


def recover_quaternion_from_field_manifold(
    field_time: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    w0: float = 1.0,
    *,
    bmgl: PWaveBMGL | None = None,
    noise_scale: float = 1.0,
    warm_start: Quaternion | None = None,
    reps: int = QEC_REPS,
    orb_field: np.ndarray | None = None,
    carrier_w: float | None = None,
    carrier_grid_steps: int = 41,
    carrier_refine_top_k: int = 5,
    slsqp_maxiter: int = 120,
    force_carrier_top_k: int | None = None,
    auto_retry_early_exit: bool = True,
) -> ManifoldRecoveryResult:
    """Reference-free path: OAM stats → S³ manifold optimization."""
    field_mid = triplet_centre_field(field_time, reps=reps)
    if orb_field is not None:
        observed = project_quaternion_oam(field_mid, rho, phi, w0=w0)
    else:
        observed = triplet_median_oam_weights(
            field_time, rho, phi, reps=reps, w0=w0
        )
    weights = mode_weights_from_bmgl(bmgl, noise_scale=noise_scale)
    if warm_start is None:
        warm_start = oam_weights_to_quaternion(observed, field=field_mid, phi=phi)
    return recover_quaternion_manifold(
        observed,
        rho,
        phi,
        w0=w0,
        warm_start=warm_start,
        field=field_mid,
        phi_grid=phi,
        mode_weights=weights,
        orb_field=orb_field,
        carrier_w=carrier_w,
        carrier_grid_steps=carrier_grid_steps,
        carrier_refine_top_k=carrier_refine_top_k,
        slsqp_maxiter=slsqp_maxiter,
        force_carrier_top_k=force_carrier_top_k,
        auto_retry_early_exit=auto_retry_early_exit,
    )