"""Geometry-aware PWM demixing — replaces pixel FastICA at high orb counts."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA

from .stable_fonts import EmergentConstants, fisher_rao_distance
from .typehead import build_orbs_from_duties


def _orb_ring_masks(
    num_orbs: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_series: np.ndarray,
    t_max: float,
    constants: EmergentConstants,
    *,
    w0: float = 1.0,
) -> np.ndarray:
    """Soft Gaussian masks per orb and time, shape ``(n_orbs, n_t, ny, nx)``."""
    orbs = build_orbs_from_duties(np.full(num_orbs, 0.5), num_orbs, constants)
    sigma = w0 * 0.35
    n_t = len(t_series)
    masks = np.zeros((num_orbs, n_t, *x_grid.shape), dtype=np.float64)
    for i, orb in enumerate(orbs):
        for ti, t_val in enumerate(t_series):
            theta = orb.phase0 + orb.omega * float(t_val)
            x0 = orb.radius * np.cos(theta)
            y0 = orb.radius * np.sin(theta)
            masks[i, ti] = np.exp(
                -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * sigma**2)
            )
    return masks


def extract_orb_ring_channels(
    intensity: np.ndarray,
    num_orbs: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_series: np.ndarray,
    t_max: float,
    constants: EmergentConstants,
    *,
    w0: float = 1.0,
) -> np.ndarray:
    """Masked ring energy time series per orb, shape ``(n_orbs, n_t)``."""
    masks = _orb_ring_masks(
        num_orbs, x_grid, y_grid, t_series, t_max, constants, w0=w0
    )
    channels = np.zeros((num_orbs, intensity.shape[0]), dtype=np.float64)
    for i in range(num_orbs):
        for ti in range(intensity.shape[0]):
            mask = masks[i, ti]
            denom = float(mask.sum()) + 1e-12
            channels[i, ti] = float((intensity[ti] * mask).sum() / denom)
    return channels


def extract_orb_ring_channels_complex(
    field_time: np.ndarray,
    reference_field: np.ndarray,
    num_orbs: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_series: np.ndarray,
    t_max: float,
    constants: EmergentConstants,
    *,
    w0: float = 1.0,
) -> np.ndarray:
    """
    Reference-normalized complex ring channels (noise-sensitive under phase turbulence).

    Each sample is ``⟨E_noisy · E_ref*⟩_mask / ⟨|E_ref|²⟩_mask``.
    """
    masks = _orb_ring_masks(
        num_orbs, x_grid, y_grid, t_series, t_max, constants, w0=w0
    )
    channels = np.zeros((num_orbs, field_time.shape[0]), dtype=np.complex128)
    for i in range(num_orbs):
        for ti in range(field_time.shape[0]):
            mask = masks[i, ti]
            denom = float((np.abs(reference_field[ti]) ** 2 * mask).sum()) + 1e-12
            channels[i, ti] = complex(
                (field_time[ti] * np.conj(reference_field[ti]) * mask).sum() / denom
            )
    return channels


def _pwm_template(orb_omega: float, t_series: np.ndarray, t_max: float, duty: float) -> np.ndarray:
    gates = np.empty(len(t_series), dtype=np.float64)
    for ti, t_val in enumerate(t_series):
        phase = (np.sin(2.0 * np.pi * orb_omega * float(t_val) / t_max) + 1.0) / 2.0
        gates[ti] = 1.0 if phase < duty else 0.15
    return gates


def pwm_demodulate_duties(
    channels: np.ndarray,
    num_orbs: int,
    t_series: np.ndarray,
    t_max: float,
    constants: EmergentConstants,
    *,
    duty_grid: int = 28,
) -> np.ndarray:
    """Estimate PWM duties by correlating each orb channel with duty-gated templates."""
    orbs = build_orbs_from_duties(np.full(num_orbs, 0.5), num_orbs, constants)
    duties = np.zeros(num_orbs, dtype=np.float64)
    candidates = np.linspace(0.1, 0.95, duty_grid)
    for i, orb in enumerate(orbs):
        series = np.real(channels[i]) if np.iscomplexobj(channels) else channels[i]
        series = series.astype(np.float64, copy=False)
        series = series - float(series.mean())
        series_norm = float(np.linalg.norm(series)) + 1e-12
        best_duty = 0.5
        best_corr = -1.0
        for duty in candidates:
            template = _pwm_template(orb.omega, t_series, t_max, float(duty))
            template = template - float(template.mean())
            template_norm = float(np.linalg.norm(template)) + 1e-12
            corr = float(series @ template / (series_norm * template_norm))
            if corr > best_corr:
                best_corr = corr
                best_duty = float(duty)
        duties[i] = best_duty
    duties = np.clip(duties, 0.0, None)
    total = float(duties.sum())
    if total <= 0:
        duties[:] = 1.0 / max(num_orbs, 1)
        return duties
    return duties / total


def recover_duties_ring_energy(
    intensity: np.ndarray,
    num_orbs: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_series: np.ndarray,
    t_max: float,
    constants: EmergentConstants,
    *,
    w0: float = 1.0,
) -> np.ndarray:
    """Mean ring-channel energy per orb (robust coarse duty proxy)."""
    channels = extract_orb_ring_channels(
        intensity, num_orbs, x_grid, y_grid, t_series, t_max, constants, w0=w0
    )
    energies = np.mean(channels, axis=1)
    energies = np.clip(energies, 0.0, None)
    total = float(energies.sum()) + 1e-12
    return energies / total


def recover_duties_channel_ica(
    channels: np.ndarray,
    num_orbs: int,
    *,
    seeds: tuple[int, ...] = (42, 7, 13, 99),
) -> np.ndarray | None:
    """FastICA on orb-ring channels ``(n_orbs, n_t)`` — well-conditioned vs. pixel ICA."""
    n_t = channels.shape[1]
    n_comp = min(num_orbs, n_t)
    if n_comp < 2:
        return None
    if np.iscomplexobj(channels):
        matrix = np.abs(channels.T).astype(np.float64, copy=False)
    else:
        matrix = channels.T.astype(np.float64, copy=False)
    best_duties: np.ndarray | None = None
    best_err = float("inf")
    for seed in seeds:
        ica = FastICA(
            n_components=n_comp,
            random_state=seed,
            max_iter=4000,
            tol=5e-4,
            whiten="unit-variance",
        )
        try:
            sources = ica.fit_transform(matrix)
            recon = ica.inverse_transform(sources)
            err = float(np.mean((matrix - recon) ** 2))
            duties = np.clip(np.mean(np.abs(sources), axis=0)[:num_orbs], 0.0, 1.0)
            duties = duties / (float(duties.sum()) + 1e-12)
            if err < best_err:
                best_err = err
                best_duties = duties
        except Exception:
            continue
    return best_duties


def recover_duties_pixel_ica(
    intensity: np.ndarray,
    num_orbs: int,
    *,
    seeds: tuple[int, ...] = (42, 7, 13),
) -> np.ndarray | None:
    """Legacy pixel FastICA with multi-restart selection (kept for low orb counts)."""
    n_t, _, _ = intensity.shape
    flat = intensity.reshape(n_t, -1).T
    n_comp = min(num_orbs, n_t)
    if n_comp < 2:
        return None
    best_duties: np.ndarray | None = None
    best_err = float("inf")
    for seed in seeds:
        ica = FastICA(
            n_components=n_comp,
            random_state=seed,
            max_iter=5000 if num_orbs >= 8 else 2000,
            tol=5e-4 if num_orbs >= 8 else 1e-4,
            whiten="unit-variance",
        )
        try:
            sources = ica.fit_transform(flat)
            recon = ica.inverse_transform(sources)
            err = float(np.mean((flat - recon) ** 2))
            duties = np.clip(np.mean(np.abs(sources), axis=0)[:num_orbs], 0.0, 1.0)
            duties = duties / (float(duties.sum()) + 1e-12)
            if err < best_err:
                best_err = err
                best_duties = duties
        except Exception:
            continue
    return best_duties


def recover_pwm_duties(
    intensity: np.ndarray,
    num_orbs: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_series: np.ndarray,
    t_max: float,
    constants: EmergentConstants,
    *,
    field_time: np.ndarray | None = None,
    reference_field: np.ndarray | None = None,
    w0: float = 1.0,
) -> np.ndarray:
    """
    Fuse ring PWM demod, ring energy, channel ICA, and pixel ICA (low orbs).

    Complex reference-normalized channels are preferred when a pilot field exists.
    """
    if field_time is not None and reference_field is not None:
        channels = extract_orb_ring_channels_complex(
            field_time,
            reference_field,
            num_orbs,
            x_grid,
            y_grid,
            t_series,
            t_max,
            constants,
            w0=w0,
        )
    else:
        channels = extract_orb_ring_channels(
            intensity,
            num_orbs,
            x_grid,
            y_grid,
            t_series,
            t_max,
            constants,
            w0=w0,
        )

    pwm = pwm_demodulate_duties(channels, num_orbs, t_series, t_max, constants)
    ring = recover_duties_ring_energy(
        intensity, num_orbs, x_grid, y_grid, t_series, t_max, constants, w0=w0
    )
    channel_ica = recover_duties_channel_ica(channels, num_orbs)

    if num_orbs >= 6:
        weights = {"pwm": 0.5, "ring": 0.35, "ica": 0.15, "pixel": 0.0}
    elif num_orbs >= 4:
        weights = {"pwm": 0.35, "ring": 0.25, "ica": 0.2, "pixel": 0.2}
    else:
        weights = {"pwm": 0.2, "ring": 0.2, "ica": 0.2, "pixel": 0.4}

    fused = weights["pwm"] * pwm + weights["ring"] * ring
    if channel_ica is not None:
        fused = fused + weights["ica"] * channel_ica

    if weights["pixel"] > 0:
        pixel_ica = recover_duties_pixel_ica(intensity, num_orbs)
        if pixel_ica is not None:
            fused = fused + weights["pixel"] * pixel_ica

    fused = np.clip(fused, 0.0, None)
    total = float(fused.sum())
    if total <= 0:
        return ring
    return fused / total


def nearest_glyph(
    recovered_duties: np.ndarray,
    font: np.ndarray,
) -> tuple[int, float]:
    """Font index with smallest Fisher-Rao distance and duty fidelity."""
    best_idx = 0
    best_dist = float("inf")
    for g in range(font.shape[0]):
        dist = fisher_rao_distance(recovered_duties, font[g])
        if dist < best_dist:
            best_dist = dist
            best_idx = g
    fidelity = max(0.0, 1.0 - best_dist / np.pi)
    return best_idx, fidelity


def refine_glyph_from_candidates(
    recovered_duties: np.ndarray,
    font: np.ndarray,
    candidate_indices: list[int],
) -> tuple[int, float]:
    """Pick best glyph among ranked candidates using duty Fisher-Rao distance."""
    if not candidate_indices:
        return nearest_glyph(recovered_duties, font)
    best_idx = candidate_indices[0]
    best_dist = float("inf")
    for g in candidate_indices:
        dist = fisher_rao_distance(recovered_duties, font[g])
        if dist < best_dist:
            best_dist = dist
            best_idx = g
    return best_idx, max(0.0, 1.0 - best_dist / np.pi)