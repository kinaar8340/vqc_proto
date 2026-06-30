"""Orb crowding / OAM spectral interference metrics for stress testing."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from .stable_fonts import EmergentConstants
from .typehead import build_orbs_from_duties, synthesize_per_orb_intensity_maps


def mean_pairwise_intensity_correlation(maps: np.ndarray) -> float:
    """
    Mean absolute Pearson correlation over unique orb-intensity pairs.

    Higher values indicate stronger spatial overlap (crowded superposition).
    """
    n = maps.shape[0]
    if n < 2:
        return 0.0
    flats = [maps[i].ravel().astype(np.float64, copy=False) for i in range(n)]
    corrs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = flats[i], flats[j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            corrs.append(abs(float(pearsonr(a, b)[0])))
    return float(np.mean(corrs)) if corrs else 0.0


def effective_oam_modes(oam_weights: dict[int, complex]) -> float:
    """
    Participation-ratio effective mode count from projected |w_ℓ|² spectrum.

    Returns ~1 for a single dominant mode; rises as energy spreads across ℓ.
    """
    powers = np.array([abs(w) ** 2 for w in oam_weights.values()], dtype=np.float64)
    total = float(powers.sum())
    if total < 1e-18:
        return 0.0
    p = powers / total
    denom = float((p**2).sum())
    if denom < 1e-18:
        return 0.0
    return float(1.0 / denom)


def measure_orb_interference(
    font: np.ndarray,
    glyph_index: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_val: float,
    t_max: float,
    num_orbs: int,
    oam_weights: dict[int, complex],
    *,
    constants: EmergentConstants | None = None,
    w0: float = 1.0,
) -> tuple[float, float]:
    """Return ``(mean_pairwise_orb_corr, effective_oam_modes)``."""
    constants = constants or EmergentConstants()
    orbs = build_orbs_from_duties(font[glyph_index], num_orbs, constants)
    maps = synthesize_per_orb_intensity_maps(
        orbs, x_grid, y_grid, t_val, t_max, w0=w0
    )
    return (
        mean_pairwise_intensity_correlation(maps),
        effective_oam_modes(oam_weights),
    )