"""Fisher-Rao font optimizer — maximize glyph separation under emergent invariants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stable_fonts import EmergentConstants, fisher_rao_distance, font_separation


@dataclass
class FontOptResult:
    font: np.ndarray
    mean_separation: float
    min_separation: float
    wg_penalty: float
    braiding_penalty: float


def _hopf_penalty(phases: np.ndarray, constants: EmergentConstants) -> float:
    """Penalize phase ladders that drift from W_g = wg_base/pi spacing."""
    expected_step = 2 * np.pi / constants.Wg
    if len(phases) < 2:
        return 0.0
    steps = np.diff(np.sort(phases))
    return float(np.mean(np.abs(steps - expected_step)))


def _braiding_penalty(duties: np.ndarray, constants: EmergentConstants) -> float:
    """Penalize duty vectors far from braiding-linking stable manifold."""
    mean_duty = float(np.mean(duties))
    target = 0.35 + constants.braiding_linking
    return abs(mean_duty - target)


def optimize_font(
    num_orbs: int = 4,
    num_glyphs: int = 16,
    constants: EmergentConstants | None = None,
    n_iter: int = 200,
    seed: int = 42,
) -> FontOptResult:
    """
    Gradient-free search for PWM duty matrix maximizing Fisher-Rao separation
    while respecting emergent TOE invariants (350/pi, braiding 0.084).
    """
    constants = constants or EmergentConstants()
    rng = np.random.default_rng(seed)
    phases = constants.stable_phase_ladder(num_orbs)

    best_font = np.zeros((num_glyphs, num_orbs))
    for g in range(num_glyphs):
        for k in range(num_orbs):
            duty = 0.35 + 0.55 * np.sin(phases[k] + g * constants.theta_crit / num_glyphs)
            best_font[g, k] = np.clip(duty, 0.1, 0.95)

    best_sep = font_separation(best_font)
    best_wg = _hopf_penalty(phases, constants)
    best_braid = np.mean([_braiding_penalty(best_font[g], constants) for g in range(num_glyphs)])

    for _ in range(n_iter):
        candidate = best_font.copy()
        g = rng.integers(0, num_glyphs)
        k = rng.integers(0, num_orbs)
        candidate[g, k] = np.clip(candidate[g, k] + rng.normal(0, 0.05), 0.1, 0.95)

        row_sum = candidate[g].sum()
        if row_sum > 0:
            candidate[g] /= row_sum

        sep = font_separation(candidate)
        wg_p = _hopf_penalty(phases, constants)
        braid_p = _braiding_penalty(candidate[g], constants)
        loss = -sep + 0.5 * wg_p + 0.3 * braid_p

        best_loss = -best_sep + 0.5 * best_wg + 0.3 * best_braid
        if loss < best_loss:
            best_font = candidate
            best_sep = sep
            best_wg = wg_p
            best_braid = braid_p

    min_sep = float("inf")
    n = best_font.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            min_sep = min(min_sep, fisher_rao_distance(best_font[i], best_font[j]))

    return FontOptResult(
        font=best_font,
        mean_separation=best_sep,
        min_separation=min_sep if min_sep != float("inf") else 0.0,
        wg_penalty=best_wg,
        braiding_penalty=best_braid,
    )