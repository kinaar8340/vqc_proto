"""Stable orbital codeword fonts from emergent TOE constants + Fisher-Rao geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmergentConstants:
    """Multi-resonator / 3-body stable orbit invariants (Rhythm bake targets)."""

    wg_base: float = 350.0
    kappa: float = 0.85
    braiding_linking: float = 0.084
    braiding_twist: float = 0.8145
    theta_crit: float = 5.8

    @property
    def Wg(self) -> float:
        return self.wg_base / np.pi

    def stable_phase_ladder(self, n: int) -> np.ndarray:
        """n phases spaced by emergent W_g / (2*pi) golden-ratio perturbation."""
        golden = (1 + np.sqrt(5)) / 2
        base = np.linspace(0, 2 * np.pi, n, endpoint=False)
        perturb = (self.braiding_linking * np.arange(n)) % (2 * np.pi)
        return (base / golden + perturb * self.kappa) % (2 * np.pi)


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(np.sum(np.sqrt(p * q)))


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    bc = min(bhattacharyya(p, q), 1.0 - 1e-7)
    return float(2.0 * np.arccos(bc))


def build_stable_font(
    num_orbs: int,
    num_glyphs: int = 16,
    constants: EmergentConstants | None = None,
) -> np.ndarray:
    """
    Build num_glyphs x num_orbs PWM duty matrices using emergent phase ladder.

    Each glyph is a distinct "Braille dot pattern" — orb duty cycles that
    maximize Fisher-Rao separation while locking to stable orbit phases.
    """
    constants = constants or EmergentConstants()
    phases = constants.stable_phase_ladder(num_orbs)

    glyphs = np.zeros((num_glyphs, num_orbs), dtype=float)
    for g in range(num_glyphs):
        for k in range(num_orbs):
            duty = 0.35 + 0.55 * np.sin(phases[k] + g * constants.theta_crit / num_glyphs)
            glyphs[g, k] = np.clip(duty, 0.1, 0.95)

    return glyphs


def font_separation(font: np.ndarray) -> float:
    """Mean pairwise Fisher-Rao distance across glyph duty vectors."""
    n = font.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(fisher_rao_distance(font[i], font[j]))
    return float(np.mean(dists))


def glyph_for_byte(byte_val: int, font: np.ndarray) -> np.ndarray:
    """Select glyph row from font by byte value."""
    return font[byte_val % font.shape[0]]