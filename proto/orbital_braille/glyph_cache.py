"""Cached glyph orb-intensity templates for fast Level-2 manifold glyph ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stable_fonts import EmergentConstants
from .typehead import build_orbs_from_duties, synthesize_orb_field

_GLYPH_BANK_CACHE: dict[tuple, GlyphTemplateBank] = {}


@dataclass(frozen=True)
class GlyphTemplateBank:
    """Precomputed per-glyph orb intensity and complex fields at a fixed time slice."""

    intensity_centered: np.ndarray
    intensity_norms: np.ndarray
    orb_fields: np.ndarray


def _cache_key(
    font: np.ndarray,
    grid_shape: tuple[int, int],
    t_val: float,
    t_max: float,
    num_orbs: int,
    constants: EmergentConstants,
    w0: float,
) -> tuple:
    phases = constants.stable_phase_ladder(num_orbs)
    return (
        font.tobytes(),
        grid_shape,
        round(t_val, 15),
        round(t_max, 15),
        num_orbs,
        phases.tobytes(),
        round(w0, 12),
    )


def get_glyph_template_bank(
    font: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_val: float,
    t_max: float,
    num_orbs: int,
    constants: EmergentConstants | None = None,
    *,
    w0: float = 1.0,
) -> GlyphTemplateBank:
    """Return cached (or build) intensity templates and orb fields for all font glyphs."""
    constants = constants or EmergentConstants()
    ny, nx = x_grid.shape
    key = _cache_key(font, (ny, nx), t_val, t_max, num_orbs, constants, w0)
    cached = _GLYPH_BANK_CACHE.get(key)
    if cached is not None:
        return cached

    n_glyphs = font.shape[0]
    flat_len = ny * nx
    intensity_centered = np.zeros((n_glyphs, flat_len), dtype=np.float64)
    intensity_norms = np.zeros(n_glyphs, dtype=np.float64)
    orb_fields = np.zeros((n_glyphs, ny, nx), dtype=np.complex64)

    for g in range(n_glyphs):
        orbs = build_orbs_from_duties(font[g], num_orbs, constants)
        orb_field = synthesize_orb_field(orbs, x_grid, y_grid, t_val, t_max, w0=w0)
        orb_fields[g] = orb_field.astype(np.complex64, copy=False)
        flat = (np.abs(orb_field) ** 2).ravel()
        centered = flat - float(flat.mean())
        norm = float(np.linalg.norm(centered))
        if norm < 1e-12:
            continue
        intensity_centered[g] = centered
        intensity_norms[g] = norm

    bank = GlyphTemplateBank(
        intensity_centered=intensity_centered,
        intensity_norms=intensity_norms,
        orb_fields=orb_fields,
    )
    _GLYPH_BANK_CACHE[key] = bank
    return bank


def rank_glyphs_by_orb_intensity(
    intensity_mid: np.ndarray,
    bank: GlyphTemplateBank,
    *,
    k: int = 24,
) -> list[int]:
    """Return top ``k`` glyph indices by Pearson correlation (vectorized)."""
    flat = intensity_mid.ravel().astype(np.float64, copy=False)
    centered = flat - float(flat.mean())
    norm = float(np.linalg.norm(centered))
    if norm < 1e-12:
        centered = centered + np.random.normal(0.0, 1e-10, centered.shape)
        norm = float(np.linalg.norm(centered)) + 1e-12

    denom = bank.intensity_norms * norm
    valid = denom > 1e-12
    scores = np.full(bank.intensity_centered.shape[0], -np.inf, dtype=np.float64)
    scores[valid] = bank.intensity_centered[valid] @ centered / denom[valid]
    k = min(k, scores.shape[0])
    return np.argsort(scores)[-k:][::-1].tolist()


def clear_glyph_template_cache() -> None:
    """Drop cached banks (tests / font rebuilds)."""
    _GLYPH_BANK_CACHE.clear()