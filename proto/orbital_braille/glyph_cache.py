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
    field_centered: np.ndarray
    field_norms: np.ndarray


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
    field_centered = np.zeros((n_glyphs, flat_len), dtype=np.complex128)
    field_norms = np.zeros(n_glyphs, dtype=np.float64)
    orb_fields = np.zeros((n_glyphs, ny, nx), dtype=np.complex64)

    for g in range(n_glyphs):
        orbs = build_orbs_from_duties(font[g], num_orbs, constants)
        orb_field = synthesize_orb_field(orbs, x_grid, y_grid, t_val, t_max, w0=w0)
        orb_fields[g] = orb_field.astype(np.complex64, copy=False)
        flat_int = (np.abs(orb_field) ** 2).ravel()
        centered_int = flat_int - float(flat_int.mean())
        int_norm = float(np.linalg.norm(centered_int))
        if int_norm >= 1e-12:
            intensity_centered[g] = centered_int
            intensity_norms[g] = int_norm
        flat_field = orb_field.ravel()
        centered_field = flat_field - np.mean(flat_field)
        field_norm = float(np.linalg.norm(centered_field))
        if field_norm >= 1e-12:
            field_centered[g] = centered_field
            field_norms[g] = field_norm

    bank = GlyphTemplateBank(
        intensity_centered=intensity_centered,
        intensity_norms=intensity_norms,
        orb_fields=orb_fields,
        field_centered=field_centered,
        field_norms=field_norms,
    )
    _GLYPH_BANK_CACHE[key] = bank
    return bank


def glyph_field_coherence(
    field_mid: np.ndarray,
    template_field: np.ndarray,
) -> float:
    """Normalized complex inner product — degrades under phase noise (unlike intensity Pearson)."""
    received = field_mid.ravel()
    template = template_field.ravel()
    denom = float(np.linalg.norm(received) * np.linalg.norm(template))
    if denom < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(received, template)) / denom)


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


def rank_glyphs_by_orb_field(
    field_mid: np.ndarray,
    bank: GlyphTemplateBank,
    *,
    k: int = 24,
    intensity_weight: float = 0.35,
) -> list[int]:
    """
    Rank glyphs by blended intensity Pearson and complex-field coherence.

    Complex coherence is noise-sensitive under BMGL phase turbulence; intensity
    Pearson alone is flat when ``|E|²`` is unchanged.
    """
    intensity_mid = np.abs(field_mid) ** 2
    flat = intensity_mid.ravel().astype(np.float64, copy=False)
    centered_int = flat - float(flat.mean())
    int_norm = float(np.linalg.norm(centered_int))
    if int_norm < 1e-12:
        centered_int = centered_int + np.random.normal(0.0, 1e-10, centered_int.shape)
        int_norm = float(np.linalg.norm(centered_int)) + 1e-12

    received = field_mid.ravel()
    recv_norm = float(np.linalg.norm(received)) + 1e-12
    coh_weight = 1.0 - float(np.clip(intensity_weight, 0.0, 1.0))

    scores = np.full(bank.intensity_centered.shape[0], -np.inf, dtype=np.float64)
    for g in range(bank.intensity_centered.shape[0]):
        int_score = -np.inf
        if bank.intensity_norms[g] > 1e-12 and int_norm > 1e-12:
            int_score = float(
                bank.intensity_centered[g] @ centered_int
                / (bank.intensity_norms[g] * int_norm)
            )
        coh_score = 0.0
        if bank.field_norms[g] > 1e-12:
            coh_score = float(
                np.abs(np.vdot(received, bank.field_centered[g]))
                / (recv_norm * bank.field_norms[g])
            )
        scores[g] = intensity_weight * int_score + coh_weight * coh_score

    k = min(k, scores.shape[0])
    return np.argsort(scores)[-k:][::-1].tolist()


def adaptive_glyph_rank_k(num_orbs: int, *, base: int = 24) -> int:
    """Widen glyph search as orb count grows (ICA crosstalk rises)."""
    return max(base, min(64, base + max(0, num_orbs - 4) * 6))


def clear_glyph_template_cache() -> None:
    """Drop cached banks (tests / font rebuilds)."""
    _GLYPH_BANK_CACHE.clear()