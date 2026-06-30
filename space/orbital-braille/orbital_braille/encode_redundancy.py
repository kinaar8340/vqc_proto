"""Transmit-side [[3,1,3]] repetition encoding for Orbital Braille."""

from __future__ import annotations

import numpy as np

QEC_REPS = 3


def effective_num_times(num_times: int, reps: int = QEC_REPS) -> int:
    """Pad frame count up to a multiple of ``reps`` for triplet codewords."""
    if num_times <= 0:
        return reps
    return ((num_times + reps - 1) // reps) * reps


def repeat_triplets_along_time(
    field_time: np.ndarray,
    intensity_time: np.ndarray,
    *,
    reps: int = QEC_REPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse each group of ``reps`` consecutive frames to one logical snapshot,
    then broadcast it back so every triplet is an exact repetition codeword.
    """
    n_t, ny, nx = field_time.shape
    n_eff = effective_num_times(n_t, reps)
    if n_eff > n_t:
        pad = n_eff - n_t
        field_time = np.concatenate([field_time, np.tile(field_time[-1:], (pad, 1, 1))], axis=0)
        intensity_time = np.concatenate(
            [intensity_time, np.tile(intensity_time[-1:], (pad, 1, 1))], axis=0
        )

    n_logical = n_eff // reps
    field_out = np.empty_like(field_time)
    intensity_out = np.empty_like(intensity_time)

    for li in range(n_logical):
        sl = slice(li * reps, (li + 1) * reps)
        # Centre frame of each triplet is the logical snapshot.
        centre = li * reps + reps // 2
        field_out[sl] = field_time[centre]
        intensity_out[sl] = intensity_time[centre]

    return field_out, intensity_out


def repeat_triplets_1d(values: np.ndarray, *, reps: int = QEC_REPS) -> np.ndarray:
    """Repeat 1-D samples (e.g. pulse envelope) in ``reps``-fold triplets."""
    values = np.asarray(values, dtype=float)
    n_eff = effective_num_times(values.size, reps)
    if n_eff > values.size:
        values = np.concatenate([values, np.full(n_eff - values.size, values[-1])])

    n_logical = n_eff // reps
    out = np.empty(n_eff, dtype=float)
    for li in range(n_logical):
        centre = li * reps + reps // 2
        out[li * reps : (li + 1) * reps] = values[centre]
    return out


def triplet_codeword_check(intensity_time: np.ndarray, *, reps: int = QEC_REPS, atol: float = 1e-12) -> bool:
    """Return True when every time triplet is an exact repetition codeword."""
    n_eff = effective_num_times(intensity_time.shape[0], reps)
    data = intensity_time
    if data.shape[0] < n_eff:
        data = np.concatenate([data, np.tile(data[-1:], (n_eff - data.shape[0], 1, 1))], axis=0)
    n_logical = n_eff // reps
    for li in range(n_logical):
        sl = slice(li * reps, (li + 1) * reps)
        triplet = data[sl]
        if not np.allclose(triplet[0], triplet[1], atol=atol) or not np.allclose(
            triplet[1], triplet[2], atol=atol
        ):
            return False
    return True