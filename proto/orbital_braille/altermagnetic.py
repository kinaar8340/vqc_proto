"""p-wave altermagnetic BMGL turbulence suppression (from VQC encode_decode)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PWaveBMGL:
    """p-wave altermagnetic parameters for BMGL error inhibition."""

    lambda_soc: float = 0.4
    p_odd_parity: float = 1.2
    gamma_1: float = 1.5
    detune_scale: float = 0.01
    alpha_chemical: float = 0.015

    @property
    def inhibition_boost(self) -> float:
        return 1.0 + (self.lambda_soc / self.p_odd_parity) * (self.gamma_1 - 1.0)

    @property
    def effective_inhibition(self) -> float:
        return self.gamma_1 * self.inhibition_boost


def apply_turbulence(
    field: np.ndarray,
    bmgl: PWaveBMGL,
    phi: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply phase noise with p-wave BMGL inhibition."""
    rng = rng or np.random.default_rng()
    phase_noise = rng.normal(0, bmgl.detune_scale, field.shape)
    phase_noise /= bmgl.effective_inhibition

    if phi is not None:
        sin_phi = np.abs(np.sin(phi))
        while sin_phi.ndim < phase_noise.ndim:
            sin_phi = sin_phi[np.newaxis, ...]
        mod_factor = np.clip(1.0 + bmgl.p_odd_parity * sin_phi, 1.0, None)
        phase_noise /= mod_factor

    if np.iscomplexobj(field):
        return field * np.exp(1j * phase_noise)
    return field * np.cos(phase_noise)


def repetition_qec(data: np.ndarray, reps: int = 16, error_rate: float = 0.015, rng=None) -> np.ndarray:
    """Majority-vote repetition code proxy (16-qubit QEC)."""
    rng = rng or np.random.default_rng()
    errors = rng.binomial(reps, error_rate, data.shape)
    correction = rng.normal(0, error_rate / reps, data.shape) * (errors % 2)
    return data + correction