"""Laguerre-Gaussian mode generation and OAM projection."""

from __future__ import annotations

import numpy as np
from scipy.special import genlaguerre, factorial


def lg_radial(p: int, ell: int, rho: np.ndarray, w0: float) -> np.ndarray:
    """Radial LG_{p}^{|ell|} factor (p=0 donut modes by default)."""
    L = abs(ell)
    norm = np.sqrt(2 * factorial(L) / (np.pi * w0**2 * factorial(p)))
    rw = np.sqrt(2) * rho / w0
    lag = genlaguerre(p, L)(rw**2)
    radial = norm * (rw**L) * np.exp(-rw**2 / 2) * lag
    return radial


def lg_mode(ell: int, rho: np.ndarray, phi: np.ndarray, w0: float = 1.0, p: int = 0) -> np.ndarray:
    """Scalar LG mode field E(rho, phi) with helical phase exp(i ell phi)."""
    radial = lg_radial(p, ell, rho, w0)
    return radial * np.exp(1j * ell * phi)


def lg_mode_full(
    ell: int,
    x: np.ndarray,
    y: np.ndarray,
    w0: float = 1.0,
    p: int = 0,
) -> np.ndarray:
    """LG mode on a Cartesian grid."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return lg_mode(ell, rho, phi, w0=w0, p=p)


def project_oam_spectrum(
    field: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    ell_range: range | list[int],
    w0: float = 1.0,
) -> dict[int, complex]:
    """Project a complex field onto LG basis modes (inner product proxy)."""
    weights: dict[int, complex] = {}
    dr = rho[1, 0] - rho[0, 0] if rho.ndim == 2 else rho[1] - rho[0]
    for ell in ell_range:
        basis = lg_mode(ell, rho, phi, w0=w0)
        if field.ndim == 2:
            integrand = field * np.conj(basis) * rho
            weights[ell] = np.sum(integrand) * dr**2
        else:
            weights[ell] = np.vdot(field.flatten(), basis.flatten())
    return weights


def dominant_ell(weights: dict[int, complex]) -> int:
    """Return ell with largest projection magnitude."""
    return max(weights, key=lambda k: abs(weights[k]))