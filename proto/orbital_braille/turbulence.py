"""Free-space turbulence: Kolmogorov scintillation + pointing jitter on top of p-wave BMGL."""

from __future__ import annotations

import numpy as np

from .altermagnetic import PWaveBMGL, apply_turbulence


def kolmogorov_phase_screen(
    shape: tuple[int, ...],
    r0_m: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 2D Kolmogorov phase screen (Fried parameter r0)."""
    rng = rng or np.random.default_rng()
    ny, nx = shape[-2], shape[-1]
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1e-12

    cn2_proxy = 0.023 * r0_m ** (-5 / 3)
    psd = cn2_proxy * k ** (-11 / 3)
    psd[0, 0] = 0

    noise = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    screen = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(psd)))
    screen -= screen.mean()
    return screen


def pointing_jitter(
    field: np.ndarray,
    sigma_pixels: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shift field by sub-pixel pointing jitter (LEO link proxy)."""
    from scipy.ndimage import shift

    rng = rng or np.random.default_rng()
    dy, dx = rng.normal(0, sigma_pixels, 2)
    if field.ndim == 2:
        return shift(field, (dy, dx), mode="nearest")
    out = np.zeros_like(field)
    for i in range(field.shape[0]):
        out[i] = shift(field[i], (dy, dx), mode="nearest")
    return out


def apply_free_space_channel(
    field: np.ndarray,
    bmgl: PWaveBMGL,
    phi: np.ndarray | None = None,
    r0_m: float = 0.1,
    pointing_sigma: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Full channel: BMGL p-wave inhibition + Kolmogorov + pointing jitter."""
    rng = rng or np.random.default_rng()

    if field.ndim == 2:
        screens = [kolmogorov_phase_screen(field.shape, r0_m=r0_m, rng=rng)]
        out = field * np.exp(1j * screens[0])
    else:
        out = np.zeros_like(field)
        for i in range(field.shape[0]):
            screen = kolmogorov_phase_screen(field[i].shape, r0_m=r0_m, rng=rng)
            out[i] = field[i] * np.exp(1j * screen)

    out = apply_turbulence(out, bmgl, phi=phi, rng=rng)
    out = pointing_jitter(out, sigma_pixels=pointing_sigma, rng=rng)
    return out