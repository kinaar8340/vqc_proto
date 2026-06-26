"""Phase-only SLM patterns for a 4-orb virtual typehead (no mechanical rotation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lg_modes import lg_mode
from .typehead import OrbConfig


@dataclass
class SLMConfig:
    resolution: int = 512
    pitch_um: float = 8.0
    wavelength_nm: float = 1550.0
    extent_mm: float = 4.0


def orb_phase_at_time(orb: OrbConfig, t: float, t_max: float) -> float:
    pwm_on = (np.sin(2 * np.pi * orb.omega * t / t_max) + 1) / 2 < orb.pwm_duty
    gate = 1.0 if pwm_on else 0.15
    return gate * (orb.omega * t + orb.phase0)


def virtual_orb_field(
    orbs: list[OrbConfig],
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    t_max: float,
    w0: float = 1.0,
) -> np.ndarray:
    """Superpose Gaussian spots at instantaneous virtual orbital positions."""
    field = np.zeros_like(x, dtype=complex)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    lg_carrier = lg_mode(1, rho, phi, w0=w0)

    for orb in orbs:
        theta = orb.phase0 + orb.omega * t
        x0 = orb.radius * np.cos(theta)
        y0 = orb.radius * np.sin(theta)
        sigma = w0 * 0.35
        gauss = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        helical = np.exp(1j * orb.ell * np.arctan2(y - y0, x - x0))
        phase = orb_phase_at_time(orb, t, t_max)
        field += orb.amplitude * gauss * helical * np.exp(1j * phase)

    return field * lg_carrier


def slm_phase_pattern(
    orbs: list[OrbConfig],
    t: float,
    t_max: float,
    cfg: SLMConfig | None = None,
    w0: float = 1.0,
) -> np.ndarray:
    """
    Generate SLM phase map in radians, wrapped to [-pi, pi].

    Hardware: load as 8-bit or 16-bit phase hologram after scaling to device range.
    """
    cfg = cfg or SLMConfig()
    half = cfg.extent_mm / 2
    x = np.linspace(-half, half, cfg.resolution)
    y = np.linspace(-half, half, cfg.resolution)
    X, Y = np.meshgrid(x, y)
    field = virtual_orb_field(orbs, X, Y, t, t_max, w0=w0)
    phase = np.angle(field)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def slm_phase_sequence(
    orbs: list[OrbConfig],
    num_frames: int,
    t_max: float,
    cfg: SLMConfig | None = None,
) -> np.ndarray:
    """Return (num_frames, H, W) phase stack for SLM playback."""
    cfg = cfg or SLMConfig()
    t = np.linspace(0, t_max, num_frames)
    stack = np.zeros((num_frames, cfg.resolution, cfg.resolution), dtype=float)
    for i, ti in enumerate(t):
        stack[i] = slm_phase_pattern(orbs, ti, t_max, cfg=cfg)
    return stack


def save_phase_hologram(phase: np.ndarray, path: str, bit_depth: int = 8) -> None:
    """Save phase map as grayscale PNG for SLM upload."""
    import matplotlib.pyplot as plt

    normalized = (phase + np.pi) / (2 * np.pi)
    if bit_depth == 8:
        img = (normalized * 255).astype(np.uint8)
    else:
        img = (normalized * 65535).astype(np.uint16)

    plt.imsave(path, img, cmap="gray", vmin=0, vmax=255 if bit_depth == 8 else 65535)