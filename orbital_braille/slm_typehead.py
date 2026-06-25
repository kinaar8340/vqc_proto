"""Phase-only SLM patterns for 4-orb virtual typehead (hardware-ready export)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from .lg_modes import lg_mode
from .quaternion_codec import Quaternion
from .typehead import OrbConfig


@dataclass
class SLMDevicePreset:
    """Common phase-only SLM parameters."""

    name: str
    resolution_x: int
    resolution_y: int
    pitch_um: float
    wavelength_nm: float = 1550.0
    bit_depth: int = 8
    phase_max_rad: float = 2 * np.pi
    notes: str = ""

    @property
    def resolution(self) -> int:
        return self.resolution_x if self.resolution_x == self.resolution_y else max(
            self.resolution_x, self.resolution_y
        )


SLM_PRESETS: dict[str, SLMDevicePreset] = {
    "generic_512": SLMDevicePreset(
        name="generic_512",
        resolution_x=512,
        resolution_y=512,
        pitch_um=8.0,
        notes="Default simulation grid; good for algorithm validation.",
    ),
    "holoeye_pluto_2": SLMDevicePreset(
        name="holoeye_pluto_2",
        resolution_x=1920,
        resolution_y=1080,
        pitch_um=8.0,
        wavelength_nm=1550.0,
        bit_depth=8,
        notes="Holoeye PLUTO-2 class (1920×1080, 8 µm). Use 8-bit BMP upload.",
    ),
    "meadowlark_512": SLMDevicePreset(
        name="meadowlark_512",
        resolution_x=512,
        resolution_y=512,
        pitch_um=15.0,
        wavelength_nm=1550.0,
        bit_depth=16,
        notes="Meadowlark 512×512 high-resolution phase (16-bit).",
    ),
    "thorlabs_1080p": SLMDevicePreset(
        name="thorlabs_1080p",
        resolution_x=1920,
        resolution_y=1080,
        pitch_um=6.4,
        wavelength_nm=1550.0,
        bit_depth=8,
        notes="1080p LCOS class (Exulus / similar).",
    ),
}


@dataclass
class SLMConfig:
    resolution_x: int = 512
    resolution_y: int = 512
    pitch_um: float = 8.0
    wavelength_nm: float = 1550.0
    extent_mm: float = 4.0
    bit_depth: int = 8
    phase_wrap: Literal["0_2pi", "neg_pi_pi"] = "0_2pi"
    w0_mm: float = 0.8

    @classmethod
    def from_preset(cls, preset: str | SLMDevicePreset, extent_mm: float = 4.0) -> SLMConfig:
        p = SLM_PRESETS[preset] if isinstance(preset, str) else preset
        return cls(
            resolution_x=p.resolution_x,
            resolution_y=p.resolution_y,
            pitch_um=p.pitch_um,
            wavelength_nm=p.wavelength_nm,
            bit_depth=p.bit_depth,
            extent_mm=extent_mm,
        )


@dataclass
class SLMPackageMeta:
    """Sidecar metadata for bench reproduction."""

    payload: str
    num_orbs: int
    frames: int
    t_max_ns: float
    wavelength_nm: float
    device_preset: str
    quaternion: dict[str, float]
    glyph_duties: list[float]
    orbs: list[dict]
    font_separation_rad: float | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generator: str = "vqc_proto/generate_slm_holograms.py"


def orb_phase_at_time(orb: OrbConfig, t: float, t_max: float) -> float:
    pwm_on = (np.sin(2 * np.pi * orb.omega * t / t_max) + 1) / 2 < orb.pwm_duty
    gate = 1.0 if pwm_on else 0.15
    return gate * (orb.omega * t + orb.phase0)


def _slm_grid(cfg: SLMConfig) -> tuple[np.ndarray, np.ndarray]:
    half = cfg.extent_mm / 2
    x = np.linspace(-half, half, cfg.resolution_x)
    y = np.linspace(-half, half, cfg.resolution_y)
    return np.meshgrid(x, y)


def virtual_orb_field(
    orbs: list[OrbConfig],
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    t_max: float,
    w0: float = 1.0,
    quat_phase: float = 0.0,
) -> np.ndarray:
    """Superpose Gaussian spots at virtual orbital positions × LG carrier."""
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

    return field * lg_carrier * np.exp(1j * quat_phase)


def slm_phase_pattern(
    orbs: list[OrbConfig],
    t: float,
    t_max: float,
    cfg: SLMConfig | None = None,
    quat_phase: float = 0.0,
) -> np.ndarray:
    """Phase map in radians for one animation frame."""
    cfg = cfg or SLMConfig()
    X, Y = _slm_grid(cfg)
    field = virtual_orb_field(orbs, X, Y, t, t_max, w0=cfg.w0_mm, quat_phase=quat_phase)
    phase = np.angle(field)
    if cfg.phase_wrap == "0_2pi":
        return np.mod(phase, 2 * np.pi)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def slm_target_intensity(
    orbs: list[OrbConfig],
    cfg: SLMConfig,
    t: float,
    t_max: float,
) -> np.ndarray:
    """Target far-field intensity proxy (for GS refinement)."""
    X, Y = _slm_grid(cfg)
    field = virtual_orb_field(orbs, X, Y, t, t_max, w0=cfg.w0_mm)
    return np.abs(field) ** 2


def gerchberg_saxton(
    target_amp: np.ndarray,
    n_iter: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """
    Gerchberg-Saxton phase retrieval for SLM plane.

    Returns phase in [0, 2π) at the SLM plane that approximates target_amp
    in the Fourier (far-field) plane.
    """
    rng = np.random.default_rng(seed)
    ny, nx = target_amp.shape
    target_amp = target_amp / (target_amp.max() + 1e-12)

    slm_phase = rng.uniform(0, 2 * np.pi, (ny, nx))
    f_amp = np.ones_like(target_amp)

    for _ in range(n_iter):
        slm_field = np.exp(1j * slm_phase)
        far = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
        far = target_amp * np.exp(1j * np.angle(far))
        slm_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far)))
        slm_phase = np.angle(slm_field)
        slm_phase = np.mod(slm_phase, 2 * np.pi)

        far_check = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * slm_phase))))
        f_amp = np.abs(far_check)

    return slm_phase


def slm_phase_sequence(
    orbs: list[OrbConfig],
    num_frames: int,
    t_max: float,
    cfg: SLMConfig | None = None,
    quat_phase: float = 0.0,
    use_gs: bool = False,
    gs_iter: int = 24,
) -> np.ndarray:
    """Return (num_frames, H, W) phase stack in radians."""
    cfg = cfg or SLMConfig()
    t = np.linspace(0, t_max, num_frames)
    stack = np.zeros((num_frames, cfg.resolution_y, cfg.resolution_x), dtype=float)
    for i, ti in enumerate(t):
        if use_gs:
            target = slm_target_intensity(orbs, cfg, ti, t_max)
            stack[i] = gerchberg_saxton(target, n_iter=gs_iter, seed=i)
        else:
            stack[i] = slm_phase_pattern(orbs, ti, t_max, cfg=cfg, quat_phase=quat_phase)
    return stack


def phase_to_levels(phase: np.ndarray, bit_depth: int = 8, wrap: str = "0_2pi") -> np.ndarray:
    """Convert radians to SLM gray levels."""
    if wrap == "0_2pi":
        p = np.mod(phase, 2 * np.pi)
        norm = p / (2 * np.pi)
    else:
        p = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        norm = (p + np.pi) / (2 * np.pi)
    max_val = (1 << bit_depth) - 1
    return np.round(norm * max_val).astype(np.uint16 if bit_depth > 8 else np.uint8)


def save_phase_hologram(
    phase: np.ndarray,
    path: str | Path,
    bit_depth: int = 8,
    wrap: str = "0_2pi",
) -> None:
    """Save phase map as hardware-ready grayscale image."""
    from PIL import Image

    levels = phase_to_levels(phase, bit_depth=bit_depth, wrap=wrap)
    path = Path(path)
    if bit_depth > 8:
        Image.fromarray(levels, mode="I;16").save(path)
    else:
        Image.fromarray(levels, mode="L").save(path)


def save_phase_raw(phase: np.ndarray, path: str | Path, bit_depth: int = 8, wrap: str = "0_2pi") -> None:
    """Save raw little-endian phase levels for custom SLM drivers."""
    levels = phase_to_levels(phase, bit_depth=bit_depth, wrap=wrap)
    Path(path).write_bytes(levels.astype(np.uint16 if bit_depth > 8 else np.uint8).tobytes())


def save_preview_montage(stack: np.ndarray, path: str | Path, max_frames: int = 8) -> None:
    """Save tiled preview of phase frames for quick visual check."""
    import matplotlib.pyplot as plt

    n = min(max_frames, stack.shape[0])
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            ax.imshow(stack[i], cmap="twilight", vmin=0, vmax=2 * np.pi)
            ax.set_title(f"frame {i}")
        ax.axis("off")
    fig.suptitle("SLM phase sequence — 4-orb virtual typehead", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def orb_to_dict(orb: OrbConfig) -> dict:
    return {
        "radius": orb.radius,
        "omega": orb.omega,
        "ell": orb.ell,
        "amplitude": orb.amplitude,
        "phase0": orb.phase0,
        "pwm_duty": orb.pwm_duty,
    }


def write_manifest(path: Path, meta: SLMPackageMeta) -> None:
    path.write_text(json.dumps(asdict(meta), indent=2))


def export_hologram_package(
    orbs: list[OrbConfig],
    t_max: float,
    out_dir: Path,
    cfg: SLMConfig,
    payload: str,
    quaternion: Quaternion,
    glyph_duties: np.ndarray,
    num_frames: int = 32,
    device_preset: str = "generic_512",
    use_gs: bool = False,
    gs_iter: int = 24,
    export_raw: bool = True,
    font_separation: float | None = None,
) -> dict:
    """
    Export complete SLM upload package: frames, manifest, preview, LUT note.

    Returns summary dict for CLI printing.
    """
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    quat_phase = float(quaternion.w * np.pi / 2)
    stack = slm_phase_sequence(
        orbs,
        num_frames,
        t_max,
        cfg=cfg,
        quat_phase=quat_phase,
        use_gs=use_gs,
        gs_iter=gs_iter,
    )

    ext = "png" if cfg.bit_depth <= 8 else "tiff"
    for i in range(num_frames):
        save_phase_hologram(
            stack[i],
            frames_dir / f"phase_{i:04d}.{ext}",
            bit_depth=cfg.bit_depth,
            wrap=cfg.phase_wrap,
        )
        if export_raw:
            save_phase_raw(
                stack[i],
                frames_dir / f"phase_{i:04d}.raw",
                bit_depth=cfg.bit_depth,
                wrap=cfg.phase_wrap,
            )

    meta = SLMPackageMeta(
        payload=payload,
        num_orbs=len(orbs),
        frames=num_frames,
        t_max_ns=float(t_max * 1e9),
        wavelength_nm=cfg.wavelength_nm,
        device_preset=device_preset,
        quaternion={
            "w": quaternion.w,
            "x": quaternion.x,
            "y": quaternion.y,
            "z": quaternion.z,
        },
        glyph_duties=[float(x) for x in glyph_duties],
        orbs=[orb_to_dict(o) for o in orbs],
        font_separation_rad=font_separation,
    )
    write_manifest(out_dir / "manifest.json", meta)

    lut_note = out_dir / "LUT_calibration.txt"
    lut_note.write_text(
        f"""SLM phase LUT calibration notes
================================
Device preset: {device_preset}
Bit depth: {cfg.bit_depth}
Phase mapping: {cfg.phase_wrap}
  0 gray   -> 0 rad
  max gray -> 2*pi rad ({cfg.bit_depth}-bit full scale)

Measure at {cfg.wavelength_nm} nm:
1. Upload uniform ramps (0, 64, 128, 192, 255 for 8-bit).
2. Record interferometric phase shift per gray level.
3. Replace linear mapping in your SLM driver if deviation > 5%.

Frame timing: {num_frames} frames over {meta.t_max_ns:.3f} ns
  -> {meta.t_max_ns / num_frames:.3f} ns per frame (adjust SLM refresh to match chirp)
"""
    )

    save_preview_montage(stack, out_dir / "preview_montage.png")
    np.save(out_dir / "phase_stack.npy", stack)

    return {
        "out_dir": str(out_dir),
        "frames": num_frames,
        "resolution": f"{cfg.resolution_x}×{cfg.resolution_y}",
        "bit_depth": cfg.bit_depth,
        "use_gs": use_gs,
        "device": device_preset,
    }