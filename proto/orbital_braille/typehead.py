"""Multi-orb orbital typehead encoder — Selectric-ball analog for VQC shards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import chirp, welch

from .altermagnetic import PWaveBMGL, apply_turbulence, noise_level_to_scale
from .encode_redundancy import (
    QEC_REPS,
    effective_num_times,
    repeat_triplets_1d,
    repeat_triplets_along_time,
)
from .lg_modes import lg_mode
from .quaternion_codec import Quaternion, encode_shard, rodrigues_rotation
from .quaternion_oam import IMPRINT_SCALE, PHI_SCALE, encode_imprint_field
from .stable_fonts import EmergentConstants, build_stable_font, glyph_for_byte


@dataclass
class OrbConfig:
    """Single orbiting point-source on the typehead."""

    radius: float
    omega: float
    ell: int
    amplitude: float = 1.0
    phase0: float = 0.0
    pwm_duty: float = 0.5


@dataclass
class TypeheadConfig:
    """Full typehead geometry."""

    num_orbs: int = 4
    grid_size: int = 80
    extent: float = 2.5
    w0: float = 1.0
    num_times: int = 64
    pulse_duration_ns: float = 1.0
    f_start_hz: float = 1e9
    f_end_hz: float = 10e9
    chirp_rate: float = 0.5
    bmgl: PWaveBMGL = field(default_factory=PWaveBMGL)
    constants: EmergentConstants = field(default_factory=EmergentConstants)
    qec_reps: int = QEC_REPS
    qec_transmit_redundancy: bool = True


def build_orbs_from_duties(
    glyph_duties: np.ndarray,
    num_orbs: int,
    constants: EmergentConstants | None = None,
) -> list[OrbConfig]:
    """Reconstruct orb configs from recovered PWM duties (decode-side)."""
    constants = constants or EmergentConstants()
    phases = constants.stable_phase_ladder(num_orbs)
    duties = np.asarray(glyph_duties, dtype=float).ravel()
    orbs: list[OrbConfig] = []
    for k in range(num_orbs):
        duty = float(duties[k]) if k < duties.size else 0.5
        orbs.append(
            OrbConfig(
                radius=0.4 + 0.15 * k,
                omega=1.0 + 0.3 * k,
                ell=k if k % 2 == 0 else -(k // 2 + 1),
                amplitude=0.8 + 0.1 * k,
                phase0=phases[k],
                pwm_duty=duty,
            )
        )
    return orbs


def synthesize_orb_field(
    orbs: list[OrbConfig],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_val: float,
    t_max: float,
    *,
    w0: float = 1.0,
) -> np.ndarray:
    """Sum PWM-gated point sources at time ``t_val`` (matches encode geometry)."""
    field = np.zeros_like(x_grid, dtype=complex)
    sigma = w0 * 0.35
    for orb in orbs:
        theta = orb.phase0 + orb.omega * t_val
        x0 = orb.radius * np.cos(theta)
        y0 = orb.radius * np.sin(theta)
        pwm_on = (np.sin(2 * np.pi * orb.omega * t_val / t_max) + 1) / 2 < orb.pwm_duty
        gate = 1.0 if pwm_on else 0.15
        phase = orb.omega * t_val + orb.phase0
        amp = orb.amplitude * gate * np.exp(1j * phase)
        gauss = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma**2))
        helical = np.exp(1j * orb.ell * np.arctan2(y_grid - y0, x_grid - x0))
        field += amp * gauss * helical
    return field


def synthesize_per_orb_intensity_maps(
    orbs: list[OrbConfig],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_val: float,
    t_max: float,
    *,
    w0: float = 1.0,
) -> np.ndarray:
    """Per-orb |field|² maps at ``t_val``, shape ``(n_orbs, ny, nx)``."""
    maps = np.zeros((len(orbs), *x_grid.shape), dtype=np.float64)
    sigma = w0 * 0.35
    for i, orb in enumerate(orbs):
        theta = orb.phase0 + orb.omega * t_val
        x0 = orb.radius * np.cos(theta)
        y0 = orb.radius * np.sin(theta)
        pwm_on = (np.sin(2 * np.pi * orb.omega * t_val / t_max) + 1) / 2 < orb.pwm_duty
        gate = 1.0 if pwm_on else 0.15
        phase = orb.omega * t_val + orb.phase0
        amp = orb.amplitude * gate * np.exp(1j * phase)
        gauss = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma**2))
        helical = np.exp(1j * orb.ell * np.arctan2(y_grid - y0, x_grid - x0))
        maps[i] = np.abs(amp * gauss * helical) ** 2
    return maps


@dataclass
class EncodeResult:
    """Encoded field + metadata for decoding."""

    field_time: np.ndarray
    intensity_time: np.ndarray
    pulse: np.ndarray
    spectral_shards: np.ndarray
    freqs: np.ndarray
    quaternion: Quaternion
    orbs: list[OrbConfig]
    glyph_duties: np.ndarray
    t: np.ndarray
    rho: np.ndarray
    phi: np.ndarray
    payload: bytes
    qec_reps: int = QEC_REPS


class OrbitalTypehead:
    """
    Dynamic multi-orb modulator: N co-rotating laser spots whose interference
    creates pyramidal pulses, spectral shards, and OAM helical content.

    Analog: IBM Selectric typeball — angular positioning selects character;
    here orbital phases + PWM duties select shard/glyph.
    """

    def __init__(self, config: TypeheadConfig | None = None, seed: int = 42):
        self.config = config or TypeheadConfig()
        self.rng = np.random.default_rng(seed)
        self.font = build_stable_font(
            self.config.num_orbs,
            num_glyphs=256,
            constants=self.config.constants,
        )
        self._setup_grid()

    def _setup_grid(self) -> None:
        cfg = self.config
        x = np.linspace(-cfg.extent, cfg.extent, cfg.grid_size)
        y = np.linspace(-cfg.extent, cfg.extent, cfg.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.rho = np.sqrt(self.X**2 + self.Y**2)
        self.phi = np.arctan2(self.Y, self.X)

    def _build_orbs(self, glyph_duties: np.ndarray) -> list[OrbConfig]:
        """Map glyph PWM duties to orbiting sources with distinct ell charges."""
        return build_orbs_from_duties(
            glyph_duties, self.config.num_orbs, self.config.constants
        )

    def _orb_position(self, orb: OrbConfig, t: float) -> tuple[float, float]:
        theta = orb.phase0 + orb.omega * t
        x = orb.radius * np.cos(theta)
        y = orb.radius * np.sin(theta)
        return x, y

    def _orb_amplitude(self, orb: OrbConfig, t: float, t_max: float) -> complex:
        """PWM-gated complex amplitude with helical OAM phase."""
        pwm_on = (np.sin(2 * np.pi * orb.omega * t / t_max) + 1) / 2 < orb.pwm_duty
        gate = 1.0 if pwm_on else 0.15
        phase = orb.omega * t + orb.phase0
        return orb.amplitude * gate * np.exp(1j * phase)

    def _point_source_field(
        self,
        x0: float,
        y0: float,
        amp: complex,
        ell: int,
    ) -> np.ndarray:
        """Gaussian point source modulated by LG helical envelope."""
        sigma = self.config.w0 * 0.35
        gauss = np.exp(-((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (2 * sigma**2))
        helical = np.exp(1j * ell * np.arctan2(self.Y - y0, self.X - x0))
        return amp * gauss * helical

    def pyramidal_pulse(self, t: np.ndarray) -> np.ndarray:
        """Triangular FM envelope (pyramidal pulse) across the time window."""
        cfg = self.config
        return chirp(
            t,
            f0=cfg.f_start_hz,
            f1=cfg.f_end_hz,
            t1=t[-1],
            method="linear",
        )

    def encode(self, payload: bytes | str) -> EncodeResult:
        """Encode payload into superposed orbital field + spectral shards."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        q = encode_shard(payload)
        glyph = glyph_for_byte(payload[0] if payload else 0, self.font)
        orbs = self._build_orbs(glyph)

        cfg = self.config
        reps = cfg.qec_reps if cfg.qec_transmit_redundancy else 1
        n_times = effective_num_times(cfg.num_times, reps) if reps > 1 else cfg.num_times
        t = np.linspace(0, cfg.pulse_duration_ns * 1e-9, n_times)
        pulse = self.pyramidal_pulse(t)

        field_time = np.zeros((n_times, cfg.grid_size, cfg.grid_size), dtype=complex)
        intensity_time = np.zeros_like(field_time, dtype=float)

        if cfg.qec_transmit_redundancy and reps > 1:
            n_logical = n_times // reps
            t_logical = t[np.arange(n_logical) * reps + reps // 2]
            pulse_logical = self.pyramidal_pulse(t_logical)
            for li, ti_val in enumerate(t_logical):
                field = np.zeros((cfg.grid_size, cfg.grid_size), dtype=complex)
                mod = 1.0 + 0.15 * pulse_logical[li] / (np.max(np.abs(pulse_logical)) + 1e-12)
                for orb in orbs:
                    x0, y0 = self._orb_position(orb, ti_val)
                    amp = self._orb_amplitude(orb, ti_val, t[-1])
                    field += self._point_source_field(x0, y0, amp * mod, orb.ell)
                field += encode_imprint_field(q, self.rho, self.phi, w0=cfg.w0)
                lg_carrier = lg_mode(1, self.rho, self.phi, w0=cfg.w0)
                axis = rodrigues_rotation(
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    q.w * np.pi / 2,
                )
                quat_phase = np.exp(1j * axis[0] * PHI_SCALE)
                field *= lg_carrier * quat_phase
                sl = slice(li * reps, (li + 1) * reps)
                field_time[sl] = field
                intensity_time[sl] = np.abs(field) ** 2
            pulse = repeat_triplets_1d(pulse, reps=reps)
        else:
            for ti, ti_val in enumerate(t):
                field = np.zeros((cfg.grid_size, cfg.grid_size), dtype=complex)
                mod = 1.0 + 0.15 * pulse[ti] / (np.max(np.abs(pulse)) + 1e-12)
                for orb in orbs:
                    x0, y0 = self._orb_position(orb, ti_val)
                    amp = self._orb_amplitude(orb, ti_val, t[-1])
                    field += self._point_source_field(x0, y0, amp * mod, orb.ell)
                field += encode_imprint_field(q, self.rho, self.phi, w0=cfg.w0)
                lg_carrier = lg_mode(1, self.rho, self.phi, w0=cfg.w0)
                axis = rodrigues_rotation(
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    q.w * np.pi / 2,
                )
                quat_phase = np.exp(1j * axis[0] * PHI_SCALE)
                field *= lg_carrier * quat_phase
                field_time[ti] = field
                intensity_time[ti] = np.abs(field) ** 2

        nperseg = min(32, max(n_times // 2, 2))
        freqs, psd = welch(
            pulse,
            fs=1.0 / (t[1] - t[0]) if n_times > 1 else 1.0,
            nperseg=nperseg,
        )

        return EncodeResult(
            field_time=field_time,
            intensity_time=intensity_time,
            pulse=pulse,
            spectral_shards=psd,
            freqs=freqs,
            quaternion=q,
            orbs=orbs,
            glyph_duties=glyph,
            t=t,
            rho=self.rho,
            phi=self.phi,
            payload=bytes(payload),
            qec_reps=reps if cfg.qec_transmit_redundancy else 1,
        )

    def propagate_with_turbulence(
        self,
        encoded: EncodeResult,
        *,
        noise_level: float | None = None,
    ) -> np.ndarray:
        """Apply p-wave BMGL turbulence to encoded field."""
        noise_scale = noise_level_to_scale(noise_level) if noise_level is not None else 1.0
        noisy = np.zeros_like(encoded.field_time)
        for ti in range(encoded.field_time.shape[0]):
            noisy[ti] = apply_turbulence(
                encoded.field_time[ti],
                self.config.bmgl,
                phi=self.phi,
                rng=self.rng,
                noise_scale=noise_scale,
            )
        return noisy