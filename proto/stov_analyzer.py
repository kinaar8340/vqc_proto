"""STOV (spatiotemporal OAM) spectrum analysis for the VQC Gradio demo."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

STOV_BG = "#1a1a2e"
STOV_AX = "#0f0f23"
STOV_GRID = "#c8b820"

STOV_PRESETS: dict[str, dict] = {
    "vqc_carrier": {
        "label": "VQC LG₁ carrier",
        "m_min": -4,
        "m_max": 4,
        "noise_level": 0.08,
        "n_modes": 5,
        "seed": 42,
        "peak_m": 1,
    },
    "multi_vortex": {
        "label": "Multi-vortex superposition",
        "m_min": -8,
        "m_max": 8,
        "noise_level": 0.12,
        "n_modes": 9,
        "seed": 7,
        "peak_m": None,
    },
    "braille_shards": {
        "label": "Orbital Braille shard modes",
        "m_min": -6,
        "m_max": 6,
        "noise_level": 0.1,
        "n_modes": 7,
        "seed": 42,
        "peak_m": 2,
    },
}


@dataclass(frozen=True)
class STOVAnalysisResult:
    m_values: np.ndarray
    powers: np.ndarray
    powers_x: np.ndarray
    powers_y: np.ndarray
    powers_z: np.ndarray
    dominant_m: int
    purity: float
    crest_factor: float
    fidelity: float
    metrics_text: str


@dataclass
class STOVSession:
    """Cached analysis for reconstruct, animation, and cross-tab bridge."""

    m_values: np.ndarray
    powers: np.ndarray
    coeffs: np.ndarray
    weights: np.ndarray
    x: np.ndarray
    t: np.ndarray
    field: np.ndarray
    result: STOVAnalysisResult
    noise_level: float
    seed: int
    preset_key: str | None = None

    def to_cache(self) -> dict[str, Any]:
        return {
            "m_values": self.m_values.tolist(),
            "powers": self.powers.tolist(),
            "coeffs_real": np.real(self.coeffs).tolist(),
            "coeffs_imag": np.imag(self.coeffs).tolist(),
            "weights": self.weights.tolist(),
            "x": self.x.tolist(),
            "t": self.t.tolist(),
            "field_real": np.real(self.field).tolist(),
            "field_imag": np.imag(self.field).tolist(),
            "noise_level": self.noise_level,
            "seed": self.seed,
            "preset_key": self.preset_key,
            "result": {
                "dominant_m": self.result.dominant_m,
                "purity": self.result.purity,
                "crest_factor": self.result.crest_factor,
                "fidelity": self.result.fidelity,
                "metrics_text": self.result.metrics_text,
                "powers_x": self.result.powers_x.tolist(),
                "powers_y": self.result.powers_y.tolist(),
                "powers_z": self.result.powers_z.tolist(),
            },
        }

    @classmethod
    def from_cache(cls, data: dict[str, Any] | None) -> STOVSession | None:
        if not data:
            return None
        m_values = np.asarray(data["m_values"], dtype=int)
        result_blob = data["result"]
        result = STOVAnalysisResult(
            m_values=m_values,
            powers=np.asarray(data["powers"], dtype=float),
            powers_x=np.asarray(result_blob["powers_x"], dtype=float),
            powers_y=np.asarray(result_blob["powers_y"], dtype=float),
            powers_z=np.asarray(result_blob["powers_z"], dtype=float),
            dominant_m=int(result_blob["dominant_m"]),
            purity=float(result_blob["purity"]),
            crest_factor=float(result_blob["crest_factor"]),
            fidelity=float(result_blob["fidelity"]),
            metrics_text=str(result_blob["metrics_text"]),
        )
        coeffs = np.asarray(data["coeffs_real"]) + 1j * np.asarray(data["coeffs_imag"])
        field = np.asarray(data["field_real"]) + 1j * np.asarray(data["field_imag"])
        return cls(
            m_values=m_values,
            powers=np.asarray(data["powers"], dtype=float),
            coeffs=coeffs,
            weights=np.asarray(data["weights"], dtype=float),
            x=np.asarray(data["x"], dtype=float),
            t=np.asarray(data["t"], dtype=float),
            field=field,
            result=result,
            noise_level=float(data["noise_level"]),
            seed=int(data["seed"]),
            preset_key=data.get("preset_key"),
        )


def load_stov_preset(preset_key: str) -> tuple[int, int, float, int, int]:
    """Return slider values for a named STOV preset."""
    preset = STOV_PRESETS[preset_key]
    return (
        int(preset["m_min"]),
        int(preset["m_max"]),
        float(preset["noise_level"]),
        int(preset["n_modes"]),
        int(preset["seed"]),
    )


def _stov_basis(m: int, x: np.ndarray, t: np.ndarray, *, w0: float = 1.8) -> np.ndarray:
    """Helical STOV basis on the space-time plane."""
    r2 = x**2 + t**2
    amp = np.exp(-r2 / (2.0 * w0**2)) * (1.0 + 0.15 * np.cos(m * np.pi / 6.0))
    phase = m * np.arctan2(t, x)
    return amp * np.exp(1j * phase)


def _build_weights(
    m_values: np.ndarray,
    *,
    n_modes: int,
    seed: int,
    peak_m: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros(len(m_values), dtype=float)
    n_active = max(1, min(int(n_modes), len(m_values)))

    if peak_m is not None and peak_m in m_values:
        weights[m_values == peak_m] = 1.0
        remaining = n_active - 1
        if remaining > 0:
            others = [i for i, m in enumerate(m_values) if m != peak_m]
            pick = rng.choice(others, size=min(remaining, len(others)), replace=False)
            weights[pick] = 0.35 * rng.random(len(pick))
    else:
        pick = rng.choice(len(m_values), size=n_active, replace=False)
        weights[pick] = rng.random(n_active)

    total = weights.sum()
    if total <= 0:
        weights[len(weights) // 2] = 1.0
        total = 1.0
    return weights / total


def generate_stov_superposition(
    m_range: tuple[int, int],
    *,
    weights: np.ndarray | None = None,
    noise_level: float = 0.1,
    n_modes: int = 9,
    seed: int = 42,
    peak_m: int | None = None,
    n_points: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a superposition of STOV modes with topological orders m."""
    m_values = np.arange(m_range[0], m_range[1] + 1)
    if weights is None:
        weights = _build_weights(m_values, n_modes=n_modes, seed=seed, peak_m=peak_m)

    x_1d = np.linspace(-5.0, 5.0, n_points)
    t_1d = np.linspace(-5.0, 5.0, n_points)
    x_grid, t_grid = np.meshgrid(x_1d, t_1d)

    field = np.zeros_like(x_grid, dtype=complex)
    for m, weight in zip(m_values, weights):
        field += weight * _stov_basis(int(m), x_grid, t_grid)

    rng = np.random.default_rng(seed + 17)
    if noise_level > 0:
        field += noise_level * (
            rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        )
    return x_grid, t_grid, field, m_values, weights


def project_stov_coefficients(
    field: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal STOV basis projection — complex coefficients and normalized powers."""
    coeffs = np.empty(len(m_values), dtype=np.complex128)
    for idx, m in enumerate(m_values):
        basis = _stov_basis(int(m), x, t)
        denom = np.sum(np.abs(basis) ** 2) + 1e-12
        coeffs[idx] = np.sum(field * np.conj(basis)) / denom
    powers = np.abs(coeffs) ** 2
    total = float(powers.sum())
    if total > 0:
        powers = powers / total
    return coeffs, powers


def project_stov_spectrum(
    field: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project the field onto STOV basis modes; return normalized powers."""
    _, powers = project_stov_coefficients(field, m_values, x, t)
    return powers


def reconstruct_stov_field(
    coeffs: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Synthesize a clean STOV field from projected mode coefficients."""
    field = np.zeros_like(x, dtype=np.complex128)
    for coeff, m in zip(coeffs, m_values):
        field += coeff * _stov_basis(int(m), x, t)
    return field


def estimate_local_topological_charge(phase: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Local m estimate from phase gradient in the (x, t) plane.

    For φ ≈ m·arctan2(t, x), use m ≈ (x·∂φ/∂t − t·∂φ/∂x) with phase unwrapping guard.
    """
    dphi_dt = np.gradient(phase, axis=0)
    dphi_dx = np.gradient(phase, axis=1)
    denom = x**2 + t**2 + 1e-3
    m_local = (x * dphi_dt - t * dphi_dx) / denom
    return np.clip(np.round(m_local), -12, 12)


def decompose_vector_spectra(
    field: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proxy Lx / Ly / Lz spectra via quadrature projections on the STOV basis."""
    phase = np.angle(field)
    intensity = np.abs(field)
    field_x = intensity * np.cos(phase)
    field_y = intensity * np.sin(phase)
    field_z = intensity * (0.5 + 0.5 * np.sin(2.0 * phase))
    return (
        project_stov_spectrum(field_x.astype(complex), m_values, x, t),
        project_stov_spectrum(field_y.astype(complex), m_values, x, t),
        project_stov_spectrum(field_z.astype(complex), m_values, x, t),
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(STOV_AX)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def plot_colorful_stov_spectrogram(
    x: np.ndarray,
    t: np.ndarray,
    field: np.ndarray,
    *,
    title: str = "STOV Space-Time Spectrogram",
) -> plt.Figure:
    """Vibrant RGB-style space-time view (Lx / Ly / Lz channel metaphor)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=STOV_BG)
    intensity = np.abs(field)
    phase = np.angle(field)
    peak = np.percentile(intensity, 99) + 1e-9
    norm = intensity / peak

    red = np.clip(norm * (0.55 + 0.45 * np.sin(phase)), 0, 1)
    green = np.clip(norm * (0.55 + 0.45 * np.cos(phase * 1.5)), 0, 1)
    blue = np.clip(norm * (0.55 + 0.45 * np.sin(phase * 2.0)), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)

    x_1d = x[0, :]
    t_1d = t[:, 0]
    ax.imshow(
        rgb,
        extent=[float(x_1d.min()), float(x_1d.max()), float(t_1d.min()), float(t_1d.max())],
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
    )
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Time (t)")
    ax.set_title(title, fontsize=13)
    ax.grid(True, color=STOV_GRID, alpha=0.28, linestyle="--", linewidth=0.6)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _downsample_stov_grid(
    x: np.ndarray,
    t: np.ndarray,
    field: np.ndarray,
    *,
    max_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample space-time grids for lightweight Plotly payloads."""
    n_rows, n_cols = field.shape
    if n_rows <= max_size and n_cols <= max_size:
        return x, t, field
    row_step = max(1, n_rows // max_size)
    col_step = max(1, n_cols // max_size)
    return x[::row_step, ::col_step], t[::row_step, ::col_step], field[::row_step, ::col_step]


def plot_stov_spectrogram_plotly(
    x: np.ndarray,
    t: np.ndarray,
    field: np.ndarray,
    *,
    max_size: int = 128,
) -> dict[str, Any]:
    """Interactive space-time spectrogram with intensity, phase, and local m on hover."""
    x_ds, t_ds, field_ds = _downsample_stov_grid(x, t, field, max_size=max_size)
    intensity = np.abs(field_ds)
    phase = np.angle(field_ds)
    m_local = estimate_local_topological_charge(phase, x_ds, t_ds)
    x_1d = x_ds[0, :].tolist()
    t_1d = t_ds[:, 0].tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=intensity.tolist(),
            x=x_1d,
            y=t_1d,
            colorscale="Viridis",
            colorbar=dict(title="|E|", tickfont=dict(color="#d8d0f0")),
            customdata=np.stack([phase, m_local], axis=-1).tolist(),
            hovertemplate=(
                "x=%{x:.2f}<br>t=%{y:.2f}<br>|E|=%{z:.4f}<br>"
                "phase=%{customdata[0]:.3f} rad<br>m_local=%{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text="STOV space-time field (hover: |E|, phase, local m)", font=dict(color="#e8e0f8")),
        paper_bgcolor=STOV_BG,
        plot_bgcolor=STOV_AX,
        font=dict(color="#d8d0f0"),
        xaxis=dict(title="Space (x)", gridcolor=STOV_GRID),
        yaxis=dict(title="Time (t)", gridcolor=STOV_GRID),
        margin=dict(l=48, r=24, t=48, b=40),
    )
    return fig.to_plotly_json()


def plot_stov_spectrum_bars(
    m_values: np.ndarray,
    powers: np.ndarray,
    *,
    title: str = "Spatiotemporal OAM Spectrum",
) -> plt.Figure:
    """DSP-style bar spectrum (power vs topological order m)."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=STOV_BG)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(m_values)))
    ax.bar(m_values, powers, color=colors, edgecolor="white", linewidth=0.5, width=0.82)
    ax.set_xlabel("Spatiotemporal order m")
    ax.set_ylabel("Relative power / weight")
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3, color="white")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_vector_spectra(
    m_values: np.ndarray,
    powers_x: np.ndarray,
    powers_y: np.ndarray,
    powers_z: np.ndarray,
) -> plt.Figure:
    """Three-component STOV spectra (Lx / Ly / Lz proxies)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=STOV_BG)
    specs = (
        (powers_x, "#ff5555", "Lx (x-comp)"),
        (powers_y, "#55dd55", "Ly (y-comp)"),
        (powers_z, "#5599ff", "Lz (z-comp)"),
    )
    for ax, (powers, color, label) in zip(axes, specs):
        ax.bar(m_values, powers, color=color, edgecolor="white", linewidth=0.4, width=0.82)
        ax.set_title(label, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25, color="white")
        _style_axes(ax)
    fig.tight_layout()
    return fig


def format_meter_gauges_html(
    *,
    purity: float,
    dominant_m: float,
    fidelity: float,
    crest_factor: float,
) -> str:
    """Visual bar meters for analyzer readouts."""

    def bar(label: str, value: float, *, max_val: float = 1.0, color: str) -> str:
        pct = float(np.clip(100.0 * value / max_val, 0.0, 100.0))
        return (
            f'<div class="vqc-stov-gauge">'
            f'<div class="vqc-stov-gauge-label"><span>{label}</span><span>{value:.3f}</span></div>'
            f'<div class="vqc-stov-gauge-track"><div class="vqc-stov-gauge-fill" '
            f'style="width:{pct:.1f}%;background:{color};"></div></div></div>'
        )

    crest_norm = float(np.clip(crest_factor / 10.0, 0.0, 1.0))
    return (
        '<div class="vqc-stov-gauges">'
        + bar("Mode purity", purity, color="#ffb347")
        + bar("Vector fidelity", fidelity, color="#6ecf9c")
        + bar("Crest factor", crest_norm, color="#7eb8ff")
        + (
            f'<div class="vqc-stov-gauge vqc-stov-gauge-dominant">'
            f'<div class="vqc-stov-gauge-label"><span>Dominant m</span>'
            f'<span class="vqc-stov-m-value">{int(dominant_m):+d}</span></div></div>'
        )
        + "</div>"
    )


def analyze_stov_field(
    field: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    *,
    noise_level: float,
    target_weights: np.ndarray | None = None,
) -> tuple[STOVAnalysisResult, np.ndarray]:
    """Compute STOV metrics, component spectra, and projection coefficients."""
    coeffs, powers = project_stov_coefficients(field, m_values, x, t)
    powers_x, powers_y, powers_z = decompose_vector_spectra(field, m_values, x, t)
    dominant_idx = int(np.argmax(powers))
    dominant_m = int(m_values[dominant_idx])
    purity = float(powers[dominant_idx])

    amplitude = np.abs(field)
    rms = float(np.sqrt(np.mean(amplitude**2)) + 1e-12)
    crest_factor = float(amplitude.max() / rms)

    if target_weights is not None and target_weights.sum() > 0:
        target = target_weights / target_weights.sum()
        fidelity = float(1.0 - 0.5 * np.sum(np.abs(powers - target)))
        fidelity = max(0.0, min(1.0, fidelity))
    else:
        fidelity = float(np.sum(np.sort(powers)[-3:]))

    metrics = "\n".join(
        [
            f"Orders m ∈ [{m_values[0]}, {m_values[-1]}]  ({len(m_values)} modes)",
            f"Dominant order: m = {dominant_m}",
            f"Mode purity: {purity:.4f}",
            f"Crest factor: {crest_factor:.3f}",
            f"Vector fidelity: {fidelity:.4f}",
            f"Channel noise σ: {noise_level:.3f}",
            f"Top-3 share: {np.sum(np.sort(powers)[-3:]):.4f}",
        ]
    )
    result = STOVAnalysisResult(
        m_values=m_values,
        powers=powers,
        powers_x=powers_x,
        powers_y=powers_y,
        powers_z=powers_z,
        dominant_m=dominant_m,
        purity=purity,
        crest_factor=crest_factor,
        fidelity=fidelity,
        metrics_text=metrics,
    )
    return result, coeffs


def build_stov_session(
    m_min: float,
    m_max: float,
    noise_level: float,
    n_modes: float,
    seed: float,
    *,
    preset_key: str | None = None,
) -> STOVSession:
    """Generate field, decompose, and cache a full STOV session."""
    m_lo, m_hi = int(m_min), int(m_max)
    if m_lo > m_hi:
        m_lo, m_hi = m_hi, m_lo
    m_range = (m_lo, m_hi)
    peak_m = STOV_PRESETS[preset_key].get("peak_m") if preset_key in STOV_PRESETS else None

    m_values = np.arange(m_range[0], m_range[1] + 1)
    target_weights = _build_weights(
        m_values,
        n_modes=int(n_modes),
        seed=int(seed),
        peak_m=peak_m,
    )
    x, t, field, m_values, weights = generate_stov_superposition(
        m_range,
        weights=target_weights,
        noise_level=float(noise_level),
        n_modes=int(n_modes),
        seed=int(seed),
        peak_m=peak_m,
    )
    result, coeffs = analyze_stov_field(
        field,
        m_values,
        x,
        t,
        noise_level=float(noise_level),
        target_weights=target_weights,
    )
    return STOVSession(
        m_values=m_values,
        powers=result.powers,
        coeffs=coeffs,
        weights=weights,
        x=x,
        t=t,
        field=field,
        result=result,
        noise_level=float(noise_level),
        seed=int(seed),
        preset_key=preset_key,
    )


def run_stov_analysis(
    m_min: float,
    m_max: float,
    noise_level: float,
    n_modes: float,
    seed: float,
    *,
    preset_key: str | None = None,
) -> tuple:
    """Gradio handler: plots, metrics, gauges, and session cache."""
    session = build_stov_session(
        m_min, m_max, noise_level, n_modes, seed, preset_key=preset_key
    )
    fig_plotly = plot_stov_spectrogram_plotly(session.x, session.t, session.field)
    fig_color = plot_colorful_stov_spectrogram(session.x, session.t, session.field)
    fig_spec = plot_stov_spectrum_bars(session.m_values, session.result.powers)
    fig_vec = plot_vector_spectra(
        session.m_values,
        session.result.powers_x,
        session.result.powers_y,
        session.result.powers_z,
    )
    gauges = format_meter_gauges_html(
        purity=session.result.purity,
        dominant_m=float(session.result.dominant_m),
        fidelity=session.result.fidelity,
        crest_factor=session.result.crest_factor,
    )
    return (
        fig_plotly,
        fig_color,
        fig_spec,
        fig_vec,
        session.result.metrics_text,
        gauges,
        session.result.purity,
        float(session.result.dominant_m),
        session.result.fidelity,
        session.result.crest_factor,
        session.to_cache(),
    )


def _field_coherence(a: np.ndarray, b: np.ndarray) -> float:
    av = a.ravel()
    bv = b.ravel()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(av, bv)) / denom)


def run_stov_reconstruct_decode(cache: dict[str, Any] | None) -> str:
    """Reconstruct clean field from spectrum and run lightweight Orbital Braille pipeline."""
    session = STOVSession.from_cache(cache)
    if session is None:
        return "Run **Analyze** first to populate the STOV session cache."

    recon = reconstruct_stov_field(session.coeffs, session.m_values, session.x, session.t)
    recon_coh = _field_coherence(recon, session.field)

    bridge = stov_bridge_to_demo_params(session)
    try:
        from demo_core import run_pipeline

        _, _, _, decoded, metrics, font_sep = run_pipeline(
            bridge["payload"],
            bridge["num_orbs"],
            quick=True,
            seed=bridge["seed"],
            gamma_1=bridge["gamma_1"],
            noise_level=bridge["noise_level"],
        )
        pipeline_block = (
            f"\n\n--- Orbital Braille pipeline (STOV-mapped settings) ---\n"
            f"Payload: {bridge['payload']!r}\n"
            f"Orbs: {bridge['num_orbs']}  γ₁: {bridge['gamma_1']:.2f}  "
            f"noise: {bridge['noise_level']:.2f}\n"
            f"Shard fidelity: {decoded.shard_fidelity:.4f}  "
            f"Glyph fidelity: {decoded.glyph_fidelity:.4f}  "
            f"field coherence: {getattr(decoded, 'glyph_field_coherence', 0):.4f}\n"
            f"Font separation: {font_sep:.4f} rad"
        )
    except Exception as exc:
        pipeline_block = f"\n\nPipeline preview failed: {exc}"

    return (
        f"STOV spectrum → field reconstruction\n"
        f"  Reconstruction coherence vs. noisy field: {recon_coh:.4f}\n"
        f"  Dominant m: {session.result.dominant_m:+d}  "
        f"purity: {session.result.purity:.4f}  "
        f"crest: {session.result.crest_factor:.3f}\n"
        f"  Mapped orbs={bridge['num_orbs']}, γ₁={bridge['gamma_1']:.2f}, "
        f"noise={bridge['noise_level']:.2f}"
        f"{pipeline_block}"
    )


def stov_bridge_to_demo_params(session: STOVSession) -> dict[str, Any]:
    """Map STOV session to Live Demo slider values."""
    m = abs(int(session.result.dominant_m))
    num_orbs = int(np.clip(4 + m // 2, 4, 8))
    noise_level = float(np.clip(session.noise_level * 2.0, 0.05, 0.85))
    gamma_1 = float(np.clip(1.4 + 0.08 * m + 0.15 * session.result.purity, 1.0, 2.0))
    payload = f"STOV m={session.result.dominant_m:+d} purity={session.result.purity:.2f}"
    return {
        "payload": payload,
        "num_orbs": num_orbs,
        "noise_level": noise_level,
        "gamma_1": gamma_1,
        "seed": session.seed,
    }


def bridge_stov_to_demo(cache: dict[str, Any] | None) -> tuple[str, float, float, float, str]:
    """Return demo payload/orbs/noise/gamma plus status message."""
    session = STOVSession.from_cache(cache)
    if session is None:
        return (
            "STOV m=0",
            4.0,
            0.35,
            1.5,
            "⚠ Run **Analyze** first, then send settings to Live Demo.",
        )
    params = stov_bridge_to_demo_params(session)
    msg = (
        f"✓ Copied STOV settings → Live Demo: orbs={params['num_orbs']}, "
        f"γ₁={params['gamma_1']:.2f}, noise={params['noise_level']:.2f}. "
        f"Switch to **Live Demo** and click **Run demo**."
    )
    return (
        params["payload"],
        float(params["num_orbs"]),
        float(params["noise_level"]),
        float(params["gamma_1"]),
        msg,
    )


def _stov_rgb_array(field: np.ndarray) -> np.ndarray:
    """RGB space-time view matching the static matplotlib spectrogram."""
    intensity = np.abs(field)
    phase = np.angle(field)
    peak = np.percentile(intensity, 99) + 1e-9
    norm = intensity / peak
    red = np.clip(norm * (0.55 + 0.45 * np.sin(phase)), 0, 1)
    green = np.clip(norm * (0.55 + 0.45 * np.cos(phase * 1.5)), 0, 1)
    blue = np.clip(norm * (0.55 + 0.45 * np.sin(phase * 2.0)), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def _stov_frame_pil(x: np.ndarray, t: np.ndarray, field: np.ndarray, row_idx: int):
    from PIL import Image, ImageDraw

    rgb = _stov_rgb_array(field)
    window = min(64, field.shape[0])
    row = int(np.clip(row_idx, 0, field.shape[0] - 1))
    r0 = int(np.clip(row - window // 2, 0, max(0, field.shape[0] - window)))
    crop = rgb[r0 : r0 + window, :, :]
    img = Image.fromarray(crop)
    draw = ImageDraw.Draw(img)
    cursor = row - r0
    if 0 <= cursor < window:
        draw.line([(0, cursor), (crop.shape[1] - 1, cursor)], fill=(255, 220, 80), width=2)
    return img.resize((640, 240), Image.Resampling.BILINEAR)


def _stov_full_frame_pil(field: np.ndarray, row_idx: int):
    """Full-field RGB frame with a time-axis cursor (for GIF/MP4 export)."""
    from PIL import Image, ImageDraw

    rgb = _stov_rgb_array(field)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    row = int(np.clip(row_idx, 0, field.shape[0] - 1))
    draw.line([(0, row), (rgb.shape[1] - 1, row)], fill=(255, 220, 80), width=2)
    return img.resize((640, 360), Image.Resampling.BILINEAR)


def render_stov_animation_bundle(
    cache: dict[str, Any] | None,
    *,
    n_frames: int = 48,
    fps: float = 11.0,
) -> tuple[str | None, str | None, str]:
    """Export STOV space-time scroll animation as GIF + optional MP4."""
    session = STOVSession.from_cache(cache)
    if session is None:
        return None, None, "⚠ Run **Analyze** first before exporting animation."

    n_frames = int(np.clip(n_frames, 8, 60))
    indices = np.linspace(0, session.field.shape[0] - 1, n_frames, dtype=int)
    frames = [_stov_full_frame_pil(session.field, int(i)) for i in indices]
    if not frames:
        return None, None, "⚠ No frames generated — re-run **Analyze** and try again."

    out_dir = Path(tempfile.mkdtemp(prefix="vqc_stov_anim_"))
    gif_path = out_dir / "stov_field.gif"
    frame_ms = max(40, int(1000.0 / fps))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    mp4_path: str | None = None
    mp4_note = ""
    try:
        from demo_core import _encode_frames_mp4

        encoded = _encode_frames_mp4(frames, out_dir / "stov_field.mp4", fps=fps)
        if encoded is not None:
            mp4_path = str(encoded)
        else:
            mp4_note = " MP4 skipped (ffmpeg not available on this host)."
    except Exception as exc:
        mp4_note = f" MP4 export failed: {exc}"

    note = (
        f"✓ Exported {n_frames} full-field RGB frames "
        f"(dominant m={session.result.dominant_m:+d}, {fps:.0f} fps).{mp4_note}"
    )
    return str(gif_path), mp4_path, note