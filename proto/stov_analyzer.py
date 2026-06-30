"""STOV (spatiotemporal OAM) spectrum analysis for the VQC Gradio demo."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

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

    x = np.linspace(-5.0, 5.0, n_points)
    t = np.linspace(-5.0, 5.0, n_points)
    x_grid, t_grid = np.meshgrid(x, t)

    field = np.zeros_like(x_grid, dtype=complex)
    for m, weight in zip(m_values, weights):
        field += weight * _stov_basis(int(m), x_grid, t_grid)

    rng = np.random.default_rng(seed + 17)
    if noise_level > 0:
        field += noise_level * (
            rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        )
    return x_grid, t_grid, field, m_values, weights


def project_stov_spectrum(
    field: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project the field onto STOV basis modes; return normalized powers."""
    powers = np.empty(len(m_values), dtype=float)
    for idx, m in enumerate(m_values):
        basis = _stov_basis(int(m), x, t)
        denom = np.sum(np.abs(basis) ** 2) + 1e-12
        coeff = np.sum(field * np.conj(basis)) / denom
        powers[idx] = float(np.abs(coeff) ** 2)
    total = powers.sum()
    if total > 0:
        powers /= total
    return powers


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

    ax.imshow(
        rgb,
        extent=[float(x.min()), float(x.max()), float(t.min()), float(t.max())],
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


def analyze_stov_field(
    field: np.ndarray,
    m_values: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    *,
    noise_level: float,
    target_weights: np.ndarray | None = None,
) -> STOVAnalysisResult:
    """Compute STOV metrics and component spectra from a simulated field."""
    powers = project_stov_spectrum(field, m_values, x, t)
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
    return STOVAnalysisResult(
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


def run_stov_analysis(
    m_min: float,
    m_max: float,
    noise_level: float,
    n_modes: float,
    seed: float,
    *,
    preset_key: str | None = None,
) -> tuple[plt.Figure, plt.Figure, plt.Figure, str, float, float, float]:
    """Gradio handler: generate STOV field, decompose, and render analyzer plots."""
    m_lo, m_hi = int(m_min), int(m_max)
    if m_lo > m_hi:
        m_lo, m_hi = m_hi, m_lo
    m_range = (m_lo, m_hi)
    peak_m = None
    if preset_key and preset_key in STOV_PRESETS:
        peak_m = STOV_PRESETS[preset_key].get("peak_m")

    m_values = np.arange(m_range[0], m_range[1] + 1)
    target_weights = _build_weights(
        m_values,
        n_modes=int(n_modes),
        seed=int(seed),
        peak_m=peak_m,
    )
    x, t, field, m_values, _ = generate_stov_superposition(
        m_range,
        weights=target_weights,
        noise_level=float(noise_level),
        n_modes=int(n_modes),
        seed=int(seed),
        peak_m=peak_m,
    )
    result = analyze_stov_field(
        field,
        m_values,
        x,
        t,
        noise_level=float(noise_level),
        target_weights=target_weights,
    )
    fig_color = plot_colorful_stov_spectrogram(x, t, field)
    fig_spec = plot_stov_spectrum_bars(m_values, result.powers)
    fig_vec = plot_vector_spectra(m_values, result.powers_x, result.powers_y, result.powers_z)
    return (
        fig_color,
        fig_spec,
        fig_vec,
        result.metrics_text,
        result.purity,
        float(result.dominant_m),
        result.fidelity,
    )