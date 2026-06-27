"""Shared encode/decode/plot helpers for run_demo, Gradio, and HF Spaces."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbital_braille import (
    OrbitalTypehead,
    TypeheadConfig,
    PWaveBMGL,
    EmergentConstants,
    decode_field,
    build_stable_font,
    font_separation,
)
from orbital_braille.slm_typehead import SLM_PRESETS, SLMConfig, export_hologram_package

PATENT_FIGURE1_PAYLOAD = "I live in Oregon"

QUICK_GRID_SIZE = 32
QUICK_NUM_TIMES = 16
FULL_GRID_SIZE = 80
FULL_NUM_TIMES = 64
QUICK_SLM_FRAMES = 16
FULL_SLM_FRAMES = 32

VQC_CLAIMS_MD = """
| VQC claim element | Demo shows… |
|-------------------|-------------|
| **Pyramidal FM pulses** | A triangular time-envelope chirp whose amplitude rises and falls symmetrically (bottom-left panel). |
| **Spectral shards** | Discrete Welch PSD peaks that barcode the payload as frequency subcarriers (bottom-middle panel). |
| **Quaternion encoding** | A unit quaternion derived from payload bytes, printed in the metrics block after each run. |
| **OAM mode multiplex** | Distinct topological charges ℓ on each PWM-gated orb in the layout scatter plot (bottom-right). |
| **Nested helical shielding** | Concentric orbit rings with differential phase structure in the clean encoded phase map (top-left). |
| **p-wave BMGL (γ₁)** | Phase noise suppression that you can tune live — compare clean vs. turbulent phase panels (γ₁ slider). |
| **16-qubit QEC proxy** | Majority-vote error correction during decode; higher shard fidelity after BMGL denoise (metrics). |
| **SLM virtual typehead** | A hardware-ready zip with `manifest.json`, `phase_stack.npy`, and optional `frames/` PNG sequence. |

**Patent Figure 1 payload:** `"I live in Oregon"` (4 orbs) — use **Load example from paper**.

Full mapping: [proto/README.md — Patent claim alignment](https://github.com/kinaar8340/vqc_proto/blob/main/proto/README.md#patent-claim-alignment)
"""


def get_build_label() -> str:
    """Return a short last-updated line for the Gradio footer."""
    try:
        from build_info import BUILD_COMMIT, BUILD_UPDATED_UTC  # noqa: WPS433

        return f"Last updated: {BUILD_UPDATED_UTC} UTC · commit `{BUILD_COMMIT}`"
    except ImportError:
        pass

    import subprocess

    try:
        root = Path(__file__).resolve().parent.parent
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        if commit:
            return f"Build: commit `{commit}` (local git)"
    except (OSError, subprocess.CalledProcessError):
        pass

    return "Build: development"


def _slm_readme_text(
    *,
    payload: str,
    device_preset: str,
    resolution: str,
    bit_depth: int,
    num_frames: int,
    t_max_ns: float,
    include_frames: bool,
) -> str:
    frame_period = t_max_ns / num_frames if num_frames else 0.0
    frames_note = (
        f"frames/phase_XXXX.png — {num_frames} 8-bit grayscale phase masks (linear, no gamma)"
        if include_frames
        else "frames/ — not included; re-run with 'Include SLM-ready phase frames' checked"
    )
    return f"""Orbital Braille — SLM upload package
=====================================
Payload: {payload!r}
Device preset: {device_preset} ({resolution}, {bit_depth}-bit)
Frames: {num_frames} over {t_max_ns:.2f} ns  (~{frame_period:.2f} ns per frame)

FILES
-----
manifest.json      Orb radii, ℓ charges, PWM duties, quaternion, timing
phase_stack.npy    NumPy array [frames, H, W] of phase in radians (0–2π)
preview_montage.png  Quick visual check of the phase sequence
LUT_calibration.txt  Linear gray→phase mapping notes; measure if deviation > 5%
README.txt           This file
{frames_note}

QUICK START BY DRIVER
---------------------
Holoeye PLUTO-2:
  Upload frames/phase_0000.png (or BMP) in phase-only mode.
  Set refresh to match manifest t_max_ns / frames. Disable gamma correction.

Meadowlark LCOS:
  Load 16-bit TIFF if available; map full scale → 0–2π. Disable dithering.

Thorlabs Exulus / 1080p LCOS:
  8-bit BMP/PNG; lock min/max gray to 0–255. Match 6.4 µm pitch optics.

Generic / custom:
  Read phase_stack.npy in Python/NumPy, or frames/*.raw (uint8 little-endian).
  Scale: gray = phase / (2π) × (2^bit_depth − 1).

See SLM_QUICKSTART.md in the vqc_proto repo for bench setup and pitfalls.
Repo: https://github.com/kinaar8340/vqc_proto/blob/main/proto/SLM_QUICKSTART.md
"""


def build_config(
    num_orbs: int,
    *,
    quick: bool = False,
    gamma_1: float = 1.5,
) -> TypeheadConfig:
    return TypeheadConfig(
        num_orbs=num_orbs,
        grid_size=QUICK_GRID_SIZE if quick else FULL_GRID_SIZE,
        num_times=QUICK_NUM_TIMES if quick else FULL_NUM_TIMES,
        bmgl=PWaveBMGL(gamma_1=gamma_1),
        constants=EmergentConstants(),
    )


def plot_results(encoded, noisy, decoded, out_dir: Path, payload: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = encoded.field_time.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Orbital Braille — VQC Typehead Prototype\n"
        f'Payload: "{payload[:32]}" | Glyph {decoded.glyph_index} | '
        f"Shard FID {decoded.shard_fidelity:.3f}",
        fontsize=11,
    )

    extent = [-2.5, 2.5, -2.5, 2.5]

    im0 = axes[0, 0].imshow(np.angle(encoded.field_time[mid]), cmap="hsv", extent=extent, origin="lower")
    axes[0, 0].set_title("Encoded phase (clean)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.angle(noisy[mid]), cmap="hsv", extent=extent, origin="lower")
    axes[0, 1].set_title("Phase after p-wave BMGL turbulence")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(encoded.intensity_time[mid], cmap="inferno", extent=extent, origin="lower")
    axes[0, 2].set_title("Intensity — OAM donut + orbital Braille")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].plot(encoded.t * 1e9, encoded.pulse, "b-", lw=1.5)
    axes[1, 0].fill_between(encoded.t * 1e9, 0, encoded.pulse, alpha=0.3)
    axes[1, 0].set_title("Pyramidal FM pulse (time)")
    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(encoded.freqs / 1e9, encoded.spectral_shards + 1e-20, "r-")
    axes[1, 1].set_title("Spectral shards (Welch PSD)")
    axes[1, 1].set_xlabel("Frequency (GHz)")
    axes[1, 1].set_ylabel("PSD")
    axes[1, 1].grid(True, alpha=0.3)

    orb_x = [o.radius * np.cos(o.phase0) for o in encoded.orbs]
    orb_y = [o.radius * np.sin(o.phase0) for o in encoded.orbs]
    axes[1, 2].scatter(orb_x, orb_y, s=120, c=range(len(encoded.orbs)), cmap="tab10")
    for i, o in enumerate(encoded.orbs):
        theta = np.linspace(0, 2 * np.pi, 64)
        axes[1, 2].plot(
            o.radius * np.cos(theta),
            o.radius * np.sin(theta),
            "--",
            alpha=0.4,
            color=plt.cm.tab10(i / max(len(encoded.orbs), 1)),
        )
        axes[1, 2].annotate(
            f"ℓ={o.ell}\nd={o.pwm_duty:.2f}",
            (orb_x[i], orb_y[i]),
            fontsize=7,
            ha="center",
        )
    axes[1, 2].set_title("Typehead orb layout (Braille dots)")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    axes[1, 2].set_aspect("equal")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "orbital_braille_demo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _style_dark_axes(axes) -> None:
    for ax in np.atleast_1d(axes).flat:
        ax.set_facecolor("#0a0818")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#444")


def _render_animation_frame(
    encoded,
    payload: str,
    frame_idx: int,
    n_frames: int,
    trail: list[list[tuple[float, float]]],
) -> "Image.Image":
    from PIL import Image

    t_idx = min(frame_idx, encoded.intensity_time.shape[0] - 1)
    t = float(encoded.t[t_idx])
    t_max = float(encoded.t[-1])
    extent = [-2.5, 2.5, -2.5, 2.5]
    t_ns = encoded.t * 1e9

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), facecolor="#0a0818")
    short_payload = payload[:28] + ("…" if len(payload) > 28 else "")
    fig.suptitle(
        f"Orbital Braille typehead  ·  \"{short_payload}\"  ·  "
        f"frame {frame_idx + 1}/{n_frames}",
        color="#f0e6ff",
        fontsize=10,
        fontweight="bold",
    )
    ax_phase, ax_int, ax_pulse, ax_orb = axes.flat
    _style_dark_axes(axes)

    im0 = ax_phase.imshow(
        np.angle(encoded.field_time[t_idx]),
        cmap="twilight",
        extent=extent,
        origin="lower",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax_phase.set_title("Helical phase (OAM carrier)", color="#ddd", fontsize=9)
    plt.colorbar(im0, ax=ax_phase, fraction=0.046)

    im1 = ax_int.imshow(
        encoded.intensity_time[t_idx],
        cmap="inferno",
        extent=extent,
        origin="lower",
    )
    ax_int.set_title("Intensity — donut + Braille lobes", color="#ddd", fontsize=9)
    plt.colorbar(im1, ax=ax_int, fraction=0.046)

    ax_pulse.plot(t_ns, encoded.pulse, color="#6eb5ff", lw=1.8)
    ax_pulse.fill_between(t_ns, 0, encoded.pulse, alpha=0.25, color="#6eb5ff")
    ax_pulse.axvline(t_ns[t_idx], color="#ff8c42", lw=2, alpha=0.9)
    ax_pulse.scatter([t_ns[t_idx]], [encoded.pulse[t_idx]], c="#ff8c42", s=40, zorder=5)
    ax_pulse.set_xlim(t_ns[0], t_ns[-1])
    ax_pulse.set_ylim(0, max(float(encoded.pulse.max()) * 1.1, 0.1))
    ax_pulse.set_title("Pyramidal FM pulse", color="#ddd", fontsize=9)
    ax_pulse.set_xlabel("Time (ns)", color="#aaa")
    ax_pulse.grid(True, alpha=0.2, color="#555")

    for i, orb in enumerate(encoded.orbs):
        theta = orb.phase0 + orb.omega * t
        x0 = orb.radius * np.cos(theta)
        y0 = orb.radius * np.sin(theta)
        trail[i].append((x0, y0))
        if len(trail[i]) > 8:
            trail[i] = trail[i][-8:]

        ring = plt.Circle((0, 0), orb.radius, fill=False, linestyle="--", alpha=0.3, color="#888")
        ax_orb.add_patch(ring)

        for age, (tx, ty) in enumerate(trail[i][:-1]):
            alpha = 0.15 + 0.07 * age
            ax_orb.scatter(tx, ty, s=30, c="#8888aa", alpha=alpha, zorder=1)

        pwm_on = (np.sin(2 * np.pi * orb.omega * t / t_max) + 1) / 2 < orb.pwm_duty
        color = "#ff8c42" if pwm_on else "#4a5568"
        ax_orb.scatter(x0, y0, s=160, c=color, edgecolors="white", linewidths=0.7, zorder=3)
        ax_orb.annotate(f"ℓ={orb.ell}", (x0, y0), fontsize=7, ha="center", va="bottom", color="#eee")

    ax_orb.set_xlim(-1.15, 1.15)
    ax_orb.set_ylim(-1.15, 1.15)
    ax_orb.set_aspect("equal")
    ax_orb.set_title("PWM-gated orbs (orange = ON)", color="#ddd", fontsize=9)
    ax_orb.grid(True, alpha=0.2, color="#555")

    fig.text(
        0.5,
        0.01,
        f"BMGL phase snapshot frame {t_idx + 1}  ·  shard carrier evolving in time",
        ha="center",
        fontsize=7,
        color="#999",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_typehead_animation(
    encoded,
    noisy: np.ndarray,
    payload: str,
    out_path: Path,
    *,
    duration_ms: int = 110,
    max_frames: int | None = None,
) -> Path:
    """Build a shareable GIF from an encode result (unique per run/settings)."""
    from PIL import Image

    n_frames = encoded.intensity_time.shape[0]
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    trail: list[list[tuple[float, float]]] = [[] for _ in encoded.orbs]
    frames = [
        _render_animation_frame(encoded, payload, i, n_frames, trail)
        for i in range(n_frames)
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def run_pipeline(
    payload: str,
    num_orbs: int,
    *,
    quick: bool = True,
    seed: int = 42,
    gamma_1: float = 1.5,
) -> tuple[TypeheadConfig, object, np.ndarray, object, str, float]:
    """Encode → turbulence → decode. Returns cfg, encoded, noisy, decoded, metrics, font_sep."""
    cfg = build_config(num_orbs, quick=quick, gamma_1=gamma_1)
    typehead = OrbitalTypehead(cfg, seed=seed)
    font_sep = font_separation(build_stable_font(num_orbs, num_glyphs=16))

    encoded = typehead.encode(payload)
    noisy = typehead.propagate_with_turbulence(encoded)
    decoded = decode_field(
        noisy,
        reference_intensity=encoded.intensity_time,
        font=typehead.font,
        orbs_ells=[o.ell for o in encoded.orbs],
        bmgl=cfg.bmgl,
        rho=encoded.rho,
        phi=encoded.phi,
        pulse_ref=encoded.pulse,
        t=encoded.t,
    )

    bmgl = cfg.bmgl
    mode = "QUICK" if quick else "FULL"
    metrics = "\n".join(
        [
            f"Mode: {mode} (grid={cfg.grid_size}, times={cfg.num_times})",
            f"Payload: {payload!r}",
            f"Orbs: {num_orbs}",
            f"p-wave BMGL γ₁ = {bmgl.gamma_1:.2f}  inhibition boost = {bmgl.inhibition_boost:.4f}",
            f"Font separation: {font_sep:.4f} rad",
            f"Shard fidelity: {decoded.shard_fidelity:.4f}",
            f"Glyph: index={decoded.glyph_index}  fidelity={decoded.glyph_fidelity:.4f}",
            f"Quaternion: w={encoded.quaternion.w:.3f} "
            f"x={encoded.quaternion.x:.3f} y={encoded.quaternion.y:.3f} "
            f"z={encoded.quaternion.z:.3f}",
            f"Dominant ℓ: {decoded.recovered_ells}",
        ]
    )
    return cfg, encoded, noisy, decoded, metrics, font_sep


def _zip_directory(src_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(src_dir))
    return zip_path


def export_slm_bundle(
    encoded,
    *,
    payload: str,
    num_orbs: int,
    font_sep: float,
    quick: bool = True,
    include_frames: bool = False,
    device_preset: str = "generic_512",
    out_dir: Path,
) -> tuple[Path, str]:
    """Build manifest.json + phase_stack.npy zip; optionally include SLM frame PNGs."""
    preset = SLM_PRESETS[device_preset]
    slm_cfg = SLMConfig.from_preset(preset, extent_mm=4.0)
    num_frames = QUICK_SLM_FRAMES if quick else FULL_SLM_FRAMES

    export_hologram_package(
        orbs=encoded.orbs,
        t_max=float(encoded.t[-1]),
        out_dir=out_dir,
        cfg=slm_cfg,
        payload=payload,
        quaternion=encoded.quaternion,
        glyph_duties=encoded.glyph_duties,
        num_frames=num_frames,
        device_preset=device_preset,
        use_gs=False,
        export_raw=include_frames,
    )

    if not include_frames:
        frames_dir = out_dir / "frames"
        if frames_dir.exists():
            for path in frames_dir.iterdir():
                path.unlink()
            frames_dir.rmdir()

    t_max_ns = float(encoded.t[-1]) * 1e9
    (out_dir / "README.txt").write_text(
        _slm_readme_text(
            payload=payload,
            device_preset=device_preset,
            resolution=f"{slm_cfg.resolution_x}×{slm_cfg.resolution_y}",
            bit_depth=slm_cfg.bit_depth,
            num_frames=num_frames,
            t_max_ns=t_max_ns,
            include_frames=include_frames,
        )
    )

    zip_path = out_dir.parent / "slm_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    _zip_directory(out_dir, zip_path)

    contents = "manifest.json, phase_stack.npy, preview_montage.png, LUT_calibration.txt, README.txt"
    if include_frames:
        contents += f", frames/ ({num_frames} phase PNGs)"
    summary = (
        f"SLM package: {contents}\n"
        f"Device: {device_preset} ({slm_cfg.resolution_x}×{slm_cfg.resolution_y}, "
        f"{slm_cfg.bit_depth}-bit)\n"
        f"Frames: {num_frames} over {float(encoded.t[-1]) * 1e9:.2f} ns"
    )
    return zip_path, summary