"""Shared encode/decode/plot helpers for run_demo, Gradio, and HF Spaces."""

from __future__ import annotations

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