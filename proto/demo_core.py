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
| VQC claim element | What you see in this demo |
|-------------------|---------------------------|
| **Pyramidal FM pulses** | Bottom-left panel — triangular chirp envelope in time |
| **Spectral shards** | Bottom-middle Welch PSD — discrete barcode peaks |
| **Quaternion encoding** | Metrics block — unit quaternion from payload bytes |
| **OAM mode multiplex** | Orb layout panel — ℓ charge per PWM-gated source |
| **Nested helical shielding** | Encoded phase panel — concentric orbit geometry |
| **p-wave BMGL (γ₁)** | Turbulent phase panel — inhibition strength via γ₁ slider |
| **16-qubit QEC proxy** | Decoder majority-vote repetition in `altermagnetic.py` |
| **SLM virtual typehead** | Download zip — `manifest.json` + `phase_stack.npy` (+ optional frames) |

**Patent Figure 1 payload:** `"I live in Oregon"` (4 orbs) — use **Load example from paper**.

Full mapping: [proto/README.md — Patent claim alignment](https://github.com/kinaar8340/vqc_proto/blob/main/proto/README.md#patent-claim-alignment)
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

    zip_path = out_dir.parent / "slm_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    _zip_directory(out_dir, zip_path)

    contents = "manifest.json, phase_stack.npy, preview_montage.png, LUT_calibration.txt"
    if include_frames:
        contents += f", frames/ ({num_frames} phase PNGs)"
    summary = (
        f"SLM package: {contents}\n"
        f"Device: {device_preset} ({slm_cfg.resolution_x}×{slm_cfg.resolution_y}, "
        f"{slm_cfg.bit_depth}-bit)\n"
        f"Frames: {num_frames} over {float(encoded.t[-1]) * 1e9:.2f} ns"
    )
    return zip_path, summary