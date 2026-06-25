#!/usr/bin/env python3
"""
Generate hardware-ready SLM hologram package for 4-orb virtual typehead.

Exports phase frames (PNG/TIFF + raw), manifest.json, LUT notes, and preview
montage — load directly onto phase-only SLMs (Holoeye, Meadowlark, Thorlabs).

Example:
    python generate_slm_holograms.py --payload "I live in Oregon" --device holoeye_pluto_2
    python generate_slm_holograms.py --device generic_512 --frames 64 --gerchberg-saxton
"""

from __future__ import annotations

import argparse
from pathlib import Path

from orbital_braille import OrbitalTypehead, TypeheadConfig, PWaveBMGL, EmergentConstants
from orbital_braille.font_optimizer import optimize_font
from orbital_braille.slm_typehead import (
    SLM_PRESETS,
    SLMConfig,
    export_hologram_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SLM phase holograms for VQC Orbital Braille virtual typehead",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--payload", default="I live in Oregon", help="Text payload to encode")
    parser.add_argument("--num-orbs", type=int, default=4, help="Orb count (4 recommended)")
    parser.add_argument("--frames", type=int, default=32, help="Animation frames in sequence")
    parser.add_argument(
        "--device",
        choices=list(SLM_PRESETS.keys()),
        default="generic_512",
        help="SLM device preset (resolution, pitch, bit depth)",
    )
    parser.add_argument("--wavelength-nm", type=float, default=None, help="Override laser wavelength")
    parser.add_argument("--extent-mm", type=float, default=4.0, help="SLM active area width")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--gerchberg-saxton",
        action="store_true",
        help="Apply Gerchberg-Saxton phase retrieval per frame (slower, sharper far-field)",
    )
    parser.add_argument("--gs-iter", type=int, default=24, help="GS iterations per frame")
    parser.add_argument("--no-raw", action="store_true", help="Skip .raw binary frame export")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preset = SLM_PRESETS[args.device]
    slm_cfg = SLMConfig.from_preset(preset, extent_mm=args.extent_mm)
    if args.wavelength_nm is not None:
        slm_cfg.wavelength_nm = args.wavelength_nm

    cfg = TypeheadConfig(
        num_orbs=args.num_orbs,
        bmgl=PWaveBMGL(gamma_1=1.5),
        constants=EmergentConstants(),
    )
    th = OrbitalTypehead(cfg, seed=args.seed)
    font = optimize_font(num_orbs=args.num_orbs, num_glyphs=16)
    th.font = font.font

    enc = th.encode(args.payload)
    out_dir = args.out_dir or (
        Path(__file__).parent / "outputs" / "slm" / f"{args.device}_{args.num_orbs}orb"
    )

    summary = export_hologram_package(
        orbs=enc.orbs,
        t_max=float(enc.t[-1]),
        out_dir=out_dir,
        cfg=slm_cfg,
        payload=args.payload,
        quaternion=enc.quaternion,
        glyph_duties=enc.glyph_duties,
        num_frames=args.frames,
        device_preset=args.device,
        use_gs=args.gerchberg_saxton,
        gs_iter=args.gs_iter,
        export_raw=not args.no_raw,
        font_separation=font.mean_separation,
    )

    print("=" * 60)
    print("SLM HOLOGRAM PACKAGE — 4-ORB VIRTUAL TYPEHEAD")
    print("=" * 60)
    print(f"Payload:     {args.payload!r}")
    print(f"Device:      {summary['device']} ({summary['resolution']}, {summary['bit_depth']}-bit)")
    print(f"Frames:      {summary['frames']}")
    print(f"GS refine:   {summary['use_gs']}")
    print(f"Font sep:    {font.mean_separation:.4f} rad")
    print(f"Quaternion:  w={enc.quaternion.w:.3f} x={enc.quaternion.x:.3f} "
          f"y={enc.quaternion.y:.3f} z={enc.quaternion.z:.3f}")
    print(f"Output:      {summary['out_dir']}/")
    print("  frames/          phase_XXXX.png (+ .raw)")
    print("  manifest.json    orb geometry + timing for bench")
    print("  preview_montage.png")
    print("  LUT_calibration.txt")
    print("  phase_stack.npy")
    print()
    print("Next: see SLM_QUICKSTART.md for optical bench setup.")


if __name__ == "__main__":
    main()