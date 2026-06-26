"""Discover and load Orbital Braille prototype outputs for the Streamlit dashboard."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProtoAsset:
    path: str
    kind: str
    label: str
    mtime: float


@dataclass
class ProtoBundle:
    root: str
    demo_png: str | None = None
    slm_montages: list[ProtoAsset] = field(default_factory=list)
    slm_manifests: list[ProtoAsset] = field(default_factory=list)
    meta_json: list[ProtoAsset] = field(default_factory=list)
    all_pngs: list[ProtoAsset] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.demo_png or self.all_pngs or self.slm_manifests or self.meta_json)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _asset(path: Path, kind: str, label: str | None = None) -> ProtoAsset:
    st = path.stat()
    return ProtoAsset(
        path=str(path),
        kind=kind,
        label=label or path.name,
        mtime=st.st_mtime,
    )


def discover_proto_outputs(proto_root: str | Path | None = None) -> ProtoBundle:
    """Scan proto/outputs for demo figures, SLM packages, and meta-optimization JSON."""
    root = Path(proto_root) if proto_root else _repo_root() / "proto" / "outputs"
    bundle = ProtoBundle(root=str(root))

    if not root.is_dir():
        return bundle

    demo = root / "orbital_braille_demo.png"
    if demo.is_file():
        bundle.demo_png = str(demo)

    for png in sorted(root.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = png.relative_to(root)
        if png == demo:
            continue
        kind = "slm_montage" if png.name == "preview_montage.png" else "png"
        asset = _asset(png, kind, str(rel))
        bundle.all_pngs.append(asset)
        if kind == "slm_montage":
            bundle.slm_montages.append(asset)

    for manifest in sorted(root.rglob("manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        bundle.slm_manifests.append(_asset(manifest, "manifest", str(manifest.relative_to(root))))

    meta_dir = root / "meta"
    if meta_dir.is_dir():
        for js in sorted(meta_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            bundle.meta_json.append(_asset(js, "meta", js.name))

    return bundle


def load_manifest_summary(path: str) -> dict:
    """Return a small summary dict from an SLM manifest.json."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        k: data[k]
        for k in ("device", "num_orbs", "payload", "frames", "wavelength_nm", "glyph_duties")
        if k in data
    }


def latest_proto_demo(proto_root: str | Path | None = None) -> str | None:
    """Return path to the newest orbital_braille_demo.png if present."""
    bundle = discover_proto_outputs(proto_root)
    if bundle.demo_png and os.path.isfile(bundle.demo_png):
        return bundle.demo_png
    pngs = sorted(bundle.all_pngs, key=lambda a: a.mtime, reverse=True)
    return pngs[0].path if pngs else None