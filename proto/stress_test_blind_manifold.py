#!/usr/bin/env python3
"""Stress-test Level-2 blind quaternion manifold recovery under noise / orbs / γ₁."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from demo_core import build_config, run_pipeline
from orbital_braille import (
    OrbitalTypehead,
    quaternion_recovery_error,
    triplet_centre_field,
)
from orbital_braille.encode_redundancy import QEC_REPS, effective_num_times
from orbital_braille.orb_interference import measure_orb_interference


DEFAULT_NOISE_LEVELS = (0.35, 0.5, 0.7, 1.0)
DEFAULT_NUM_ORBS = (4, 6, 8, 12)
DEFAULT_GAMMA_1 = (1.5, 2.0)
QUICK_NOISE_LEVELS = (0.35, 0.7)
QUICK_NUM_ORBS = (4, 8)
QUICK_GAMMA_1 = (1.5,)
CLIFF_NOISE_LEVELS = (0.35, 0.5, 0.7)
CLIFF_NUM_ORBS = (4, 6, 8)
CLIFF_GAMMA_1 = (1.5, 2.0)
BASELINE_NOISE = 0.35
BASELINE_ORBS = 4
EARLY_EXIT_NFEV_THRESHOLD = 50


@dataclass(frozen=True)
class BlindManifoldStressRow:
    """One blind-manifold pipeline run with structured metrics."""

    noise_level: float
    num_orbs: int
    gamma_1: float
    quick: bool
    seed: int
    s3_error: float
    s3_degradation_ratio: float
    manifold_loss: float | None
    converged: bool | None
    used_fallback: bool | None
    orb_subtracted: bool | None
    carrier_w: float | None
    optimizer_nfev: int | None
    optimizer_nit: int | None
    optimizer_nfev_total: int | None
    carrier_grid_evals: int | None
    carrier_search_retried: bool
    orb_pairwise_corr: float
    effective_oam_modes: float
    shard_fidelity: float
    glyph_fidelity: float
    glyph_intensity_correlation: float
    glyph_index: int
    qec_logical_error: float | None
    runtime_s: float

    @property
    def loss_flag(self) -> str:
        if self.manifold_loss is None:
            return "n/a"
        if self.manifold_loss < 1e-5:
            return "ok"
        if self.manifold_loss < 1e-3:
            return "warn"
        return "bad"

    @property
    def s3_flag(self) -> str:
        if self.s3_error < 0.15:
            return "ok"
        if self.s3_error < 0.35:
            return "warn"
        return "bad"

    @property
    def early_exit(self) -> bool:
        return (
            self.optimizer_nfev_total is not None
            and self.optimizer_nfev_total < EARLY_EXIT_NFEV_THRESHOLD
        )


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _format_carrier_w(carrier_w: float | None) -> str:
    if carrier_w is None:
        return "n/a"
    return f"{carrier_w:.3f}"


def _manifold_params(
    num_orbs: int,
    glyph_rank_k: int,
    glyph_refine_k: int,
    carrier_grid_steps: int,
) -> tuple[int, int, int]:
    """Scale manifold search with orb count when using auto-tuning (capped)."""
    refine = max(glyph_refine_k, min(14, 6 + num_orbs // 2))
    grid = max(carrier_grid_steps, min(17, 9 + num_orbs // 3))
    rank = max(glyph_rank_k, refine * 2)
    return rank, refine, grid


def _centre_time_index(n_t: int) -> int:
    n_eff = effective_num_times(n_t, QEC_REPS)
    n_logical = n_eff // QEC_REPS
    mid_logical = n_logical // 2
    return mid_logical * QEC_REPS + QEC_REPS // 2


def _glyph_intensity_correlation(noisy: np.ndarray, encoded) -> float:
    """
    Complex-field coherence at the triplet centre slice vs clean transmit.

    Intensity Pearson is invariant under phase-only BMGL turbulence (|E|² unchanged),
    so this uses normalized complex inner product, which degrades with noise level.
    """
    centre_idx = _centre_time_index(noisy.shape[0])
    received = triplet_centre_field(noisy).ravel()
    clean = encoded.field_time[centre_idx].ravel()
    denom = float(np.linalg.norm(received) * np.linalg.norm(clean))
    if denom < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(received, clean)) / denom)


def _interference_metrics(encoded, decoded, typehead) -> tuple[float, float]:
    centre_idx = _centre_time_index(encoded.field_time.shape[0])
    t_val = float(encoded.t[centre_idx]) if centre_idx < encoded.t.shape[0] else float(encoded.t[-1] * 0.5)
    x_grid = encoded.rho * np.cos(encoded.phi)
    y_grid = encoded.rho * np.sin(encoded.phi)
    return measure_orb_interference(
        typehead.font,
        decoded.glyph_index,
        x_grid,
        y_grid,
        t_val,
        float(encoded.t[-1]),
        typehead.config.num_orbs,
        decoded.oam_weights,
        constants=typehead.config.constants,
    )


def run_blind_stress_case(
    payload: str,
    num_orbs: int,
    *,
    noise_level: float = 0.35,
    gamma_1: float = 1.5,
    seed: int = 42,
    quick: bool = True,
    glyph_rank_k: int = 24,
    glyph_refine_k: int = 12,
    carrier_grid_steps: int = 15,
    carrier_refine_top_k: int = 3,
    slsqp_maxiter: int = 120,
    auto_scale_manifold: bool = True,
    baseline_s3: float | None = None,
    pin_carrier_w: float | None = None,
    force_carrier_top_k: int | None = None,
    auto_retry_early_exit: bool = True,
) -> BlindManifoldStressRow:
    """Run one blind Level-2 case and return structured metrics."""
    rank, refine, grid = (
        _manifold_params(num_orbs, glyph_rank_k, glyph_refine_k, carrier_grid_steps)
        if auto_scale_manifold
        else (glyph_rank_k, glyph_refine_k, carrier_grid_steps)
    )

    t0 = time.perf_counter()
    _, encoded, noisy, decoded, _, _ = run_pipeline(
        payload,
        num_orbs,
        quick=quick,
        seed=seed,
        gamma_1=gamma_1,
        noise_level=noise_level,
        blind_quaternion=True,
        glyph_rank_k=rank,
        glyph_refine_k=refine,
        carrier_grid_steps=grid,
        carrier_refine_top_k=carrier_refine_top_k,
        slsqp_maxiter=slsqp_maxiter,
        pin_carrier_w=pin_carrier_w,
        force_carrier_top_k=force_carrier_top_k,
        auto_retry_early_exit=auto_retry_early_exit,
    )
    runtime_s = time.perf_counter() - t0

    cfg = build_config(num_orbs, quick=quick, gamma_1=gamma_1)
    typehead = OrbitalTypehead(cfg, seed=seed)

    mr = decoded.manifold_recovery
    qec_logical = (
        decoded.qec_stats.logical_error_rate if decoded.qec_stats is not None else None
    )
    s3_error = quaternion_recovery_error(encoded.quaternion, decoded.quaternion)
    orb_corr, oam_eff = _interference_metrics(encoded, decoded, typehead)
    glyph_ic = _glyph_intensity_correlation(noisy, encoded)
    base = baseline_s3 if baseline_s3 and baseline_s3 > 1e-12 else s3_error
    s3_ratio = s3_error / base

    return BlindManifoldStressRow(
        noise_level=noise_level,
        num_orbs=num_orbs,
        gamma_1=gamma_1,
        quick=quick,
        seed=seed,
        s3_error=s3_error,
        s3_degradation_ratio=s3_ratio,
        manifold_loss=None if mr is None else float(mr.loss),
        converged=None if mr is None else bool(mr.converged),
        used_fallback=None if mr is None else bool(mr.used_fallback),
        orb_subtracted=None if mr is None else bool(mr.orb_subtracted),
        carrier_w=None if mr is None else mr.carrier_w,
        optimizer_nfev=None if mr is None else mr.optimizer_nfev,
        optimizer_nit=None if mr is None else mr.optimizer_nit,
        optimizer_nfev_total=None if mr is None else mr.optimizer_nfev_total,
        carrier_grid_evals=None if mr is None else mr.carrier_grid_evals,
        carrier_search_retried=False if mr is None else bool(mr.carrier_search_retried),
        orb_pairwise_corr=orb_corr,
        effective_oam_modes=oam_eff,
        shard_fidelity=float(decoded.shard_fidelity),
        glyph_fidelity=float(decoded.glyph_fidelity),
        glyph_intensity_correlation=glyph_ic,
        glyph_index=int(decoded.glyph_index),
        qec_logical_error=qec_logical,
        runtime_s=runtime_s,
    )


def measure_blind_manifold_stress(
    payload: str = "I live in Oregon",
    *,
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    num_orbs_list: tuple[int, ...] = DEFAULT_NUM_ORBS,
    gamma_1_list: tuple[float, ...] = DEFAULT_GAMMA_1,
    seed: int = 42,
    quick: bool = True,
    glyph_rank_k: int = 24,
    glyph_refine_k: int = 12,
    carrier_grid_steps: int = 15,
    auto_scale_manifold: bool = True,
    carrier_refine_top_k: int = 3,
    slsqp_maxiter: int = 120,
    pin_carrier_w: float | None = None,
    force_carrier_top_k: int | None = None,
    auto_retry_early_exit: bool = True,
    on_progress: callable | None = None,
) -> list[BlindManifoldStressRow]:
    """Run the full 2D (× γ₁) stress matrix."""
    cases: list[tuple[float, int, float]] = []
    for gamma_1 in gamma_1_list:
        for num_orbs in num_orbs_list:
            for noise_level in noise_levels:
                cases.append((noise_level, num_orbs, gamma_1))
    total = len(cases)
    done = 0

    baseline_rows: dict[float, BlindManifoldStressRow] = {}
    for gamma_1 in gamma_1_list:
        row = run_blind_stress_case(
            payload,
            BASELINE_ORBS,
            noise_level=BASELINE_NOISE,
            gamma_1=gamma_1,
            seed=seed,
            quick=quick,
            glyph_rank_k=glyph_rank_k,
            glyph_refine_k=glyph_refine_k,
            carrier_grid_steps=carrier_grid_steps,
            carrier_refine_top_k=carrier_refine_top_k,
            slsqp_maxiter=slsqp_maxiter,
            auto_scale_manifold=auto_scale_manifold,
            pin_carrier_w=pin_carrier_w,
            force_carrier_top_k=force_carrier_top_k,
            auto_retry_early_exit=auto_retry_early_exit,
        )
        baseline_rows[gamma_1] = replace(row, s3_degradation_ratio=1.0)
        done += 1
        if on_progress is not None:
            on_progress(done, total, baseline_rows[gamma_1])

    rows: list[BlindManifoldStressRow] = []
    for gamma_1 in gamma_1_list:
        baseline_s3 = baseline_rows[gamma_1].s3_error
        for num_orbs in num_orbs_list:
            for noise_level in noise_levels:
                if noise_level == BASELINE_NOISE and num_orbs == BASELINE_ORBS:
                    rows.append(baseline_rows[gamma_1])
                    continue
                row = run_blind_stress_case(
                    payload,
                    num_orbs,
                    noise_level=noise_level,
                    gamma_1=gamma_1,
                    seed=seed,
                    quick=quick,
                    glyph_rank_k=glyph_rank_k,
                    glyph_refine_k=glyph_refine_k,
                    carrier_grid_steps=carrier_grid_steps,
                    carrier_refine_top_k=carrier_refine_top_k,
                    slsqp_maxiter=slsqp_maxiter,
                    auto_scale_manifold=auto_scale_manifold,
                    baseline_s3=baseline_s3,
                    pin_carrier_w=pin_carrier_w,
                    force_carrier_top_k=force_carrier_top_k,
                    auto_retry_early_exit=auto_retry_early_exit,
                )
                rows.append(row)
                done += 1
                if on_progress is not None:
                    on_progress(done, total, row)
    return rows


def format_stress_table(rows: list[BlindManifoldStressRow]) -> str:
    """Render a fixed-width table for terminal / logs."""
    headers = (
        "noise",
        "orbs",
        "γ₁",
        "S³ err",
        "S³ ratio",
        "loss",
        "conv",
        "nfev",
        "carr_w",
        "retry",
        "orbρ",
        "OAM_eff",
        "shard",
        "glyph",
        "glyphρ",
        "time(s)",
    )
    lines = [
        "Level-2 blind manifold stress (reference-free quaternion recovery)",
        f"Baseline: noise={BASELINE_NOISE}, orbs={BASELINE_ORBS} → S³ ratio=1.0 per γ₁",
        "",
        "  ".join(f"{h:>8}" for h in headers),
        "  ".join("-" * 8 for _ in headers),
    ]
    for r in rows:
        loss_s = f"{r.manifold_loss:.2e}" if r.manifold_loss is not None else "n/a"
        conv_s = "Y" if r.converged else ("N" if r.converged is not None else "-")
        nfev_s = (
            f"{r.optimizer_nfev_total}"
            if r.optimizer_nfev_total is not None
            else "n/a"
        )
        carrier_s = _format_carrier_w(r.carrier_w)
        if r.early_exit:
            carrier_s = f"{carrier_s}*"
        retry_s = "Y" if r.carrier_search_retried else "N"
        mode = "Q" if r.quick else "F"
        lines.append(
            "  ".join(
                [
                    f"{r.noise_level:8.2f}",
                    f"{r.num_orbs:8d}",
                    f"{r.gamma_1:8.1f}",
                    f"{r.s3_error:8.3f}",
                    f"{r.s3_degradation_ratio:8.2f}",
                    f"{loss_s:>8}",
                    f"{conv_s:>8}",
                    f"{nfev_s:>8}",
                    f"{carrier_s:>8}",
                    f"{retry_s:>8}",
                    f"{r.orb_pairwise_corr:8.3f}",
                    f"{r.effective_oam_modes:8.2f}",
                    f"{r.shard_fidelity:8.3f}",
                    f"{r.glyph_fidelity:8.3f}",
                    f"{r.glyph_intensity_correlation:8.4f}",
                    f"{r.runtime_s:8.1f}",
                ]
            )
            + f"  [{mode}]"
        )

    lines.extend(
        [
            "",
            "Legend: S³ ratio = s3_error / baseline(0.35,4,γ₁)  orbρ = mean pairwise orb |E|² corr",
            "        glyphρ = centre complex-field coherence vs clean transmit (noise-sensitive)",
            "        carr_w = winning carrier weight (* = early exit, nfev < 50)",
            "        retry = adaptive carrier search retried after shallow early exit",
            "        nfev = total SLSQP evals across carrier grid  OAM_eff = participation-ratio mode count",
            "Loss: ok <1e-5 | warn <1e-3 | bad ≥1e-3   S³: ok <0.15 | warn <0.35 | bad ≥0.35",
        ]
    )

    bad_loss = sum(1 for r in rows if r.loss_flag == "bad")
    bad_s3 = sum(1 for r in rows if r.s3_flag == "bad")
    fallbacks = sum(1 for r in rows if r.used_fallback)
    max_ratio = max((r.s3_degradation_ratio for r in rows), default=1.0)
    lines.extend(
        [
            "",
            f"Summary: {len(rows)} runs | loss-bad={bad_loss} | S³-bad={bad_s3} | "
            f"fallbacks={fallbacks} | max S³ ratio={max_ratio:.2f}",
            f"Total runtime: {sum(r.runtime_s for r in rows):.1f}s",
        ]
    )
    return "\n".join(lines)


def format_early_exit_summary(rows: list[BlindManifoldStressRow]) -> str:
    """List runs where SLSQP exited early (low nfev) with carrier_w for basin comparison."""
    early = [r for r in rows if r.early_exit]
    if not early:
        return "Early-exit runs (nfev < 50): none"

    lines = [
        f"Early-exit runs (nfev < {EARLY_EXIT_NFEV_THRESHOLD}): {len(early)}",
        "  noise    orbs      γ₁    S³ err   nfev   carr_w      loss",
        "  " + "-" * 54,
    ]
    for r in sorted(early, key=lambda row: (row.gamma_1, row.noise_level, row.num_orbs)):
        loss_s = f"{r.manifold_loss:.2e}" if r.manifold_loss is not None else "n/a"
        lines.append(
            f"  {r.noise_level:5.2f}  {r.num_orbs:5d}  {r.gamma_1:5.1f}  "
            f"{r.s3_error:7.3f}  {r.optimizer_nfev_total:5d}  "
            f"{_format_carrier_w(r.carrier_w):6s}  {loss_s:>8}"
        )
    return "\n".join(lines)


def _case_key(row: BlindManifoldStressRow) -> tuple[float, int, float]:
    return (row.noise_level, row.num_orbs, row.gamma_1)


def run_auto_retry_comparison(
    payload: str = "I live in Oregon",
    *,
    noise_levels: tuple[float, ...] = CLIFF_NOISE_LEVELS,
    num_orbs_list: tuple[int, ...] = CLIFF_NUM_ORBS,
    gamma_1_list: tuple[float, ...] = CLIFF_GAMMA_1,
    seed: int = 42,
    quick: bool = True,
    glyph_rank_k: int = 24,
    glyph_refine_k: int = 12,
    carrier_grid_steps: int = 15,
    auto_scale_manifold: bool = True,
    carrier_refine_top_k: int = 3,
    slsqp_maxiter: int = 120,
    on_progress: callable | None = None,
) -> tuple[list[BlindManifoldStressRow], list[BlindManifoldStressRow]]:
    """Run cliff-region matrix with auto-retry off vs on (paired before/after)."""
    baseline_s3: dict[float, float] = {}
    for gamma_1 in gamma_1_list:
        row = run_blind_stress_case(
            payload,
            BASELINE_ORBS,
            noise_level=BASELINE_NOISE,
            gamma_1=gamma_1,
            seed=seed,
            quick=quick,
            glyph_rank_k=glyph_rank_k,
            glyph_refine_k=glyph_refine_k,
            carrier_grid_steps=carrier_grid_steps,
            carrier_refine_top_k=carrier_refine_top_k,
            slsqp_maxiter=slsqp_maxiter,
            auto_scale_manifold=auto_scale_manifold,
            auto_retry_early_exit=False,
        )
        baseline_s3[gamma_1] = row.s3_error

    cases: list[tuple[float, int, float]] = []
    for gamma_1 in gamma_1_list:
        for num_orbs in num_orbs_list:
            for noise_level in noise_levels:
                cases.append((noise_level, num_orbs, gamma_1))

    before_rows: list[BlindManifoldStressRow] = []
    after_rows: list[BlindManifoldStressRow] = []
    total = len(cases) * 2
    done = 0

    for noise_level, num_orbs, gamma_1 in cases:
        kwargs = dict(
            payload=payload,
            num_orbs=num_orbs,
            noise_level=noise_level,
            gamma_1=gamma_1,
            seed=seed,
            quick=quick,
            glyph_rank_k=glyph_rank_k,
            glyph_refine_k=glyph_refine_k,
            carrier_grid_steps=carrier_grid_steps,
            carrier_refine_top_k=carrier_refine_top_k,
            slsqp_maxiter=slsqp_maxiter,
            auto_scale_manifold=auto_scale_manifold,
            baseline_s3=baseline_s3[gamma_1],
        )
        before = run_blind_stress_case(**kwargs, auto_retry_early_exit=False)
        done += 1
        if on_progress is not None:
            on_progress(done, total, before)
        before_rows.append(before)

        after = run_blind_stress_case(**kwargs, auto_retry_early_exit=True)
        done += 1
        if on_progress is not None:
            on_progress(done, total, after)
        after_rows.append(after)

    return before_rows, after_rows


def format_auto_retry_comparison(
    before_rows: list[BlindManifoldStressRow],
    after_rows: list[BlindManifoldStressRow],
) -> str:
    """Side-by-side before/after table for adaptive carrier retry."""
    after_by_key = {_case_key(row): row for row in after_rows}
    lines = [
        "Adaptive auto-retry comparison (no_retry → auto_retry)",
        "",
        "  noise    orbs      γ₁   S³ before  S³ after   ΔS³    nfev B  nfev A  w B    w A   retry",
        "  " + "-" * 78,
    ]
    improved = 0
    retried = 0
    worsened = 0
    unchanged = 0
    total_delta = 0.0

    for before in sorted(before_rows, key=_case_key):
        after = after_by_key.get(_case_key(before))
        if after is None:
            continue
        delta = after.s3_error - before.s3_error
        total_delta += delta
        if after.carrier_search_retried:
            retried += 1
        if delta < -0.005:
            improved += 1
        elif delta > 0.005:
            worsened += 1
        else:
            unchanged += 1
        lines.append(
            f"  {before.noise_level:5.2f}  {before.num_orbs:5d}  {before.gamma_1:5.1f}  "
            f"{before.s3_error:8.3f}  {after.s3_error:8.3f}  {delta:+7.3f}  "
            f"{before.optimizer_nfev_total:6d}  {after.optimizer_nfev_total:6d}  "
            f"{_format_carrier_w(before.carrier_w):>5}  {_format_carrier_w(after.carrier_w):>5}  "
            f"{'Y' if after.carrier_search_retried else 'N':>5}"
        )

    n = len(before_rows)
    lines.extend(
        [
            "",
            f"Summary: {n} pairs | improved={improved} | unchanged={unchanged} | "
            f"worsened={worsened} | retries={retried} | mean ΔS³={total_delta / max(n, 1):+.4f}",
            f"Runtime: before={sum(r.runtime_s for r in before_rows):.1f}s  "
            f"after={sum(r.runtime_s for r in after_rows):.1f}s",
        ]
    )
    return "\n".join(lines)


def format_gamma_carrier_comparison(
    rows: list[BlindManifoldStressRow],
    *,
    noise_level: float = 0.5,
) -> str:
    """Side-by-side carrier_w / S³ / nfev at a fixed noise level across γ₁ values."""
    gamma_vals = sorted({r.gamma_1 for r in rows})
    if len(gamma_vals) < 2:
        return ""

    orb_vals = sorted({r.num_orbs for r in rows})
    lines = [
        "",
        f"γ₁ carrier comparison at noise={noise_level:.2f}",
        "  orbs      γ₁    S³ err   nfev   carr_w      loss   early?",
        "  " + "-" * 58,
    ]
    for n_orb in orb_vals:
        for gamma_1 in gamma_vals:
            match = [
                r for r in rows
                if r.num_orbs == n_orb
                and r.gamma_1 == gamma_1
                and abs(r.noise_level - noise_level) < 1e-9
            ]
            if not match:
                continue
            r = match[0]
            loss_s = f"{r.manifold_loss:.2e}" if r.manifold_loss is not None else "n/a"
            early_s = "Y" if r.early_exit else "N"
            lines.append(
                f"  {n_orb:5d}  {gamma_1:5.1f}  {r.s3_error:7.3f}  "
                f"{r.optimizer_nfev_total:5d}  {_format_carrier_w(r.carrier_w):6s}  "
                f"{loss_s:>8}  {early_s:>6}"
            )
    return "\n".join(lines)


def write_stress_plots(rows: list[BlindManifoldStressRow], out_dir: Path) -> list[Path]:
    """Generate S³, glyph fidelity, and loss heatmap figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    gamma_vals = sorted({r.gamma_1 for r in rows})
    for gamma_1 in gamma_vals:
        sub = [r for r in rows if r.gamma_1 == gamma_1]
        orb_vals = sorted({r.num_orbs for r in sub})
        noise_vals = sorted({r.noise_level for r in sub})

        # S³ error vs noise (lines per orb count)
        fig, ax = plt.subplots(figsize=(8, 5))
        for n_orb in orb_vals:
            pts = sorted(
                (r.noise_level, r.s3_error) for r in sub if r.num_orbs == n_orb
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{n_orb} orbs")
        ax.set_xlabel("Channel noise level")
        ax.set_ylabel("S³ recovery error")
        ax.set_title(f"Blind manifold S³ error vs noise (γ₁={gamma_1})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / f"s3_vs_noise_gamma{gamma_1:.1f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

        # Glyph fidelity vs orbs (template metric; noise-independent)
        fig, ax = plt.subplots(figsize=(8, 5))
        by_orbs: dict[int, float] = {}
        for r in sub:
            by_orbs.setdefault(r.num_orbs, r.glyph_fidelity)
        pts = sorted(by_orbs.items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color="C0")
        ax.set_xlabel("Number of orbs")
        ax.set_ylabel("Glyph fidelity (template)")
        ax.set_title(f"Glyph fidelity vs orb count (γ₁={gamma_1}, noise-independent)")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"glyph_vs_orbs_gamma{gamma_1:.1f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

        # Noise-sensitive glyph intensity correlation vs noise
        fig, ax = plt.subplots(figsize=(8, 5))
        for n_orb in orb_vals:
            pts = sorted(
                (r.noise_level, r.glyph_intensity_correlation)
                for r in sub
                if r.num_orbs == n_orb
            )
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker="o",
                label=f"{n_orb} orbs",
            )
        ax.set_xlabel("Channel noise level")
        ax.set_ylabel("Centre field coherence")
        ax.set_title(f"Centre field coherence vs noise (γ₁={gamma_1})")
        vals = [r.glyph_intensity_correlation for r in sub]
        if vals:
            lo, hi = min(vals), max(vals)
            span = hi - lo
            pad = max(2e-4, span * 0.35)
            ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.ticklabel_format(useOffset=False, axis="y")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax.legend(title="Orbs")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"glyph_intensity_vs_noise_gamma{gamma_1:.1f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

        # Manifold loss heatmap (orbs × noise)
        grid = np.full((len(orb_vals), len(noise_vals)), np.nan)
        for i, n_orb in enumerate(orb_vals):
            for j, noise in enumerate(noise_vals):
                match = [
                    r for r in sub
                    if r.num_orbs == n_orb and r.noise_level == noise
                ]
                if match and match[0].manifold_loss is not None:
                    grid[i, j] = np.log10(max(match[0].manifold_loss, 1e-20))
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(noise_vals)), [f"{n:.2f}" for n in noise_vals])
        ax.set_yticks(range(len(orb_vals)), [str(n) for n in orb_vals])
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Orbs")
        ax.set_title(f"log₁₀(manifold loss) heatmap (γ₁={gamma_1})")
        fig.colorbar(im, ax=ax, label="log₁₀ loss")
        path = out_dir / f"loss_heatmap_gamma{gamma_1:.1f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

        # Orb interference vs S³ ratio (colour by orbs, size by S³ ratio)
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(orb_vals), vmax=max(orb_vals))
        sizes = [
            50.0 + 150.0 * max(0.0, r.s3_degradation_ratio - 1.0)
            for r in sub
        ]
        sc = ax.scatter(
            [r.orb_pairwise_corr for r in sub],
            [r.s3_degradation_ratio for r in sub],
            c=[r.num_orbs for r in sub],
            cmap=cmap,
            norm=norm,
            s=sizes,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
        )
        for n_orb in orb_vals:
            ax.scatter([], [], c=[cmap(norm(n_orb))], s=80, label=f"{n_orb} orbs")
        ax.set_xlabel("Mean pairwise orb intensity correlation (orbρ)")
        ax.set_ylabel("S³ degradation ratio")
        ax.set_title(f"Orb crowding vs quaternion degradation (γ₁={gamma_1})")
        ax.legend(title="Orbs", loc="upper left")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"interference_vs_s3_gamma{gamma_1:.1f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

        # Carrier weight vs noise (basin selection diagnostic)
        fig, ax = plt.subplots(figsize=(8, 5))
        for n_orb in orb_vals:
            pts = sorted(
                (r.noise_level, r.carrier_w)
                for r in sub
                if r.num_orbs == n_orb and r.carrier_w is not None
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=f"{n_orb} orbs")
            for x, y, row in [
                (r.noise_level, r.carrier_w, r)
                for r in sub
                if r.num_orbs == n_orb and r.carrier_w is not None and r.early_exit
            ]:
                ax.scatter([x], [y], s=120, facecolors="none", edgecolors="red", linewidths=1.5)
        ax.set_xlabel("Channel noise level")
        ax.set_ylabel("Winning carrier weight")
        ax.set_title(f"Carrier basin vs noise (γ₁={gamma_1}; red ring = early exit)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Orbs")
        ax.grid(True, alpha=0.3)
        path = out_dir / f"carrier_w_vs_noise_gamma{gamma_1:.1f}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", default="I live in Oregon")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--full",
        action="store_true",
        help="Use full-resolution grid/times",
    )
    p.add_argument(
        "--subset",
        action="store_true",
        help="Smaller matrix: noise 0.35/0.7, orbs 4/8, γ₁=1.5 only",
    )
    p.add_argument(
        "--noise-levels",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated noise levels (e.g. 0.35,0.5,0.7)",
    )
    p.add_argument(
        "--orbs",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated orb counts (e.g. 4,6,8)",
    )
    p.add_argument(
        "--gamma-values",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated γ₁ values (e.g. 1.5,2.0)",
    )
    p.add_argument(
        "--pin-carrier-w",
        type=float,
        default=None,
        metavar="W",
        help="Pin carrier weight (skip grid search; debug basin hypothesis)",
    )
    p.add_argument(
        "--force-carrier-top-k",
        type=int,
        default=None,
        metavar="K",
        help="Always refine top-K warm carriers + clean-basin anchors",
    )
    p.add_argument(
        "--no-auto-retry",
        action="store_true",
        help="Disable adaptive carrier retry on shallow early exit",
    )
    p.add_argument(
        "--compare-auto-retry",
        action="store_true",
        help="Cliff-region before/after: no_retry vs auto_retry (paired runs)",
    )
    p.add_argument(
        "--cliff-region",
        action="store_true",
        help="Matrix preset: noise 0.35/0.5/0.7, orbs 4/6/8, γ₁=1.5/2.0",
    )
    p.add_argument("--glyph-rank-k", type=int, default=24)
    p.add_argument("--glyph-refine-k", type=int, default=12)
    p.add_argument("--carrier-grid-steps", type=int, default=15)
    p.add_argument("--carrier-refine-top-k", type=int, default=5)
    p.add_argument("--slsqp-maxiter", type=int, default=120)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster sweep: fewer glyphs/carrier SLSQP (≈5–8× speedup)",
    )
    p.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable orb-count scaling of glyph/carrier search",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Write PNG plots to --out-dir",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stress_output"),
        help="Directory for plot output (default: stress_output)",
    )
    return p.parse_args()


def _print_progress(done: int, total: int, row: BlindManifoldStressRow) -> None:
    carrier_s = _format_carrier_w(row.carrier_w)
    nfev_s = (
        str(row.optimizer_nfev_total)
        if row.optimizer_nfev_total is not None
        else "n/a"
    )
    early_s = " early" if row.early_exit else ""
    retry_s = " retry" if row.carrier_search_retried else ""
    print(
        f"[{done:2d}/{total}] noise={row.noise_level:.2f} orbs={row.num_orbs:2d} "
        f"γ₁={row.gamma_1:.1f}  S³={row.s3_error:.3f} ratio={row.s3_degradation_ratio:.2f} "
        f"loss={row.manifold_loss:.1e}  nfev={nfev_s}  w={carrier_s}{early_s}{retry_s}  "
        f"{row.runtime_s:.0f}s",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    quick = not args.full

    glyph_rank_k = args.glyph_rank_k
    glyph_refine_k = args.glyph_refine_k
    carrier_grid_steps = args.carrier_grid_steps
    carrier_refine_top_k = args.carrier_refine_top_k
    slsqp_maxiter = args.slsqp_maxiter
    auto_scale = not args.no_auto_scale
    if args.fast:
        glyph_rank_k = min(glyph_rank_k, 16)
        glyph_refine_k = min(glyph_refine_k, 6)
        carrier_grid_steps = min(carrier_grid_steps, 11)
        carrier_refine_top_k = min(carrier_refine_top_k, 2)
        slsqp_maxiter = min(slsqp_maxiter, 80)
        auto_scale = False

    use_cliff = args.cliff_region or (
        args.compare_auto_retry
        and args.noise_levels is None
        and args.orbs is None
        and args.gamma_values is None
    )
    if use_cliff:
        noise_levels = CLIFF_NOISE_LEVELS
        num_orbs_list = CLIFF_NUM_ORBS
        gamma_1_list = CLIFF_GAMMA_1
    elif args.noise_levels is not None:
        noise_levels = _parse_float_list(args.noise_levels)
    elif args.subset:
        noise_levels = QUICK_NOISE_LEVELS
    else:
        noise_levels = DEFAULT_NOISE_LEVELS

    if not use_cliff:
        if args.orbs is not None:
            num_orbs_list = _parse_int_list(args.orbs)
        elif args.subset:
            num_orbs_list = QUICK_NUM_ORBS
        else:
            num_orbs_list = DEFAULT_NUM_ORBS

        if args.gamma_values is not None:
            gamma_1_list = _parse_float_list(args.gamma_values)
        elif args.subset:
            gamma_1_list = QUICK_GAMMA_1
        else:
            gamma_1_list = DEFAULT_GAMMA_1

    n_cases = len(noise_levels) * len(num_orbs_list) * len(gamma_1_list)
    pin_carrier_w = args.pin_carrier_w
    force_carrier_top_k = args.force_carrier_top_k
    extra_notes: list[str] = []
    if pin_carrier_w is not None:
        extra_notes.append(f"pin_w={pin_carrier_w:.3f}")
    if force_carrier_top_k is not None:
        extra_notes.append(f"force_k={force_carrier_top_k}")
    if args.no_auto_retry:
        extra_notes.append("no_retry")
    extra_note = f", {', '.join(extra_notes)}" if extra_notes else ""
    if args.compare_auto_retry:
        n_pairs = n_cases
        print(
            f"Starting {n_pairs}-pair auto-retry comparison ({n_pairs * 2} runs) "
            f"(quick={quick}, refine_k={glyph_refine_k}, grid={carrier_grid_steps}, "
            f"top_k={carrier_refine_top_k}{extra_note})",
            flush=True,
        )
        before_rows, after_rows = run_auto_retry_comparison(
            args.payload,
            noise_levels=noise_levels,
            num_orbs_list=num_orbs_list,
            gamma_1_list=gamma_1_list,
            seed=args.seed,
            quick=quick,
            glyph_rank_k=glyph_rank_k,
            glyph_refine_k=glyph_refine_k,
            carrier_grid_steps=carrier_grid_steps,
            carrier_refine_top_k=carrier_refine_top_k,
            slsqp_maxiter=slsqp_maxiter,
            auto_scale_manifold=auto_scale,
            on_progress=_print_progress,
        )
        print(format_auto_retry_comparison(before_rows, after_rows))
        return

    print(
        f"Starting {n_cases}-case blind manifold sweep "
        f"(quick={quick}, fast={args.fast}, refine_k={glyph_refine_k}, "
        f"grid={carrier_grid_steps}, top_k={carrier_refine_top_k}{extra_note})",
        flush=True,
    )

    rows = measure_blind_manifold_stress(
        args.payload,
        noise_levels=noise_levels,
        num_orbs_list=num_orbs_list,
        gamma_1_list=gamma_1_list,
        seed=args.seed,
        quick=quick,
        glyph_rank_k=glyph_rank_k,
        glyph_refine_k=glyph_refine_k,
        carrier_grid_steps=carrier_grid_steps,
        carrier_refine_top_k=carrier_refine_top_k,
        slsqp_maxiter=slsqp_maxiter,
        auto_scale_manifold=auto_scale,
        pin_carrier_w=pin_carrier_w,
        force_carrier_top_k=force_carrier_top_k,
        auto_retry_early_exit=not args.no_auto_retry,
        on_progress=_print_progress,
    )
    print(format_stress_table(rows))
    print()
    print(format_early_exit_summary(rows))
    gamma_cmp = format_gamma_carrier_comparison(rows, noise_level=0.5)
    if gamma_cmp:
        print(gamma_cmp)

    if args.plots:
        paths = write_stress_plots(rows, args.out_dir)
        print(f"\nWrote {len(paths)} plots to {args.out_dir.resolve()}/")
        for p in paths:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()