"""Minimal [[3,1,3]] bit-flip repetition code for Orbital Braille decode.

Replaces the stochastic repetition_qec proxy with a real stabilizer decoder:
  - Stabilizers: Z₀Z₁, Z₁Z₂  (detect X / bit-flip errors on physical qubits)
  - Syndrome: (b₀⊕b₁, b₁⊕b₂)
  - Correction: single-bit lookup then majority vote

Continuous intensity fields are grouped in triplets; each triplet encodes one
logical bit (median threshold).  When a clean ``reference`` is supplied, error
rates are measured against it.  Optional ``simulate_channel`` injects bit-flips
for isolated code benchmarks (BMGL turbulence remains the primary channel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .altermagnetic import PWaveBMGL


@dataclass(frozen=True)
class QECStats:
    """Measured error rates for one decode pass."""

    physical_error_rate: float
    logical_error_rate: float
    n_groups: int
    code_name: str = "[[3,1,3]] bit-flip repetition"
    syndromes_detected: int = 0
    p_physical_effective: float = 0.0
    logical_error_rate_pre: float = 0.0


@dataclass(frozen=True)
class BitFlipRepetitionCode:
    """[[3,1,3]] bit-flip repetition code with explicit stabilizer decode."""

    n_physical: int = 3

    def encode(self, logical: int) -> np.ndarray:
        bit = int(logical) & 1
        return np.full(self.n_physical, bit, dtype=np.int8)

    def syndrome(self, physical: np.ndarray) -> tuple[int, int]:
        b = physical.astype(np.int8) & 1
        return int(b[0] ^ b[1]), int(b[1] ^ b[2])

    def correct(self, physical: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Apply syndrome-based correction for a single bit-flip."""
        b = physical.astype(np.int8).copy() & 1
        s0, s1 = int(b[0] ^ b[1]), int(b[1] ^ b[2])
        if s0 and not s1:
            b[0] ^= 1
        elif s0 and s1:
            b[1] ^= 1
        elif not s0 and s1:
            b[2] ^= 1
        return b, (s0, s1)

    def decode_logical(self, physical: np.ndarray) -> int:
        corrected, _ = self.correct(physical)
        return int(np.round(corrected.mean())) & 1

    def apply_bitflip_channel(
        self,
        physical: np.ndarray,
        p_flip: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        corrupted = physical.astype(np.int8).copy() & 1
        flips = rng.random(self.n_physical) < p_flip
        corrupted ^= flips.astype(np.int8)
        return corrupted


def physical_error_rate(bmgl: PWaveBMGL, *, noise_scale: float = 1.0) -> float:
    """Map BMGL parameters to per-physical-qubit bit-flip probability."""
    p = bmgl.alpha_chemical * noise_scale
    p /= max(bmgl.effective_inhibition, 1e-12)
    return float(np.clip(p, 0.0, 0.5))


def _logical_majority(bits: np.ndarray) -> int:
    return int(np.round(bits.mean())) & 1


def _group_triplets_tensor(data: np.ndarray, reps: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Group along time axis: each (y,x) pixel gets ``reps`` temporal samples."""
    if data.ndim != 3:
        flat = data.ravel()
        pad = (-len(flat)) % reps
        if pad:
            flat = np.concatenate([flat, np.full(pad, flat[-1] if flat.size else 0.0)])
        groups = flat.reshape(-1, reps)
        return groups, groups, data.shape

    n_t, ny, nx = data.shape
    pad = (-n_t) % reps
    if pad:
        data = np.concatenate([data, np.tile(data[-1:], (pad, 1, 1))], axis=0)
    n_groups_t = data.shape[0] // reps
    # (n_groups_t, reps, ny, nx) → (n_groups_t*ny*nx, reps)
    stacked = data.reshape(n_groups_t, reps, ny, nx).transpose(0, 2, 3, 1).reshape(-1, reps)
    return stacked, data, (n_t, ny, nx)


def _logical_bit(g: np.ndarray, threshold: float) -> int:
    return int(np.mean(g > threshold) >= 0.5)


def _bits_from_group(
    g: np.ndarray,
    code: BitFlipRepetitionCode,
    *,
    threshold: float | None = None,
    as_codeword: bool = False,
) -> np.ndarray:
    thr = float(np.median(g)) if threshold is None else threshold
    if as_codeword:
        return code.encode(_logical_bit(g, thr))
    if g.std() < 1e-15:
        return code.encode(int(thr > 0.0))
    return (g > thr).astype(np.int8)


def bitflip_repetition_qec(
    data: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    error_rate: float | None = None,
    bmgl: PWaveBMGL | None = None,
    noise_scale: float = 1.0,
    reps: int = 3,
    simulate_channel: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, QECStats]:
    """
    Apply [[3,1,3]] bit-flip repetition decode to a continuous intensity tensor.

    Default (decode) mode treats ``data`` as noisy channel output (after BMGL).
    Stabilizer syndromes correct single bit-flips per triplet.

    With ``reference`` (clean intensity), physical/logical error rates are
    measured before and after correction.
    """
    if reps != 3:
        raise ValueError("bitflip_repetition_qec requires reps=3 for [[3,1,3]] code")

    rng = rng or np.random.default_rng()
    p_effective = error_rate
    if p_effective is None:
        p_effective = 0.015 if bmgl is None else physical_error_rate(bmgl, noise_scale=noise_scale)

    code = BitFlipRepetitionCode()
    orig_shape = data.shape
    data = np.asarray(data, dtype=np.float64)
    groups, padded_data, shape_key = _group_triplets_tensor(data, reps)

    ref_groups = None
    global_thresh: float | None = None
    if reference is not None:
        ref_arr = np.asarray(reference, dtype=np.float64)
        if ref_arr.shape != orig_shape:
            ref_arr = np.broadcast_to(ref_arr, orig_shape)
        global_thresh = float(np.median(ref_arr))
        ref_groups, _, _ = _group_triplets_tensor(ref_arr, reps)

    out = np.empty_like(groups)
    phys_err = 0.0
    log_err_pre = 0.0
    log_err_post = 0.0
    syndromes = 0
    n_groups = len(groups)

    for i, g in enumerate(groups):
        ref_threshold = float(np.median(ref_groups[i])) if ref_groups is not None else None
        if ref_groups is not None:
            ref_g = ref_groups[i]
            ref_val = float(ref_g[0])
            assert global_thresh is not None
            logical_ref = 1 if ref_val > global_thresh else 0
            ref_bits = code.encode(logical_ref)
            # Binarize against global field median so protected triplets share one logical bit.
            received_bits = (g > global_thresh).astype(np.int8)
        else:
            received_bits = _bits_from_group(g, code)
            logical_ref = code.decode_logical(received_bits)

        if simulate_channel:
            received_bits = code.apply_bitflip_channel(received_bits, p_effective, rng)

        if ref_groups is not None:
            phys_err += float(np.mean(received_bits != ref_bits))
            log_err_pre += float(_logical_majority(received_bits) != logical_ref)

        corrected, (s0, s1) = code.correct(received_bits)
        if s0 or s1:
            syndromes += 1
        logical_post = code.decode_logical(corrected)
        log_err_post += float(logical_post != logical_ref)

        if ref_groups is not None:
            consensus = float(np.median(ref_groups[i]))
        else:
            consensus = float(np.median(g))
        out[i, :] = consensus

    if len(shape_key) == 3:
        n_t, ny, nx = shape_key
        pad = padded_data.shape[0] - n_t
        n_groups_t = padded_data.shape[0] // reps
        out_tensor = out.reshape(n_groups_t, ny, nx, reps).transpose(0, 3, 1, 2).reshape(-1, ny, nx)
        result = out_tensor[:n_t]
    else:
        result = out.ravel()[: np.prod(orig_shape)].reshape(orig_shape)
    stats = QECStats(
        physical_error_rate=phys_err / max(n_groups, 1),
        logical_error_rate=log_err_post / max(n_groups, 1),
        logical_error_rate_pre=log_err_pre / max(n_groups, 1),
        n_groups=n_groups,
        syndromes_detected=syndromes,
        p_physical_effective=p_effective,
    )
    return result, stats


def simulate_code_memory(
    p_physical: float,
    n_trials: int = 10_000,
    seed: int = 0,
) -> QECStats:
    """Monte Carlo logical vs physical error rate for the bare [[3,1,3]] code."""
    rng = np.random.default_rng(seed)
    code = BitFlipRepetitionCode()
    phys = 0.0
    logi = 0.0
    syn = 0

    for _ in range(n_trials):
        logical = int(rng.integers(0, 2))
        encoded = code.encode(logical)
        corrupted = code.apply_bitflip_channel(encoded, p_physical, rng)
        phys += float(np.mean(encoded != corrupted))
        corrected, (s0, s1) = code.correct(corrupted)
        if s0 or s1:
            syn += 1
        logi += float(code.decode_logical(corrected) != logical)

    return QECStats(
        physical_error_rate=phys / n_trials,
        logical_error_rate=logi / n_trials,
        n_groups=n_trials,
        syndromes_detected=syn,
        p_physical_effective=p_physical,
    )


def measure_qec_threshold(
    *,
    gamma_1_values: list[float] | None = None,
    noise_scales: list[float] | None = None,
    n_trials: int = 5_000,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Sweep γ₁ and noise_scale; return physical/logical error rate rows."""
    gamma_1_values = gamma_1_values or [1.0, 1.25, 1.5, 1.75, 2.0]
    noise_scales = noise_scales or [0.25, 0.5, 0.75, 1.0]
    rows: list[dict[str, float]] = []

    for gamma_1 in gamma_1_values:
        bmgl = PWaveBMGL(gamma_1=gamma_1)
        for noise_scale in noise_scales:
            p_phys = physical_error_rate(bmgl, noise_scale=noise_scale)
            stats = simulate_code_memory(p_phys, n_trials=n_trials, seed=seed)
            rows.append(
                {
                    "gamma_1": gamma_1,
                    "noise_scale": noise_scale,
                    "effective_inhibition": bmgl.effective_inhibition,
                    "p_physical": p_phys,
                    "physical_error_rate": stats.physical_error_rate,
                    "logical_error_rate": stats.logical_error_rate,
                    "syndrome_rate": stats.syndromes_detected / max(stats.n_groups, 1),
                }
            )
    return rows


def format_threshold_table(rows: list[dict[str, float]]) -> str:
    """Pretty-print threshold sweep for logs / metrics blocks."""
    header = (
        f"{'γ₁':>5}  {'noise':>5}  {'p_phys':>8}  "
        f"{'phys_err':>9}  {'log_err':>9}  {'syndrome':>9}"
    )
    lines = ["[[3,1,3]] bit-flip QEC threshold sweep", header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['gamma_1']:5.2f}  {r['noise_scale']:5.2f}  {r['p_physical']:8.5f}  "
            f"{r['physical_error_rate']:9.5f}  {r['logical_error_rate']:9.5f}  "
            f"{r['syndrome_rate']:9.5f}"
        )
    return "\n".join(lines)