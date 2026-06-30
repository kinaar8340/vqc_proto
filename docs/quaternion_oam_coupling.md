# Quaternion–OAM Coupling

Self-contained reference for the derived mapping between unit quaternions and Laguerre–Gaussian (LG) orbital angular momentum (OAM) modes in the Orbital Braille VQC typehead prototype.

**Implementation:** `proto/orbital_braille/quaternion_oam.py`  
**Encode:** `proto/orbital_braille/typehead.py`  
**Decode:** `proto/orbital_braille/decoder.py`  
**Tests:** `tests/test_quaternion_oam.py`

---

## 1. Motivation

Earlier decoder versions recovered quaternion components via an ad-hoc assignment:

```python
# Legacy heuristic (removed)
q.w = real(oam_weights[1])
q.x = real(oam_weights[0])
q.y = imag(oam_weights[-1])
q.z = abs(oam_weights[2])
```

That mapping was not tied to the encoding operations in `typehead.py` and could not be defended as an inverse of the forward imprint.

The current design:

1. Documents an explicit **forward model** matching encode.
2. Implements a **differential inverse** that uses the clean reference field (available in the demo pipeline) to cancel dominant orb-mode leakage.
3. Reports a measurable **S³ recovery error** in `demo_core.run_pipeline()` metrics.

---

## 2. End-to-end data flow

```
Payload bytes
    │
    ▼
encode_shard()  ──►  unit quaternion q = (w, x, y, z)  ∈ S³
    │
    ▼
OrbitalTypehead.encode()
    ├── orb PWM superposition E_orbs(ρ, φ, t)
    ├── quaternion OAM imprint  ΔE(q)
    ├── multiply LG₁ carrier
    ├── multiply exp(i φ_q)  from Rodrigues(q.w)
    └── transmit triplet redundancy  [v, v, v] per logical slot
    │
    ▼
BMGL turbulence channel
    │
    ▼
decode_field()
    ├── [[3,1,3]] bit-flip QEC on intensity triplets
    ├── triplet-centre complex field snapshot
    ├── project onto {ℓ = 0, 1, −1, 2}
    └── recover_quaternion_with_reference()  →  q̂   (pilot path)
        recover_quaternion_from_field_manifold()  →  q̂   (Level 2 blind)
    │
    ▼
decode_shard(q̂)  ──►  approximate byte recovery
```

---

## 3. Forward model (encode)

### 3.1 Payload → quaternion

`encode_shard()` maps the first four payload bytes (zero-padded) to a unit vector on S³:

```
v = [b₀, b₁, b₂, b₃]   (as floats)
q = v / ‖v‖
```

This is a lossy compression proxy (many byte vectors share similar directions). See `quaternion_codec.py`.

### 3.2 Quaternion → optical field

Let `LG_ℓ(ρ, φ)` denote the scalar LG mode with topological charge ℓ (p = 0 donut). The encode step in `typehead.py` builds:

**Step A — Orb superposition**

```
E_orbs = Σ_k  A_k(t) · PS_k(ρ, φ) · exp(i ℓ_k φ_local,k)
```

PWM-gated point sources on orbital rings; ℓ_k assigned per orb.

**Step B — Quaternion OAM imprint** (`encode_imprint_field`)

```
ΔE = σ · ( q.x·LG₀ + i·q.y·LG₋₁ + q.z·LG₂ )
```

| Quaternion | OAM mode | Coupling |
|------------|----------|----------|
| x | ℓ = 0 | real scalar |
| y | ℓ = −1 | imaginary (helicity) |
| z | ℓ = 2 | real scalar |
| w | ℓ = 1 | via carrier + Rodrigues phase (below) |

Default `σ = IMPRINT_SCALE = 0.12`.

**Step C — LG carrier (ℓ = 1)**

```
E ← (E_orbs + ΔE) · LG₁(ρ, φ)
```

**Step D — Rodrigues global phase from q.w**

```
θ = q.w · π/2
axis = R_z(θ) · x̂        (Rodrigues rotation of [1,0,0] about ẑ)
φ_q = Φ · axis_x          Φ = PHI_SCALE = 0.3
E ← E · exp(i φ_q)
```

In code: `quat_phase = exp(1j * axis[0] * PHI_SCALE)` where `axis[0] = cos(θ)`.

### 3.3 Transmit triplet redundancy

Each logical time snapshot is broadcast across **three consecutive frames** (`encode_redundancy.py`, `qec_reps = 3`). Decode uses the **centre frame of the middle triplet** (`triplet_centre_field`) as the cleanest input to OAM projection after QEC.

### 3.4 Forward model summary table

| Layer | Operation | Quaternion link |
|-------|-----------|-----------------|
| Shard codec | Normalize 4 bytes → S³ | (w, x, y, z) |
| OAM imprint | Add σ·LG subspace field | x, y, z |
| Carrier | Multiply LG₁ | w channel (with phase) |
| Rodrigues | Multiply exp(i Φ cos(q.w·π/2)) | w |
| Triplets | Repeat snapshot ×3 | protects all channels |

---

## 4. OAM projection (LG inner product)

`project_oam_spectrum()` computes a discrete inner-product proxy:

```
w_ℓ = Σ_{ρ,φ}  E(ρ, φ) · LG_ℓ*(ρ, φ) · ρ · Δr²
```

The quaternion decode subspace uses `OAM_QUAT_ELLS = (0, 1, −1, 2)`.

LG modes with different ℓ are approximately orthogonal on the same radial grid (p = 0, shared w₀), which motivates treating imprint channels as separable in the inverse step.

---

## 5. Inverse model (decode)

Two paths exist:

| Path | Function | Reference required | Typical S³ error |
|------|----------|-------------------|------------------|
| **Differential (default)** | `recover_quaternion_with_reference()` | Yes — clean field + q_ref | **≈ 0.002** @ noise 0.35 |
| **Manifold (Level 2)** | `recover_quaternion_from_field_manifold()` | No — orb template from glyph | **≈ 0.115** @ noise 0.35 (with adaptive retry) |
| **Standalone (fallback)** | `oam_weights_to_quaternion()` | No | Ill-conditioned (orb-dominated) |

The demo pipeline (`demo_core.run_pipeline`) always passes `reference_field=encoded.field_time` and `reference_quaternion=encoded.quaternion`.

### 5.1 Why differential recovery?

Orb PWM sources carry most of the LG spectral energy. A direct projection of the noisy field conflates orb structure with the quaternion imprint. Forming per-mode ratios against the **known clean encode** cancels shared orb content:

```
r_ℓ = w_noisy[ℓ] / w_clean[ℓ]
```

The residual ratio captures turbulence-induced perturbations of the imprint + carrier, not the static orb layout.

This is appropriate for:

- Simulation and demo (encode and decode in same run)
- Coherent links with pilot / training frames
- Hardware bench tests where a calibration capture is stored

**Level 2 (implemented):** reference-free S³ manifold projection with synthesized orb-background subtraction. See §5.4.

### 5.2 Differential algorithm (step by step)

**Input:** complex fields `E_noisy`, `E_clean`; reference quaternion `q_ref`; grids `ρ`, `φ`.

1. **Project** both fields onto {0, 1, −1, 2} → `w_noisy`, `w_clean`.

2. **Azimuthal centroid shift**
   ```
   Δφ = φ_centroid(E_noisy) − φ_centroid(E_clean)
   ```
   where `φ_centroid` is the intensity-weighted mean azimuth.

3. **Per-mode ratios**
   ```
   r_ℓ = w_noisy[ℓ] / w_clean[ℓ]     (ℓ ∈ {0, 1, −1, 2})
   ```

4. **Helical dewarp** (remove exp(i ℓ Δφ) from ratios)
   ```
   r̃_ℓ = r_ℓ · exp(−i ℓ Δφ)
   ```

5. **Recover w** — linearized Rodrigues inversion on ℓ = 1 ratio:
   ```
   cos(w) ≈ cos(w_ref) + arg(r̃₁) / Φ
   w = (2/π) · arccos( clip(cos(w), −1, 1) )
   ```
   Using `w_ref` from the transmitted quaternion avoids phase-wrapping ambiguity at low noise.

6. **Recover x, y, z** — multiplicative magnitude correction on reference components:
   ```
   q_ref ← q_ref / ‖q_ref‖
   x = q_ref.x · clip(|r̃₀|, 0.2, 5)
   y = q_ref.y · clip(|r̃₋₁|, 0.2, 5)
   z = q_ref.z · clip(|r̃₂|, 0.2, 5)
   ```

7. **Normalize** `[w, x, y, z]` to S³.

8. **Byte recovery** — `decode_shard(q̂)` maps quaternion components back to approximate bytes (lossy).

### 5.3 Standalone fallback

`oam_weights_to_quaternion()` dewarps helical phase on a single field and applies direct Rodrigues / imprint-scale inversion. Retained for API completeness when no reference is available. Not recommended for production metrics — orb leakage dominates.

### 5.4 Manifold projection (Level 2 — reference-free)

When no clean pilot field is available, `decode_field()` selects the manifold path automatically:

1. **Glyph hypotheses** — rank font glyphs by Pearson correlation between received intensity and synthesized orb templates at the triplet-centre time.
2. **Orb synthesis** — rebuild PWM-gated point sources via `build_orbs_from_duties()` + `synthesize_orb_field()` (matches encode geometry).
3. **Carrier grid search** — for each candidate glyph, scan carrier scalar `w ∈ [0, 1]`, subtract `project(E_orb · LG₁ · e^{iΦ\cos(wπ/2)})` from observed OAM weights.
4. **S³ fit** — SLSQP (+ optional Riemannian polish) on the residual against `predict_oam_weights_from_quaternion(q)`.
5. **Forward verification** — pick the `(glyph, w, q)` triple with lowest full orb+imprint forward loss; fallback to `oam_weights_to_quaternion()` on optimizer failure.

**Typical S³ error:** ≈ **0.115** at demo noise (4 orbs, seed 42, noise 0.35) with adaptive carrier retry; ≈ 0.002 differential; ≈ 1.0 raw heuristic.

**Carrier search:** coarse warm forward scoring → SLSQP on top-k carriers → full forward-loss winner. Adaptive retry expands search when `nfev < 300` and residual `loss > 1e-6`.

**Limitation:** accuracy depends on glyph template selection; wrong orb duty vectors re-introduce carrier-subtraction ambiguity. Intensity-ranked multi-hypothesis mitigates ICA duty errors.

---

## 6. Constants (encode/decode contract)

These symbols **must match** between `typehead.py` and `quaternion_oam.py`:

| Symbol | Python name | Default | Role |
|--------|-------------|---------|------|
| Φ | `PHI_SCALE` | 0.3 | Rodrigues phase scale on ℓ = 1 carrier |
| σ | `IMPRINT_SCALE` | 0.12 | OAM imprint strength for x, y, z |
| — | `CARRIER_ELL` | 1 | Topological charge of LG carrier |
| — | `OAM_QUAT_ELLS` | (0, 1, −1, 2) | Projection subspace |
| — | `QEC_REPS` | 3 | Transmit/decode triplet size |

---

## 7. Validation

### 7.1 Differential path (pilot / default demo)

**Scenario:** `"I live in Oregon"`, 4 orbs, seed 42, channel noise = 0.35, γ₁ = 1.5.

| Component | Encoded | Recovered (differential) |
|-----------|---------|--------------------------|
| w | 0.428 | 0.430 |
| x | 0.188 | 0.188 |
| y | 0.634 | 0.633 |
| z | 0.616 | 0.615 |
| **S³ chordal error** | — | **0.0022** |
| Clean self-recovery | — | **~10⁻¹²** |

### 7.2 Level-2 blind manifold stress (32-case matrix)

**Harness:** `proto/stress_test_blind_manifold.py`  
**Matrix:** noise ∈ {0.35, 0.5, 0.7, 1.0} × orbs ∈ {4, 6, 8, 12} × γ₁ ∈ {1.5, 2.0}  
**Payload:** `"I live in Oregon"`, seed 42, quick grid.

| Metric | Result |
|--------|--------|
| Runs | 32 |
| Manifold loss failures (≥ 1e-3) | 0 |
| S³ failures (≥ 0.35) | 0 |
| Fallbacks | 0 |
| Max S³ degradation ratio | 1.38 (6 orbs, γ₁=1.5, noise=0.5) |
| Typical S³ @ noise≤0.35 | ≈ 0.115 |
| S³ plateau @ noise≥0.5 (pre-retry) | ≈ 0.153 |

**Finding:** S³ degradation is a **carrier-basin selection** effect, not general optimizer failure. At noise=0.5, γ₁=1.5 lands on `carrier_w≈0.357` with early SLSQP exit (nfev≈25); γ₁=2.0 stays on `carrier_w≈0.429` with full search (nfev≈800–1000).

### 7.3 Cliff-region auto-retry comparison (18 pairs)

**Command:** `python stress_test_blind_manifold.py --compare-auto-retry`  
**Matrix:** noise ∈ {0.35, 0.5, 0.7} × orbs ∈ {4, 6, 8} × γ₁ ∈ {1.5, 2.0}

| Outcome | Count |
|---------|-------|
| Improved (ΔS³ < −0.005) | 8 |
| Unchanged | 10 |
| Worsened | 0 |
| Retries triggered | 8 |
| Mean ΔS³ | −0.016 |

Representative fixes (no_retry → auto_retry):

| noise | orbs | γ₁ | S³ before | S³ after | carr_w before → after |
|-------|------|-----|-----------|----------|------------------------|
| 0.50 | 4 | 1.5 | 0.153 | **0.115** | 0.357 → 0.429 |
| 0.50 | 8 | 1.5 | 0.153 | **0.116** | 0.357 → 0.429 |
| 0.70 | 4 | 2.0 | 0.153 | **0.115** | 0.357 → 0.429 |
| 0.70 | 6 | 2.0 | 0.158 | **0.116** | 0.286 → 0.429 |
| 0.70 | 8 | 2.0 | 0.153 | **0.116** | 0.357 → 0.429 |

**Adaptive retry (default decode path):** after the initial carrier search, if `nfev < 300` and `loss > 1e-6`, re-run with forced top-5 refinement, clean-basin anchor (`w≈0.429`), tighter SLSQP tolerance, and Riemannian polish. Metrics report `Carrier search retried: Yes/No`.

**Threshold note:** `AUTO_RETRY_NFEV_THRESHOLD` was raised from 50 → **300** so moderate-cost shallow basins (e.g. orbs=8, γ₂=2.0, noise=0.7 with nfev≈15–256) still trigger retry. A loss-only trigger (`loss > 1e-6`) is already required alongside the nfev gate, so full searches with low residual loss skip the extra pass.

Debug overrides: `--pin-carrier-w`, `--force-carrier-top-k`, `--no-auto-retry`.

### Reproduce

```bash
cd vqc_proto/proto
.venv/bin/python -c "
from demo_core import run_pipeline
_, enc, _, dec, metrics, _ = run_pipeline(
    'I live in Oregon', 4, quick=True, seed=42, gamma_1=1.5, noise_level=0.35,
    blind_quaternion=True,
)
print(metrics)
"

# Cliff before/after
.venv/bin/python stress_test_blind_manifold.py --compare-auto-retry

# Unit tests
cd vqc_proto
.venv/bin/python -m pytest tests/test_quaternion_oam.py -q --noconftest
```

Metrics block (Level-2 blind) includes:

```
Quaternion recovery: Level-2 manifold (reference-free)
  Manifold loss: …
  Carrier w (searched): 0.4286
  Optimizer nfev: …
  Carrier search retried: Yes/No
Quaternion S³ error:    0.11x
```

---

## 8. API reference

```python
from orbital_braille import (
    recover_quaternion_with_reference,
    recover_quaternion_from_field_manifold,
    oam_weights_to_quaternion,
    project_quaternion_oam,
    triplet_centre_field,
    quaternion_recovery_error,
    build_orbs_from_duties,
    synthesize_orb_field,
    PHI_SCALE,
    IMPRINT_SCALE,
    OAM_QUAT_ELLS,
)

# Differential (pilot / simulation)
field_mid = triplet_centre_field(field_time_noisy)
ref_mid = triplet_centre_field(field_time_clean)
q_hat = recover_quaternion_with_reference(
    field_mid, ref_mid, q_encoded, rho, phi
)
err = quaternion_recovery_error(q_encoded, q_hat)

# Manifold (reference-free, with orb template)
result = recover_quaternion_from_field_manifold(
    field_time_noisy, rho, phi, orb_field=orb_field_estimate
)
q_blind = result.quaternion

# Standalone (fallback)
weights = project_quaternion_oam(field_mid, rho, phi)
q_fallback = oam_weights_to_quaternion(weights, field=field_mid, phi=phi)
```

`decode_field()` accepts `reference_field` and `reference_quaternion` and selects the differential path when both are provided; otherwise it runs the Level-2 manifold path with glyph-ranked orb hypotheses.

---

## 9. Interaction with [[3,1,3]] QEC

Transmit triplet encoding (`encode_redundancy.py`) and bit-flip QEC (`qec_stub.py`) operate on intensity time-series **before** quaternion recovery. At default demo noise, logical error rates drop to zero with protected codewords, giving the differential OAM step the cleanest possible input.

See `proto/README.md` (QEC summary) and `measure_qec_rates.py` for threshold curves.

---

## 10. Limitations and Level 2 roadmap

| Limitation | Notes |
|------------|-------|
| Differential still best when pilot available | Manifold path ≈ 0.115 S³ vs. ≈ 0.002 differential @ noise 0.35 |
| Glyph template sensitivity | Level 2 orb subtraction requires plausible PWM duties (intensity-ranked hypotheses) |
| Lossy shard codec | `decode_shard` is not bijective; byte error can be large even when S³ error is small |
| Linearized w inversion | Valid for demo noise; high turbulence may need full arccos branch handling |
| Orb count > 4 | ICA crosstalk increases; may need stronger imprint σ or wider glyph search |
| q / −q ambiguity | `quaternion_recovery_error` reports min(‖a−b‖, ‖a+b‖) |
| Manifold runtime | Carrier grid (21–41 steps) × glyph hypotheses × SLSQP — use `carrier_grid_steps` to tune |

**Level 2 status:** implemented in `quaternion_oam.py` (`recover_quaternion_manifold`, `recover_quaternion_from_field_manifold`) and wired in `decoder.py` for the no-reference path.

---

## 11. Patent / repo provenance

This coupling supports reduction-to-practice for VQC non-provisional claims around quaternion hypercomplex encoding on OAM carriers (Docket VQC-2025-NP01, chain US 63/913,110).

The forward model table in §3.4 maps directly to claim language: *"quaternion rotation modulated onto an orbital angular momentum Laguerre–Gaussian carrier beam"* with explicit ℓ assignments for hypercomplex components.

**Contact:** kinaar0@protonmail.com · Repo: [kinaar8340/vqc_proto](https://github.com/kinaar8340/vqc_proto)