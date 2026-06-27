# VQC Proto Roadmap

Phased plan for usability, performance, and collaboration. **Phase 0** merged in PR #1.

---

## Phase 0 — Onboarding & CI ✅ (merged)

| Item | Status |
|------|--------|
| README <5 min quick-start | ✅ |
| `run_demo_quick.py` + `--quick` | ✅ |
| Docker + docker-compose | ✅ |
| GLOSSARY, IP_NOTICE, CONTRIBUTING | ✅ |
| GitHub Actions (ruff, black, pytest) | ✅ |
| `tests/test_orbital_braille.py` | ✅ |
| Jupyter notebook | ✅ |

---

## Phase 1 — Hot-path profiling 🚧 (in progress)

**Goal:** Identify compute bottlenecks in LG mode generation, OAM projection, and FastICA demix.

| Task | Status | Notes |
|------|--------|-------|
| `proto/profile_hotpaths.py` CLI | ✅ | Timed benches + cProfile report |
| Baseline report in `proto/outputs/profile/` | ✅ | Run locally after merge |
| Vectorize `lg_radial` batch over ℓ | ⬜ | Phase 1b — if LG > 30% of encode time |
| Cache LG basis on grid in `TypeheadConfig` | ⬜ | Phase 1b |
| ICA `max_iter` / tol tuning from profile | ⬜ | Phase 1b |
| CI smoke: `--quick` profile < 30s | ⬜ | Optional gate |

**Run:**
```bash
cd proto
python profile_hotpaths.py --quick          # fast baseline
python profile_hotpaths.py --grid-size 80 # full-res
```

---

## Phase 2 — Gradio web demo ✅

**Goal:** Let visitors try Orbital Braille without cloning Python envs.

| Task | Status | Notes |
|------|--------|-------|
| `proto/gradio_demo.py` | ✅ | Payload, orbs, quick/full, 6-panel output |
| `proto/demo_core.py` | ✅ | Shared with run_demo + HF Space |
| `proto/requirements-web.txt` | ✅ | `pip install -r requirements-web.txt` |
| docker-compose `gradio` service | ✅ | Port 7860 |

## Phase 2b — Hugging Face Spaces 🚧 (in progress)

**Goal:** Public zero-install browser demo at maximum visibility.

| Task | Status | Notes |
|------|--------|-------|
| `space/orbital-braille/` bundle | ✅ | app.py, requirements, README frontmatter |
| `scripts/sync_hf_space.sh` | ✅ | Copy proto → space folder |
| `scripts/deploy_hf_space.sh` | ✅ | `HF_TOKEN` + huggingface_hub upload |
| README live-demo badge + section | ✅ | Top of main README |
| Space live at HF URL | ✅ | [kinaar111/orbital-braille-vqc](https://huggingface.co/spaces/kinaar111/orbital-braille-vqc) |
| X thread / social update | ⬜ | After Space is live |

**Deploy:**
```bash
# 1. Create Space at huggingface.co/new-space (Gradio SDK) — or let deploy script create it
# 2. Sync + upload
export HF_TOKEN=hf_...
./scripts/sync_hf_space.sh
./scripts/deploy_hf_space.sh kinaar111/orbital-braille-vqc
```

**Local:**
```bash
cd proto && pip install -r requirements-web.txt && python gradio_demo.py
# or: docker compose up gradio
```

---

## Phase 3 — Dashboard proto auto-load 🚧 (in progress)

**Goal:** Make Streamlit dashboard discover proto outputs without manual paths.

| Task | Status | Notes |
|------|--------|-------|
| `analysis/proto_loader.py` | ✅ | Scans demo PNG, SLM montages, manifests, meta JSON |
| Dashboard "Orbital Braille" tab | ✅ | Auto-load latest `proto/outputs/` |
| Sidebar view mode (Pipeline / Proto / Both) | ✅ | |
| `tests/test_proto_loader.py` | ✅ | |
| In-dashboard "Run quick demo" button | ⬜ | Phase 3b — subprocess `run_demo_quick.py` |
| Filter pipeline PNGs by L## / run_* | ⬜ | Phase 3b |
| Proto metrics card (shard FID from meta JSON) | ⬜ | Phase 3b |

**Run:**
```bash
streamlit run analysis/dashboard.py
# or: docker compose up dashboard
```

---

## Phase 4 — Future (not started)

| Item | Priority |
|------|----------|
| `click` unified CLI across proto scripts | Medium |
| Structured `logging` config (file + level) | Medium |
| Expand pytest to `src/` photonics LG paths | Medium |
| SLM bench validation checklist + sample datasets | High (hardware) |
| HF Spaces / Gradio public URL | Medium |
| GitHub repo topics (manual Settings) | Low |

---

## Suggested merge order

1. **feat/phase2-profiling-gradio-dashboard** — Phases 1–3 initial delivery
2. Phase 1b — optimizations from profile data
3. Phase 2b — Hugging Face Spaces
4. Phase 3b — dashboard live-run + filters