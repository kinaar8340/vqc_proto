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

## Phase 2 — Gradio web demo 🚧 (in progress)

**Goal:** Let visitors try Orbital Braille without cloning Python envs.

| Task | Status | Notes |
|------|--------|-------|
| `proto/gradio_demo.py` | ✅ | Payload, orbs, quick/full, 6-panel output |
| `proto/requirements-web.txt` | ✅ | `pip install -r requirements-web.txt` |
| docker-compose `gradio` service | ✅ | Port 7860 |
| Hugging Face Spaces deploy | ⬜ | Phase 2b — optional public hosting |
| Share demo link in README / X | ⬜ | After HF or tunnel |

**Run:**
```bash
cd proto && pip install -r requirements-web.txt
python gradio_demo.py
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