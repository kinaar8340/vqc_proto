# Contributing to VQC Proto

Thank you for your interest in the Vortex Quaternion Conduit simulation framework. This project welcomes **non-commercial research** contributions.

## Before you start

1. Read [`IP_NOTICE.md`](IP_NOTICE.md) — CC-BY-NC-SA-4.0 + patent restrictions apply.
2. Browse [`GLOSSARY.md`](GLOSSARY.md) if terms like BMGL or Fisher-Rao are unfamiliar.
3. Try the prototype: `cd proto && python run_demo_quick.py` (seconds) or `python run_demo.py` (full quality).

## Development setup

```bash
git clone https://github.com/kinaar8340/vqc_proto.git
cd vqc_proto
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff black pytest
cd proto && pip install -r requirements.txt
```

## Running tests

```bash
# From repo root
pytest -q

# Proto-only roundtrip tests
pytest tests/test_orbital_braille.py -v
```

## Code style

- Format with **Black** (line length 100).
- Lint with **Ruff** (`ruff check .`).
- Match existing module layout: `src/` (full pipeline), `proto/orbital_braille/` (typehead prototype).

```bash
ruff check .
black --check .
```

## Pull request checklist

- [ ] Tests pass (`pytest -q`)
- [ ] Ruff + Black clean on changed files
- [ ] README or `proto/README.md` updated if behavior or CLI flags change
- [ ] No commercial-use claims or license changes without explicit maintainer approval

## Good first issues

- Expand `tests/test_orbital_braille.py` (encoder/decoder roundtrips, font separation bounds)
- Device-specific notes for SLM presets in `SLM_QUICKSTART.md`
- Dashboard UX (filtering, proto output auto-load)
- Documentation clarifications in `GLOSSARY.md`

## Questions

Open a [GitHub issue](https://github.com/kinaar8340/vqc_proto/issues) or email [kinaar0@protonmail.com](mailto:kinaar0@protonmail.com).