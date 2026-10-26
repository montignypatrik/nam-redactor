# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, single-user web app (FastAPI + React) that black-box redacts Quebec health insurance numbers (NAM/RAMQ) and patient names from PDFs, then helps turn appointment rows into billing lines. Redaction runs fully offline on CPU (PaddleOCR + a local spaCy French model). The only cloud step is optional row extraction/interpretation through OpenRouter, which receives already-redacted row images only. The documents contain health data, so privacy is the main design constraint throughout.

## Commands

```bash
pip install -e .[dev]                 # Python 3.11+; pulls CPU Paddle and the spaCy model wheel
pytest                                # full suite (CI sets OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
pytest tests/test_locator.py::test_name   # single test
python -m nam_redactor.web            # run the app (or `nam-redact-web`); --no-open, --port, --data-dir

cd frontend && npm ci && npm run build   # tsc --noEmit + vite build into nam_redactor/web/dist
cd frontend && npm run dev               # hot reload on :5173; proxies /api to the backend on :8000
```

There is no linter configured. The TypeScript type-check runs as part of `npm run build`.

**The compiled frontend is committed** (`nam_redactor/web/dist/`) so `pip install` needs no Node. After any change under `frontend/`, run `npm run build` and commit the dist output. CI rebuilds it and fails on `git diff --exit-code -- nam_redactor/web/dist`.

## Architecture

Two layers: a library (`nam_redactor/`) and a web layer (`nam_redactor/web/`) that wraps it.

**Library pipeline** (`redactor.py`): detection is split from rendering. `detect_nam_pages` is slow: it rasterizes and OCRs each page once and returns normalized boxes. `render_redacted_pdf` is fast: it re-rasterizes and burns in whatever boxes it receives, so reviewer edits cost no OCR. The output is always a fresh image-only PDF, so the original text layer is gone.
- `validation.py`: deterministic NAM rules (4 letters + 8 digits, encoded birth date, OCR-confusion tolerant). No LLM is ever used for detection.
- `locator.py`: searches each OCR line run for NAM-length spans (labels are often fused to the number) and narrows line boxes to the matched characters by interpolating character offsets. It also merges NAMs split across adjacent blocks.
- `names.py`: name boxes come from three sources: NAM-anchored (the NAM encodes the surname prefix and first initial), labelled (`Nom :` etc.), and spaCy NER. NER is opt-in from Python but always on in the web app.
- `zones.py` / `provenance.py`: dense pages are cut into zones, and each zone becomes its own *working page*. Working-page numbers change whenever zones change, so every working page carries a `PageProvenance` that maps back to the source page and pixel crop. `provenance.py` documents the coordinate conventions (normalized 0..1, top-left origin, `/Rotate` applied).
- `nam_redactor/__init__.py` sets Paddle thread env vars before any `import paddle`. Keep that ordering.
- `extraction.py`: the `OpenRouterExtractor` adapter. Anything with `extract(image_png, model)` can replace it, and tests do that. Privacy routing (ZDR only, `data_collection: deny`, `require_parameters`) must never be relaxed.

**Web layer**:
- `app.py`: `create_app(engine_factory=..., name_model_factory=..., dir_root=...)`. The factories let tests inject a `FakeEngine` and no NER, so web tests run in seconds without Paddle. The hardening is deliberate: no OpenAPI/docs routes, an Origin-vs-Host check on mutations (which is why the Vite proxy uses `changeOrigin: false`), TrustedHost, magic-byte upload checks, and no PHI text in any response.
- `jobs.py`: one job per document. All OCR is serialized through a single-worker executor, and job state plus SSE fan-out live in memory, so uvicorn must stay at `workers=1`. State is saved to disk after every change and reopened on restart. `python -m nam_redactor.web` preloads the OCR and NER models on that worker at startup (`create_app(preload_models=True)`); tests leave it off.
- `storage.py`: per-user data dir (override with `--data-dir` or `NAM_REDACTOR_DATA`), with one folder per document under `documents/<id>/`. `document.json` holds no NAM, name or OCR text. The NAM→pseudonym map lives only in `private/identities.json` and must never be exported.
- Row workflow: `rows.py` (appointment rows, one per zone, with preparation/readiness) → `evidence.py` (redacted row crops rendered from the *reviewed* boxes, frozen with SHA-256 at approval) → `web/extraction.py` (transcription attempts and reviews; a row's first pass is one combined call that also yields its interpretation attempt, re-runs call `interpret` alone) → `interpretation.py` (proposed billing acts from a doctor profile, marked stale when any input changes) → `billing.py` (manual edits and confirmations stored beside the model output) → `billing_csv.py` (17-column `;`-delimited CSV, values copied verbatim with no mapping) and `manifest.py` (allow-list JSON export).
- `doctors.py`: versioned doctor profiles in `doctors.sqlite3`. Versions are immutable (enforced by triggers), and profiles are archived, never deleted. A document pins an exact version, and newer versions must be adopted explicitly.

**Recurring invariants**: model outputs, attempts and reviews are append-only and never edited in place. Corrections and edits are stored beside them. Every derived result records the exact inputs it was based on and becomes "stale" or "outdated" when they change. Billing values are never normalized or mapped by code (no case changes, no leading-zero stripping, no inventing 0/1 units).

**Frontend** (`frontend/src`): a React 18 + Vite SPA with no router or state library. `App.tsx` drives a stage machine (upload → zones → detecting → review → rows, plus doctors). "Confirm review" saves the review and opens the appointments page, which processes every row once. All HTTP and SSE calls are in `api.ts`, and their types are in `types.ts`.

## Tests

- `tests/privacy.py` provides shared assertions for exports. Use them rather than substring-searching serialized manifests: IDs and hashes can contain digit fragments by chance.
- `*.csv` and `*.pdf` are gitignored because they may contain patient data. Tests build their PDFs in code (see `make_pdf_bytes` in `tests/test_web_api.py`).
- `docs/doctor-profiles/` holds real profile instructions. `docs/future-improvements.md` lists deferred work and doctor conventions that are still open. Do not invent billing mappings that the user has not supplied.
