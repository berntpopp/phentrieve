# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to pre-1.0 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
conventions: breaking changes bump the minor version during the 0.x series
(`major_on_zero = false` in `[tool.semantic_release]`).

Each release has three independent component versions which are all bumped
together:

- `phentrieve` — Python CLI/library (root `pyproject.toml`)
- `phentrieve-api` — FastAPI backend (`api/pyproject.toml`)
- `phentrieve-frontend` — Vue.js frontend (`frontend/package.json`)

---

## [Unreleased]

### Added

- **`--profile NAME`** flag on `query`, `text process`, `text interactive` (issue #28).
  Apply a named profile from `phentrieve.yaml` to preset CLI options. See
  [Configuration Profiles](docs/user-guide/configuration-profiles.md). Both root
  placement (`phentrieve --profile X cmd`) and per-command placement
  (`phentrieve cmd --profile X`) work; subcommand-level wins on conflict.
- **`phentrieve config`** subcommand group with `list-profiles`, `show`,
  `validate`, `path` subcommands.
- **`--show-resolved-config`** debug flag on every command. Prints resolved
  option values with source labels (profile/yaml/const/commandline) to stderr
  before running.
- **`PHENTRIEVE_PROFILE`** environment variable, equivalent to `--profile`.
- New `phentrieve.yaml` sections: `profiles:` and `extraction:`.
- Built-in profiles: `default` (API-matching strict defaults) and `interactive`
  (legacy `text interactive` defaults).
- **Adaptive re-chunking** (issue #148): an opt-in mechanism that detects
  per-chunk retrieval quality and, when poor, subdivides the chunk into
  sentence-bounded sub-chunks, re-queries them, and merges results. Enable
  on the CLI via `phentrieve text process FILE --adaptive-rechunking` with
  three threshold flags (`--adaptive-rechunking-quality-threshold`,
  `--adaptive-rechunking-margin-threshold`,
  `--adaptive-rechunking-max-depth`), or in `phentrieve.yaml` under the new
  `extraction.adaptive_rechunking:`
  block (also surfaced as `Profile.adaptive_rechunking` for per-profile
  overrides). The API `/text/process` request schema gains an
  `adaptive_rechunking` field (request-time override), and the response's
  `meta` block gains an `adaptive_rechunking` summary
  (`triggered_chunks`, `subdivided_chunks`, `recursion_depth`, etc.) when
  the feature runs. See
  [docs/user-guide/adaptive-rechunking.md](docs/user-guide/adaptive-rechunking.md).

### Fixed

- **`phentrieve text interactive`** now uses config-driven defaults (issue #171).
  Previously hardcoded `language="en"`, `chunk_retrieval_threshold=0.3`,
  `aggregated_term_confidence=0.35`, `num_results=5` are now read from the new
  built-in `interactive` profile (preserving prior behavior — no migration
  needed). Pass `--profile default` to switch to API-matching strict defaults.
- **Frontend `DEFAULT_SIMILARITY_THRESHOLD`** aligned with the API: `0.5` → `0.3`.
  This is a behavior change for users who relied on the frontend's stricter
  cutoff. To recover, pass an explicit threshold in the UI.

### Changed

- The previously-documented `--config-profile` flag (which was never
  implemented) is replaced by the now-real `--profile`.
  `docs/user-guide/configuration-profiles.md` is rewritten in place to reflect
  the actual design.
- **`orchestrate_hpo_extraction` return type**: now returns an
  `OrchestrationResult` dataclass instead of a plain tuple. Backward
  compatible — legacy 2-tuple unpacking via `__iter__` continues to work
  for all existing call sites. New attribute `raw_query_results` exposes
  the unfiltered top-K from `query_batch` for callers (e.g. adaptive
  re-chunking) that need scores below `chunk_retrieval_threshold`.

## [0.18.2] — 2026-04-25

**Component versions**: phentrieve `0.18.2`, phentrieve-api `0.9.3`, phentrieve-frontend `0.8.3`

Patch release for release-bundle workflow stabilization.

### Fixed

- fixed data bundle creation for minimal database-only bundles
- aligned the release data bundle workflow with the repository data directory
  used by `phentrieve data prepare`

## [0.18.1] — 2026-04-25

**Component versions**: phentrieve `0.18.1`, phentrieve-api `0.9.2`, phentrieve-frontend `0.8.2`

Patch release for the deprecated second-stage ranking removal.

### Changed

- removed the obsolete second-stage ranking implementation, public options,
  stale benchmark artifacts, and related documentation/frontend copy
- added a tracked-source guard test for removed ranking terminology

### Benchmarks

- `gemini-3.1-flash-lite-preview` full 10-doc GeneReviews run:
  - micro precision `0.820`
  - micro recall `0.806`
  - micro F1 `0.813`

## [0.15.0] — 2026-04-17

**Component versions**: phentrieve `0.15.0`, phentrieve-api `0.8.2`, phentrieve-frontend `0.7.3`

Post-merge stabilization and release follow-up.

### Fixed

- repaired the GitHub Pages documentation deployment by aligning the docs
  workflow Python version with the project baseline and fixing a strict MkDocs
  warning caused by an invalid repository-root link in the test documentation

### Release

- bumped the Phentrieve CLI/library minor version to `0.15.0`

## [0.14.0] — 2026-04-17

**Component versions**: phentrieve `0.14.0`, phentrieve-api `0.8.2`, phentrieve-frontend `0.7.0`

Focused LLM CLI stabilization and provider-quality release.

### Changed

- default LLM model changed to `gemini-3.1-flash-lite-preview` after full-run
  GeneReviews benchmarking
- benchmark observability now records richer Gemini usage and cost accounting,
  including thought and cached-token usage
- Gemini retries now cover transient transport/network failures so long-running
  CLI and benchmark executions do not fail on a single connectivity blip
- grounded routing, grouped Phase-1 extraction, and shared mapping prompt
  behavior were stabilized after regression investigation and reruns
- invalid trusted-proxy CIDR values are no longer logged in clear text

### Benchmarks

- `gemini-3.1-flash-lite-preview` full 10-doc GeneReviews run:
  - wall clock `109.640s`
  - micro precision `0.8291`
  - micro recall `0.8186`
  - micro F1 `0.8238`
  - estimated cost `$0.0546`

## [0.13.0] — 2026-04-10

**Component versions**: phentrieve `0.13.0`, phentrieve-api `0.8.0`, phentrieve-frontend `0.7.0`

A 90-commit refactor-and-perf release that closes the 2026-04-09 code quality
review, removes a deprecated second-stage ranking layer, and speeds up the full
CI pipeline by ~17% on warm caches.

### ⚠ BREAKING CHANGES

- **Deprecated second-stage ranking removed** across every layer (CLI, API,
  frontend, config, tests, docs). It was unvalidated against benchmark data and
  added significant complexity without measured accuracy gains.
- **`sys.path.append`** in `api/main.py` removed. The API must now be installed
  via `pip install -e .` / `uv sync` before running `uvicorn api.main:app` or
  the module import will fail. Docker images already did this correctly.

### ✨ Features

- **`ErrorResponse` API contract**: every 4xx/5xx response from the Phentrieve
  API now conforms to a single `ErrorResponse` Pydantic schema
  (`status_code`, `error`, `detail`, `request_id`). Three exception handlers
  registered in `create_app()`:
  - `StarletteHTTPException` — routing 404s + explicit `HTTPException` raises
  - `RequestValidationError` — Pydantic validation failures (422)
  - `Exception` — catch-all 500 with server-side traceback logging and a
    non-leaky `internal_server_error` payload
  - Dict and list detail payloads are preserved through the handler (the
    similarity router returns structured error dicts on 404 that clients
    can parse directly).

### ♻ Refactoring

- **Backend query orchestration**: `phentrieve/retrieval/query_orchestrator.py`
  reduced from 1057 → 715 LOC (−32%). Extracted shared retrieval and formatting
  flow.
  Extracted `InteractiveState` to its own module with proper instance attributes.
  Extracted `convert_multi_vector_to_chromadb_format()` to `phentrieve/retrieval/utils.py`.
- **HPO extraction orchestrator**: `phentrieve/text_processing/hpo_extraction_orchestrator.py`
  reduced from 298 → 103 LOC via decomposition into 4 private helpers in
  `_hpo_extraction_helpers.py`: `process_chunk_matches`, `load_term_details`,
  `build_evidence_map`, `aggregate_and_rank`. 9 characterization tests lock
  the public behavior.
- **Chunker pipeline**: `_create_chunkers()` in `phentrieve/text_processing/pipeline.py`
  reduced from 158 → ~37 LOC via a registry-based factory (`_chunker_registry.py`).
  Dead duplicate sliding_window branch removed.
- **Language resource loading**: `FinalChunkCleaner.__init__` DRY'd up —
  three near-identical word-list loading blocks collapsed to calls on a new
  `_load_language_word_list()` helper.
- **API application factory**: `create_app()` pattern introduced; MCP mount
  moved from import-time to `lifespan` startup; `similarity_router` HPO graph
  loads lazily via `@lru_cache` on first use (was eager at import).
- **Model caches hardened**: `LOADED_SBERT_MODELS`, `LOADED_RETRIEVERS`,
  `MODEL_LOADING_STATUS`, `MODEL_LOAD_LOCKS` replaced with bounded
  `cachetools.TTLCache` instances protected by a `threading.Lock`. Race
  condition between `in` check and indexed access fixed with `.get()` /
  `.pop(default)` patterns. `cleanup_model_caches()` wired into lifespan
  shutdown — cancels in-flight loads via `asyncio.shield` + `gather` before
  clearing.
- **SBERT loader inlined**: `_load_model_with_status_tracking()` was a 2-caller
  abstraction; after model-loading simplification the only remaining call site
  was inlined into `_load_sbert_in_background()` and the `is_sbert` dead
  parameter dropped.
- **Frontend mega-components decomposed**:
  - `QueryInterface.vue` 1483 → 801 LOC (−46%)
  - `ResultsDisplay.vue` 1079 → 634 LOC (−41%)
  - `App.vue` ~600 → 570 LOC (−5%)
  - 6 new composables: `useAdvancedOptions`, `usePhenotypeCollection`,
    `useFileDownload` (eliminates 3× download duplication), `useVersionCheck`,
    plus query/text-processing composables
  - 5 new sub-components: `AdvancedOptionsPanel`, `PhenotypeCollectionPanel`,
    `ResultItem`, `ChunkResultsView`, `AggregatedTermsView`
- **Frontend store consistency**: `queryPreferences` store migrated from
  Options API to setup store — all 4 Pinia stores now use the same pattern.
- **Hardcoded constants**: thresholds extracted to `frontend/src/constants/defaults.js`,
  URLs to `frontend/src/constants/urls.js`. 7+ `document.querySelector` calls
  replaced with `data-tutorial-step` attributes and reactive refs.
- **Dead code removed**: `phentrieve/visualization/` module deleted;
  `useDisclaimer.js` composable deleted; 6 unused chunker class imports removed
  from `pipeline.py`.

### ⚡ Performance

**CI pipeline (Tier 1 + Tier 2 speedup — 17 tasks, plan at
`.planning/active/CI-SPEEDUP-PLAN-2026-04-10.md`):**

- **`pytest-xdist -n auto`** enabled in `pyproject.toml` addopts →
  `make test` 25.9s → 18.5s (−29%), 927 tests parallelized across CPU cores
- **mypy cache** persisted across CI runs via `actions/cache@v4`
- **`COVERAGE_CORE=sysmon`** on Python 3.12+ (~53% faster coverage
  instrumentation per Trail of Bits benchmark; branch coverage not enabled
  so the sysmon gap is a non-issue)
- **Vite esbuild minifier** replaces terser (30-90× faster, ~0.5–2%
  compression loss) → frontend build 4.0s → 2.1s (−47%). `console` and
  `debugger` statements still stripped in production via `esbuild.drop`
  gated on `mode === 'production'` (dev server preserves them).
- **Brotli compression + bundle visualizer** gated behind `!process.env.CI`
  to skip deploy-time work in CI
- **ESLint `--cache`** with `--cache-strategy content`, cached in CI
- **Prettier `--cache`** with `--cache-strategy content`
- **Workflow concurrency groups** with `cancel-in-progress: true` on
  `pull_request` events only (main branch pushes run to completion)
- **`astral-sh/setup-uv@v7`** with `enable-cache: true` replaces the
  3-step `setup-python + install uv + actions/cache` pattern
- **Vitest `pool: 'threads'`** + `test:ci` script with `--no-coverage` for
  PR runs (main branch still uploads coverage to Codecov)
- **`timeout-minutes`** on every job (caps runaway cost at ~2× normal)
- **`fail-fast: false`** on Python matrix so cross-version regressions
  aren't masked by the first failure
- **`HF_HOME` cache** for SBERT (`paraphrase-MiniLM-L3-v2`, `BioLORD-2023-M`)
  model downloads in integration tests
- **`make ci-local`** meta-target reproduces full CI on one command
  (format-check, lint, typecheck, test-ci, frontend-lint, format-check,
  test-ci, build-ci in the same order as CI)

**Net CI impact**: full wall-clock 5.1 min → 4.3 min on warm caches (−17%)

### 🐛 Bug Fixes

- **Phenopacket DOB field**: frontend export composable was writing the
  subject's date of birth to `subject.timeAtLastEncounter` (semantically
  incorrect per Phenopacket v2). Fixed to populate `subject.dateOfBirth`.
- **`ErrorResponse.detail`** widened to `str | dict[str, Any] | list[Any]`
  so structured error payloads (e.g., the similarity router's 404 dict)
  pass through unchanged instead of being stringified.
- **Phenopacket composable**: replaced blocking `alert()` with a thrown
  `Error` so the consuming component can present an accessible snackbar.
- **TTLCache race conditions**: all get/set/del on model caches now
  wrapped in a single `_cache_lock`.
- **Ruff `target-version`**: aligned with project minimum (`py39` → `py310`).
- **Frontend download composable**: try/finally + `setTimeout(() => URL.revokeObjectURL(url), 0)`
  for Safari/WebKit URL revoke race fix.

### 🧪 Testing

- **Coverage threshold re-enabled** at 40% baseline; actual coverage now
  **55.81%** (was ~45%).
- **`norecursedirs` fix in `pyproject.toml`**: stray `phentrieve`/`api`
  entries were matching at every directory level and silently excluding
  `tests/unit/api/`, hiding 29 broken tests. Removing them exposed the
  failures, the bugs got fixed, and the test count jumped from 796 → **927**.
- **+44 Python characterization tests** pinning behavior through the
  `query_orchestrator`, `api/dependencies`, `api/main`, `hpo_extraction_orchestrator`,
  and `_create_chunkers` refactors.
- **+39 frontend tests** covering composables, stores, and components.
- **`@pytest.mark.parametrize` consolidation** across assertion_detection
  and hpo_parser tests.
- **Centralized fixtures**: `mock_sbert_model` fixture now lives in a
  shared conftest.
- **18 zero-assertion tests** fixed with meaningful assertions.
- **15 previously unmarked test files** now have `@pytest.mark.unit`.

### 📚 Documentation

- Pre-commit hooks (`ruff-check`, `ruff-format`, trailing whitespace,
  end-of-file, YAML, merge-conflicts, large-files) added via
  `.pre-commit-config.yaml`.
- TTLCache vs `@lru_cache` caching decisions documented in
  `docs/architecture/caching.md`.
- Review dashboard at `.planning/analysis/CODE-QUALITY-REVIEW-2026-04-09.md`
  updated with post-refactor status (all Critical findings closed).
- Phase 2 plan (`.planning/completed/2026-04-10-code-quality-phase-2.md`)
  and CI speedup plan (`.planning/active/CI-SPEEDUP-PLAN-2026-04-10.md`)
  saved for future reference.
- Stale deprecated-ranking references removed from docs and config.
- `AGENTS.md` notes the `pytest -n 0` escape hatch for single-threaded
  debugging when the default `-n auto` causes issues.

### 🧹 Chore

- `terser` devDependency removed from `frontend/package.json` after the
  Vite minifier switch (orphaned optional peer dependency).
- `Makefile` gains 6 new CI-parity targets: `format-check`, `test-ci`,
  `frontend-build-ci`, `frontend-format-check`, `frontend-test-ci`, `ci-local`.

---

## [0.12.1] — 2025-11-xx

Previous release — see git history for details. Notable work:
configuration resolver refactoring, HPO graph caching via `@lru_cache`,
SQLite migration for HPO data, model caching optimization, benchmark data
reorganization.

---

[0.13.0]: https://github.com/berntpopp/phentrieve/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/berntpopp/phentrieve/releases/tag/v0.12.1
