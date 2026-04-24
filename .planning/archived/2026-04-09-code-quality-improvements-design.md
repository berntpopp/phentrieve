# Code Quality Improvements — Design Spec

**Date**: 2026-04-09
**Goal**: Improve maintainability, performance, and usability across the phentrieve codebase
**Approach**: Domain-Parallel Streams with worktree isolation
**Timeline**: 2-3 weeks (full cycle)
**Regression policy**: Zero tolerance — characterization tests precede all refactoring

---

## Context

A comprehensive code quality review (see `plan/05-analysis/CODE-QUALITY-REVIEW-2026-04-09.md`) identified structural issues across backend, frontend, and test infrastructure. Grand average score: 6.6/10. The codebase has strong DevOps maturity (8.7/10) and domain awareness, but orchestration layers are overloaded, global state is hidden, mega-components resist testing, and test coverage gives false confidence (47 tests with zero assertions, 24+ untested modules).

Research into 2025-2026 best practices for FastAPI, Vue 3, and RAG systems confirmed actionable patterns: lifespan/factory for API startup, `cachetools` for bounded caching, composable extraction for Vue mega-components, and template refs to replace DOM queries.

## Priorities

Balanced across three axes:
1. **Ship reliability** — characterization tests, pre-commit hooks, coverage gates
2. **Developer velocity** — decompose orchestrators, clarify module boundaries
3. **Production performance** — bounded caching, startup optimization, frontend rendering

## Architecture: Three Parallel Streams

```
Week 1              Week 2              Week 3
STREAM A (Backend) ─────────────────────────────────
  char tests ──────► orchestrator ─────► caching &
                     decomposition       API lifecycle

STREAM B (Frontend) ────────────────────────────────
  component tests ─► decompose mega ───► v-memo,
                     components          shallowRef

STREAM C (Infrastructure) ──────────────────────────
  quick wins + ────► test infra ───────► coverage
  pre-commit         fixes               gates
```

Each stream runs in its own git worktree. File overlap between streams is <5%. Merge coordination at end of each week.

---

## Stream A: Backend Refactoring

### A1. Characterization Tests

Write tests that lock current behavior for all modules we will refactor. These tests run against the unmodified code and must continue to pass after refactoring.

**Targets:**

| Module | File | LOC | Test focus |
|--------|------|-----|------------|
| Query orchestrator | `phentrieve/retrieval/query_orchestrator.py` | 1057 | All 3 code paths (sentence, fallback, full-text), `_InteractiveState`, multi-vector routing |
| HPO extraction orchestrator | `phentrieve/text_processing/hpo_extraction_orchestrator.py` | 346 | Batch retrieval, threshold filtering, reranking bypass, evidence aggregation, attribution |
| API dependencies | `api/dependencies.py` | 417 | Double-check locking, status transitions (not_loaded/loading/loaded/failed), timeout, cache hit |
| API main | `api/main.py` | 248 | App creation, MCP mount, startup sequence |

**Strategy**: Mock at the boundary (retriever, reranker, DB), assert on outputs. Use `pytest.mark.parametrize` for code path variations.

**Target**: ~80-120 new test cases.

### A2. Structural Refactoring

#### Orchestrator decomposition

Extract the 3x duplicated retrieve-convert-rerank-format sequence into a single pipeline function:

```python
# phentrieve/retrieval/pipeline.py (new)
def execute_retrieval_pipeline(
    retriever: DenseRetriever,
    reranker: CrossEncoder | None,
    text: str,
    config: RetrievalConfig,
) -> list[RetrievalResult]:
    """Single pipeline replacing 3 duplicated sequences at lines 554-606, 637-679, 737-799."""
    results = retriever.query(text, n_results=config.query_count)
    candidates = convert_results_to_candidates(results)
    if reranker:
        candidates = reranker.protected_dense_rerank(candidates, text, config)
    return candidates
```

Split `query_orchestrator.py` (1057 LOC) into:
- `phentrieve/retrieval/pipeline.py` (~100 LOC) — `execute_retrieval_pipeline()`, shared by all modes
- `phentrieve/retrieval/interactive_state.py` (~60 LOC) — `_InteractiveState` class
- `phentrieve/retrieval/query_orchestrator.py` (~300 LOC) — thin coordinator calling pipeline + formatters

Use shared utility (extracted in Stream C quick wins):
- Import `_convert_multi_vector_to_chromadb_format()` from `phentrieve/retrieval/utils.py`
- Note: Stream C extracts this utility on Day 1; Stream A consumes it

#### API dependency unification

Extract shared model loading logic:

```python
# api/dependencies.py
async def _load_model_with_status_tracking(
    model_name: str,
    loader_fn: Callable,
    cache_dict: dict,
    status_dict: dict,
    lock_dict: dict,
    timeout: float,
) -> Any:
    """Replaces ~140 lines of duplicated logic in get_sbert_model_dependency
    and get_cross_encoder_dependency."""
```

#### API lifecycle

- Add `create_app()` factory function
- Add `@asynccontextmanager` lifespan for startup/shutdown
- Move `sys.path` mutation (main.py:11) into factory
- Move eager `load_hpo_graph_data()` (similarity_router.py:61) into lifespan
- Move MCP mounting (_try_mount_mcp at main.py:171) into lifespan
- Fix version: replace hardcoded `"0.1.0"` at main.py:106,189 with `get_api_version()`

### A3. Cache Ownership Model (Backend)

**Primary model: bounded module-level caches.** The current code in `api/dependencies.py:26-36` and all existing tests are built around module-level global dicts. Moving to `app.state` would require API signature churn (every dependency function needs `Request` parameter) plus rewriting all dependency tests. That's a separate refactor.

This cycle:
- Replace unbounded global dicts (`LOADED_SBERT_MODELS`, `LOADED_RETRIEVERS`, `LOADED_CROSS_ENCODERS`) with `cachetools.TTLCache` (bounded maxsize, configurable TTL) — **same module-level location, same access pattern, just bounded**
- Add `threading.Lock` wrapper on each cache (required by cachetools for thread safety)
- Add explicit cache cleanup callable, invoked from lifespan shutdown handler via import (no `app.state` needed)
- Keep `MODEL_LOADING_STATUS` and `MODEL_LOAD_LOCKS` as module-level dicts (they are coordination state, not cached resources)

**Deferred to future cycle:** Migrate from module-level caches to `app.state` + `Request`-based dependency injection. This is a clean follow-on once the bounded caching and lifespan patterns are proven stable.

### A: Files touched

```
phentrieve/retrieval/query_orchestrator.py    (split)
phentrieve/retrieval/pipeline.py              (new)
phentrieve/retrieval/interactive_state.py     (new)
phentrieve/retrieval/utils.py                 (new)
phentrieve/evaluation/runner.py               (import change)
phentrieve/text_processing/hpo_extraction_orchestrator.py  (minor)
api/main.py                                   (factory + lifespan)
api/dependencies.py                           (unify loaders)
api/routers/similarity_router.py              (remove import-time load)
tests/unit/retrieval/                         (new char tests)
tests/unit/api/                               (new char tests)
```

---

## Stream B: Frontend Refactoring

### B1. Characterization Tests

Write Vitest component tests locking current behavior before decomposition:

| Component | LOC | Test focus |
|-----------|-----|------------|
| `QueryInterface.vue` | 1483 | Query submission, mode switching, advanced options, phenotype collection, export |
| `ResultsDisplay.vue` | 1079 | All 3 render paths (query/textProcess/aggregated), attribution, expand/collapse |
| `App.vue` | 598 | Disclaimer flow, version fetch, health monitoring, tutorial DOM management |
| `queryPreferences.js` | store | Current options-API behavior before migration |

**Strategy**: Mount with `@vue/test-utils`, mock Pinia stores and API service, assert on emitted events and rendered output.

**Target**: ~40-60 new test cases.

### B2. Structural Decomposition

#### QueryInterface.vue (1483 LOC -> ~300 LOC coordinator)

Extract composables:
- `useQueryExecution()` — query submission, API calls, loading state
- `useTextProcessing()` — text/document processing mode, chunk handling
- `useAdvancedOptions()` — model selection, language, thresholds, parameters
- `usePhenotypeCollection()` — collection management, assertion toggling, phenopacket export
- `useFileDownload()` — shared download utility (replaces 3x duplication in QueryInterface + LogViewer)

Extract sub-components:
- `AdvancedOptionsPanel.vue`
- `PhenotypeCollectionPanel.vue`
- `SpecimenMetadata.vue`

#### ResultsDisplay.vue (1079 LOC -> ~400 LOC + sub-components)

Extract sub-components:
- `ResultItem.vue` — single result with details expansion
- `ChunkResultsView.vue` — chunk-based text processing display
- `AggregatedTermsView.vue` — aggregated HPO terms display
- `TextAttribution.vue` — text attribution highlighting

#### App.vue cleanup

- Replace all `document.querySelector` calls (lines 445, 448, 461, 471, 475, 519, 546) with template refs and event-driven patterns
- Extract `useVersionCheck()` composable — version fetching + health monitoring
- Extract `useTutorial()` composable — decouple tutorial from DOM

#### Store and code cleanup

- Migrate `queryPreferences.js` from options API to composition API (setup store)
- Note: `composables/useDisclaimer.js` deleted in Stream C quick wins
- Move hardcoded thresholds (0.5, 0.7, 0.75, 0.85, 0.9, 120, 10, 3, etc.) to `frontend/src/constants/defaults.js`
- Move HPO URL template to `frontend/src/constants/urls.js`

### B3. Performance (Frontend)

Apply only where profiling or Vue DevTools render-count inspection confirms a measurable benefit. Do not blindly add optimization annotations.

- **Deep watcher fix** (confirmed issue): Replace deep watcher on `conversationStore.queryHistory` (QueryInterface.vue:926) with shallow watch on `.length` — this fires on every nested property change and is always wasteful
- **JSON.stringify fix** (confirmed issue): Replace `JSON.stringify` in LogViewer's `filteredLogs` with targeted field search — stringify on every filter keystroke is unconditionally expensive
- **Route-level code splitting**: Add `defineAsyncComponent` for views — standard practice, no profiling needed
- **Conditional** (apply if profiling shows benefit): `v-memo` on result list items, `shallowRef` for large read-only datasets

### B: Files touched

```
frontend/src/components/QueryInterface.vue      (decompose)
frontend/src/components/ResultsDisplay.vue       (decompose)
frontend/src/App.vue                             (cleanup)
frontend/src/components/LogViewer.vue            (download fix)
frontend/src/composables/useQueryExecution.js    (new)
frontend/src/composables/useTextProcessing.js    (new)
frontend/src/composables/useAdvancedOptions.js   (new)
frontend/src/composables/usePhenotypeCollection.js (new)
frontend/src/composables/useFileDownload.js      (new)
frontend/src/composables/useVersionCheck.js      (new)
frontend/src/composables/useTutorial.js          (new)
frontend/src/composables/useDisclaimer.js        (delete)
frontend/src/components/AdvancedOptionsPanel.vue (new)
frontend/src/components/PhenotypeCollectionPanel.vue (new)
frontend/src/components/SpecimenMetadata.vue     (new)
frontend/src/components/ResultItem.vue           (new)
frontend/src/components/ChunkResultsView.vue     (new)
frontend/src/components/AggregatedTermsView.vue  (new)
frontend/src/components/TextAttribution.vue      (new)
frontend/src/stores/queryPreferences.js          (migrate to setup store)
frontend/src/constants/defaults.js               (new)
frontend/src/constants/urls.js                   (new)
frontend/src/test/                               (new component tests)
```

---

## Stream C: Infrastructure & Test Quality

### C1. Quick Wins (Day 1)

Each is an independent commit:

| Fix | File(s) | Change |
|-----|---------|--------|
| Delete dead composable | `frontend/src/composables/useDisclaimer.js` | Delete file |
| Delete disabled tests | `tests/unit/cli/test_query_commands.py.disabled`, `conftest.py.disabled` | Delete files |
| Fix Ruff target | `pyproject.toml:92` | `py39` -> `py310` |

Note: Two quick wins from the original review moved into **Stream A** to eliminate cross-stream file overlap:
- Fix version hardcode (`api/main.py`) — now part of A2 API lifecycle work
- Extract shared utility (`phentrieve/retrieval/utils.py`) — now part of A2 orchestrator decomposition

### C2. Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.4
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        additional_dependencies: ["pydantic>=2", "types-requests"]
        args: ["--ignore-missing-imports"]
```

No pytest in pre-commit (too slow). Tests stay in `make test`.

### C3. Test Infrastructure Fixes

**Import path: keep current approach (ACCEPTED).**
The repo has already tried and rejected 6 approaches to fixing pytest import paths for the `api` module (see `tests/unit/api/README.md:67-86`). Root cause: pytest assertion rewriting runs before any path configuration is processed. The current Makefile/PYTHONPATH solution (`make test-api`) works reliably.
- Keep the `sys.path` hack in `tests/unit/api/test_dependencies_model_loading.py:17-19` for this cycle
- Keep `make test-api` as the official way to run API tests
- Defer `src` layout / packaging cleanup to a dedicated future refactor (noted in Out of Scope)

**Fix 47 zero-assertion tests:**
- Audit each test — add meaningful value assertions or convert bare `pytest.raises` to include `match=` parameter
- Priority: `test_dependencies_model_loading.py`, `test_text_processing_router_performance.py`, `test_benchmark_integration.py`

**Add parametrize:**
- `test_assertion_detection.py` — parametrize across 122 rules x 5 languages
- `test_hpo_parser_edge_cases.py` — parametrize similar edge case structures
- `test_text_processing_router.py` — parametrize validation variations

**Re-enable coverage threshold:**
- Un-comment `--cov-fail-under` in `pyproject.toml`
- Set to current baseline (~45%), ratchet up as streams A/B add tests
- New/refactored modules target 80%+ coverage

**Marker consistency:**
- Add `@pytest.mark.unit` to 4 unmarked files in `tests/unit/`

**Centralize fixtures:**
- Move duplicated `mock_sbert_model`, `mock_cross_encoder` to `tests/unit/conftest.py`

### C: Files touched

```
.pre-commit-config.yaml                          (new)
pyproject.toml                                    (ruff target, coverage threshold)
tests/unit/conftest.py                            (shared fixtures)
tests/unit/core/test_assertion_detection.py       (parametrize)
tests/unit/data_processing/test_hpo_parser_edge_cases.py (parametrize)
tests/unit/api/test_text_processing_router.py     (parametrize + assertions)
~15 files with zero-assertion tests               (add assertions)
4 unmarked test files                             (add markers)
frontend/src/composables/useDisclaimer.js         (delete)
tests/unit/cli/test_query_commands.py.disabled    (delete)
tests/unit/cli/conftest.py.disabled               (delete)
```

Note: `api/main.py` version fix and `phentrieve/retrieval/utils.py` extraction are now in Stream A. API test import path hack is kept as-is (see C3).

---

## Out of Scope (Deferred)

| Item | Reason |
|------|--------|
| Accessibility (heading hierarchy, skip links, ARIA) | Separate focused cycle |
| Evaluation module tests (24 untested modules) | Too large, separate cycle |
| Benchmark module tests | Separate cycle |
| Config unification (3 YAML systems) | Needs architectural decision first |
| `src` layout / packaging cleanup | 6 workarounds tried and failed (see `tests/unit/api/README.md:67-86`); needs dedicated refactor |
| Migrate caches to `app.state` | Requires API signature churn + test rewrites; do after bounded caching is stable |
| Semantic query caching | Needs benchmarking data |
| Character-offset provenance | Feature addition, not refactoring |
| Assertion-aware chunking | Historical plan exists (`.planning/archived/CHUNKING-OPTIMIZATION-PLAN.md`) |
| Virtual scrolling | Only if list sizes become a real problem |
| `eslint-plugin-vuejs-accessibility` | Deferred with accessibility work |

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| `query_orchestrator.py` LOC | 1057 | <350 (coordinator) |
| `QueryInterface.vue` LOC | 1483 | <350 (coordinator) |
| `ResultsDisplay.vue` LOC | 1079 | <450 (coordinator) |
| Duplicated retrieve-rerank-format sequences | 3 | 1 |
| Files >1000 LOC (non-test) | 6 | 2 (chunkers.py, assertion_detection.py deferred) |
| Tests with zero assertions | 47 | 0 |
| `@pytest.mark.parametrize` usage | 3 files | 8+ files |
| Coverage threshold | disabled | enabled at baseline, ratcheting up |
| Pre-commit hooks | none | ruff + mypy + standard hooks |
| Module-level unbounded resource caches | 3 (SBERT, retrievers, cross-encoders) | 0 (all bounded via cachetools.TTLCache, same module location) |
| `document.querySelector` in Vue components | 7+ locations | 0 |
| Dead code files | 3 | 0 |
| `make all` passes | yes | yes (at every commit) |

---

## Verification Gates

Each stream has explicit required checks beyond `make all`. A stream is not complete until all its gates pass.

### Stream A Gates
1. `make check` (ruff format + lint) — zero errors
2. `make typecheck-fast` (mypy daemon) — zero errors
3. `make test` — all existing tests pass
4. **New orchestrator unit tests**: all characterization tests from A1 pass against refactored code, covering all 3 query paths (sentence, fallback, full-text), `_InteractiveState`, and multi-vector routing
5. **New API dependency tests**: status transitions (not_loaded/loading/loaded/failed), timeout behavior, cache hit/miss for unified loader
6. **New API lifecycle tests**: `create_app()` returns configured app, lifespan initializes/cleans up, no import-time side effects
7. **Bounded cache verification**: `cachetools.TTLCache` maxsize is respected, eviction works, cleanup callable runs at shutdown

### Stream B Gates
1. `make frontend-lint` + `make frontend-format` — zero errors
2. `make frontend-build` — production build succeeds
3. `make frontend-test` — all existing store tests pass
4. **New component tests**: Vitest suite for QueryInterface, ResultsDisplay, App covering query submission, mode switching, all 3 render paths, disclaimer flow, export
5. **Manual smoke test**: run `make dev-frontend` + `make dev-api`, exercise query flow, text processing, phenotype collection, and export end-to-end in browser
6. `make frontend-i18n-check` — all locale keys still valid after decomposition

### Stream C Gates
1. `make check` + `make typecheck-fast` — zero errors
2. `make test` — all tests pass (including newly-fixed zero-assertion tests)
3. **Pre-commit verification**: `pre-commit run --all-files` passes
4. **Zero-assertion audit**: grep confirms no test functions with empty bodies or bare `pytest.raises` without `match=`
5. **Marker audit**: all files in `tests/unit/` have `@pytest.mark.unit`, all in `tests/integration/` have `@pytest.mark.integration`
6. **Coverage gate**: `--cov-fail-under` enabled and passing at baseline

---

## Worktree Strategy

Each stream runs in its own worktree branched from `main`:

```
main
 ├── worktree: improve/backend-refactor     (Stream A)
 ├── worktree: improve/frontend-refactor    (Stream B)
 └── worktree: improve/infrastructure       (Stream C)
```

**Cross-stream dependencies:**
- Stream A and Stream C have **zero file overlap** — backend quick wins (version fix, utility extraction) were moved into Stream A to eliminate the original conflict
- Stream B depends on Stream C's dead composable deletion — or B simply skips that file (trivial)
- No dependency between Stream A and Stream B
- All three streams can branch from `main` simultaneously

Merge order at end of cycle: C first (smallest, foundational), then A, then B. File overlap between streams is <5% (only `pyproject.toml` is touched by both A and C — A for coverage config changes, C for Ruff target).

Within each stream, every logical unit is an atomic commit with `make all` passing.

## Research Sources

- FastAPI lifespan/factory: [FastAPI Events](https://fastapi.tiangolo.com/advanced/events/), [LSST SQR-072](https://sqr-072.lsst.io/)
- Bounded caching: [cachetools docs](https://cachetools.readthedocs.io/), [cachebox](https://github.com/awolverp/cachebox)
- Vue composables: [Vue Composables Guide](https://vuejs.org/guide/reusability/composables.html), [Enterprise Patterns](https://www.yeasirarafat.com/posts/vue-composition-api-advanced-patterns)
- Vue performance: [Vue Performance Guide](https://vuejs.org/guide/best-practices/performance)
- Pinia setup stores: [Pinia Core Concepts](https://pinia.vuejs.org/core-concepts/)
- Template refs: [Vue Template Refs](https://vuejs.org/guide/essentials/template-refs)
- Pre-commit: [ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit), [pydevtools guide](https://pydevtools.com/handbook/how-to/how-to-set-up-pre-commit-hooks-for-a-python-project/)
- RAG architecture: [kapa.ai RAG Guide](https://www.kapa.ai/blog/how-to-build-a-rag-pipeline-from-scratch-in-2026), [Chroma Performance](https://docs.trychroma.com/guides/deploy/performance)
- Clinical chunking: [PMC RAG Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12649634/)
