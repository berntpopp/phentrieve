# Codebase Maintainability Review — 2026-04-15

**Scope:** `phentrieve/`, `api/`, `frontend/`, `tests/`, packaging, CI, docs
**Goal:** improve maintainability and development pace; define the fastest path to a codebase score above `8/10`
**Method:** direct repo inspection, focused subsystem reviews via parallel agents, lightweight local checks, and current primary-source guidance from Google code review guidance, FastAPI, Vue, PyPA packaging, and SentenceTransformers docs.

---

## Executive Summary

**Current score: `5.8/10`**

The codebase has real strengths: domain separation is visible, there is meaningful test coverage, the Makefile is broad, the frontend already uses Pinia/composables, and the API has a coherent error envelope. The main drag on maintainability is not style quality. It is **architectural concentration**: too much orchestration, mutable state, configuration, and fallback logic sits in a handful of oversized modules and components.

The fastest route to `>8/10` is **not** a repo-wide cleanup. It is a short, focused program:

1. Break up the core orchestration hotspots.
2. Re-centralize configuration and state boundaries.
3. Restore trust in tests, docs, and CI as the source of truth.
4. Remove a small number of backend/API operational risks.

If those four areas are addressed first, the score can realistically move into the low `8s` without a rewrite.

---

## Review Rubric

The rubric used here follows current primary guidance:

- Google code review guidance prioritizes **design**, **complexity**, **tests**, **documentation**, and whether changes improve overall code health: <https://google.github.io/eng-practices/review/reviewer/looking-for.html>
- FastAPI’s own docs recommend structuring larger apps across **multiple files**, with separate routers and dependencies rather than centralizing everything: <https://fastapi.tiangolo.com/tutorial/bigger-applications/>
- Vue’s scaling guide recommends extracting shared state out of components and centralizing state-mutating logic; it also recommends Pinia for larger apps: <https://vuejs.org/guide/scaling-up/state-management.html>
- PyPA packaging guidance treats `pyproject.toml` as the central place for project metadata and tool config, so docs and commands should align with that source of truth: <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>
- SentenceTransformers documents the retrieval/rerank split explicitly, which is relevant to reviewing RAG boundaries and orchestration design: <https://sbert.net/>

---

## Scorecard

| Area | Score | Notes |
|---|---:|---|
| Architecture and modularization | 5 | Good top-level domain folders, but too many central orchestrators and god-modules |
| DRY / KISS / SOLID | 5 | Repeated setup logic, hidden mutable state, mixed responsibilities |
| Core Python maintainability | 6 | Functional, but major hotspots are hard to test and change safely |
| API / MCP backend | 6 | Good schema/error structure, but some async and safety boundaries leak |
| Frontend maintainability | 6 | Solid primitives, but key UI flows are still over-concentrated |
| RAG / ML architecture | 6 | Retrieval concepts are sound; execution path is too dict-heavy and orchestration-heavy |
| Tests and CI signal quality | 5 | Some good coverage, but current signal is weaker than it appears |
| Docs and developer experience | 4 | Multiple doc pages are out of sync with the repo and toolchain |
| Security and operational safety | 5 | Generally reasonable, but one model-loading path is too permissive |

**Summary score: `5.8/10`**

---

## What Is Working

- The repo is organized by domain, not as one flat package.
- The frontend already uses the right scaling primitives: Pinia, composables, and subcomponents.
- The API has a centralized error-response contract in [api/main.py](/home/bernt-popp/development/phentrieve/api/main.py:125).
- The Makefile is broad and gives the repo a discoverable workflow surface.
- The RAG stack itself is directionally sensible: embedding retrieval, optional reranking concepts, chunking, attribution, and evaluation are all present.
- `ruff` is currently clean locally.

---

## Highest-Priority Findings

### P0

1. **Core RAG orchestration is too concentrated.**
   [phentrieve/retrieval/query_orchestrator.py](/home/bernt-popp/development/phentrieve/phentrieve/retrieval/query_orchestrator.py:45) combines interactive state, segmentation, retrieval mode switching, reranking flow, formatting, and orchestration. That violates separation of concerns and makes the most business-critical path harder to test and evolve.

2. **Configuration is too import-time and too global.**
   [phentrieve/config.py](/home/bernt-popp/development/phentrieve/phentrieve/config.py:1) is acting as both constants module and runtime config loader. Defaults also drift into retrieval modules. This raises coupling and makes tests/config overrides less reliable.

3. **The API exposes an unsafe model-loading path.**
   `trust_remote_code` is user-controlled in [api/schemas/text_processing_schemas.py](/home/bernt-popp/development/phentrieve/api/schemas/text_processing_schemas.py:66) and used in model loading from [api/routers/text_processing_router.py](/home/bernt-popp/development/phentrieve/api/routers/text_processing_router.py:263). Combined with caller-supplied model names, this is an avoidable operational and security risk.

4. **Docs and CI are not a dependable source of truth right now.**
   There is a likely undeclared pytest marker in [tests/unit/cli/test_benchmark_integration.py](/home/bernt-popp/development/phentrieve/tests/unit/cli/test_benchmark_integration.py:15) versus marker registration in [pyproject.toml](/home/bernt-popp/development/phentrieve/pyproject.toml:175). CI also treats mypy as advisory with `continue-on-error: true` in [.github/workflows/ci.yml](/home/bernt-popp/development/phentrieve/.github/workflows/ci.yml:115). Several docs pages still describe commands and extras that no longer exist.

### P1

5. **The assertion-detection and HPO-preparation paths are monolithic.**
   [phentrieve/text_processing/assertion_detection.py](/home/bernt-popp/development/phentrieve/phentrieve/text_processing/assertion_detection.py:131) and [phentrieve/data_processing/hpo_parser.py](/home/bernt-popp/development/phentrieve/phentrieve/data_processing/hpo_parser.py:66) both mix parsing, loading, orchestration, and policy in ways that slow change and discourage focused tests.

6. **The frontend’s main flow is still too centralized.**
   [frontend/src/components/QueryInterface.vue](/home/bernt-popp/development/phentrieve/frontend/src/components/QueryInterface.vue:1) is still the biggest maintainability hotspot. It mixes route/query parsing, request assembly, auto-submit behavior, scroll handling, and UI composition.

7. **Result rendering has duplication and drift risk.**
   [frontend/src/components/ResultsDisplay.vue](/home/bernt-popp/development/phentrieve/frontend/src/components/ResultsDisplay.vue:1) still overlaps with `ChunkResultsView`, `AggregatedTermsView`, and `ResultItem` around detail toggles, HPO link behavior, and collection actions.

8. **Async boundaries in the API are inconsistent.**
   Some heavy work is correctly moved to thread pools, but some blocking paths remain synchronous inside async request flows, including retriever construction and ontology/config loading. See [api/dependencies.py](/home/bernt-popp/development/phentrieve/api/dependencies.py:238), [api/routers/similarity_router.py](/home/bernt-popp/development/phentrieve/api/routers/similarity_router.py:113), and [api/routers/config_info_router.py](/home/bernt-popp/development/phentrieve/api/routers/config_info_router.py:132).

### P2

9. **State and persistence boundaries are blurred in the frontend.**
   [frontend/src/stores/conversation.js](/home/bernt-popp/development/phentrieve/frontend/src/stores/conversation.js:18), [frontend/src/i18n.js](/home/bernt-popp/development/phentrieve/frontend/src/i18n.js:19), and [frontend/src/components/LanguageSwitcher.vue](/home/bernt-popp/development/phentrieve/frontend/src/components/LanguageSwitcher.vue:57) all participate in persistence decisions. That works, but it makes the state model harder to test and reason about.

10. **The test tree mixes unit, integration, performance, and real-dependency tests in ways that weaken signal.**
    `tests/unit/` currently contains multiple non-unit categories, making fast selective runs and coverage interpretation less trustworthy.

11. **Local checks are close, but not fully green.**
    `uv run ruff check phentrieve api tests` passed. `uv run mypy phentrieve api` currently fails on one issue: `api/dependencies.py:6: error: Unused "type: ignore" comment  [unused-ignore]`.

---

## Concrete Evidence

- Oversized hotspots include:
  [phentrieve/retrieval/query_orchestrator.py](/home/bernt-popp/development/phentrieve/phentrieve/retrieval/query_orchestrator.py:386), [phentrieve/text_processing/assertion_detection.py](/home/bernt-popp/development/phentrieve/phentrieve/text_processing/assertion_detection.py:325), [phentrieve/data_processing/hpo_parser.py](/home/bernt-popp/development/phentrieve/phentrieve/data_processing/hpo_parser.py:900), [frontend/src/components/QueryInterface.vue](/home/bernt-popp/development/phentrieve/frontend/src/components/QueryInterface.vue:1), and [frontend/src/components/ResultsDisplay.vue](/home/bernt-popp/development/phentrieve/frontend/src/components/ResultsDisplay.vue:1).
- There is duplicate interactive-state ownership in both [phentrieve/retrieval/query_orchestrator.py](/home/bernt-popp/development/phentrieve/phentrieve/retrieval/query_orchestrator.py:45) and [phentrieve/retrieval/interactive_state.py](/home/bernt-popp/development/phentrieve/phentrieve/retrieval/interactive_state.py:1).
- Numeric defaults are applied with `or` in [api/routers/text_processing_router.py](/home/bernt-popp/development/phentrieve/api/routers/text_processing_router.py:348), which can discard valid `0.0` values that the schema explicitly allows.
- Several docs are stale:
  [docs/development/running-tests.md](/home/bernt-popp/development/phentrieve/docs/development/running-tests.md:1),
  [docs/development/dev-environment.md](/home/bernt-popp/development/phentrieve/docs/development/dev-environment.md:186),
  [docs/getting-started/installation.md](/home/bernt-popp/development/phentrieve/docs/getting-started/installation.md:87).

---

## Fastest Path To `>8/10`

### Phase 1: Restore trust and remove risk

1. Remove request-level `trust_remote_code` and replace caller-supplied models with a server-side allowlist.
2. Fix the pytest marker mismatch and add a collection-only smoke check to CI.
3. Make mypy blocking in CI after fixing the current one-line error.
4. Update docs so `README`, `Makefile`, `pyproject.toml`, and docs all describe the same install/test workflow.

**Expected score impact:** `+0.8 to +1.0`

### Phase 2: Break up the orchestration hotspots

5. Split `query_orchestrator.py` into:
   - retrieval execution
   - rerank/result scoring
   - result normalization
   - interactive session state
   - presentation formatting
6. Convert config from import-time globals to validated config objects/settings passed at boundaries.
7. Break `assertion_detection.py` and `hpo_parser.py` into smaller services with injected dependencies.
8. Move `QueryInterface.vue` orchestration into composables/services and reduce `ResultsDisplay.vue` duplication.

**Expected score impact:** `+1.3 to +1.8`

### Phase 3: Improve change velocity

9. Reclassify tests so `unit` really means fast, isolated, and low-flake.
10. Add targeted characterization tests around refactored orchestration seams.
11. Consolidate frontend persistence/state mutation patterns.
12. Consolidate API route/config ownership and move remaining blocking work off async request paths.

**Expected score impact:** `+0.5 to +0.8`

**Realistic post-phase score:** `8.1 to 8.4`

---

## Recommended Priority Order

1. Lock down model loading in the API.
2. Fix CI/test/doc truthfulness.
3. Refactor `query_orchestrator.py`.
4. Refactor `assertion_detection.py` and `hpo_parser.py`.
5. Split `QueryInterface.vue` and deduplicate result rendering.
6. Normalize configuration ownership across CLI/API/frontend/docs.

---

## Suggested Success Metrics

- `make check`, `make typecheck-fast`, and `make test` are all green and match CI behavior.
- No user-controlled remote model code execution path remains.
- No file over roughly `400-500` lines in critical orchestration paths without a clear reason.
- `unit` tests are fast and isolated; performance/real/integration tests are explicitly separated.
- Docs describe only commands that work in the current repo.
- New features land by extending small modules/composables/services instead of expanding central orchestrators.

---

## Verification Notes

- Completed: repo inspection, parallel subsystem reviews, `ruff` check, `mypy` run.
- Not completed: full `make all`, full frontend test/build run, full Docker/E2E verification.
- Interpretation note: this is a maintainability and development-pace review, not a release-certification report.
