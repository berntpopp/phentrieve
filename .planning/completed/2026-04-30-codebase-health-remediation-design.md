# Codebase Health Remediation Design

## Goal

Raise Phentrieve from the 2026-04-30 health review score of `6.5/10` to a
credible `>8/10` by fixing the review's safety, correctness, modularity,
frontend maintainability, and developer-experience findings in one coordinated
program.

## Scope

This design covers all 13 findings from
`.planning/analysis/2026-04-30-codebase-health-review.md`:

1. `/query` arbitrary retrieval model names.
2. `CombinedAssertionDetector.preference` not honored.
3. Raw clinical query history persisted in browser storage.
4. Concentrated LLM pipeline/provider modules.
5. Unused HPO extraction helper module.
6. Overloaded `text_processing_router.py`.
7. Blocking retriever/model initialization in async dependency paths.
8. Prompt-injection guard inconsistency across prompt families.
9. Oversized `QueryInterface.vue`.
10. Logging side effects in `ResultsDisplay.vue` prop validation.
11. Split/import-time-bound configuration.
12. Packaging/docs/Makefile drift.
13. Dead frontend `trustRemoteCode` intent.

## Non-Goals

- Do not change extraction ranking semantics except where required to honor
  documented assertion preferences.
- Do not replace the current Vue/Vuetify stack.
- Do not replace FastAPI, ChromaDB, SentenceTransformers, or the LLM provider
  integrations.
- Do not introduce remote telemetry or server-side storage of raw clinical text.
- Do not broaden model choices beyond the existing safe/allowlisted choices.

## Design Principles

- Safety-sensitive policy must be server-owned and shared across endpoints.
- Clinical text must not be durably persisted without explicit user intent.
- Refactors must be characterization-test driven.
- Public API route files should be thin HTTP adapters.
- LLM extraction components should be small enough to review independently.
- Planning and documentation artifacts remain under `.planning/` according to
  `AGENTS.md`.

## Target Architecture

### Shared model policy

Create one shared retrieval model policy module used by both `/query` and
`/text/process`.

Responsibility:

- Own the allowlist of model IDs.
- Own the `trust_remote_code` decision per allowed model.
- Reject unsupported model IDs before dependency/model construction.
- Remove substring-based trust decisions from lower-level model loading.

Expected boundary:

- `phentrieve/model_policy.py` or `phentrieve/retrieval/model_policy.py` owns
  policy.
- API routers and dependencies consume this policy.
- `phentrieve/embeddings.py` receives an explicit trust flag and does not infer
  trust from model-name substrings.

### Assertion detector preferences

Make `CombinedAssertionDetector.preference` behavior explicit and testable.

Supported strategies:

- `dependency`: dependency result wins unless unavailable.
- `keyword`: keyword result wins unless unavailable.
- `any_negative`: any negative/uncertain detector result wins over normal.

The default behavior should preserve existing behavior unless existing docs or
configuration indicate a different intended default.

### Browser privacy

Raw clinical notes should be transient by default.

Target behavior:

- Persist metadata, IDs, timestamps, phenotypes, and redacted query text.
- Keep raw query text in memory/session state only unless the user explicitly
  opts into durable local history.
- Continuing through a PII warning must not silently write raw PII to durable
  browser storage.

### HPO extraction orchestration

Convert the current unused helper module into the real internal boundary, or
delete it if a better split emerges during implementation.

Target internal units:

- chunk match processing
- term detail loading
- evidence map creation
- aggregation and ranking
- final response assembly

The public behavior of `orchestrate_hpo_extraction()` should stay stable.

### API boundary

Keep `text_processing_router.py` focused on HTTP concerns.

Move to service/dependency helpers:

- request context preparation
- backend/model policy resolution
- standard backend execution
- LLM backend execution
- response adaptation
- timeout/threadpool coordination

Heavy dependency construction should happen in a threadpool or startup preload
path so async dependency functions do not block the event loop on cache misses.

### Prompt safety

Make untrusted-document handling a reusable prompt primitive.

Every extraction-capable prompt family should include:

- explicit system/developer instruction hierarchy
- untrusted clinical text boundaries
- instruction to ignore commands embedded in user-provided clinical text
- tests that fail if those boundaries disappear

### Frontend maintainability

Split `QueryInterface.vue` into focused pieces while preserving user-visible
behavior.

Target units:

- query input/form shell
- full-text note workspace
- PII review orchestration
- result/collection actions
- mode/model controls
- export and history helpers

State ownership:

- durable conversation data stays in Pinia stores
- transient UI state stays in local component/composable state
- service payload normalization stays in `PhentrieveService`

Remove logging side effects from prop validators and remove dead
`trustRemoteCode` client payload intent.

### LLM modularization

Split LLM modules by responsibility without changing provider semantics.

Target units:

- phase 1 extraction
- phase 2 mapping
- candidate retrieval/mapping support
- trace/result assembly
- retry/error classification
- provider-specific request/response adapters
- tool execution
- provider resolver

Use characterization tests before moving code.

### Configuration and docs

Make `pyproject.toml` the source of truth for package metadata and optional
extras.

Target cleanup:

- remove or rename stale Makefile extras
- update docs away from stale `pip install` and nonexistent extra names
- align Python-version documentation with `pyproject.toml`
- add a static check that documented extras referenced by Makefile/docs exist

## Rollout Order

1. P1 safety/correctness changes.
2. HPO/API backend boundary cleanup.
3. Frontend privacy and component cleanup.
4. LLM modularization.
5. Config/docs/CI cleanup.
6. Full verification.

This order prioritizes user safety and extraction correctness before structural
refactors.

## Testing Strategy

Required focused tests:

- API model policy tests for `/query` and `/text/process`.
- Unit tests for assertion detector preference modes.
- Frontend store persistence tests for raw/redacted query history.
- HPO orchestrator characterization tests.
- API service/dependency tests for request preparation and threadpool behavior.
- Prompt rendering tests for all extraction-capable prompt families.
- Frontend component/composable tests for split `QueryInterface` behavior.
- LLM characterization tests around provider/pipeline refactors.
- Static docs/config checks for optional extras and documented install commands.

Required final verification:

- `make check`
- `make typecheck-fast`
- `make test`
- `make frontend-test-ci`
- `make frontend-build-ci`

## Risks And Controls

- LLM refactoring can change subtle provider behavior. Control with
  characterization tests before moving code.
- Frontend decomposition can break UI flows. Control with component tests around
  submit, PII review, mode switching, and result collection.
- Assertion preference changes can alter extraction output. Control by making
  default behavior explicit and adding regression cases.
- Browser privacy changes can surprise users who expect history persistence.
  Control by preserving redacted history and making durable raw history an
  explicit setting.

## Success Criteria

- No public API path accepts arbitrary remote model IDs.
- Raw clinical notes are not durably persisted by default.
- Assertion detector preferences are meaningful and tested.
- Core extraction and LLM logic are decomposed behind tested boundaries.
- `QueryInterface.vue`, `pipeline.py`, and `provider.py` are materially smaller
  and easier to review.
- Docs, Makefile, and packaging metadata agree.
- Required checks pass.
