# Phentrieve Codebase Health Review - 2026-04-30

## Executive Summary

Reviewed the current Phentrieve codebase on branch `main` against the archived
quality reviews, current source code, CI/configuration, API/backend design,
frontend maintainability, security/privacy posture, and RAG/HPO extraction
architecture.

Current architecture, in brief:

- Python domain code is split across `retrieval`, `text_processing`, `llm`,
  `evaluation`, `benchmark`, `cli`, and `data_processing`.
- FastAPI uses `create_app()`, lifespan startup/cleanup, routers, and dependency
  caches.
- `/text/process` now has a safer service boundary and server-owned model
  policy.
- `/query` remains a direct route-to-retriever path with weaker model
  validation.
- LLM extraction is two-phase and grounded, but concentrated in very large
  pipeline/provider modules.
- Frontend uses Vue 3, Pinia, composables, and services, but
  `QueryInterface.vue` still owns too much UI and workflow state.
- CI is materially stronger than older reviews: Ruff, mypy, pytest collection,
  Python matrix, and frontend jobs are present.
- Privacy/safety has improved with local PII detection and prompt-injection
  defenses, but browser persistence and model-loading policy still need
  tightening.

No P0 release-stopper was found. Several P1 issues still cap the codebase below
a high-confidence health score.

## Overall Score: `6.5/10`

## Scorecard

| Area | Score |
| --- | ---: |
| Architecture/modularization | 6.4 |
| Python core maintainability | 5.8 |
| API/backend design | 7.0 |
| Frontend maintainability | 6.4 |
| Security/privacy/safety | 6.8 |
| RAG/HPO extraction quality architecture | 6.2 |
| Tests/CI signal | 7.2 |
| Packaging/config/docs/developer experience | 6.3 |

## Highest Priority Findings

### 1. P1: `/query` still accepts arbitrary retrieval model names

Affected files:

- `api/schemas/query_schemas.py:22`
- `api/routers/query_router.py:66`
- `api/dependencies.py:238`
- `phentrieve/embeddings.py:184`

Evidence:

- `QueryRequest.model_name` is a free string.
- `query_router.py` passes the request model name into retriever/model
  construction.
- `load_sentence_transformer_model()` enables `trust_remote_code` when the
  model name is the default BioLORD model or contains `"BioLORD"`.
- `/text/process` has a model allowlist and server-owned trust policy, but
  `/query` does not.

Why it matters:

Model loading is security-sensitive and operationally expensive. One API path
should not bypass the stricter policy added to another public endpoint.

Recommended fix:

Centralize retrieval model policy in one allowlist object with explicit
`trust_remote_code` metadata per allowed model. Remove substring-based trust
decisions.

Suggested tests:

- Reject non-allowlisted query models.
- Reject suspicious model names containing `"BioLORD"` unless exactly
  allowlisted.
- Verify each allowed model receives the expected trust flag.

### 2. P1: assertion detector preference is stored but not honored

Affected files:

- `phentrieve/text_processing/assertion_detection.py:1106`
- `phentrieve/text_processing/assertion_detection.py:1183`

Evidence:

- `CombinedAssertionDetector.__init__()` accepts and stores `preference`.
- `detect()` always applies a fixed dependency/keyword priority order.
- The result does not branch on `self.preference`.

Why it matters:

Negation and uncertainty handling directly affect HPO extraction correctness in
clinical text. A configuration knob that silently does not work is risky.

Recommended fix:

Implement explicit behavior for `dependency`, `keyword`, and `any_negative`, or
remove the option.

Suggested tests:

- Use fake keyword/dependency detectors that intentionally disagree.
- Verify each preference mode returns the documented result.

### 3. P1: raw clinical query history is persisted in browser storage

Affected files:

- `frontend/src/stores/conversation.js:138`
- `frontend/src/stores/conversation.js:380`
- `frontend/src/composables/useQueryInterfaceController.js:298`

Evidence:

- Local PII scanning and review exist before submit.
- `queryHistory` stores `queryItem.query`.
- Pinia persisted state includes `queryHistory` and `collectedPhenotypes`.

Why it matters:

Clinical text can persist beyond the active browser session. Browser-local
storage is still durable storage, and "continue anyway" can leave sensitive
notes behind.

Recommended fix:

Make raw-text history session-only by default. Persist only redacted text or
metadata unless the user explicitly opts into durable raw history.

Suggested tests:

- Store persistence tests for redacted PII flow.
- Store persistence tests for "continue anyway" flow.
- Store persistence tests proving raw clinical text is not durable by default.

### 4. P1: LLM pipeline/provider modules are too concentrated for safety review

Affected files:

- `phentrieve/llm/pipeline.py:510`
- `phentrieve/llm/provider.py:368`
- `phentrieve/llm/provider.py:1549`

Evidence:

- `phentrieve/llm/pipeline.py` is about 2400 LOC.
- `phentrieve/llm/provider.py` is about 1965 LOC.
- Provider classes, schema compaction, tool execution, retry/error handling, and
  provider resolution live together.

Why it matters:

Public LLM extraction is one of the riskiest flows. Concentrated files slow
review and make regressions harder to isolate.

Recommended fix:

Split by responsibility:

- extraction phase
- mapping phase
- trace/result assembly
- retry policy
- provider implementations
- tool executor

Suggested tests:

- Characterization tests around existing two-phase outputs before refactoring.
- Provider-specific tests for request shaping and retry classification.

### 5. P1: HPO extraction helper extraction appears unfinished or dead

Affected files:

- `phentrieve/text_processing/hpo_extraction_orchestrator.py:21`
- `phentrieve/text_processing/_hpo_extraction_helpers.py:1`

Evidence:

- `_hpo_extraction_helpers.py` defines helper boundaries.
- Current source search found no production references to those helpers.
- `hpo_extraction_orchestrator.py` still contains the full inlined extraction
  flow.

Why it matters:

This is the core HPO extraction path. Dead helper modules create false
confidence and duplicate design intent without reducing risk.

Recommended fix:

Either wire the helpers into the orchestrator or delete them. Prefer moving
chunk matching, term enrichment, evidence mapping, aggregation, and ranking
behind focused functions.

Suggested tests:

- Golden extraction tests before and after the refactor.
- Cases covering negated text, synonyms, overlapping mentions, and duplicate
  evidence.

### 6. P2: `text_processing_router.py` still mixes too many responsibilities

Affected files:

- `api/routers/text_processing_router.py:292`
- `api/routers/text_processing_router.py:634`

Evidence:

- The router is about 733 LOC.
- It contains allowlist validation, retriever/model setup, service execution,
  LLM target resolution, timeout handling, and response shaping.

Why it matters:

The public API boundary is better than before, but future changes still land in
a high-coupling file.

Recommended fix:

Move request preparation, model policy, dependency construction, and response
adaptation into service/dependency modules. Keep the router as an HTTP adapter.

Suggested tests:

- Direct service tests without FastAPI.
- Route tests focused on HTTP status, schema, and auth/quota behavior.

### 7. P2: blocking retriever/model initialization still happens in async paths

Affected files:

- `api/dependencies.py:232`

Evidence:

- `get_dense_retriever_dependency()` is async.
- It constructs `DenseRetriever.from_model_name()` synchronously on cache miss.

Why it matters:

Cold starts or cache misses can block the event loop.

Recommended fix:

Run heavy construction in a threadpool or make initialization an explicit
startup/preload step with bounded concurrency.

Suggested tests:

- Monkeypatch a slow constructor and verify dependency resolution does not block
  the event loop.

### 8. P2: prompt-injection protections are not consistently applied to all prompt families

Affected files:

- `phentrieve/llm/security_policy.py:54`
- `phentrieve/llm/prompts/templates/two_phase/en.yaml:9`

Evidence:

- Public LLM target policy rejects client provider/model/base URL overrides.
- Two-phase prompts mark document text as untrusted.
- Other extraction-capable prompt families do not all carry equivalent
  boundaries.

Why it matters:

CLI or internal paths can later become public. Inconsistent safety text invites
drift.

Recommended fix:

Make untrusted-document delimiters a shared prompt primitive and require it in
prompt tests for all extraction-capable templates.

Suggested tests:

- Prompt rendering tests across direct, tool-guided, agentic, mapping, and
  postprocess templates.

### 9. P2: `QueryInterface.vue` remains a high-coupling frontend component

Affected files:

- `frontend/src/components/QueryInterface.vue:446`
- `frontend/src/components/QueryInterface.vue:542`
- `frontend/src/composables/useQueryInterfaceController.js:247`

Evidence:

- `QueryInterface.vue` is about 1244 LOC.
- It includes Options API state, a manual composable bridge, PII actions, query
  submission, model state, note rendering, export, and collection logic.

Why it matters:

UI state ownership is hard to reason about. Future privacy and safety changes
will be brittle if they must pass through one large component.

Recommended fix:

Split into query form, note workspace, result shell, PII workflow, and
collection actions. Move durable state to stores and transient UI state to
focused composables.

Suggested tests:

- Component tests for submit, PII review, model switching, full-text mode, and
  collection actions.

### 10. P2: Vue prop validator has logging side effects

Affected files:

- `frontend/src/components/ResultsDisplay.vue:156`

Evidence:

- The `results` prop validator logs details about received results.

Why it matters:

Vue prop validators should be pure. Dev/prod behavior and invocation frequency
can differ.

Recommended fix:

Remove validator logging. Use a watcher or explicit debug helper if needed.

Suggested tests:

- Mount test ensuring prop validation does not write to `console`.

### 11. P2: configuration is still split and partly import-time bound

Affected files:

- `phentrieve/config.py:354`
- `api/config.py:149`

Evidence:

- Core config lazily loads YAML and exports many module constants.
- API config separately binds env/YAML values.

Why it matters:

Clinical, API, and model policy settings should be inspectable, overrideable,
and testable from one typed source.

Recommended fix:

Introduce typed settings objects, inject them into API/services, and keep
module constants as compatibility shims only where needed.

Suggested tests:

- Settings precedence tests for env, YAML, defaults, and API/core consistency.

### 12. P2: packaging/docs/Makefile drift creates unreliable developer instructions

Affected files:

- `Makefile:25`
- `pyproject.toml:16`
- `docs/development/dev-environment.md:186`
- `api/README.md:47`

Evidence:

- `Makefile` references `uv sync --extra text_processing`.
- `pyproject.toml` has no `text_processing` extra.
- Docs still mention `uv sync --extra text` and direct `pip install`
  commands.
- `AGENTS.md` says mypy targets Python 3.10, while `pyproject.toml` requires
  Python `>=3.11`.

Why it matters:

Setup drift lowers trust in CI parity and onboarding instructions.

Recommended fix:

Make `pyproject.toml` the source of truth. Update Makefile/docs and add a small
CI check for advertised extras.

Suggested tests:

- Smoke test documented install targets.
- Static check that Makefile extras exist in `pyproject.toml`.

### 13. P3: text-processing frontend still carries dead `trustRemoteCode` intent

Affected files:

- `frontend/src/composables/useQueryInterfaceController.js:247`
- `frontend/src/services/PhentrieveService.js:136`

Evidence:

- Controller adds `trustRemoteCode: true`.
- Service normalization does not forward it.
- Backend now owns model trust policy.

Why it matters:

Dead security-related flags confuse future maintainers.

Recommended fix:

Remove the frontend field and document that model trust is server-owned.

Suggested tests:

- Payload normalization test proving no client trust flag is sent.

## Fixed/Changed Since Old Reviews

- API application structure improved. `create_app()` and lifespan cleanup are
  present in `api/main.py`.
- `api/main.py` no longer has the old direct `sys.path` manipulation. Local
  runner isolation remains in `api/run_api_local.py`.
- Model/retriever caches are now bounded TTL caches with cleanup in
  `api/dependencies.py`.
- `/text/process` model allowlisting and server-owned trust policy are
  materially improved.
- CI mypy is now blocking.
- `AnnotatedDocumentPane.vue` is no longer the old oversized component.
- Still present: orchestration concentration, `QueryInterface.vue` size,
  config/docs drift, and core extraction maintainability risk.
- Newer PII and prompt-injection work is real and useful, but still needs
  architectural hardening around browser persistence and shared prompt policy.

## Strengths To Preserve

- Server-owned public LLM provider/model selection.
- Local browser PII detector/redactor with multi-locale rules.
- Bounded model caches and lifespan cleanup.
- CI coverage across lint, format, typecheck, pytest collection, Python
  versions, and frontend jobs.
- Domain concepts for ontology-aware evaluation and grounded extraction.
- API response/error consistency and explicit research-use acknowledgement.
- Logging patterns that often prefer lengths/counts over raw text.

## Roadmap To >8/10

### Immediate P1 work

- Apply the same model allowlist/trust policy to `/query` that `/text/process`
  now uses.
- Fix or remove `CombinedAssertionDetector.preference`.
- Change browser persistence to avoid durable raw clinical text by default.
- Wire or delete `_hpo_extraction_helpers.py` and decompose the HPO orchestrator
  behind tests.

### 1-2 week maintainability workstream

- Split `phentrieve/llm/pipeline.py` by extraction phase, mapping phase,
  retries, result assembly, and trace handling.
- Split `phentrieve/llm/provider.py` by provider plus shared schema/tool
  utilities.
- Move API request preparation and model construction out of
  `text_processing_router.py`.
- Break `QueryInterface.vue` into focused components/composables with clear
  Pinia ownership.
- Consolidate config into typed settings and update docs/Makefile to match
  `pyproject.toml`.

### Longer-term quality investments

- Raise coverage gates after adding characterization tests for LLM/full-text/HPO
  extraction.
- Productize ontology-aware scoring for semantic parent/child HPO equivalence.
- Add ASVS-style regression tests for logs, error surfaces, model policy, public
  LLM target locking, and browser PII storage.
- Add CI checks that documented install commands and optional extras are valid.

## Verification Notes

Commands run:

- `git status --short --branch`
- Required review reads for `AGENTS.md`, `.planning/README.md`, and the archived
  review/analysis files requested in the review prompt.
- Current-code inspection with `rg`, `wc -l`, `sed`, and `nl`.
- `make check` - passed.
- `make typecheck-fast` - passed after `dmypy` restarted fresh:
  `Success: no issues found in 130 source files`.

Commands not run:

- `make test`
- `make frontend-test-ci`
- `make frontend-build-ci`

No product files were modified as part of the review. This Markdown artifact was
created after the review at the user's request.

## Remediation Status

The active remediation plan has been implemented through Task 12 and final
verification has been recorded in
[`2026-04-30-codebase-health-remediation-verification.md`](2026-04-30-codebase-health-remediation-verification.md).

Verified head: `b70d8d1` on `codebase-health-remediation`.

Final required checks passed:

- `make check`
- `make typecheck-fast`
- `make test`
- `make frontend-test-ci`
- `make frontend-build-ci`
