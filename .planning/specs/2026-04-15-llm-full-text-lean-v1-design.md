# Lean V1 Design: LLM Full-Text Processing

Date: 2026-04-15

Supersedes the older auth- and account-oriented implementation plan now archived
at `.planning/archived/2026-04-15-llm-auth-quota-benchmark-implementation-plan.md`.

## Goal

Add LLM-based full-text phenotype extraction to `phentrieve` in a lean way across:

- CLI
- FastAPI API
- Vue frontend
- benchmark workflows

The first version must:

- avoid registration, login, sessions, and billing
- keep the existing product shape intact
- clearly separate development from production behavior
- enforce a strict anonymous production limit of three LLM full-text analyses per subject per UTC day

## Scope

This design covers:

- selective porting of the reusable LLM framework from `../phentrieve-bench`
- integration into the existing full-text processing path
- lean production safeguards for anonymous public use
- scriptable LLM benchmarking inside the existing benchmark family

This design does not cover:

- user accounts
- email verification
- password reset
- registered-user quotas
- payment or billing
- multi-node distributed quota infrastructure
- broad agentic ensembles as the default user-facing mode

## Current Codebase Findings

### Existing product surface

`phentrieve` already has a mature full-text path:

- CLI full-text entrypoint: `phentrieve text process` in `phentrieve/cli/text_commands.py`
- API full-text entrypoint: `POST /api/v1/text/process` in `api/routers/text_processing_router.py`
- frontend full-text UX: the `textProcess` branch in `frontend/src/components/QueryInterface.vue` and `frontend/src/components/ResultsDisplay.vue`

This means the product already thinks in two modes:

- query
- full-text

The lean LLM feature should extend full-text mode instead of introducing a parallel product surface.

### Existing technical substrate

The current shared full-text pipeline is already split well:

- `TextProcessingPipeline` handles chunking and assertion detection
- `orchestrate_hpo_extraction` handles retrieval, ranking, and aggregation

The benchmark stack already reuses the same extraction path via `phentrieve/benchmark/extraction_benchmark.py`.

The API already uses:

- lifespan startup/shutdown hooks
- cached dependency loading
- `run_in_threadpool` for blocking ML operations

### Findings from `../phentrieve-bench`

The sibling repo contains two relevant branches:

- `feature/llm-annotation-system`
- `feature/agentic-judge-mode`

The reusable foundation is in `feature/llm-annotation-system`:

- `phentrieve/llm/types.py`
- `phentrieve/llm/provider.py`
- `phentrieve/llm/pipeline.py`
- prompt templates
- `phentrieve/benchmark/llm_benchmark.py`

The `feature/agentic-judge-mode` branch contains useful experimental follow-up work, but it is not a good direct product baseline because it includes:

- large benchmark artifacts
- branch-specific analysis files
- heavier configuration complexity
- more dataset-specific tuning

The initial port should therefore be selective and subsystem-based, not a branch cherry-pick.

## Product Shape

### User-facing modes

The product should keep exactly two user-facing modes:

- `query`
- `full-text`

Inside `full-text`, add an extraction backend selector:

- `standard`
- `llm`

`standard` maps to the current retrieval-based pipeline.

`llm` maps to the new LLM-assisted extraction pipeline.

There should be no third top-level frontend mode.

### CLI shape

The CLI should extend the existing full-text command rather than forcing the main user path through a new top-level LLM command.

Primary user path:

- `phentrieve text process --extraction-backend standard|llm`

Developer and evaluation path:

- `phentrieve benchmark llm ...`

### API shape

The API should keep one main full-text product endpoint:

- extend `POST /api/v1/text/process`

The request schema should gain:

- `extraction_backend`
- optional nested or flat LLM settings for model and mode

The route should stay a thin adapter around a shared full-text service.

### Frontend shape

The frontend should preserve the existing single full-text UX:

- keep full-text inside `QueryInterface`
- add an LLM toggle or selector in advanced options that only appears when full-text mode is active
- submit through the existing service layer to the same full-text API endpoint

The result should still render in the current full-text results surface.

## Architecture

### New modules

Add a shared LLM subsystem under `phentrieve/llm/` with a lean initial scope:

- provider abstraction
- typed request and response models
- prompt loading
- structured-output parsing
- one primary annotation mode for v1

Add a shared full-text orchestration layer under `phentrieve/text_processing/`, for example:

- `phentrieve/text_processing/full_text_service.py`

This layer will be the single backend-facing integration point for:

- CLI
- API
- future benchmark adapters

### Initial LLM mode

The recommended initial LLM mode is:

- `two_phase`

Reasoning:

- it is the strongest balanced foundation from the sibling repo
- it is more product-ready than the experimental agentic judge flow
- it avoids shipping a broad set of modes before the first integration stabilizes

`agentic_judge` should be treated as a later follow-up, not part of lean v1.

### Shared service contract

The shared full-text service should:

1. accept normalized full-text configuration
2. decide between `standard` and `llm`
3. execute the selected backend
4. map both backends into one stable response contract

The CLI and API should remain thin and should not embed LLM orchestration logic directly.

## Data Flow

### Standard backend

1. receive text and config
2. build chunking/assertion pipeline
3. process text into chunks
4. run retrieval aggregation
5. return the existing full-text response shape

### LLM backend

1. receive text and config
2. normalize language and backend settings
3. run the LLM annotation pipeline
4. validate and normalize HPO IDs
5. apply minimal post-processing where useful
6. map the result into the same general response shape as standard mode

The LLM backend may still call parts of the existing retrieval system for grounding and validation, but that should remain an internal implementation detail of the shared service.

## API And Response Contract

The top-level full-text response contract should remain stable:

- `meta`
- `processed_chunks`
- `aggregated_hpo_terms`

For `standard`, behavior should remain as close as possible to current output.

For `llm`, the same top-level shape should be preserved, with `meta` extended to include:

- `extraction_backend`
- `llm_model`
- `llm_mode`
- token usage summary when available
- timing summary when available
- cost estimate when available
- `quota_limit`
- `quota_used`
- `quota_remaining`
- `quota_reset_at`

The frontend should not be forced to parse a separate incompatible response family.

In LLM mode, `processed_chunks` must not contain invented or misleading chunk
boundaries. The lean v1 should choose one of two valid behaviors only:

- return faithful chunks with correct source text and span offsets, or
- return an empty chunk list and require the frontend to degrade gracefully

The implementation plan should prefer faithful chunks if they can be produced
cleanly from the shared service boundary. If not, empty chunks are preferable to
synthetic evidence views.

## Environment Separation And Production Controls

### Environment modes

The API should distinguish environments explicitly using configuration:

- `development`
- `production`

Recommended environment variable:

- `PHENTRIEVE_ENV=development|production`

Development behavior:

- no anonymous daily quota for LLM full-text processing

Production behavior:

- strict anonymous daily quota for LLM full-text processing
- reverse-proxy request-rate protection

The production quota applies only to the `llm` backend. Standard retrieval-based
full-text processing remains uncapped by this daily application quota.

### Anonymous quota rule

Initial production rule:

- maximum `3` successful LLM full-text analyses per subject per UTC day

The quota applies only when:

- the request uses `extraction_backend = llm`
- the request comes through the public API path
- the environment is `production`

The quota does not apply to:

- CLI local execution
- benchmark commands
- retrieval-only full-text requests

### Quota storage

For the current stack, the leanest durable solution is a small SQLite-backed quota store.

Reasoning:

- the repo already uses SQLite successfully
- the deployment shape is currently single-node Docker Compose
- there is no existing Redis, Postgres, or worker infrastructure
- a scheduler is not required if the quota key includes the UTC date

Recommended subject key:

- trusted client IP or forwarded client IP hash

The API must not trust arbitrary forwarded headers directly. The reverse-proxy
chain must be wired explicitly:

- Nginx forwards `X-Real-IP` and `X-Forwarded-For`
- FastAPI application code trusts forwarded client IP only when the immediate
  sender is a configured trusted proxy
- trusted proxy CIDRs are configured explicitly, for example via
  `PHENTRIEVE_TRUSTED_PROXY_CIDRS`

If a trusted forwarded client IP cannot be resolved safely, the request should
be rejected for `llm` production usage with a `503` or `400` style operational
error rather than falling back to a shared proxy bucket. Failing closed is
preferable to silently collapsing all users into one quota subject.

Recommended logical key:

- `(subject_key, usage_date_utc, feature_name)`

### Reverse-proxy protection

Nginx should still be used for coarse request-rate limiting and abuse protection.

That protection is complementary to, not a replacement for, the application-layer daily quota.

Nginx-side work should therefore include:

- request-rate limiting for `/api/v1/text/process` and any LLM-specific path
- forwarding the real client IP headers needed by the application quota layer
- explicit trusted-proxy configuration so the application does not bucket by the
  proxy container IP

## Frontend Behavior

The frontend should be informative, not authoritative, about limits.

Required behavior:

- expose LLM extraction only within full-text mode
- label clearly when LLM extraction is enabled
- show a concise production notice for limited public LLM usage
- render quota exhaustion errors from structured `429` responses

The frontend should be able to read quota state from successful `2xx` full-text
LLM responses using:

- `quota_limit`
- `quota_used`
- `quota_remaining`
- `quota_reset_at`

The frontend should also be able to read quota failure details from a structured
`429` response that includes at minimum:

- `error`
- `detail`
- `quota_limit`
- `quota_used`
- `quota_remaining`
- `quota_reset_at`
- `retry_after_seconds`

Optional but reasonable:

- show an environment banner such as demo or limited production instance when configured

The frontend should not attempt to maintain quota state independently beyond displaying API-provided status.

## Benchmark Integration

LLM benchmarking should be integrated into the existing benchmark family, not bolted on as a completely separate experimental framework.

Recommended CLI shape:

- `phentrieve benchmark llm ...`

The benchmark path should:

- instantiate the LLM pipeline directly
- bypass public API quota enforcement
- reuse existing extraction-style evaluation metrics where possible
- write structured output compatible with current result storage patterns

The initial benchmark scope should cover:

- a tiny smoke dataset for fast validation
- existing PhenoBERT-derived full-text datasets where appropriate
- comparison between `standard` and `llm`

Do not port giant result artifacts or branch-era benchmark payloads into the main repo.

## Testing Strategy

The implementation should include targeted tests, not a large unrelated refactor.

Required test areas:

- unit tests for the new LLM service boundary
- unit tests for request and response schema extensions
- unit tests for production quota logic
- unit tests for frontend service and advanced-options behavior
- integration tests for full-text standard vs LLM execution paths
- integration or smoke tests for `phentrieve benchmark llm`

The CLI, API, and frontend should all be verified against the existing project commands before completion.

## Porting Strategy

Port by subsystem, not by branch.

### Port now

From `feature/llm-annotation-system`:

- core `phentrieve/llm/` framework
- prompt loading and templates
- typed result models
- provider abstraction
- benchmark harness ideas and tests

### Do not port now

From `feature/agentic-judge-mode`:

- large benchmark artifacts
- heavy experimental tuning
- multi-judge default behavior
- research reports as product docs

### Follow-up candidate

Later, after the lean integration is stable:

- selectively import `agentic_judge` as an advanced experimental backend

## Risks And Mitigations

### Risk: response-shape drift between standard and LLM backends

Mitigation:

- define one shared response adapter layer
- keep the same top-level contract
- test frontend rendering against both backends

### Risk: blocking LLM calls degrade API responsiveness

Mitigation:

- keep long-running sync work behind the current threadpool pattern
- avoid embedding blocking logic directly inside async handlers

### Risk: quota logic becomes too large for lean v1

Mitigation:

- keep quota to one feature only: anonymous LLM full-text API usage
- use a date-derived reset instead of scheduled cleanup
- store only minimal counters and timestamps

### Risk: over-porting branch experiments into the main repo

Mitigation:

- port only the foundation and one mode
- exclude benchmark artifacts and branch analysis output

## Recommended Implementation Order

1. Port the minimal reusable `phentrieve/llm/` foundation
2. Add shared full-text service with `standard|llm` backend selection
3. Extend CLI full-text command
4. Extend API full-text schema and route
5. Add production-only quota enforcement for LLM full-text API requests
6. Extend frontend advanced options and full-text submission flow
7. Add benchmark namespace and smoke dataset support
8. Verify with unit, integration, CLI, API, and frontend checks

## Success Criteria

Lean v1 is successful when:

- users can run LLM full-text extraction from the CLI through the existing full-text command family
- API clients can request LLM full-text extraction through the existing `/api/v1/text/process` route
- frontend users can enable LLM extraction inside the existing full-text workflow
- production deployments enforce a strict anonymous quota of three LLM full-text analyses per UTC day
- development remains friction-light and quota-free
- benchmarks can compare LLM full-text extraction through an existing-style benchmark namespace
- the implementation remains scoped and does not introduce registration or account infrastructure
