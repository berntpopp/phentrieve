# Multi-Provider LLM Quality Stabilization Design

- Date: 2026-04-19
- Status: Draft for review
- Scope: Python-only stabilization and benchmark improvements for the existing multi-provider LLM pipeline
- Out of scope: API or frontend changes, new provider families, broad prompt rewrites, automatic model routing in production, and hardcoded vendor performance assumptions

## Goal

Improve quality, reliability, and performance consistency across the existing
Gemini, Anthropic, OpenAI, and Ollama provider paths for the current two-phase
GeneReviews-style extraction workflow.

The design should target the highest-leverage causes of benchmark divergence:

- provider-specific structured-output recovery asymmetry
- phase 1 instability on long schema-constrained extraction requests
- incomplete benchmark observability around retries, fallback behavior, and
  failure classes

## Current benchmark signal

Recent 10-document GeneReviews benchmarks on this branch showed:

- `gemini-3.1-flash-lite-preview` was the strongest overall result at F1 `0.828`
- `gemini-3.1-pro-preview` was close on quality but much slower
- Anthropic and OpenAI were competitive but consistently below the best Gemini
  results
- `ollama/qwen3:32b` was usable after timeout fixes but materially slower and
  weaker on recall
- `gemini-3-flash-preview` repeatedly failed phase 1 with
  `504 DEADLINE_EXCEEDED`

The strongest explanation is not prompt favoritism. The prompt template is
shared. The more important differences are:

1. model/provider capability differences on long structured extraction
2. adapter-level asymmetry in recovery behavior
3. benchmark observability that is not yet granular enough to isolate which
   failures were transport, schema, timeout, or semantic extraction problems

## Current implementation diagnosis

### Shared prompt and pipeline

The grounded phase 1 extraction prompt is shared across providers, and the
pipeline uses the same two-phase flow for all providers.

Relevant code paths:

- `phentrieve/llm/prompts/templates/two_phase/en.yaml`
- `phentrieve/llm/pipeline.py`

This means the current benchmark gap is not primarily caused by a Gemini-only
prompt template.

### Real adapter asymmetry

Gemini currently has a stronger structured-output recovery path than the other
providers.

In `GeminiStructuredOutputProvider.run_structured_prompt(...)`, the provider:

- validates structured output locally
- retries on retryable structured parsing/validation failures
- increases output tokens on retry

The other providers do not currently get equivalent post-parse recovery:

- Ollama retries transport-level failures, but not structured validation
  failures
- Anthropic retries transient transport/provider errors, but not structured
  validation failures
- OpenAI retries transient transport/provider errors, but not structured
  validation failures or empty-but-schema-shaped outputs

This is a real quality/reliability asymmetry and should be treated as a design
bug, not a benchmark curiosity.

### Provider-specific request differences already in place

The provider layer already uses native structured-output APIs:

- Gemini: `response_json_schema`
- Anthropic: `output_config.format`
- OpenAI: Responses API `text.format`
- Ollama: `/api/chat` with `format`

Ollama additionally grounds the schema in the user prompt, which aligns with
Ollama's official recommendation.

This design should preserve native structured-output paths rather than moving
providers behind an OpenAI-compatible shim.

## Design principles

- Keep the shared two-phase architecture.
- Preserve native structured-output APIs per provider.
- Remove reliability asymmetries before introducing larger abstractions.
- Favor small provider-aware behavior differences over provider-specific prompt
  forks.
- Improve benchmark signal quality so reliability gains are measurable.
- Stay quality-first and evidence-driven; do not add configuration layers that
  are not yet justified by benchmark signal.

## Scope

Included:

- provider-layer structured-output recovery parity
- provider-aware phase 1 stabilization for large structured extraction requests
- benchmark artifact improvements for retries, fallback behavior, and failure
  classes
- Python-only tests for the new stabilization behavior
- optional benchmark comparison modes that directly support stabilization work

Excluded:

- API changes
- frontend changes
- new providers
- speculative model routing in production
- large prompt redesigns or provider-specific prompt templates
- automatic provider/model recommendations hardcoded into runtime behavior

## Functional requirements

1. Structured-output recovery behavior must become provider-parity rather than
   Gemini-only.
2. The pipeline must support a bounded fallback strategy for phase 1 when a
   single long structured extraction request is unstable or too slow.
3. The fallback strategy must preserve the current shared prompt semantics and
   grounded chunk behavior.
4. Benchmark artifacts must distinguish:
   - transport/provider retries
   - structured-output validation retries
   - phase 1 fallback activation
   - failure class
5. The benchmark path must make it easier to compare stabilization modes
   without requiring API or frontend changes.
6. Existing provider selection and backward compatibility for bare Gemini model
   names must remain unchanged.

## Non-functional requirements

1. Keep edits scoped to Python provider, pipeline, benchmark, and tests.
2. Preserve existing successful code paths where no stabilization is needed.
3. Keep new defaults conservative and benchmark-justified.
4. Avoid introducing a large provider strategy framework in this iteration.

## Proposed design

### 1. Shared structured-response recovery layer

Add a shared structured-response recovery behavior that providers can opt into
through the same interface.

The recovery contract should cover failures that happen after a successful API
response but before a valid `BaseModel` can be returned, including:

- malformed or truncated JSON
- empty structured payload
- schema-valid envelope with unusable content
- provider refusal represented as structured-output absence rather than a
  transport error

The design should standardize:

- retry count
- retry eligibility rules
- output-token expansion on retry where the provider supports it
- retry accounting in metadata and benchmark artifacts

Gemini's current implementation is the starting point, but it should be
generalized instead of copied ad hoc into each adapter.

### 2. Provider-aware phase 1 stabilization

Phase 1 is the dominant instability point because it is a long,
schema-constrained extraction pass over grounded clinical chunks.

The pipeline should support a bounded fallback path:

1. try the current grouped or non-grouped phase 1 request shape
2. if it fails with a retryable structured or timeout failure, switch to a
   smaller grouped extraction mode
3. continue collecting partial successes when some groups fail
4. surface the fallback clearly in observability and benchmark artifacts

This is intentionally narrower than full provider-specific routing. The goal is
to reduce failure blast radius for fragile models without changing the meaning
of the task.

### 3. Minimal semantic stabilization, not prompt forks

Do not create separate provider-specific extraction prompts in this iteration.

Instead:

- preserve the shared prompt template
- keep provider-native schema enforcement
- allow a small shared semantic reinforcement layer for structured extraction
  where needed, such as emphasizing source-faithful wording and chunk grounding

This should be minimal and compatible across providers.

Ollama should retain the current schema-in-prompt grounding because the vendor
docs explicitly recommend it.

Anthropic and OpenAI should not automatically receive full schema-text prompt
duplication in this iteration unless benchmark evidence shows it improves
quality without harming latency or schema behavior.

### 4. Provider-specific tuning knobs with benchmark justification

Allow small provider-aware tuning where it directly supports stabilization:

- timeout seconds
- structured retry count
- structured retry token multiplier
- phase 1 fallback enable/disable
- fallback group sizing where relevant

These settings should remain internal or benchmark-oriented unless there is a
clear user-facing need.

This is not a general provider strategy framework. It is a bounded
stabilization layer.

### 5. Benchmark observability upgrades

Extend benchmark artifacts so they explain why a model performed the way it
did.

Add or persist fields such as:

- `transport_retry_count`
- `structured_retry_count`
- `phase1_fallback_triggered`
- `phase1_initial_mode`
- `phase1_final_mode`
- `failure_class`
- `failure_reason_detail`
- exact-vs-estimated token count source already present in metadata

Failure classes should be normalized enough to compare runs, for example:

- `provider_timeout`
- `provider_transient_http_error`
- `structured_json_invalid`
- `structured_schema_validation_failed`
- `structured_refusal`
- `all_phase1_groups_failed`

### 6. Benchmark comparison support

Improve the benchmark path so stabilization work can be compared directly.

The benchmark should make it easy to run:

- current/default mode
- stabilization mode enabled
- per-provider overrides where needed

This can be done through targeted benchmark flags or config additions rather
than a broad runtime configuration system.

## Architecture impact

### Provider layer

Primary target:

- `phentrieve/llm/provider.py`

Expected changes:

- factor shared structured-response retry/recovery helpers
- keep native provider request shaping intact
- add provider-specific retry eligibility hooks where needed
- preserve existing transport retry behavior

### Pipeline layer

Primary target:

- `phentrieve/llm/pipeline.py`

Expected changes:

- add bounded phase 1 fallback behavior
- preserve grouped extraction semantics and partial-failure handling
- record richer retry/fallback observability into `LLMMeta.trace`,
  `phase_counts`, and `phase_request_counts`

### Benchmark layer

Primary targets:

- `phentrieve/benchmark/llm_benchmark.py`
- `phentrieve/benchmark/llm_cli.py`

Expected changes:

- expose stabilization toggles needed for benchmark comparison
- persist richer observability in benchmark summary and per-document artifacts
- support clearer failure-class reporting

## Testing strategy

### Unit tests

Add or extend unit tests for:

- provider structured retry parity across Gemini, Anthropic, OpenAI, and Ollama
- retryable vs non-retryable structured failures
- output-token expansion on structured retry where supported
- phase 1 fallback activation after timeout or structured failure
- partial-success grouped extraction under fallback mode
- benchmark artifact persistence of retry counts, fallback flags, and failure
  class

### Integration tests

Add focused integration coverage for:

- grouped fallback preserving extracted phenotype semantics
- phase 1 partial-failure behavior remaining stable
- benchmark records clearly distinguishing phase 1 instability from later-phase
  mapping behavior

### Live validation

After implementation, rerun representative GeneReviews benchmarks for at least:

- current Gemini baseline
- one Anthropic model
- one OpenAI model
- one Ollama local model

The goal is not to guarantee that all providers tie Gemini. The goal is to
verify:

- fewer phase 1 hard failures
- fewer empty or malformed structured outputs
- equal or improved F1 for non-Gemini providers where stabilization applies
- no regression to the strongest Gemini baseline

## Risks

1. Recovery retries may improve completion rates but also increase latency and
   cost.
2. Grouped phase 1 fallback may improve reliability while reducing some global
   extraction context, which could affect recall.
3. Provider-specific stabilization can drift into hidden prompt specialization
   if not kept minimal.
4. Benchmark complexity can grow quickly; observability additions should stay
   directly tied to stabilization questions.

## Recommendation

Implement this as a quality-first stabilization pass, not a full provider
strategy framework.

The first iteration should prioritize:

1. structured retry parity
2. bounded phase 1 fallback
3. richer benchmark observability

Only after those changes are benchmarked should the project consider a larger
provider strategy abstraction.

## Source notes

This design is based on current official provider documentation and observed
benchmark behavior on this branch.

Primary sources:

- Gemini structured outputs:
  https://ai.google.dev/gemini-api/docs/structured-output
- Gemini model catalog:
  https://ai.google.dev/gemini-api/docs/models
- Anthropic structured outputs:
  https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Anthropic model selection guidance:
  https://platform.claude.com/docs/en/about-claude/models/choosing-a-model
- Anthropic consistency guidance:
  https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency
- OpenAI model catalog:
  https://developers.openai.com/api/docs/models
- OpenAI GPT-5.4 model page:
  https://developers.openai.com/api/docs/models/gpt-5.4
- OpenAI GPT-5.4 mini model page:
  https://developers.openai.com/api/docs/models/gpt-5.4-mini
- OpenAI structured outputs guide:
  https://developers.openai.com/api/docs/guides/structured-outputs
- Ollama structured outputs:
  https://docs.ollama.com/capabilities/structured-outputs
