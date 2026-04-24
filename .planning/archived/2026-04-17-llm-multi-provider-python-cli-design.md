# LLM Multi-Provider Python CLI Design

- Date: 2026-04-17
- Status: Draft for review
- Scope: Python-only LLM provider support for the CLI, shared service/pipeline, and benchmark path
- Out of scope: API request/response contract changes, frontend changes, embedding provider swaps, agent loops, and non-structured-output tool orchestration

## 1. Goal

Extend the current Gemini-only Python LLM path so Phentrieve can run the
existing structured extraction workflow against multiple providers:

- Ollama local models with explicit model choice and optional base URL
- OpenAI via API
- Anthropic via API
- Gemini via API

Phase one implementation and testing will start with Ollama, using
`qwen3.5:35b` as the primary local validation model and `qwen3.5:27b` as the
fallback local validation model, assuming current Ollama library availability
at implementation time. If those tags move before implementation, the plan
should switch to the nearest official Qwen 3.5 Ollama tags rather than
hardcoding stale model names.

## 2. Current State

The current provider surface is narrower than the CLI suggests:

- The CLI and benchmark paths accept `--llm-model`, but not a first-class
  provider field.
- The provider factory in `phentrieve/llm/provider.py` reads
  `PHENTRIEVE_LLM_PROVIDER` from the environment, but explicitly rejects
  anything other than `gemini`. There is no CLI-level provider flag today.
- `LLMPipelineConfig` carries model, mode, language, and seed, but not
  provider identity or base URL.
- The shared pipeline depends mainly on the `LLMProvider` contract and is
  otherwise close to provider-agnostic.

This makes the provider factory, config layer, and CLI normalization path the
correct refactor boundary.

## 3. Design Principles

- Provider is first-class internally, even when inferred from user input.
- Model and provider are normalized once at the boundary, then propagated in
  typed form.
- Provider adapters preserve the existing pipeline contract instead of forcing
  a pipeline rewrite.
- Native structured output is preferred per provider, followed by local
  Pydantic validation of the returned payload.
- Ollama uses its native `/api/chat` structured-output path, not the OpenAI
  compatibility path, for stricter schema behavior.
- Logs, benchmark results, and internal metadata must distinguish provider from
  model.
- Validation should be strict about incompatible provider/model combinations,
  but not brittle about local Ollama model names.

## 4. Scope Boundaries

Included in this design:

- CLI provider/model/base URL configuration
- Shared normalization and validation logic
- Multi-provider factory dispatch
- Provider adapters for Gemini, OpenAI, Anthropic, and Ollama
- Benchmark path propagation of provider metadata
- Unit and targeted integration coverage for the Python path

Excluded from this design:

- FastAPI schema or router changes
- Frontend controls or settings
- Server-owned model allowlists for API clients
- Embedding backend changes
- Multi-turn agent abstractions

## 5. User-Facing Behavior

### 5.1 CLI inputs

Add these Python CLI inputs where `--llm-model` is already supported:

- `--llm-provider`
- `--llm-model`
- `--llm-base-url`

Retain provider-relevant existing options like `--llm-seed`, but only apply
them when supported by the selected provider.

### 5.2 Normalization rules

User input is normalized into a typed provider request with:

- `provider`
- `model`
- `base_url`

Normalization rules:

1. If `--llm-model` uses a known prefix such as `gemini/...`, `openai/...`,
   `anthropic/...`, or `ollama/...`, infer the provider from that prefix.
2. If `--llm-provider` is also supplied, it must match the inferred provider.
3. If the model has no prefix, use `--llm-provider` when present.
4. If neither field is explicit, fall back to environment/config defaults.
5. After parsing, the stored runtime state becomes provider-native:
   `provider="ollama"`, `model="qwen3.5:35b"`, rather than a combined string.

### 5.3 Error behavior

The CLI should fail fast with clear errors for:

- unknown provider prefixes
- unsupported providers
- explicit provider/model mismatches
- missing required API keys for cloud providers
- malformed base URLs

For Ollama, model names should remain flexible. The CLI should not attempt to
hardcode a strict allowlist of local model identifiers.

## 6. Configuration

Add or formalize these environment variables:

- `PHENTRIEVE_LLM_PROVIDER`
- `PHENTRIEVE_LLM_MODEL`
- `PHENTRIEVE_LLM_BASE_URL`
- `PHENTRIEVE_OPENAI_API_KEY` with `OPENAI_API_KEY` fallback
- `PHENTRIEVE_ANTHROPIC_API_KEY` with `ANTHROPIC_API_KEY` fallback
- existing Gemini key fallbacks remain supported

Recommended defaults:

- default Ollama base URL: `http://localhost:11434`
- default provider remains repo-configurable in `phentrieve/llm/config.py`
- default model remains repo-configurable in `phentrieve/llm/config.py`

Base URL semantics:

- Ollama gets a concrete default base URL.
- Cloud providers may also accept an explicit base URL for enterprise proxies,
  gateways, or compatible deployments, but they should not have a hardcoded
  non-standard default.

Per-provider timeout defaults should be represented explicitly instead of
reusing Gemini-tuned values across every provider. In particular, Ollama
should get a longer default timeout to tolerate cold starts and large local
model runs, with a starting design target of 300 seconds unless local testing
supports a tighter value.

Provider-specific defaults should be represented separately from global
defaults so the code can evolve without encoding Gemini assumptions as the
general case.

Dependency strategy:

- Ollama should use plain HTTP against its native API rather than adding a
  hard dependency on an Ollama SDK.
- The Python `llm` extra should be expanded to include whatever official SDKs
  or minimal client dependencies are required for OpenAI and Anthropic support.
- Dependency layout should remain simple unless optional extras become large
  enough to justify provider-specific splits later.

## 7. Internal Architecture

### 7.1 Provider resolution layer

Introduce a small typed normalization object, conceptually:

- `provider`
- `model`
- `base_url`
- `api_key`
- `seed`

This object is constructed once in the CLI/service boundary and passed into the
provider factory.

`LLMPipelineConfig` should be extended explicitly to carry at least
`provider`, and optionally `base_url` if downstream tracing or behavior needs
that field. `LLMMeta` should also gain `llm_provider` so benchmark and service
metadata preserve provider identity beyond the lifetime of the live provider
instance.

### 7.2 Factory dispatch

Replace Gemini-only branching in `phentrieve/llm/provider.py` with a dispatch
factory that selects a provider adapter based on normalized provider identity.

Planned adapters:

- `GeminiStructuredOutputProvider`
- `OpenAIStructuredOutputProvider`
- `AnthropicStructuredOutputProvider`
- `OllamaStructuredOutputProvider`

The factory must stop inferring provider behavior from raw model strings after
normalization has completed.

### 7.3 Stable pipeline contract

The current `LLMProvider` abstraction should remain the pipeline contract:

- `complete(messages)`
- `run_structured_prompt(...)`
- `count_tokens(...)`

The pipeline should continue to expect:

- a validated Pydantic object from `run_structured_prompt()`
- provider-populated `last_usage`
- provider-populated `last_finish_reason`
- provider-populated `last_request_count`

This keeps `phentrieve/llm/pipeline.py` mostly unchanged and confines most work
to provider, config, CLI, and service plumbing.

## 8. Provider-Specific Behavior

### 8.1 Ollama

Ollama is the phase one implementation and testing target.

Requirements:

- use native `POST /api/chat`
- use `format=<json schema>` for structured outputs
- set low temperature, defaulting to `0` for extraction tasks
- validate returned JSON locally with Pydantic
- support explicit `base_url`, defaulting to `http://localhost:11434`
- use an Ollama-specific retry and timeout policy rather than inheriting
  Gemini's structured retry behavior mechanically

The implementation should not rely on Ollama's OpenAI compatibility endpoint
for strict schema handling in the first phase.

### 8.2 OpenAI

Use native structured output support on the current official API path selected
at implementation time. The adapter must handle structured-output refusal or
non-schema terminal states explicitly and keep local validation in place.
OpenAI-specific retry and timeout behavior should live in the OpenAI adapter.

### 8.3 Anthropic

Use current Claude API structured outputs on the GA path, using
`output_config.format` semantics where supported by the selected model family.
Preserve local validation and provider-level error mapping. If a chosen Claude
model lacks GA structured outputs, the adapter must fail clearly or use a
documented fallback path rather than silently pretending native support exists.

### 8.4 Gemini

Retain Gemini support, but move it behind the same provider registry so Gemini
is no longer the hardcoded default behavior for the entire provider layer.
Gemini-specific retry and timeout behavior should remain adapter-owned rather
than treated as the global policy template.

## 9. Metadata And Observability

Record provider-specific metadata in the Python path:

- provider
- model
- base URL when relevant
- token count source: `exact` or `estimated`

Benchmark metadata and CLI-visible metadata should stop treating `llm_model`
alone as sufficient identity for cross-provider comparisons.
`LLMResponse.provider` is not sufficient on its own; durable pipeline and
benchmark metadata should include provider identity through `LLMMeta`.

## 10. Token Counting

Token counting behavior should be explicit:

- use exact provider-native token counting when available
- if unavailable, use a heuristic estimate
- record whether the count is exact or estimated

Token-counting uncertainty must not silently mix with exact values in
benchmarking or observability.

Existing bare model-name behavior must remain backward compatible:

- a bare model name such as `gemini-3.1-flash-lite-preview` should continue to
  resolve through the configured default provider
- prefixed model names become the new cross-provider path, not a breaking
  replacement for existing Gemini workflows

## 11. Testing Strategy

### 11.1 Unit coverage

Add or update tests for:

- model-prefix parsing
- provider/model mismatch errors
- env and CLI fallback order
- Ollama base URL normalization
- provider dispatch behavior
- provider-specific structured-output request shaping
- provider-specific retry behavior
- metadata propagation including provider identity
- exact vs estimated token count metadata
- redaction of API keys and other secrets in logged errors

### 11.2 Pipeline coverage

Keep the current provider-agnostic pipeline tests, but add at least one
non-Gemini fake provider smoke test to prove the pipeline still depends only on
the abstract contract.

### 11.3 Ollama-focused integration coverage

Add targeted Ollama tests around:

- native schema request payload shape
- successful Pydantic validation
- invalid or truncated JSON handling
- default local base URL behavior

Phase one validation should concentrate on Ollama before broadening to the
other providers.

## 12. Rollout Plan

### Phase 1

- Implement provider normalization
- Implement Ollama adapter
- Thread provider/model/base URL through CLI, service, and benchmark path
- Add Ollama-focused tests
- Validate locally with `qwen3.5:35b`

### Phase 2

- Add OpenAI adapter
- Add Anthropic adapter
- Re-home Gemini behind the shared dispatch layer
- Expand tests for cloud providers

This staged rollout reduces surface area while enabling immediate local
development and validation.

## 13. Local Model Recommendation

Primary local validation model:

- `qwen3.5:35b`

Fallback local validation model:

- `qwen3.5:27b`

Comparison model after adapter stability:

- `gpt-oss:20b`

Reasoning:

- the development machine has an NVIDIA RTX 5090 with 32 GB VRAM
- `qwen3.5` is a current official Ollama family with a practical middle tier
  suitable for local validation
- `qwen3.5:35b` is the best first attempt for capability-per-local-run on a
  quantized local deployment
- `qwen3.5:27b` gives a lower-risk fallback if memory pressure, latency, or
  schema reliability is better there
- `gpt-oss:20b` is a useful later comparison because it is also positioned for
  structured-output local use

## 14. Risks

- token preflight logic may drift if providers differ in token accounting
- provider retries and timeouts need provider-specific tuning rather than one
  global policy
- structured-output schema support differs by provider and may require
  provider-specific shaping
- benchmark result identity may become ambiguous if metadata is not updated
  consistently
- Ollama can still return schema-shaped but semantically bad data, so local
  validation and extraction-focused tests remain necessary
- normalization objects that carry secrets increase the chance of accidental
  key leakage in logs unless errors are redacted consistently

## 15. Implementation Entry Points

Expected primary files:

- `phentrieve/llm/provider.py`
- `phentrieve/llm/config.py`
- `phentrieve/llm/types.py`
- `phentrieve/text_processing/full_text_service.py`
- `phentrieve/cli/text_commands.py`
- `phentrieve/benchmark/llm_cli.py`
- `phentrieve/benchmark/llm_benchmark.py`
- `tests/unit/llm/test_provider.py`
- `tests/unit/llm/test_pipeline.py`
- `tests/unit/text_processing/test_full_text_service.py`
- `tests/unit/cli/test_text_commands.py`
- `tests/unit/cli/test_benchmark_commands.py`

## 16. Sources

- Ollama structured outputs:
  https://docs.ollama.com/capabilities/structured-outputs
- Ollama Qwen 3.5 library:
  https://ollama.com/library/qwen3.5
- Ollama Qwen 2.5 library:
  https://ollama.com/library/qwen2.5
- Ollama gpt-oss library:
  https://ollama.com/library/gpt-oss
- NVIDIA RTX 5090 official specs:
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- Anthropic release notes for GA structured outputs:
  https://platform.claude.com/docs/en/release-notes/overview
- Anthropic structured outputs docs:
  https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- OpenAI model docs showing structured-output support:
  https://developers.openai.com/api/docs/models
  https://developers.openai.com/api/docs/models/gpt-4o
  https://developers.openai.com/api/docs/models/gpt-4o-mini
  https://developers.openai.com/api/docs/models/gpt-oss-20b
