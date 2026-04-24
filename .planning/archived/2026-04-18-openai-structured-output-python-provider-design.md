# OpenAI Structured Output Python Provider Design

- Date: 2026-04-18
- Status: Draft for review
- Scope: Python-only OpenAI provider support for the CLI, shared service/pipeline, and benchmark path
- Out of scope: API request/response contract changes, frontend changes, prompt redesign outside provider request shaping, and support for non-structured-output OpenAI models

## Goal

Add native OpenAI support to the existing Python LLM provider abstraction so the CLI, shared LLM pipeline, benchmark path, and tests can run against OpenAI models that officially support structured outputs.

## Scope

In scope:
- Python-only changes under `phentrieve/` and `tests/`
- native OpenAI provider integration
- provider normalization and CLI/benchmark plumbing
- configurable token-cost accounting reuse
- live benchmark compatibility with supported OpenAI models

Out of scope:
- API changes
- frontend changes
- prompt redesign beyond provider-specific request shaping
- support for OpenAI models that do not support structured outputs

## Requirements

### Functional

1. Add `openai` as a first-class provider in the same abstraction layer as Gemini, Ollama, and Anthropic.
2. Support native OpenAI structured outputs for models that officially support them.
3. Reject unsupported OpenAI models early with a clear error message.
4. Support provider selection through both:
   - `--llm-provider openai`
   - prefixed model names like `openai/gpt-5.4`
5. Preserve existing behavior for Gemini, Ollama, and Anthropic.
6. Preserve backward compatibility for bare Gemini model names.
7. Thread OpenAI provider metadata through the existing `LLMMeta`, benchmark result payloads, and CLI outputs.
8. Support OpenAI API key discovery via:
   - `PHENTRIEVE_OPENAI_API_KEY`
   - `OPENAI_API_KEY`
   - `CHATGPT_API_KEY`
9. Reuse the existing configurable benchmark cost accounting without hardcoded vendor pricing in code.

### Non-functional

1. Keep the implementation aligned with the current provider architecture.
2. Use the native OpenAI API and official SDK, not an OpenAI-compatible shim.
3. Keep schema handling conservative and validate locally with Pydantic after model output is returned.
4. Keep edits minimal and scoped to the Python provider/CLI/benchmark/test surface.

## Provider Design

### Adapter strategy

Use a new `OpenAIStructuredOutputProvider` in `phentrieve/llm/provider.py`.

It should follow the same internal contract as the other providers:
- `complete(messages) -> LLMResponse`
- `run_structured_prompt(...) -> BaseModel`
- `count_tokens(...) -> dict[str, int]`

The OpenAI implementation should use:
- the official OpenAI Python SDK
- the native OpenAI Responses API for plain completions
- the native OpenAI Responses API `text.format` JSON Schema path for schema-constrained extraction

### Supported models

Initial supported OpenAI models:
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`

Unsupported for this provider path:
- `gpt-5.4-pro`
- any other OpenAI model that does not officially support structured outputs

If a user selects an unsupported OpenAI model for the structured extraction pipeline, fail fast with an error that explains the model is not supported because native structured outputs are required.

## Structured Output Contract

### Common schema strategy

Keep the current cross-provider pattern:
1. build a provider-safe JSON Schema from the Pydantic response model
2. send the schema through the provider's native structured-output request shape
3. parse returned JSON content
4. validate locally with `response_model.model_validate_json(...)`

The OpenAI path must reuse the same schema-normalization approach already used to keep Gemini/Ollama/Anthropic stable.
That said, OpenAI strict structured outputs have additional constraints that should be treated as provider-specific validation rules:
- the root schema must be an object and must not use top-level `anyOf`
- all fields must be listed in `required`
- `additionalProperties` should be `false` on structured objects
- only a subset of JSON Schema is supported in strict mode

The implementation should add an OpenAI-specific schema-sanitization pass if the current shared normalizer does not already satisfy those constraints.

### Native request behavior

The OpenAI provider should use the provider-native structured-output feature on the Responses API, not prompt-only JSON mode and not Chat Completions `response_format`.
The implementation target is:
- plain completions: OpenAI Responses API
- structured extraction: OpenAI Responses API with `text.format` JSON Schema, ideally through the SDK's `responses.parse(...)` helper when that fits the local validation flow cleanly

The adapter should keep model requests deterministic and conservative, matching the current extraction pipeline expectations.

If the model returns invalid JSON or schema-invalid output, raise the same kind of pipeline-visible structured extraction failure used for the other providers.
If the model returns a refusal instead of structured content, treat that as a distinct provider error path and surface it clearly rather than misclassifying it as invalid JSON.

## Configuration and Plumbing

### Provider resolution

Extend `resolve_llm_provider_request()` and `get_llm_provider()` to recognize `openai`.

Expected forms:
- `llm_provider="openai", llm_model="gpt-5.4"`
- `llm_model="openai/gpt-5.4"`

### Credentials

OpenAI key lookup order:
1. explicit `api_key`
2. `PHENTRIEVE_OPENAI_API_KEY`
3. `OPENAI_API_KEY`
4. `CHATGPT_API_KEY`

### Base URL

Allow optional `llm_base_url` to flow through to the OpenAI client for compatibility with enterprise or gateway setups, but do not rely on compatibility semantics for structured outputs.

### Timeout

Reuse the existing provider timeout plumbing so benchmark runs can tune OpenAI request timeouts through the same Python path.

## Benchmark and Cost Accounting

OpenAI benchmark runs should reuse the existing configurable accounting design:
- no hardcoded pricing in code
- costs driven by CLI flags or pricing config
- `estimated_cost` compatibility alias preserved
- `estimated_token_cost` populated for OpenAI runs

The benchmark output should include:
- `llm_provider: openai`
- `llm_model: <selected model>`
- token usage
- request counts
- estimated token cost when configured

## Testing

### Unit tests

Add focused tests for:
- provider normalization for `openai`
- environment-variable API key discovery
- supported-model acceptance
- unsupported-model rejection
- structured-output request shaping
- completion request shaping
- token counting behavior
- strict-schema request shaping for OpenAI structured outputs
- refusal handling as distinct from invalid JSON handling
- metadata propagation through `LLMMeta`, benchmark payloads, and `estimated_cost` / `estimated_token_cost`
- token usage source propagation so OpenAI runs report the same exact-vs-estimated markers used by the existing provider layer

### Verification

Required verification before completion:
- `make check`
- `make typecheck-fast`
- `make test`

If plain `make test` hits the known unrelated parallel CUDA OOM issue again, record that explicitly and also run:
- `CUDA_VISIBLE_DEVICES=\"\" make test`

### Live validation

After implementation, run real GeneReviews benchmarks for at least:
- `openai/gpt-5.4`
- `openai/gpt-5.4-mini`
- optionally `openai/gpt-5.4-nano` if time/cost permit

Use configured pricing inputs rather than code defaults so the reported costs remain user-configured.

## Risks

1. OpenAI's structured-output request shape may differ materially from Gemini/Anthropic/Ollama, so the adapter should stay isolated behind the provider seam.
2. Some OpenAI models support structured outputs while others do not; capability checks must be explicit.
3. The Responses API may expose token accounting differently from Gemini/Anthropic; usage extraction may require provider-specific parsing.
4. Long-context pricing differences for `gpt-5.4` are documented by OpenAI, but benchmark cost accounting in this branch should remain user-configured rather than automatically encoding provider pricing rules.

## Recommended initial benchmark set

For this repository's benchmark path, start with:
- `gpt-5.4` as the quality baseline
- `gpt-5.4-mini` as the likely cost/performance sweet spot
- `gpt-5.4-nano` as the cheap extraction baseline if additional comparison is useful

## Source Notes

This design is based on current official OpenAI docs as of 2026-04-18:
- latest model guidance recommends `gpt-5.4` as the default and `gpt-5.4-mini` / `gpt-5.4-nano` for smaller workloads:
  https://developers.openai.com/api/docs/guides/latest-model
- model catalog and current pricing for `gpt-5.4`, `gpt-5.4-mini`, and `gpt-5.4-nano`:
  https://developers.openai.com/api/docs/models
- `gpt-5.4` supports structured outputs:
  https://developers.openai.com/api/docs/models/gpt-5.4
- `gpt-5.4-mini` supports structured outputs:
  https://developers.openai.com/api/docs/models/gpt-5.4-mini
- `gpt-5.4-nano` supports structured outputs:
  https://developers.openai.com/api/docs/models/gpt-5.4-nano
- `gpt-5.4-pro` does not support structured outputs:
  https://developers.openai.com/api/docs/models/gpt-5.4-pro
- OpenAI structured outputs guide, including `text.format`, strict schema constraints, and refusal handling:
  https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI Responses API create reference, including `text.format` JSON Schema configuration:
  https://developers.openai.com/api/reference/resources/responses/methods/create
- pricing is published separately and should remain externally configurable in this codebase:
  https://developers.openai.com/api/docs/pricing
