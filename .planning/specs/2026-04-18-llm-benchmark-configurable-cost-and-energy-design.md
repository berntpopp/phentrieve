# LLM Benchmark Configurable Cost And Energy Design

- Date: 2026-04-18
- Status: Draft for review
- Scope: Python-only benchmark cost and energy accounting for Gemini, Anthropic, Ollama, and other provider-backed CLI runs
- Out of scope: API or frontend changes, hardcoded model pricing tables, provider-side billing reconciliation, and system-wide hardware monitoring outside benchmark execution

## 1. Goal

Extend the existing LLM benchmark reporting so cost-like benchmarking data is
configurable, explicit, and comparable across providers without hardcoding
vendor pricing into the codebase.

The design must support two distinct accounting modes:

- hosted API token pricing for providers such as Gemini and Anthropic
- local runtime energy accounting for providers such as Ollama

The benchmark should be able to report either mode independently or both at
once, while preserving the current lightweight manual pricing behavior as the
baseline.

## 2. Current State

The benchmark already supports optional token-cost estimation through three CLI
flags:

- `--input-cost-per-1m-tokens`
- `--output-cost-per-1m-tokens`
- `--cached-input-cost-per-1m-tokens`

Internally, `phentrieve/benchmark/llm_benchmark.py` computes an
`estimated_cost` block from normalized token usage. This works for Gemini-like
pricing inputs, but it has three gaps:

- pricing inputs are loose scalar arguments instead of a typed pricing config
- there is no local energy or emissions accounting path for Ollama
- output conflates a single cost estimate instead of separating token pricing
  from local electricity-style cost

## 3. Design Principles

- No hardcoded vendor pricing tables in application code.
- All pricing and energy assumptions must be user-configurable at runtime.
- Token pricing and local energy accounting are separate concepts and should be
  reported separately.
- Existing manual pricing flags should remain supported for backward
  compatibility.
- Benchmark output should make missing assumptions explicit rather than
  inventing estimates.
- Local Ollama cost should be based on measured or tool-estimated energy, not a
  fabricated per-token USD rate.

## 4. Scope Boundaries

Included in this design:

- benchmark CLI inputs
- benchmark internal config typing
- benchmark summary and per-document output schema additions
- optional local energy measurement during benchmark execution
- unit coverage for the new config and accounting behavior

Excluded from this design:

- automatic fetching of live provider pricing from the web
- built-in Gemini or Anthropic model price maps
- API service cost reporting
- frontend presentation of cost or energy data
- non-benchmark CLI commands

## 5. Research Constraints And Best Practices

Official documentation supports the underlying accounting categories but does
not justify hardcoded provider tables:

- Gemini bills by input tokens, output tokens, cached tokens, and cache storage
  duration. Prices vary by model and can change. Source:
  https://ai.google.dev/gemini-api/docs/pricing
- Gemini billing documentation explicitly frames billing around token
  categories, not a universal model-agnostic flat rate. Source:
  https://ai.google.dev/gemini-api/docs/billing/
- Anthropic pricing is likewise category-based with distinct base input, cache
  write/read, and output pricing. Source:
  https://platform.claude.com/docs/en/about-claude/pricing
- Ollama exposes runtime and model metadata such as loaded model size and VRAM
  through `/api/ps`, but does not expose official local usage pricing. Source:
  https://docs.ollama.com/api/ps
- CodeCarbon is a documented way to estimate local energy use and emissions,
  while being explicit that some measurements are direct and some are hardware
  estimates. Sources:
  https://mlco2.github.io/codecarbon/usage.html
  https://mlco2.github.io/codecarbon/methodology.html

This supports a design where the application accepts user-supplied economic
assumptions and uses documented measurement tooling for local inference.

## 6. User-Facing Behavior

### 6.1 New benchmark accounting model

Replace the current loose pricing inputs with an internal typed config that can
still be filled from the existing CLI flags.

Conceptually:

- token pricing inputs:
  - `input_cost_per_1m_tokens`
  - `output_cost_per_1m_tokens`
  - `cached_input_cost_per_1m_tokens`
  - `cache_storage_cost_per_1m_tokens_hour` optional
  - `thoughts_cost_per_1m_tokens` optional for providers that might diverge in
    the future
- local energy inputs:
  - `measure_energy`
  - `electricity_cost_per_kwh`
  - `carbon_kg_per_kwh`
  - CodeCarbon location/config fields as needed for offline estimation

### 6.2 CLI surface

Keep the existing token pricing flags for compatibility and add:

- `--pricing-config PATH`
- `--measure-energy / --no-measure-energy`
- `--electricity-cost-per-kwh FLOAT`
- `--carbon-kg-per-kwh FLOAT`
- optional CodeCarbon location flags such as country or region if needed by the
  implementation path

Precedence rules:

1. explicit CLI flags override file-based config
2. file-based config overrides defaults
3. defaults remain `None` or disabled, producing no estimate rather than an
   invented one

### 6.3 Pricing config file

Support a JSON pricing file for versionable, provider-specific benchmark
configuration without hardcoding values into code.

Example shape:

```json
{
  "token_pricing": {
    "input_cost_per_1m_tokens": 0.125,
    "output_cost_per_1m_tokens": 0.75,
    "cached_input_cost_per_1m_tokens": 0.0125
  },
  "energy_accounting": {
    "measure_energy": true,
    "electricity_cost_per_kwh": 0.30,
    "carbon_kg_per_kwh": 0.35,
    "country_iso_code": "DEU"
  }
}
```

The config is intentionally generic rather than keyed by provider/model. Users
can keep separate files per benchmark target if they want model-specific rates.

## 7. Output Design

### 7.1 Summary payload

Replace the single `estimated_cost` concept with an additive accounting block:

- `estimated_token_cost`
- `estimated_energy_cost`
- `estimated_total_cost`

Each block should be omitted or `null` when its prerequisites are missing.

`estimated_token_cost` includes:

- input, cached input, cache storage, thoughts, and output subtotals when
  available
- billable token counts used to compute them

`estimated_energy_cost` includes:

- `energy_kwh`
- `electricity_cost_usd` when `electricity_cost_per_kwh` is provided
- `carbon_kg` when enough carbon-intensity information is provided
- measurement source metadata such as `measured`, `estimated`, or `disabled`

`estimated_total_cost` should only be present when at least one cost component
  is available. It may include:

- `token_cost_usd`
- `energy_cost_usd`
- `total_cost_usd`

### 7.2 Per-document payload

Per-document prediction records should include the same accounting structure so
users can compare expensive or energy-heavy documents directly.

This preserves aggregation and document-level debugging symmetry.

## 8. Internal Architecture

### 8.1 Typed benchmark accounting config

Introduce a typed internal config model, likely in the benchmark layer rather
than the provider layer, because these settings affect reporting rather than
LLM request semantics.

Suggested units:

- token prices in USD per 1M tokens
- electricity in USD per kWh
- emissions in kg CO2e per kWh

### 8.2 Token cost estimation

Refactor `_estimate_cost(...)` into a token-specific helper that consumes the
typed config. It should continue to use normalized token usage keys:

- `prompt_tokens`
- `completion_tokens`
- `thoughts_tokens`
- `cached_content_tokens`

Cache storage cost should remain optional because the current benchmark does
not track cache TTL/storage hours directly.

### 8.3 Energy measurement

Add an optional benchmark-scoped energy tracker abstraction with two
requirements:

- no-op when energy measurement is disabled or unavailable
- measurable start/stop lifecycle around each benchmarked document and around
  the full run

The implementation should prefer a small wrapper around CodeCarbon so the rest
of the benchmark code only sees a stable local interface.

If CodeCarbon is unavailable or unsupported on the host, benchmark execution
must continue with an explicit disabled/unavailable accounting result rather
than failing the benchmark.

### 8.4 Provider independence

The benchmark should not special-case Gemini, Anthropic, or Ollama in the
costing logic. Instead:

- hosted providers become token-priced when the user supplies token pricing
- local providers become energy-priced when the user enables energy accounting
- either provider type can technically use both if the user wants combined
  accounting

This keeps the feature orthogonal to provider routing.

## 9. Error Handling

- malformed pricing config files should fail fast with a clear CLI error
- negative prices or rates should be rejected
- unsupported or unavailable energy measurement should degrade to explicit
  `unavailable` metadata, not a hard benchmark failure
- missing electricity or carbon rates should still allow energy measurement,
  but only report the measurable fields

## 10. Testing Strategy

Add unit coverage for:

- pricing config parsing and precedence
- token cost estimation with and without cached/thought tokens
- additive total-cost aggregation
- disabled energy accounting
- unavailable energy tracker behavior
- benchmark payload serialization for summary and per-document accounting data

Avoid mandatory live energy measurement in CI. The CodeCarbon integration
should be tested behind mocks or fakes so the benchmark suite remains stable
across developer machines and CI runners.

## 11. Recommended Implementation Order

1. Introduce typed benchmark accounting config and CLI/file parsing
2. Refactor existing token-cost computation onto the new config
3. Expand benchmark output schema to split token, energy, and total accounting
4. Add optional energy tracker abstraction and mocked tests
5. Wire CLI and benchmark summaries together without changing API or frontend

## 12. Open Questions Resolved

- Hardcoded vendor pricing tables: rejected
- Automatic live web pricing fetches: rejected
- Fake Ollama dollar pricing: rejected
- Optional user-supplied electricity and emissions factors: accepted
- Backward compatibility for current token-pricing flags: accepted
