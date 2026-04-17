# LLM Multi-Provider Adoption Research

- **Date:** 2026-04-17
- **Branch:** `feat/llm-full-text-lean-v1`
- **Status:** Research / pre-planning
- **Scope:** Make the LLM full-text pipeline adaptable to Anthropic Claude,
  OpenAI, and local Ollama in addition to the current Gemini-only path.
- **Out of scope:** Embedding model swap (BioLORD stays); agent loops; tool
  calling beyond what structured output requires.

## 1. Current state (audit of `feat/llm-full-text-lean-v1`)

**Verdict: Gemini-only by design. Not ready for other providers.**

- `phentrieve/llm/provider.py:784-801` — factory **explicitly rejects** any
  value of `PHENTRIEVE_LLM_PROVIDER` other than `"gemini"`.
- Single concrete impl: `GeminiStructuredOutputProvider` (provider.py:70-495).
  The `LLMProvider` ABC exists but has no other subclasses.
- Zero references to `ollama`, `localhost:11434`, `base_url`, `litellm`, or
  OpenAI-compatible endpoints anywhere in `phentrieve/llm/`.

Gemini-leaky assumptions in the current code:

- Structured output via `response_mime_type="application/json"` +
  Pydantic `response_schema` (provider.py:162-239).
- Token counting via `client.models.count_tokens()` (provider.py:241-264).
- Finish-reason parsing (provider.py:321-330) and retryable-error detection
  (provider.py:474-486) are Gemini-specific.
- `structured_retry_token_multiplier` (provider.py:92-108) is tuned to
  Gemini's behavior.

Planning intent already confirms this is deferred:

- `2026-04-16-llm-shared-chunk-pipeline-internal-refactor-plan.md:66` lists
  "replacing the provider" as **OUT OF SCOPE**.
- `2026-04-16-llm-lean-v1-comparative-review.md` and the CLI grounded design
  doc both assume Gemini throughout.

## 2. Library landscape (April 2026)

Three viable abstraction layers were evaluated.

### LiteLLM 1.83+

- One OpenAI-shaped call site for 100+ providers including Anthropic, OpenAI,
  Gemini, and Ollama (`ollama_chat/` and `ollama/` prefixes).
- `response_format=PydanticModel` is the unified entry point but quality
  varies per provider:
  - OpenAI / Azure: native strict `json_schema`. Solid.
  - Gemini 2.0+: native `responseJsonSchema`. Solid.
  - Anthropic: synthesized via tool-use. Known broken on
    `minimum`/`maximum`/`ge`/`le` constraints — issues
    [#6766](https://github.com/BerriAI/litellm/issues/6766) and
    [#21016](https://github.com/BerriAI/litellm/issues/21016) still open
    Feb 2026.
  - Ollama: PR
    [#7344](https://github.com/BerriAI/litellm/pull/7344) translates
    `response_format` to Ollama's native `format=<schema>`, but the OpenAI
    compat path is still leaky
    ([#17807](https://github.com/BerriAI/litellm/issues/17807)).
- Token counter: tiktoken fallback for non-OpenAI/Llama models — 10-30% off
  for Gemma/Mistral/Qwen on Ollama
  ([#8244](https://github.com/BerriAI/litellm/issues/8244)).
- Unified exception hierarchy and unified cost reporting
  (`litellm.completion_cost`, `_hidden_params["response_cost"]`,
  `_response_ms`) are clean.
- Common 2026 complaints: schema-coercion bugs reappear per release, model
  pricing DB lags new model launches, frequent breaking-ish minor versions.

### Instructor 1.15+

- First-class support for OpenAI, Anthropic, Gemini (`google-genai` SDK),
  Ollama, vLLM, plus a `from_litellm()` bridge.
- Init pattern is the same across providers:

  ```python
  client = instructor.from_provider("anthropic/claude-sonnet-4-5")
  client = instructor.from_provider("google/gemini-2.5-pro")
  client = instructor.from_provider("openai/gpt-4o")
  client = instructor.from_provider("ollama/qwen2.5:32b")
  ```

- Per-provider mode selection is automatic and matches the native sweet spot:
  OpenAI -> `TOOLS` or `JSON_SCHEMA`; Anthropic -> `ANTHROPIC_TOOLS`;
  Gemini -> `GENAI_TOOLS` or `GENAI_STRUCTURED_OUTPUTS`; Ollama -> `TOOLS`
  for function-calling models, `JSON` otherwise.
- Killer feature: automatic retry-on-`ValidationError` with the validation
  error fed back to the model. Works uniformly across providers.
- Gotcha: Instructor does **not** call Ollama's native
  `/api/chat` `format=<json_schema>` mode. It uses tool-calling or JSON
  mode + reparse. Acceptable in practice for modern Ollama models.

### Pydantic-AI

- Cleaner typed agent model from the Pydantic team itself.
- Built-in Logfire observability and `output_validator` + `ModelRetry`.
- Ollama via `OpenAIProvider(base_url="http://localhost:11434/v1")`.
- Heavier than needed for single-shot extraction. Wins only if Phentrieve
  later adds multi-turn agent loops, tool calls, or wants Logfire traces.

**Recommendation: use Instructor.** It is the lowest-friction path that
matches Phentrieve's actual workload — single-shot, schema-validated
extraction across cloud + local providers — without taking on
agent-framework weight.

## 3. Native structured-output state (2026)

| Provider | Status | Notes |
|---|---|---|
| OpenAI | GA | `client.chat.completions.parse(response_format=Model)`; `additionalProperties:false` required, all fields in `required`. |
| Anthropic | **Public beta since 2025-11-14** for Sonnet 4.5 / Opus 4.1+ | `anthropic-beta: structured-outputs-2025-11-13`; tool-use fallback for older models. |
| Gemini 2.5 | Stable | `response_mime_type="application/json"` + `response_schema=PydanticModel`. Pydantic `default` values can trip validation ([genai#699](https://github.com/googleapis/python-genai/issues/699)); enums returned as strings. |
| Ollama | Stable since v0.5 (Dec 2024) | `format=<schema>` on `/api/chat` uses GBNF-constrained decoding. **Avoid `/v1` for strict schemas** ([ollama#10001](https://github.com/ollama/ollama/issues/10001)). Truncated JSON returns silently if `num_predict` too small. |

Discriminated unions remain the weakest cross-provider feature — OpenAI is
strongest, Gemini partial, Anthropic beta, Ollama via grammar with reduced
reliability. If the HPO schema uses any, test all four providers
explicitly.

## 4. Local-model recommendations (Ollama)

Targeted at strict-schema HPO extraction:

| Model | VRAM (Q4_K_M) | JSON reliability | Tool calling | Biomed knowledge |
|---|---|---|---|---|
| Qwen 3 32B | ~20 GB | Excellent | Yes | Generalist, multilingual |
| Llama 3.3 70B | ~42 GB | Strong | Yes | Generalist |
| gpt-oss-20B | ~14 GB | Excellent (trained for structured output) | Yes | Generalist |
| MedGemma 27B | ~16 GB | Moderate | Limited | **Strongest biomed** |
| Mistral Nemo 12B | ~7 GB | Good | Yes | Generalist |
| Phi-4 14B | ~8 GB | Good (slips on deep nesting) | Limited | Weak biomed |
| BioMistral 7B | ~5 GB | Moderate | Weak | Strong PubMed |
| OpenBioLLM 70B | ~42 GB | Moderate | Weak | Strong biomed |

**Suggested defaults for the first round of local benchmarks:**
Qwen 3 32B (general workhorse), gpt-oss-20B (cheaper, structured-native),
MedGemma 27B (biomed comparison — but budget extra evaluation time for
schema fidelity).

**Embeddings:** keep BioLORD on the existing sentence-transformers pipeline.
Ollama's bundled embedding models (`nomic-embed-text`, `mxbai-embed-large`,
`bge-m3`) are not biomedical and routing them through Ollama adds an HTTP
hop for zero gain.

**Concurrency:** since v0.2 Ollama supports `OLLAMA_NUM_PARALLEL` (auto 1
or 4) for 3-4x throughput at the cost of `NUM_PARALLEL x CONTEXT_LENGTH`
KV cache — relevant for long clinical notes.

## 5. Refactor plan (concrete)

Files to touch on this branch (or a follow-up phase):

1. **`phentrieve/llm/provider.py`** (~500 LoC rewrite of the Gemini class)
   - Drop the hardcoded `"gemini"` reject at lines 784-801.
   - Replace `GeminiStructuredOutputProvider` with a thin
     `InstructorStructuredOutputProvider` that takes a model string and
     delegates to `instructor.from_provider(...)`.
   - Move provider-specific knobs (retry-token multiplier, finish-reason
     parsing, retryable-error detection) behind a small **capability
     descriptor** keyed by provider name.

2. **`phentrieve/llm/config.py`**
   - Per-provider config block: API key env vars, default models,
     `supports_native_structured_output`,
     `supports_native_token_counting`, default Ollama `api_base`,
     recommended `num_predict` floor for Ollama.

3. **`phentrieve/llm/pipeline.py`** (lines 170-239)
   - Already provider-agnostic at the surface — should mostly work once
     `run_structured_prompt()` is provider-neutral.
   - Add fallback path for providers without reliable token counting
     (`len(text) // 4` heuristic; flag the result as estimated in
     observability).

4. **`phentrieve/cli/text_commands.py`**
   - Add `--llm-provider` flag (today only `--llm-model` exists).
   - Validate provider/model combinations against the capability
     descriptor.

5. **`tests/unit/llm/test_pipeline.py`**
   - Replace Gemini-specific mocks with provider-agnostic Instructor mocks.
   - Add per-provider parametrized smoke tests covering structured output,
     validation-retry, and timeout behavior.

6. **`pyproject.toml`**
   - Add `instructor[anthropic,google-genai,litellm]>=1.15` (verify
     latest 1.x at implementation time).
   - Keep `google-genai` direct dep for any Gemini-native fast path we
     decide to retain.

## 6. Honest gaps to plan around

- **Anthropic native structured output is still beta-flagged.** Wire it via
  Instructor (which abstracts the header) and keep tool-use as a documented
  fallback if the header moves.
- **Ollama returns truncated JSON silently** if `num_predict` is too low.
  Always validate post-parse; raise the floor to 4096+ for HPO schemas.
- **Token counting on local models is heuristic** (10-30% error). Benchmark
  observability needs a `token_count_source: "exact" | "estimated"` field
  so dashboards do not silently mix the two.
- **Discriminated unions** require per-provider testing if the HPO schema
  uses any. Treat as a known portability risk.
- **MedGemma's structured-output behavior under heavy schemas is
  under-documented.** Budget evaluation time before committing to it as a
  benchmark target.
- **Cost / pricing data lag.** LiteLLM's `model_prices_and_context_window.json`
  trails new model launches by days/weeks. Verify per-token pricing on any
  fresh model before running large benchmarks.

## 7. Effort estimate

- **Full Instructor refactor with all four providers wired and parametrized
  tests:** 3-5 days.
- **Minimum-viable "Gemini + one new provider as proof":** ~1.5 days.
  Recommended if we want to land the abstraction cleanly first and add the
  remaining providers in a second phase.

## 8. Open questions for `/gsd-discuss-phase`

1. Do we want Instructor as the **only** code path, or keep a Gemini-native
   fast path alongside (worth ~5-10% latency on Gemini calls)?
2. Which providers do we wire in **phase one**? All four, or
   Gemini + Anthropic + Ollama with OpenAI deferred?
3. Which Ollama model(s) become the default local benchmark target?
4. Do we want benchmark observability to record per-call cost in USD using
   `litellm.completion_cost`, or stay token-only?
5. Is there a hard requirement on offline-only operation (no cloud calls)
   for any deployment target? That changes the default-provider logic.

## Sources

- LiteLLM: [docs](https://docs.litellm.ai/docs/),
  [Ollama provider](https://docs.litellm.ai/docs/providers/ollama),
  [Anthropic provider](https://docs.litellm.ai/docs/providers/anthropic),
  [structured outputs](https://docs.litellm.ai/docs/completion/json_mode),
  [token counting](https://docs.litellm.ai/docs/count_tokens),
  [exception mapping](https://docs.litellm.ai/docs/exception_mapping).
- Instructor: [docs](https://python.useinstructor.com/),
  [integrations](https://python.useinstructor.com/integrations/),
  [mode comparison](https://python.useinstructor.com/modes-comparison/),
  [Ollama integration](https://python.useinstructor.com/integrations/ollama/),
  [LiteLLM bridge](https://python.useinstructor.com/integrations/litellm/).
- Pydantic-AI: [output docs](https://ai.pydantic.dev/output/),
  [providers](https://ai.pydantic.dev/api/providers/).
- Ollama: [structured outputs blog](https://ollama.com/blog/structured-outputs),
  [structured outputs docs](https://docs.ollama.com/capabilities/structured-outputs),
  [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility),
  [issue #10001](https://github.com/ollama/ollama/issues/10001),
  [FAQ / concurrency](https://docs.ollama.com/faq).
- OpenAI: [structured outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
- Anthropic: [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
  [Nov 2025 launch writeup](https://techbytes.app/posts/claude-structured-outputs-json-schema-api/).
- Gemini: [structured output docs](https://ai.google.dev/gemini-api/docs/structured-output),
  [genai SDK issue #699](https://github.com/googleapis/python-genai/issues/699).
- Models: [MedGemma on Ollama](https://ollama.com/library/medgemma),
  [gpt-oss on Ollama](https://ollama.com/library/gpt-oss:120b),
  [BioMistral on Ollama](https://ollama.com/cniongolo/biomistral).
