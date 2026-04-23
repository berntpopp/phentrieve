# OpenAI Structured Output Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native OpenAI structured-output support to the Python LLM provider layer, benchmark path, and tests for OpenAI models that officially support structured outputs.

**Architecture:** Reuse the existing multi-provider seam in `phentrieve/llm/provider.py` and add a native OpenAI adapter that uses the OpenAI Responses API for both free-text completions and structured outputs. Keep benchmark and CLI plumbing generic, preserve existing Gemini/Ollama/Anthropic behavior, and validate OpenAI structured responses locally with the same Pydantic-based contract used elsewhere.

**Tech Stack:** Python 3.10, OpenAI Python SDK, Pydantic, Typer, pytest, Ruff, mypy.

---

## File Map

- Modify: `pyproject.toml`
  Add the official OpenAI SDK to the `llm` optional dependency set.
- Modify: `uv.lock`
  Refresh the lockfile for the new `openai` dependency.
- Modify: `phentrieve/llm/provider.py`
  Add the native OpenAI provider, model support checks, key lookup, schema sanitization, refusal handling, and provider factory wiring.
- Modify: `tests/unit/llm/test_provider.py`
  Add provider-resolution, env fallback, structured-output, completion, refusal, and token-count tests for OpenAI.
- Modify: `tests/unit/test_llm_benchmark.py`
  Add benchmark propagation coverage for resolved OpenAI provider metadata and token source markers.

### Task 1: Add OpenAI Dependency And Provider Resolution Coverage

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `phentrieve/llm/provider.py`
- Test: `tests/unit/llm/test_provider.py`

- [ ] **Step 1: Write the failing tests for OpenAI provider resolution and env fallback**

```python
def test_get_llm_provider_infers_openai_from_prefixed_model(monkeypatch) -> None:
    monkeypatch.setenv("PHENTRIEVE_OPENAI_API_KEY", "test-key")
    provider = get_llm_provider(llm_model="openai/gpt-5.4")

    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-5.4"


def test_get_llm_provider_accepts_chatgpt_api_key_fallback(monkeypatch) -> None:
    monkeypatch.delenv("PHENTRIEVE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHATGPT_API_KEY", "chatgpt-key")

    provider = get_llm_provider(
        llm_provider="openai",
        llm_model="gpt-5.4-mini",
    )

    assert provider.provider_name == "openai"


def test_get_llm_provider_rejects_unsupported_openai_structured_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PHENTRIEVE_OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError, match="structured outputs"):
        get_llm_provider(
            llm_provider="openai",
            llm_model="gpt-5.4-pro",
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/llm/test_provider.py -k "openai and (prefixed_model or chatgpt_api_key or unsupported_openai)" -v`
Expected: FAIL because no OpenAI provider implementation exists yet.

- [ ] **Step 3: Add the OpenAI SDK dependency and provider factory branch**

```toml
[project.optional-dependencies]
llm = [
    "google-genai>=1.22.0,<2.0.0",
    "anthropic>=0.73.0,<1.0.0",
    "openai>=2.0.0,<3.0.0",
]
```

```python
if request.provider == "openai":
    return OpenAIStructuredOutputProvider(
        model_name=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
        seed=request.seed,
        timeout_seconds=timeout_seconds or DEFAULT_OPENAI_TIMEOUT_SECONDS,
    )
```

- [ ] **Step 4: Add the OpenAI provider skeleton with supported-model and key validation**

```python
class OpenAIStructuredOutputProvider(LLMProvider):
    provider_name = "openai"
    token_count_source = "estimated"  # noqa: S105
    _SUPPORTED_STRUCTURED_MODELS = {
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    }

    def __init__(..., api_key: str | None = None, base_url: str | None = None, ...):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else None
        self._api_key = (
            api_key
            or os.getenv("PHENTRIEVE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("CHATGPT_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "OpenAI API key not configured. Set PHENTRIEVE_OPENAI_API_KEY, "
                "OPENAI_API_KEY, or CHATGPT_API_KEY."
            )
        self._ensure_supported_model()
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/llm/test_provider.py -k "openai and (prefixed_model or chatgpt_api_key or unsupported_openai)" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock phentrieve/llm/provider.py tests/unit/llm/test_provider.py
git commit -m "feat: add openai provider resolution"
```

### Task 2: Implement Native OpenAI Responses API Completion And Structured Output

**Files:**
- Modify: `phentrieve/llm/provider.py`
- Test: `tests/unit/llm/test_provider.py`

- [ ] **Step 1: Write the failing tests for OpenAI request shaping, refusal handling, and token extraction**

```python
def test_openai_structured_prompt_uses_responses_api_json_schema(monkeypatch) -> None:
    fake_responses = _install_fake_openai(
        monkeypatch,
        create_responses=[
            _fake_openai_response(
                output_text='{"phenotypes":[]}',
                input_tokens=42,
                output_tokens=11,
            )
        ],
    )
    provider = get_llm_provider(llm_provider="openai", llm_model="gpt-5.4")

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert isinstance(result, LLMExtractedPhenotypes)
    request = fake_responses.create_calls[0]
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True


def test_openai_complete_uses_responses_api(monkeypatch) -> None:
    fake_responses = _install_fake_openai(
        monkeypatch,
        create_responses=[_fake_openai_response(output_text="plain text")],
    )
    provider = get_llm_provider(llm_provider="openai", llm_model="gpt-5.4-mini")

    response = provider.complete(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
    )

    assert response.content == "plain text"
    assert fake_responses.create_calls[0]["model"] == "gpt-5.4-mini"


def test_openai_structured_prompt_raises_on_refusal(monkeypatch) -> None:
    _install_fake_openai(
        monkeypatch,
        create_responses=[
            _fake_openai_response(refusal="cannot comply"),
        ],
    )
    provider = get_llm_provider(llm_provider="openai", llm_model="gpt-5.4")

    with pytest.raises(RuntimeError, match="refusal"):
        provider.run_structured_prompt(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractedPhenotypes,
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/llm/test_provider.py -k "openai and (responses_api or refusal or structured_prompt)" -v`
Expected: FAIL because the provider skeleton does not yet call the OpenAI SDK or parse responses.

- [ ] **Step 3: Implement the OpenAI client wrapper and Responses API calls**

```python
def _create_client(self, *, openai_module: Any) -> Any:
    client_kwargs = {
        "api_key": self._api_key,
        "timeout": self.timeout_seconds,
        "max_retries": 0,
    }
    if self.base_url:
        client_kwargs["base_url"] = self.base_url
    return openai_module.OpenAI(**client_kwargs)


def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
    response, request_count = self._create_response_with_transient_retry(
        messages=messages,
    )
    ...


def run_structured_prompt(...):
    response_schema = self._build_openai_response_schema(response_model)
    response, request_count = self._create_response_with_transient_retry(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format={
            "type": "json_schema",
            "name": response_model.__name__,
            "strict": True,
            "schema": response_schema,
        },
    )
    refusal = self._extract_refusal(response)
    if refusal:
        raise RuntimeError(f"OpenAI structured output refusal: {refusal}")
    return response_model.model_validate_json(self._extract_text_content(response))
```

- [ ] **Step 4: Add OpenAI schema sanitization and usage extraction helpers**

```python
def _build_openai_response_schema(
    self, response_model: type[BaseModel]
) -> dict[str, Any]:
    schema = build_response_json_schema(response_model)
    return self._sanitize_openai_schema(schema)


def _sanitize_openai_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object":
        properties = dict(schema.get("properties", {}))
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False
    schema.pop("title", None)
    return schema
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/llm/test_provider.py -k "openai and (responses_api or refusal or structured_prompt)" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/llm/provider.py tests/unit/llm/test_provider.py
git commit -m "feat: add native openai structured output provider"
```

### Task 3: Add Benchmark Metadata Coverage For OpenAI Runs

**Files:**
- Modify: `tests/unit/test_llm_benchmark.py`
- Modify: `phentrieve/llm/provider.py`

- [ ] **Step 1: Write the failing benchmark test for resolved OpenAI provider metadata**

```python
def test_run_llm_benchmark_records_openai_provider_metadata(monkeypatch) -> None:
    class _FakeProvider:
        provider_name = "openai"
        model_name = "gpt-5.4-mini"
        base_url = "https://api.openai.com/v1"
        token_count_source = "estimated"

    ...

    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_provider="openai",
        llm_model="gpt-5.4-mini",
    )

    assert result["llm_provider"] == "openai"
    assert result["results"][0]["meta"]["llm_provider"] == "openai"
    assert result["results"][0]["meta"]["token_count_source"] == "estimated"
```

- [ ] **Step 2: Run the focused test to verify it fails if metadata is not wired correctly**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "openai_provider_metadata" -v`
Expected: FAIL if the benchmark payload does not preserve the resolved OpenAI provider fields.

- [ ] **Step 3: Make the minimal plumbing changes needed to preserve OpenAI metadata**

```python
resolved_provider_name = getattr(provider, "provider_name", llm_provider or "gemini")
resolved_model_name = getattr(provider, "model_name", llm_model)
token_count_source = getattr(provider, "token_count_source", None)
```

Keep any changes minimal and reuse the same generic code path used for Ollama and Anthropic.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "openai_provider_metadata" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/provider.py tests/unit/test_llm_benchmark.py
git commit -m "test: cover openai benchmark metadata"
```

### Task 4: Verify, Live-Benchmark OpenAI Models, And Capture Cost/Performance

**Files:**
- Modify if needed after live validation: `phentrieve/llm/provider.py`
- Modify if needed after live validation: `tests/unit/llm/test_provider.py`

- [ ] **Step 1: Run focused provider and benchmark tests before full verification**

Run: `uv run pytest tests/unit/llm/test_provider.py tests/unit/test_llm_benchmark.py -k "openai" -v`
Expected: PASS

- [ ] **Step 2: Run required repository verification**

Run: `make check`
Expected: PASS

Run: `make typecheck-fast`
Expected: PASS

Run: `make test`
Expected: PASS, or if the known unrelated CUDA OOM recurs, record it and run `CUDA_VISIBLE_DEVICES="" make test`.

- [ ] **Step 3: Run live GeneReviews benchmarks for OpenAI models with explicit pricing config**

Run:

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-provider openai \
  --llm-model gpt-5.4 \
  --pricing-config phentrieve/benchmark/pricing_examples/openai_gpt54.json \
  --output-path /tmp/phentrieve-openai-gpt54-genereviews-full.json
```

```bash
uv run phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-provider openai \
  --llm-model gpt-5.4-mini \
  --pricing-config phentrieve/benchmark/pricing_examples/openai_gpt54_mini.json \
  --output-path /tmp/phentrieve-openai-gpt54mini-genereviews-full.json
```

Optionally run `gpt-5.4-nano` if time and API cost permit.

- [ ] **Step 4: Fix any live-validation issues with TDD before claiming completion**

If live runs expose request-shape or refusal-handling bugs:
1. add a focused failing unit test in `tests/unit/llm/test_provider.py`
2. run it to watch it fail
3. patch `phentrieve/llm/provider.py` minimally
4. rerun the focused tests

- [ ] **Step 5: Summarize benchmark quality, latency, and configured cost**

Capture for each model:
- `id_only` micro precision, recall, F1
- average seconds per document
- total configured token cost and cost per document

- [ ] **Step 6: Commit any final OpenAI provider fixes**

```bash
git add phentrieve/llm/provider.py tests/unit/llm/test_provider.py tests/unit/test_llm_benchmark.py pyproject.toml uv.lock
git commit -m "fix: polish openai provider benchmark validation"
```

## Self-Review

- Spec coverage: this plan covers provider normalization, key lookup, Responses API usage, strict schema shaping, refusal handling, benchmark metadata propagation, and live GeneReviews benchmarking for supported OpenAI models.
- Placeholder scan: no `TODO`, `TBD`, or "similar to task N" shortcuts are left in the task steps.
- Type consistency: the plan keeps the existing `LLMProvider`, `LLMResponse`, `LLMMeta`, and benchmark payload naming instead of inventing new interfaces.
