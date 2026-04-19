# Ollama-First Multi-Provider LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class provider normalization and an Ollama native structured-output adapter to the Python CLI, shared LLM service, and benchmark path while preserving backward compatibility for existing bare Gemini model names.

**Architecture:** Introduce a typed provider request that normalizes CLI and env input into `provider`, `model`, and `base_url`, then dispatch through a provider registry instead of the current Gemini-only branch. Implement phase one with a native Ollama `/api/chat` adapter, carry provider identity through `LLMPipelineConfig` and `LLMMeta`, and keep `phentrieve/llm/pipeline.py` mostly unchanged by preserving the existing `LLMProvider` contract.

**Tech Stack:** Python 3.10, Typer CLI, Pydantic, httpx, pytest, Ruff, mypy, Ollama native HTTP API

---

## File Structure

- `phentrieve/llm/config.py`
  Responsible for global and per-provider defaults, including provider names, base URLs, and timeout values.
- `phentrieve/llm/types.py`
  Responsible for durable LLM metadata and pipeline config types shared across CLI, service, benchmark, and pipeline code.
- `phentrieve/llm/provider.py`
  Responsible for provider normalization, dispatch, provider adapters, and shared provider contract behavior.
- `phentrieve/text_processing/full_text_service.py`
  Responsible for translating CLI/service inputs into normalized provider configuration and passing provider identity into the pipeline.
- `phentrieve/cli/text_commands.py`
  Responsible for user-facing CLI flags and forwarding provider/model/base URL inputs into the service layer.
- `phentrieve/benchmark/llm_cli.py`
  Responsible for benchmark CLI flags and user-facing benchmark metadata.
- `phentrieve/benchmark/llm_benchmark.py`
  Responsible for provider-aware benchmark execution and checkpoint metadata.
- `tests/unit/llm/test_provider.py`
  Responsible for provider normalization, dispatch, Ollama adapter, and retry behavior.
- `tests/unit/llm/test_pipeline.py`
  Responsible for proving the pipeline stays provider-agnostic and preserves provider metadata.
- `tests/unit/text_processing/test_full_text_service.py`
  Responsible for provider-aware service resolution and forwarding.
- `tests/unit/cli/test_text_commands.py`
  Responsible for CLI flag parsing and service invocation.
- `tests/unit/cli/test_benchmark_commands.py`
  Responsible for benchmark CLI flag parsing and benchmark invocation.

## Task 1: Add Provider Types And Defaults

**Files:**
- Modify: `phentrieve/llm/config.py`
- Modify: `phentrieve/llm/types.py`
- Test: `tests/unit/llm/test_provider.py`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write the failing tests for provider-aware types and defaults**

```python
def test_llm_pipeline_config_includes_provider() -> None:
    config = LLMPipelineConfig(
        provider="ollama",
        model="qwen3.5:35b",
        mode="two_phase",
        language="en",
    )

    assert config.provider == "ollama"
    assert config.model == "qwen3.5:35b"


def test_llm_meta_includes_provider_identity() -> None:
    meta = LLMMeta(
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
        llm_mode="two_phase",
    )

    assert meta.llm_provider == "ollama"


def test_provider_config_exposes_ollama_defaults() -> None:
    assert DEFAULT_PROVIDER_NAME == "gemini"
    assert DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434"
    assert DEFAULT_OLLAMA_TIMEOUT_SECONDS == 300
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/llm/test_provider.py -k "provider_config_exposes_ollama_defaults or llm_pipeline_config_includes_provider or llm_meta_includes_provider_identity" -v
pytest tests/unit/llm/test_pipeline.py -k "provider" -v
```

Expected:

```text
FAIL ... unexpected keyword argument 'provider'
FAIL ... unexpected keyword argument 'llm_provider'
FAIL ... name 'DEFAULT_OLLAMA_BASE_URL' is not defined
```

- [ ] **Step 3: Add minimal config and type support**

```python
# phentrieve/llm/config.py
DEFAULT_PROVIDER_NAME = "gemini"
SUPPORTED_PROVIDER_NAMES = ("gemini", "ollama", "openai", "anthropic")

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300
DEFAULT_OPENAI_TIMEOUT_SECONDS = DEFAULT_PROVIDER_TIMEOUT_SECONDS
DEFAULT_ANTHROPIC_TIMEOUT_SECONDS = DEFAULT_PROVIDER_TIMEOUT_SECONDS
DEFAULT_GEMINI_TIMEOUT_SECONDS = DEFAULT_PROVIDER_TIMEOUT_SECONDS
```

```python
# phentrieve/llm/types.py
class LLMMeta(BaseModel):
    llm_provider: str = "gemini"
    llm_model: str
    llm_mode: str
    prompt_version: str = "v1"
    token_input: int | None = None
    token_output: int | None = None
    token_count_source: Literal["exact", "estimated"] | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    request_count: int = 0
    phase_timings: dict[str, float] = Field(default_factory=dict)
    phase_counts: dict[str, int] = Field(default_factory=dict)
    phase_request_counts: dict[str, int] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)


class LLMPipelineConfig(BaseModel):
    provider: str = "gemini"
    model: str
    base_url: str | None = None
    mode: str = DEFAULT_LLM_MODE
    language: str | None = DEFAULT_LLM_LANGUAGE
    seed: int | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/unit/llm/test_provider.py -k "provider_config_exposes_ollama_defaults or llm_pipeline_config_includes_provider or llm_meta_includes_provider_identity" -v
pytest tests/unit/llm/test_pipeline.py -k "provider" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/config.py phentrieve/llm/types.py tests/unit/llm/test_provider.py tests/unit/llm/test_pipeline.py
git commit -m "feat: add provider-aware llm config types"
```

## Task 2: Add Provider Normalization And Dispatch

**Files:**
- Modify: `phentrieve/llm/provider.py`
- Test: `tests/unit/llm/test_provider.py`

- [ ] **Step 1: Write the failing normalization and dispatch tests**

```python
def test_get_llm_provider_infers_ollama_from_prefixed_model() -> None:
    provider = get_llm_provider(llm_model="ollama/qwen3.5:35b")

    assert provider.provider_name == "ollama"
    assert provider.model_name == "qwen3.5:35b"


def test_get_llm_provider_accepts_bare_gemini_model_for_backwards_compat(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PHENTRIEVE_GEMINI_API_KEY", "test-key")
    provider = get_llm_provider(llm_model="gemini-2.5-flash")

    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-2.5-flash"


def test_get_llm_provider_rejects_mismatched_explicit_provider() -> None:
    with pytest.raises(ValueError, match="does not match"):
        get_llm_provider(
            llm_provider="anthropic",
            llm_model="ollama/qwen3.5:35b",
        )


def test_get_llm_provider_defaults_ollama_base_url() -> None:
    provider = get_llm_provider(
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
    )

    assert provider.base_url == "http://localhost:11434"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/llm/test_provider.py -k "infers_ollama or mismatched_explicit_provider or defaults_ollama_base_url or backwards_compat" -v
```

Expected:

```text
FAIL ... get_llm_provider() got an unexpected keyword argument 'llm_provider'
FAIL ... Gemini-only provider factory does not support model
```

- [ ] **Step 3: Implement a typed normalization helper and dispatch path**

```python
@dataclass(frozen=True)
class ResolvedLLMProviderRequest:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    seed: int | None = None


def resolve_llm_provider_request(
    *,
    llm_provider: str | None,
    llm_model: str,
    llm_base_url: str | None = None,
    seed: int | None = None,
) -> ResolvedLLMProviderRequest:
    known_prefixes = ("gemini", "ollama", "openai", "anthropic")
    inferred_provider: str | None = None
    model_name = llm_model

    if "/" in llm_model:
        prefix, remainder = llm_model.split("/", 1)
        if prefix in known_prefixes:
            inferred_provider = prefix
            model_name = remainder

    resolved_provider = (llm_provider or inferred_provider or os.getenv(
        "PHENTRIEVE_LLM_PROVIDER", DEFAULT_PROVIDER_NAME
    )).strip().lower()

    if inferred_provider and llm_provider and resolved_provider != inferred_provider:
        raise ValueError(
            f"Explicit llm_provider={resolved_provider!r} does not match model prefix "
            f"{inferred_provider!r}."
        )

    resolved_base_url = (
        llm_base_url
        or os.getenv("PHENTRIEVE_LLM_BASE_URL")
        or (DEFAULT_OLLAMA_BASE_URL if resolved_provider == "ollama" else None)
    )

    return ResolvedLLMProviderRequest(
        provider=resolved_provider,
        model=model_name,
        base_url=resolved_base_url,
        seed=seed,
    )
```

```python
def get_llm_provider(
    *,
    llm_model: str,
    llm_provider: str | None = None,
    llm_base_url: str | None = None,
    api_key: str | None = None,
    seed: int | None = None,
) -> LLMProvider:
    request = resolve_llm_provider_request(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        seed=seed,
    )

    if request.provider == "gemini":
        return GeminiStructuredOutputProvider(
            model_name=request.model,
            api_key=api_key,
            seed=request.seed,
        )
    if request.provider == "ollama":
        return OllamaStructuredOutputProvider(
            model_name=request.model,
            base_url=request.base_url or DEFAULT_OLLAMA_BASE_URL,
            seed=request.seed,
        )

    raise ValueError(
        f"Provider {request.provider!r} is not implemented in phase one."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/unit/llm/test_provider.py -k "infers_ollama or mismatched_explicit_provider or defaults_ollama_base_url or backwards_compat" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/provider.py tests/unit/llm/test_provider.py
git commit -m "feat: normalize llm provider selection"
```

## Task 3: Implement Native Ollama Structured Output Adapter

**Files:**
- Modify: `phentrieve/llm/provider.py`
- Test: `tests/unit/llm/test_provider.py`

- [ ] **Step 1: Write the failing Ollama adapter tests**

```python
def test_ollama_structured_prompt_posts_native_chat_schema(mocker) -> None:
    provider = OllamaStructuredOutputProvider(
        model_name="qwen3.5:35b",
        base_url="http://localhost:11434",
    )
    post = mocker.patch("httpx.Client.post")
    post.return_value = _fake_ollama_response(
        content='{"phenotypes": []}',
        prompt_eval_count=12,
        eval_count=5,
    )

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert isinstance(result, LLMExtractedPhenotypes)
    _, kwargs = post.call_args
    assert kwargs["json"]["format"]["type"] == "object"
    assert kwargs["json"]["options"]["temperature"] == 0


def test_ollama_provider_sets_estimated_token_count_source_when_counting_missing() -> None:
    provider = OllamaStructuredOutputProvider(
        model_name="qwen3.5:35b",
        base_url="http://localhost:11434",
    )

    token_counts = provider.count_tokens(
        system_prompt="system",
        user_prompt="user",
    )

    assert token_counts["total_tokens"] >= 1
    assert provider.token_count_source == "estimated"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/llm/test_provider.py -k "ollama_structured_prompt_posts_native_chat_schema or estimated_token_count_source" -v
```

Expected:

```text
FAIL ... name 'OllamaStructuredOutputProvider' is not defined
```

- [ ] **Step 3: Implement the minimal Ollama adapter**

```python
class OllamaStructuredOutputProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        seed: int | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        transient_retries: int = DEFAULT_PROVIDER_TRANSIENT_RETRIES,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.transient_retries = transient_retries
        self.token_count_source = "estimated"

    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        max_output_tokens: int | None = None,
    ) -> BaseModel:
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_predict": max_output_tokens or DEFAULT_PROVIDER_MAX_TOKENS,
            },
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        body = response.json()
        content = str(body["message"]["content"])
        self.last_usage = {
            "prompt_tokens": int(body.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(body.get("eval_count", 0) or 0),
            "total_tokens": int(body.get("prompt_eval_count", 0) or 0)
            + int(body.get("eval_count", 0) or 0),
        }
        self.last_request_count = 1
        self.last_finish_reason = body.get("done_reason")
        return response_model.model_validate_json(content)

    def count_tokens(self, *, system_prompt: str, user_prompt: str) -> dict[str, int]:
        estimated_total = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        return {
            "prompt_tokens": estimated_total,
            "completion_tokens": 0,
            "total_tokens": estimated_total,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/unit/llm/test_provider.py -k "ollama_structured_prompt_posts_native_chat_schema or estimated_token_count_source" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/llm/provider.py tests/unit/llm/test_provider.py
git commit -m "feat: add native ollama structured output provider"
```

## Task 4: Thread Provider Metadata Through Service And Pipeline

**Files:**
- Modify: `phentrieve/text_processing/full_text_service.py`
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `tests/unit/text_processing/test_full_text_service.py`
- Modify: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write the failing service and pipeline metadata tests**

```python
def test_run_llm_backend_passes_provider_into_factory(mocker) -> None:
    factory = mocker.Mock()
    factory.return_value = _FakeProvider()

    run_llm_backend(
        text="Patient has seizures.",
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
        provider_factory=factory,
    )

    factory.assert_called_once_with(
        llm_model="qwen3.5:35b",
        llm_provider="ollama",
        llm_base_url=None,
    )


def test_pipeline_result_meta_records_provider() -> None:
    provider = _FakeProvider(provider_name="ollama")
    pipeline = TwoPhaseLLMPipeline(provider=provider)

    result = pipeline.run(
        text="Patient has seizures.",
        config=LLMPipelineConfig(
            provider="ollama",
            model="qwen3.5:35b",
            mode="two_phase",
            language="en",
        ),
    )

    assert result.meta.llm_provider == "ollama"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/text_processing/test_full_text_service.py -k "passes_provider_into_factory" -v
pytest tests/unit/llm/test_pipeline.py -k "records_provider" -v
```

Expected:

```text
FAIL ... expected call not found
FAIL ... 'LLMMeta' object has no attribute 'llm_provider'
```

- [ ] **Step 3: Implement minimal service and pipeline propagation**

```python
# phentrieve/text_processing/full_text_service.py
llm_provider = kwargs.get("llm_provider") or os.getenv("PHENTRIEVE_LLM_PROVIDER")
llm_base_url = kwargs.get("llm_base_url") or os.getenv("PHENTRIEVE_LLM_BASE_URL")

provider = provider_factory(
    llm_model=llm_model,
    llm_provider=llm_provider,
    llm_base_url=llm_base_url,
)

config = LLMPipelineConfig(
    provider=getattr(provider, "provider_name", llm_provider or "gemini"),
    model=llm_model,
    base_url=llm_base_url,
    mode=llm_mode,
    language=kwargs.get("language"),
)
```

```python
# phentrieve/llm/pipeline.py
meta = LLMMeta(
    llm_provider=config.provider,
    llm_model=config.model,
    llm_mode=config.mode,
    token_input=token_input,
    token_output=token_output,
    token_usage=token_usage,
    request_count=request_count,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/unit/text_processing/test_full_text_service.py -k "passes_provider_into_factory" -v
pytest tests/unit/llm/test_pipeline.py -k "records_provider" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/text_processing/full_text_service.py phentrieve/llm/pipeline.py tests/unit/text_processing/test_full_text_service.py tests/unit/llm/test_pipeline.py
git commit -m "feat: propagate llm provider metadata"
```

## Task 5: Add CLI Provider Flags And Backward Compatibility

**Files:**
- Modify: `phentrieve/cli/text_commands.py`
- Modify: `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_text_process_passes_provider_and_base_url_to_service(runner, mocker) -> None:
    service = mocker.patch("phentrieve.cli.text_commands.run_full_text_service")

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "--extraction-backend",
            "llm",
            "--llm-provider",
            "ollama",
            "--llm-model",
            "qwen3.5:35b",
            "--llm-base-url",
            "http://localhost:11434",
            "Patient has seizures.",
        ],
    )

    assert result.exit_code == 0
    assert service.call_args.kwargs["llm_provider"] == "ollama"
    assert service.call_args.kwargs["llm_base_url"] == "http://localhost:11434"


def test_text_process_keeps_bare_gemini_model_compatible(runner, mocker) -> None:
    service = mocker.patch("phentrieve.cli.text_commands.run_full_text_service")

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "--extraction-backend",
            "llm",
            "--llm-model",
            "gemini-2.5-flash",
            "Patient has seizures.",
        ],
    )

    assert result.exit_code == 0
    assert service.call_args.kwargs["llm_model"] == "gemini-2.5-flash"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/cli/test_text_commands.py -k "passes_provider_and_base_url or keeps_bare_gemini_model_compatible" -v
```

Expected:

```text
FAIL ... no such option: --llm-provider
```

- [ ] **Step 3: Implement CLI option forwarding**

```python
llm_provider: Annotated[
    str | None,
    typer.Option("--llm-provider", help="LLM provider: gemini, ollama, openai, anthropic."),
] = None,
llm_base_url: Annotated[
    str | None,
    typer.Option("--llm-base-url", help="Optional base URL for Ollama or compatible gateways."),
] = None,
```

```python
service_result = run_full_text_service(
    text=raw_text,
    extraction_backend=extraction_backend,
    language=language,
    llm_provider=llm_provider,
    llm_model=llm_model,
    llm_base_url=llm_base_url,
    llm_mode=llm_mode,
    llm_internal_mode=llm_internal_mode,
    ...
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/unit/cli/test_text_commands.py -k "passes_provider_and_base_url or keeps_bare_gemini_model_compatible" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/cli/text_commands.py tests/unit/cli/test_text_commands.py
git commit -m "feat: add llm provider flags to text cli"
```

## Task 6: Make Benchmark Path Provider-Aware

**Files:**
- Modify: `phentrieve/benchmark/llm_cli.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `tests/unit/cli/test_benchmark_commands.py`
- Modify: `tests/unit/test_llm_benchmark.py`

- [ ] **Step 1: Write the failing benchmark tests**

```python
def test_benchmark_cli_passes_provider_to_runner(runner, mocker) -> None:
    benchmark = mocker.patch("phentrieve.benchmark.llm_cli.run_llm_benchmark_cli")

    result = runner.invoke(
        app,
        [
            "--test-file",
            "tests/fixtures/benchmark.json",
            "--llm-provider",
            "ollama",
            "--llm-model",
            "qwen3.5:35b",
        ],
    )

    assert result.exit_code == 0
    assert benchmark.call_args.kwargs["llm_provider"] == "ollama"


def test_run_llm_benchmark_passes_provider_to_factory(mocker) -> None:
    factory = mocker.patch("phentrieve.benchmark.llm_benchmark.get_llm_provider")
    factory.return_value = _FakeProvider()

    run_llm_benchmark(
        test_file="tests/fixtures/benchmark.json",
        llm_provider="ollama",
        llm_model="qwen3.5:35b",
    )

    factory.assert_called()
    assert factory.call_args.kwargs["llm_provider"] == "ollama"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit/cli/test_benchmark_commands.py -k "passes_provider_to_runner" -v
pytest tests/unit/test_llm_benchmark.py -k "passes_provider_to_factory" -v
```

Expected:

```text
FAIL ... no such option: --llm-provider
FAIL ... unexpected keyword argument 'llm_provider'
```

- [ ] **Step 3: Implement benchmark provider forwarding**

```python
# phentrieve/benchmark/llm_cli.py
llm_provider: Annotated[
    str | None,
    typer.Option("--llm-provider", help="LLM provider to benchmark."),
] = None,
llm_base_url: Annotated[
    str | None,
    typer.Option("--llm-base-url", help="Optional provider base URL for benchmark runs."),
] = None,
```

```python
# phentrieve/benchmark/llm_benchmark.py
def run_llm_benchmark(
    *,
    test_file: str,
    llm_provider: str | None = None,
    llm_model: str,
    llm_base_url: str | None = None,
    ...
) -> dict[str, Any]:
    provider = get_llm_provider(
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        seed=llm_seed,
    )
    config = LLMPipelineConfig(
        provider=getattr(provider, "provider_name", llm_provider or "gemini"),
        model=llm_model,
        base_url=llm_base_url,
        mode=llm_mode,
        language=language,
        seed=llm_seed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/unit/cli/test_benchmark_commands.py -k "passes_provider_to_runner" -v
pytest tests/unit/test_llm_benchmark.py -k "passes_provider_to_factory" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 5: Commit**

```bash
git add phentrieve/benchmark/llm_cli.py phentrieve/benchmark/llm_benchmark.py tests/unit/cli/test_benchmark_commands.py tests/unit/test_llm_benchmark.py
git commit -m "feat: add provider-aware llm benchmark path"
```

## Task 7: Verify Ollama Phase One End-To-End

**Files:**
- Modify: `tests/unit/llm/test_provider.py`
- Modify: `tests/unit/text_processing/test_full_text_service.py`
- Modify: `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Add final Ollama-focused regression tests**

```python
def test_ollama_invalid_json_raises_validation_error(mocker) -> None:
    provider = OllamaStructuredOutputProvider(
        model_name="qwen3.5:35b",
        base_url="http://localhost:11434",
    )
    mocker.patch(
        "httpx.Client.post",
        return_value=_fake_ollama_response(content='{"phenotypes": [}', prompt_eval_count=1, eval_count=1),
    )

    with pytest.raises(Exception):
        provider.run_structured_prompt(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractedPhenotypes,
        )
```

- [ ] **Step 2: Run focused test targets**

Run:

```bash
pytest tests/unit/llm/test_provider.py -v
pytest tests/unit/text_processing/test_full_text_service.py -v
pytest tests/unit/cli/test_text_commands.py -k "llm" -v
pytest tests/unit/cli/test_benchmark_commands.py -k "llm" -v
pytest tests/unit/test_llm_benchmark.py -k "llm" -v
```

Expected:

```text
PASSED
```

- [ ] **Step 3: Run repository verification commands**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected:

```text
All commands exit 0.
```

- [ ] **Step 4: Record an Ollama manual validation command**

Run:

```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:35b",
    "messages": [
      {"role": "user", "content": "Return JSON with keys provider, model, and ready."}
    ],
    "stream": false,
    "format": {
      "type": "object",
      "properties": {
        "provider": {"type": "string"},
        "model": {"type": "string"},
        "ready": {"type": "boolean"}
      },
      "required": ["provider", "model", "ready"],
      "additionalProperties": false
    },
    "options": {"temperature": 0}
  }' | jq -r '.message.content'
```

Expected:

```json
{"provider":"ollama","model":"qwen3.5:35b","ready":true}
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/llm/test_provider.py tests/unit/text_processing/test_full_text_service.py tests/unit/cli/test_text_commands.py tests/unit/cli/test_benchmark_commands.py tests/unit/test_llm_benchmark.py
git commit -m "test: verify ollama-first multi-provider llm path"
```

## Self-Review

- Spec coverage: The plan covers provider-aware types, normalization, Ollama adapter behavior, CLI/service/benchmark propagation, durable metadata, backward compatibility, per-provider timeout/retry ownership, and phase-one Ollama validation. OpenAI, Anthropic, and Gemini registry work beyond phase one are intentionally deferred by the spec's rollout section and not missing from this plan.
- Placeholder scan: No `TODO`, `TBD`, or “similar to task N” references remain. Each task names exact files, test targets, commands, and concrete code snippets.
- Type consistency: The plan consistently uses `llm_provider`, `llm_model`, `llm_base_url`, `LLMPipelineConfig.provider`, `LLMMeta.llm_provider`, and `ResolvedLLMProviderRequest`.
