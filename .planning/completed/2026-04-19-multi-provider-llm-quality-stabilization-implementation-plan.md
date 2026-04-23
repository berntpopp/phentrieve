# Multi-Provider LLM Quality Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-parity structured-output recovery, bounded phase-1 fallback, and richer benchmark observability so non-Gemini providers become more reliable on the current two-phase GeneReviews workflow without changing API or frontend behavior.

**Architecture:** Keep the existing two-phase pipeline and native provider adapters, but move structured-output recovery orchestration into the shared `LLMProvider` seam. Layer a bounded phase-1 fallback onto the pipeline using explicit mode transitions (`ungrouped -> grouped_large -> grouped_small`), then expose retry/fallback/failure-class observability through `LLMMeta`, benchmark artifacts, and the benchmark CLI so stabilization work is measurable rather than anecdotal.

**Tech Stack:** Python 3.10, Pydantic, Typer, pytest, Ruff, mypy, provider SDKs already in the `llm` extra.

---

## File Map

- Modify: `phentrieve/llm/provider.py`
  Add shared structured-recovery helpers on `LLMProvider`, provider-specific retry classification hooks, and per-provider retry/failure observability counters.
- Modify: `phentrieve/llm/config.py`
  Add conservative stabilization defaults for structured retries, phase-1 fallback, and grouped-small sizing if new constants are needed.
- Modify: `phentrieve/llm/pipeline.py`
  Add explicit phase-1 mode transitions, bounded fallback behavior, and richer retry/fallback/failure observability in `LLMMeta.trace`, `phase_counts`, and `phase_request_counts`.
- Modify: `phentrieve/llm/types.py`
  Add any small typed metadata support needed for benchmark-facing observability fields while keeping backward compatibility.
- Modify: `phentrieve/benchmark/llm_benchmark.py`
  Persist new retry/fallback/failure metadata into summary and per-document outputs, and normalize failure classes for cross-run comparison.
- Modify: `phentrieve/benchmark/llm_cli.py`
  Add targeted benchmark flags/config plumbing for stabilization-mode comparisons without broad new runtime configuration.
- Modify: `tests/unit/llm/test_provider.py`
  Add provider-parity structured-retry tests, terminal failure tests, and observability accounting checks.
- Modify: `tests/unit/llm/test_pipeline.py`
  Add phase-1 fallback, explicit mode transition, and partial-success grouped fallback tests.
- Modify: `tests/unit/test_llm_benchmark.py`
  Add benchmark artifact assertions for retry counts, fallback flags, mode transitions, and failure classes.
- Modify: `tests/unit/cli/test_benchmark_commands.py`
  Add CLI parsing and plumbing tests for the stabilization comparison flags.
- Modify: `tests/integration/llm/test_grounded_pipeline_integration.py`
  Add focused integration coverage for grouped fallback preserving grounded extraction semantics.
- Optionally modify: `tests/integration/test_benchmark_workflow.py`
  Add artifact-persistence coverage if unit coverage alone misses workflow serialization behavior.

## Task 1: Add Shared Structured-Recovery Parity In The Provider Layer

**Files:**
- Modify: `phentrieve/llm/provider.py`
- Modify: `phentrieve/llm/config.py`
- Test: `tests/unit/llm/test_provider.py`

- [ ] **Step 1: Write the failing provider tests for structured retry parity and terminal failures**

```python
def test_openai_retries_retryable_structured_payload_failure(monkeypatch) -> None:
    fake_responses = _install_fake_openai(
        monkeypatch,
        create_responses=[
            _fake_openai_response(output_text='{"phenotypes": [', input_tokens=10, output_tokens=2),
            _fake_openai_response(output_text='{"phenotypes":[]}', input_tokens=11, output_tokens=3),
        ],
    )
    provider = get_llm_provider(llm_provider="openai", llm_model="gpt-5.4-mini")

    result = provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
    )

    assert result.phenotypes == []
    assert provider.last_request_count == 2
    assert provider.last_usage["total_tokens"] == 26


def test_anthropic_does_not_retry_structured_refusal(monkeypatch) -> None:
    _install_fake_anthropic(
        monkeypatch,
        create_responses=[_fake_anthropic_response(text="", stop_reason="refusal")],
    )
    provider = get_llm_provider(llm_provider="anthropic", llm_model="claude-sonnet-4-6")

    with pytest.raises(RuntimeError, match="refusal|structured"):
        provider.run_structured_prompt(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractedPhenotypes,
        )

    assert provider.last_request_count == 1


def test_ollama_structured_retry_expands_output_budget(monkeypatch) -> None:
    fake_http = _install_fake_ollama_http(
        monkeypatch,
        json_bodies=[
            {"message": {"content": '{"phenotypes": ['}, "prompt_eval_count": 5, "eval_count": 1, "done_reason": "stop"},
            {"message": {"content": '{"phenotypes":[]}'}, "prompt_eval_count": 5, "eval_count": 2, "done_reason": "stop"},
        ],
    )
    provider = get_llm_provider(llm_provider="ollama", llm_model="qwen3:32b")

    provider.run_structured_prompt(
        system_prompt="system",
        user_prompt="user",
        response_model=LLMExtractedPhenotypes,
        max_output_tokens=8192,
    )

    assert fake_http.requests[0]["json"]["options"]["num_predict"] == 8192
    assert fake_http.requests[1]["json"]["options"]["num_predict"] == 16384
```

- [ ] **Step 2: Run the focused provider tests to verify they fail**

Run: `uv run pytest tests/unit/llm/test_provider.py -k "structured_retry or refusal or output_budget" -v`
Expected: FAIL because only Gemini currently has post-parse structured retry behavior.

- [ ] **Step 3: Add conservative stabilization defaults in config**

```python
DEFAULT_PROVIDER_STRUCTURED_RETRIES = 1
DEFAULT_PROVIDER_STRUCTURED_RETRY_TOKEN_MULTIPLIER = 2
DEFAULT_PHASE1_FALLBACK_ENABLED = True
DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS = 2
```

- [ ] **Step 4: Add shared structured-recovery orchestration on `LLMProvider`**

```python
class LLMProvider(ABC):
    ...
    def _run_structured_with_recovery(
        self,
        *,
        invoke: Callable[[int], tuple[Any, int]],
        parse: Callable[[Any], BaseModel],
        initial_output_tokens: int,
        max_output_tokens: int,
        structured_retries: int,
        structured_retry_token_multiplier: int,
    ) -> BaseModel:
        current_output_tokens = initial_output_tokens
        aggregate_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_exception: Exception | None = None

        for attempt in range(1, structured_retries + 2):
            response, request_count = invoke(current_output_tokens)
            self._record_structured_attempt(response=response, request_count=request_count, aggregate_usage=aggregate_usage)
            try:
                return parse(response)
            except Exception as exc:
                last_exception = exc
                if attempt > structured_retries or not self._is_retryable_structured_error(exc):
                    raise
                current_output_tokens = self._next_retry_output_tokens(
                    current_output_tokens,
                    max_output_tokens=max_output_tokens,
                    retry_token_multiplier=structured_retry_token_multiplier,
                )

        assert last_exception is not None
        raise last_exception
```

- [ ] **Step 5: Refactor Gemini to use the shared helper and add provider-specific hooks for Anthropic, OpenAI, and Ollama**

```python
def run_structured_prompt(...):
    response_schema = build_response_json_schema(response_model)
    return self._run_structured_with_recovery(
        invoke=lambda output_tokens: self._create_response(..., output_tokens=output_tokens, schema=response_schema),
        parse=lambda response: self._parse_structured_response(response=response, response_model=response_model),
        initial_output_tokens=max_output_tokens or self.max_tokens,
        max_output_tokens=max(self.max_tokens, DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS),
        structured_retries=self.structured_retries,
        structured_retry_token_multiplier=self.structured_retry_token_multiplier,
    )


def _is_retryable_structured_error(self, exc: Exception) -> bool:
    message = str(exc).lower()
    return "invalid json" in message or "unterminated" in message or "no structured response payload" in message
```

- [ ] **Step 6: Ensure terminal failures do not retry**

```python
def _is_retryable_structured_error(self, exc: Exception) -> bool:
    message = str(exc).lower()
    if "refusal" in message or "billing" in message or "unsupported" in message:
        return False
    ...
```

- [ ] **Step 7: Run the focused provider tests to verify they pass**

Run: `uv run pytest tests/unit/llm/test_provider.py -k "structured_retry or refusal or output_budget" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add phentrieve/llm/config.py phentrieve/llm/provider.py tests/unit/llm/test_provider.py
git commit -m "feat: add provider structured recovery parity"
```

## Task 2: Add Explicit Phase-1 Mode Transitions And Bounded Fallback

**Files:**
- Modify: `phentrieve/llm/pipeline.py`
- Modify: `phentrieve/llm/types.py`
- Modify: `phentrieve/llm/config.py`
- Test: `tests/unit/llm/test_pipeline.py`
- Test: `tests/integration/llm/test_grounded_pipeline_integration.py`

- [ ] **Step 1: Write the failing pipeline tests for mode transitions and fallback**

```python
def test_pipeline_falls_back_from_ungrouped_to_grouped_large_on_retryable_phase1_failure(
    mocker,
) -> None:
    provider = StubStructuredRetryProvider(
        phase1_failures=[LLMPipelinePhaseError("phase1", "Structured extraction failed")],
        grouped_phase1_responses=[
            LLMGroundedExtractedPhenotypes(
                phenotypes=[LLMGroundedExtractedPhenotype(phrase="seizures", category="Abnormal", chunk_ids=[1])]
            )
        ],
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider)

    result = pipeline.run(
        text="Seizures were noted.",
        grounded_chunks=[{"chunk_id": 1, "text": "Seizures were noted."}],
        extraction_groups=[],
        config=LLMPipelineConfig(provider="openai", model="gpt-5.4-mini", mode="two_phase"),
    )

    assert result.meta.trace["phase1"]["initial_mode"] == "ungrouped"
    assert result.meta.trace["phase1"]["final_mode"] == "grouped_large"
    assert result.meta.trace["phase1"]["fallback_triggered"] is True


def test_pipeline_falls_back_from_grouped_large_to_grouped_small(mocker) -> None:
    ...
    assert result.meta.trace["phase1"]["initial_mode"] == "grouped_large"
    assert result.meta.trace["phase1"]["final_mode"] == "grouped_small"


def test_pipeline_records_terminal_phase1_failure_without_fallback(mocker) -> None:
    ...
    assert result.meta.trace["phase1"]["failure_class"] == "structured_refusal"
    assert result.meta.trace["phase1"]["fallback_triggered"] is False
```

- [ ] **Step 2: Run the focused pipeline tests to verify they fail**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "fallback or initial_mode or final_mode or failure_class" -v`
Expected: FAIL because phase-1 fallback and explicit mode/failure metadata do not exist yet.

- [ ] **Step 3: Add explicit phase-1 mode helpers and fallback classification in the pipeline**

```python
def _classify_phase1_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "refusal" in message:
        return "structured_refusal"
    if "timeout" in message or "deadline_exceeded" in message:
        return "provider_timeout"
    if "json" in message:
        return "structured_json_invalid"
    return "structured_schema_validation_failed"


def _next_phase1_mode(current_mode: str) -> str | None:
    if current_mode == "ungrouped":
        return "grouped_large"
    if current_mode == "grouped_large":
        return "grouped_small"
    return None
```

- [ ] **Step 4: Teach `TwoPhaseLLMPipeline.run()` to attempt bounded phase-1 fallback**

```python
phase1_initial_mode = self._resolve_initial_phase1_mode(extraction_groups=extraction_groups)
phase1_final_mode = phase1_initial_mode
phase1_fallback_triggered = False

try:
    extracted, phase1_usage, phase1_request_count, phase1_elapsed, phase1_groups_trace = self._run_phase1_mode(...)
except LLMPipelinePhaseError as exc:
    failure_class = _classify_phase1_failure(exc)
    next_mode = self._next_phase1_mode(phase1_initial_mode)
    if self._should_fallback_phase1(exc=exc, next_mode=next_mode):
        phase1_fallback_triggered = True
        phase1_final_mode = next_mode
        extracted, phase1_usage, phase1_request_count, phase1_elapsed, phase1_groups_trace = self._run_phase1_mode(...)
    else:
        raise
```

- [ ] **Step 5: Record mode transitions and failure metadata into `LLMMeta.trace` and counts**

```python
trace["phase1"].update(
    {
        "initial_mode": phase1_initial_mode,
        "final_mode": phase1_final_mode,
        "fallback_triggered": phase1_fallback_triggered,
        "failure_class": failure_class,
    }
)
phase_counts["phase1_fallbacks"] = int(phase1_fallback_triggered)
```

- [ ] **Step 6: Add grouped-small behavior using smaller extraction groups**

```python
def _build_small_extraction_groups(self, grounded_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_extraction_groups(
        grounded_chunks=[GroundedChunk(**chunk) for chunk in grounded_chunks],
        max_chunks_per_group=DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS,
    )
```

- [ ] **Step 7: Run the focused unit and integration tests to verify they pass**

Run: `uv run pytest tests/unit/llm/test_pipeline.py -k "fallback or initial_mode or final_mode or failure_class" -v`
Expected: PASS

Run: `uv run pytest tests/integration/llm/test_grounded_pipeline_integration.py -k "fallback or grouped" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add phentrieve/llm/config.py phentrieve/llm/pipeline.py phentrieve/llm/types.py tests/unit/llm/test_pipeline.py tests/integration/llm/test_grounded_pipeline_integration.py
git commit -m "feat: add bounded phase1 fallback"
```

## Task 3: Persist Retry, Fallback, And Failure-Class Observability In Benchmark Outputs

**Files:**
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `tests/unit/test_llm_benchmark.py`
- Optionally modify: `tests/integration/test_benchmark_workflow.py`

- [ ] **Step 1: Write the failing benchmark serialization tests**

```python
def test_benchmark_persists_phase1_mode_transition_and_retry_counts() -> None:
    result = _run_stub_benchmark_with_meta(
        LLMMeta(
            llm_provider="openai",
            llm_model="gpt-5.4-mini",
            llm_mode="two_phase",
            trace={
                "phase1": {
                    "initial_mode": "ungrouped",
                    "final_mode": "grouped_large",
                    "fallback_triggered": True,
                    "failure_class": None,
                }
            },
            phase_counts={"phase1_fallbacks": 1},
            phase_request_counts={"phase1_requests": 2, "phase2b_llm_requests": 1},
            token_usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
    )

    observability = result["results"][0]["observability"]
    assert observability["phase1_initial_mode"] == "ungrouped"
    assert observability["phase1_final_mode"] == "grouped_large"
    assert observability["phase1_fallback_triggered"] is True


def test_benchmark_persists_terminal_failure_class() -> None:
    ...
    assert result["results"][0]["failure_class"] == "structured_refusal"
```

- [ ] **Step 2: Run the focused benchmark tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "phase1_initial_mode or phase1_final_mode or failure_class or fallback_triggered" -v`
Expected: FAIL because these fields are not persisted yet.

- [ ] **Step 3: Extend benchmark observability extraction from pipeline meta**

```python
phase1_trace = trace.get("phase1", {}) if isinstance(trace, dict) else {}
observability.update(
    {
        "phase1_initial_mode": phase1_trace.get("initial_mode"),
        "phase1_final_mode": phase1_trace.get("final_mode"),
        "phase1_fallback_triggered": bool(phase1_trace.get("fallback_triggered", False)),
        "failure_class": phase1_trace.get("failure_class"),
    }
)
```

- [ ] **Step 4: Normalize failure-class reporting in failed benchmark records**

```python
if isinstance(exc, LLMPipelinePhaseError):
    failure_class = _extract_failure_class_from_trace_or_exception(...)
    record["failure_class"] = failure_class
    record["error_phase"] = exc.phase
    record["error_message"] = str(exc)
```

- [ ] **Step 5: Run the focused benchmark tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "phase1_initial_mode or phase1_final_mode or failure_class or fallback_triggered" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/benchmark/llm_benchmark.py tests/unit/test_llm_benchmark.py tests/integration/test_benchmark_workflow.py
git commit -m "feat: persist llm stabilization observability"
```

## Task 4: Add Benchmark Comparison Flags And Final Validation Coverage

**Files:**
- Modify: `phentrieve/benchmark/llm_cli.py`
- Modify: `tests/unit/cli/test_benchmark_commands.py`
- Modify: `tests/unit/test_llm_benchmark.py`

- [ ] **Step 1: Write the failing CLI plumbing tests for stabilization comparison flags**

```python
def test_benchmark_cli_loads_stabilization_flags_into_run_call(mocker) -> None:
    run_mock = mocker.patch("phentrieve.benchmark.llm_cli.llm_benchmark.run_llm_benchmark")
    run_mock.return_value = {"status": "completed", "results": [], "metrics": {}}

    run_llm_benchmark_cli(
        test_file="tests/data/en/phenobert",
        llm_model="gpt-5.4-mini",
        llm_provider="openai",
        enable_stabilization=True,
        phase1_fallback=True,
        structured_retries=1,
    )

    kwargs = run_mock.call_args.kwargs
    assert kwargs["enable_stabilization"] is True
    assert kwargs["phase1_fallback"] is True
    assert kwargs["structured_retries"] == 1
```

- [ ] **Step 2: Run the focused CLI tests to verify they fail**

Run: `uv run pytest tests/unit/cli/test_benchmark_commands.py -k "stabilization" -v`
Expected: FAIL because the CLI does not yet expose or forward stabilization flags.

- [ ] **Step 3: Add targeted benchmark CLI flags and forwarding**

```python
def run_llm_benchmark_cli(
    *,
    ...
    enable_stabilization: bool = False,
    phase1_fallback: bool = False,
    structured_retries: int | None = None,
) -> dict[str, Any]:
    ...
    result = llm_benchmark.run_llm_benchmark(
        ...,
        enable_stabilization=enable_stabilization,
        phase1_fallback=phase1_fallback,
        structured_retries=structured_retries,
    )
```

- [ ] **Step 4: Add end-to-end unit coverage for default conservative behavior**

```python
def test_benchmark_defaults_do_not_enable_extra_stabilization_flags(mocker) -> None:
    ...
    assert kwargs["enable_stabilization"] is False
    assert kwargs["phase1_fallback"] is False
    assert kwargs["structured_retries"] is None
```

- [ ] **Step 5: Run the focused CLI tests to verify they pass**

Run: `uv run pytest tests/unit/cli/test_benchmark_commands.py -k "stabilization" -v`
Expected: PASS

- [ ] **Step 6: Run verification**

Run: `make check`
Expected: PASS

Run: `make typecheck-fast`
Expected: PASS

Run: `make test`
Expected: PASS

If `make test` hits the known unrelated parallel CUDA OOM issue again, also run:

Run: `CUDA_VISIBLE_DEVICES=\"\" make test`
Expected: PASS

- [ ] **Step 7: Run bounded live validation**

Run: `uv run phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews --llm-provider gemini --llm-model gemini-3.1-flash-lite-preview --enable-stabilization --phase1-fallback`
Expected: completed benchmark artifact with no regression relative to the existing Gemini baseline.

Run: `uv run phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews --llm-provider anthropic --llm-model claude-sonnet-4-6 --enable-stabilization --phase1-fallback`
Expected: fewer hard phase-1 failures than the unstabilized path, if failures were previously present.

Run: `uv run phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews --llm-provider openai --llm-model gpt-5.4-mini --enable-stabilization --phase1-fallback`
Expected: completed benchmark artifact with populated retry/fallback observability fields.

Run: `uv run phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews --llm-provider ollama --llm-model qwen3:32b --llm-base-url http://localhost:11434 --enable-stabilization --phase1-fallback --llm-timeout-seconds 900`
Expected: completed benchmark artifact with retry/fallback observability and no regression in prior Ollama timeout handling.

- [ ] **Step 8: Commit**

```bash
git add phentrieve/benchmark/llm_cli.py tests/unit/cli/test_benchmark_commands.py tests/unit/test_llm_benchmark.py
git commit -m "feat: add llm stabilization benchmark controls"
```

## Self-Review Notes

- Spec coverage:
  - structured retry parity is covered in Task 1
  - bounded phase-1 fallback and explicit mode enum are covered in Task 2
  - observability and failure-class persistence are covered in Task 3
  - benchmark comparison controls and live validation are covered in Task 4
- Placeholder scan:
  - no `TODO`, `TBD`, or cross-task shorthand placeholders remain
- Type consistency:
  - phase-1 mode names are used consistently as `ungrouped`, `grouped_large`, and `grouped_small`
  - failure class names are referenced consistently in pipeline and benchmark tasks
