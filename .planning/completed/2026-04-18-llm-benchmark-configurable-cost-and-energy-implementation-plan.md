# LLM Benchmark Configurable Cost And Energy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable benchmark accounting for token pricing and optional local energy measurement without hardcoded vendor pricing, while preserving backward compatibility for existing benchmark outputs.

**Architecture:** Keep accounting logic in the benchmark layer, not the provider layer. Introduce a typed Pydantic accounting config that can be populated from existing CLI flags and an optional JSON config file, retain `estimated_cost` as a compatibility alias to token pricing, and add a lazy optional CodeCarbon wrapper for run-level and opt-in per-document energy accounting.

**Tech Stack:** Python 3.10, Pydantic, Typer, pytest, Ruff, mypy, optional CodeCarbon integration.

---

## File Map

- Modify: `phentrieve/benchmark/llm_cli.py`
  Add pricing-config and energy CLI options, config-file parsing, and compatibility output handling.
- Modify: `phentrieve/benchmark/llm_benchmark.py`
  Add typed accounting config support, token-cost refactor, compatibility aliasing, and optional energy tracking.
- Add: `phentrieve/benchmark/energy.py`
  Provide a small lazy-import wrapper around CodeCarbon with a no-op fallback.
- Modify: `pyproject.toml`
  Add an optional `energy` extra for CodeCarbon.
- Modify: `uv.lock`
  Refresh dependency lockfile for the new optional extra.
- Add: `phentrieve/benchmark/pricing_examples/gemini.json`
  Example user-editable pricing config file.
- Add: `phentrieve/benchmark/pricing_examples/anthropic.json`
  Example user-editable pricing config file.
- Add: `phentrieve/benchmark/pricing_examples/ollama_local.json`
  Example user-editable local energy config file.
- Modify: `tests/unit/test_llm_benchmark.py`
  Add config precedence, token-accounting, compatibility alias, and energy wrapper tests.
- Modify: `tests/unit/cli/test_benchmark_commands.py`
  Add CLI parsing and config precedence coverage if benchmark CLI tests live there.

### Task 1: Add Typed Accounting Config And Preserve Token-Cost Compatibility

**Files:**
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Test: `tests/unit/test_llm_benchmark.py`

- [ ] **Step 1: Write the failing tests for typed accounting config and compatibility alias**

```python
def test_run_llm_benchmark_emits_token_cost_blocks_and_compat_alias(monkeypatch):
    result = llm_benchmark.run_llm_benchmark(
        test_file="tests/data/en/phenobert",
        llm_model="gemini-2.5-flash",
        input_cost_per_1m_tokens=0.1,
        output_cost_per_1m_tokens=0.4,
    )

    assert result["estimated_token_cost"]["total_cost"] > 0.0
    assert result["estimated_cost"] == result["estimated_token_cost"]


def test_benchmark_accounting_config_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        llm_benchmark.BenchmarkAccountingConfig(
            token_pricing=llm_benchmark.TokenPricingConfig(
                input_cost_per_1m_tokens=-1.0
            )
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "compat_alias or negative_values" -v`
Expected: FAIL because the accounting config models and split cost fields do not exist yet.

- [ ] **Step 3: Add Pydantic accounting config models and a token-cost helper**

```python
class TokenPricingConfig(BaseModel):
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    cached_input_cost_per_1m_tokens: float | None = None

    @field_validator("*")
    @classmethod
    def validate_non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("pricing values must be non-negative")
        return value


class EnergyAccountingConfig(BaseModel):
    measure_energy: bool = False
    per_document_energy: bool = False
    electricity_cost_per_kwh: float | None = None
    carbon_kg_per_kwh: float | None = None
    currency: str | None = None
    country_iso_code: str | None = None
    region: str | None = None


class BenchmarkAccountingConfig(BaseModel):
    token_pricing: TokenPricingConfig = Field(default_factory=TokenPricingConfig)
    energy_accounting: EnergyAccountingConfig = Field(default_factory=EnergyAccountingConfig)
```

- [ ] **Step 4: Refactor `_estimate_cost(...)` into token-accounting output while keeping `estimated_cost`**

```python
estimated_token_cost = _estimate_token_cost(
    token_usage=token_usage,
    pricing=accounting_config.token_pricing,
)

payload["estimated_token_cost"] = estimated_token_cost
payload["estimated_cost"] = estimated_token_cost
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "compat_alias or negative_values or estimate_cost" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/benchmark/llm_benchmark.py tests/unit/test_llm_benchmark.py
git commit -m "feat: add configurable benchmark token accounting"
```

### Task 2: Add CLI Pricing Config Parsing And Precedence

**Files:**
- Modify: `phentrieve/benchmark/llm_cli.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Test: `tests/unit/test_llm_benchmark.py`
- Test: `tests/unit/cli/test_benchmark_commands.py`

- [ ] **Step 1: Write the failing tests for config-file parsing and precedence**

```python
def test_run_llm_benchmark_cli_prefers_cli_pricing_over_file(tmp_path, monkeypatch):
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps(
            {
                "token_pricing": {
                    "input_cost_per_1m_tokens": 0.2,
                    "output_cost_per_1m_tokens": 0.3,
                }
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_run_llm_benchmark(**kwargs):
        captured.update(kwargs)
        return {"cases": 0, "llm_model": kwargs["llm_model"], "llm_mode": kwargs["llm_mode"]}

    monkeypatch.setattr(llm_cli.llm_benchmark, "run_llm_benchmark", fake_run_llm_benchmark)

    llm_cli.run_llm_benchmark_cli(
        test_file=str(tmp_path / "cases.json"),
        llm_model="gemini-2.5-flash",
        pricing_config=str(pricing_path),
        input_cost_per_1m_tokens=0.9,
    )

    assert captured["accounting_config"].token_pricing.input_cost_per_1m_tokens == 0.9
    assert captured["accounting_config"].token_pricing.output_cost_per_1m_tokens == 0.3
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm_benchmark.py tests/unit/cli/test_benchmark_commands.py -k "pricing_over_file or pricing_config" -v`
Expected: FAIL because the CLI does not accept a pricing config file or typed accounting config yet.

- [ ] **Step 3: Add JSON config loading and precedence merging in `phentrieve/benchmark/llm_cli.py`**

```python
def _load_accounting_config(
    *,
    pricing_config_path: str | None,
    input_cost_per_1m_tokens: float | None,
    output_cost_per_1m_tokens: float | None,
    cached_input_cost_per_1m_tokens: float | None,
    measure_energy: bool,
    per_document_energy: bool,
    electricity_cost_per_kwh: float | None,
    carbon_kg_per_kwh: float | None,
    currency: str | None,
) -> BenchmarkAccountingConfig:
    file_payload = {}
    if pricing_config_path is not None:
        file_payload = json.loads(Path(pricing_config_path).read_text(encoding="utf-8"))

    config = BenchmarkAccountingConfig.model_validate(file_payload or {})
    if input_cost_per_1m_tokens is not None:
        config.token_pricing.input_cost_per_1m_tokens = input_cost_per_1m_tokens
    ...
    return config
```

- [ ] **Step 4: Thread `accounting_config` into `run_llm_benchmark()` and add new CLI options**

```python
pricing_config: Annotated[
    str | None,
    typer.Option("--pricing-config", help="Optional JSON pricing/energy config file."),
] = None
```

- [ ] **Step 5: Run the focused tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_benchmark.py tests/unit/cli/test_benchmark_commands.py -k "pricing_over_file or pricing_config or run_llm_benchmark_cli" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/benchmark/llm_cli.py phentrieve/benchmark/llm_benchmark.py tests/unit/test_llm_benchmark.py tests/unit/cli/test_benchmark_commands.py
git commit -m "feat: add benchmark accounting config precedence"
```

### Task 3: Add Optional Energy Tracking Wrapper And Run-Level Accounting

**Files:**
- Add: `phentrieve/benchmark/energy.py`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/unit/test_llm_benchmark.py`

- [ ] **Step 1: Write the failing tests for disabled and unavailable energy tracking**

```python
def test_energy_tracker_returns_unavailable_when_codecarbon_missing(monkeypatch):
    monkeypatch.setattr(llm_benchmark.energy, "create_energy_tracker", lambda *args, **kwargs: llm_benchmark.energy.UnavailableEnergyTracker(reason="codecarbon_not_installed"))
    result = llm_benchmark.run_llm_benchmark(...)
    assert result["estimated_energy_cost"]["measurement_source"] == "unavailable"


def test_run_level_energy_cost_uses_electricity_rate(monkeypatch):
    fake_tracker = FakeEnergyTracker(energy_kwh=0.5, carbon_kg=0.2)
    monkeypatch.setattr(llm_benchmark.energy, "create_energy_tracker", lambda *args, **kwargs: fake_tracker)
    result = llm_benchmark.run_llm_benchmark(..., accounting_config=BenchmarkAccountingConfig(
        energy_accounting=EnergyAccountingConfig(
            measure_energy=True,
            electricity_cost_per_kwh=0.4,
            currency="EUR",
        )
    ))
    assert result["estimated_energy_cost"]["electricity_cost"] == 0.2
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "energy_tracker or estimated_energy_cost" -v`
Expected: FAIL because there is no energy wrapper or energy output path yet.

- [ ] **Step 3: Add a lazy-import energy wrapper in `phentrieve/benchmark/energy.py`**

```python
def create_energy_tracker(config: EnergyAccountingConfig) -> BaseEnergyTracker:
    if not config.measure_energy:
        return DisabledEnergyTracker()
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        return UnavailableEnergyTracker(reason="codecarbon_not_installed")
    return CodeCarbonEnergyTracker(config=config, tracker_cls=OfflineEmissionsTracker)
```

- [ ] **Step 4: Add run-level energy accounting in `phentrieve/benchmark/llm_benchmark.py` and declare the optional extra**

```python
benchmark_energy_tracker = energy.create_energy_tracker(accounting_config.energy_accounting)
benchmark_energy_tracker.start_run()
...
run_energy = benchmark_energy_tracker.stop_run()
payload["estimated_energy_cost"] = _build_energy_cost_payload(run_energy, accounting_config.energy_accounting)
```

In `pyproject.toml`:

```toml
[project.optional-dependencies]
energy = ["codecarbon>=3.2,<4"]
```

- [ ] **Step 5: Refresh the lockfile and run the focused tests**

Run: `uv lock`
Run: `uv run pytest tests/unit/test_llm_benchmark.py -k "energy_tracker or estimated_energy_cost" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add phentrieve/benchmark/energy.py phentrieve/benchmark/llm_benchmark.py pyproject.toml uv.lock tests/unit/test_llm_benchmark.py
git commit -m "feat: add optional benchmark energy accounting"
```

### Task 4: Add Example Config Files, Per-Document Energy Opt-In, And Full Verification

**Files:**
- Add: `phentrieve/benchmark/pricing_examples/gemini.json`
- Add: `phentrieve/benchmark/pricing_examples/anthropic.json`
- Add: `phentrieve/benchmark/pricing_examples/ollama_local.json`
- Modify: `phentrieve/benchmark/llm_benchmark.py`
- Modify: `phentrieve/benchmark/llm_cli.py`
- Test: `tests/unit/test_llm_benchmark.py`
- Test: `tests/unit/cli/test_benchmark_commands.py`

- [ ] **Step 1: Write the failing tests for per-document energy opt-in and example-file-friendly payloads**

```python
def test_prediction_records_only_include_energy_when_enabled(monkeypatch):
    result = llm_benchmark.run_llm_benchmark(
        ...,
        accounting_config=BenchmarkAccountingConfig(
            energy_accounting=EnergyAccountingConfig(measure_energy=True, per_document_energy=False)
        ),
    )
    assert result["prediction_records"][0]["metadata"].get("estimated_energy_cost") is None
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm_benchmark.py tests/unit/cli/test_benchmark_commands.py -k "per_document_energy or pricing_examples" -v`
Expected: FAIL because prediction-record energy opt-in and example-file expectations do not exist yet.

- [ ] **Step 3: Implement per-document energy opt-in and add example config files**

```json
{
  "token_pricing": {
    "input_cost_per_1m_tokens": 0.125,
    "output_cost_per_1m_tokens": 0.75,
    "cached_input_cost_per_1m_tokens": 0.0125
  }
}
```

```json
{
  "energy_accounting": {
    "measure_energy": true,
    "electricity_cost_per_kwh": 0.30,
    "carbon_kg_per_kwh": 0.35,
    "currency": "EUR",
    "country_iso_code": "DEU"
  }
}
```

- [ ] **Step 4: Run targeted tests, then repository verification**

Run: `uv run pytest tests/unit/test_llm_benchmark.py tests/unit/cli/test_benchmark_commands.py -v`
Expected: PASS

Run: `make check`
Expected: PASS

Run: `make typecheck-fast`
Expected: PASS

Run: `make test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add phentrieve/benchmark/llm_cli.py phentrieve/benchmark/llm_benchmark.py phentrieve/benchmark/energy.py phentrieve/benchmark/pricing_examples/*.json tests/unit/test_llm_benchmark.py tests/unit/cli/test_benchmark_commands.py pyproject.toml uv.lock
git commit -m "feat: add configurable benchmark cost and energy reporting"
```

## Self-Review

- Spec coverage:
  - compatibility alias covered in Task 1
  - optional dependency and lazy import covered in Task 3
  - CodeCarbon noise and run-level default covered in Tasks 3 and 4
  - no hardcoded vendor pricing covered by config parsing and example files in Tasks 2 and 4
- Placeholder scan:
  - no `TODO` or deferred implementation markers remain
- Type consistency:
  - `BenchmarkAccountingConfig`, `TokenPricingConfig`, and `EnergyAccountingConfig` are introduced first and reused consistently
