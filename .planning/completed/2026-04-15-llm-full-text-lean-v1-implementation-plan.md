# Lean V1 LLM Full-Text Processing Implementation Plan

**Status:** Completed
**Completed:** 2026-04-16
**Outcome:** Implemented and stabilized across CLI, API, frontend, benchmarking, and CI verification.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `standard|llm` full-text extraction to the existing `phentrieve` CLI, API, frontend, and benchmark workflows, with a production-only anonymous quota of three successful LLM full-text API analyses per UTC day.

**Architecture:** Selectively port a minimal `phentrieve.llm` foundation, then wrap both the existing retrieval pipeline and the new LLM pipeline behind one shared full-text service that preserves the current response contract. Keep quota enforcement in the FastAPI layer only, trust forwarded client IPs only from configured proxy CIDRs, and extend the current frontend full-text workflow instead of adding a third mode.

**Tech Stack:** Python 3.11, Typer, FastAPI, Pydantic, Vue 3, Vuetify, Vue I18n, SQLite, NGINX, pytest, Vitest, Ruff, mypy, `uv`, Makefile

---

## File Map

- Create: `phentrieve/llm/__init__.py`
- Create: `phentrieve/llm/types.py`
- Create: `phentrieve/llm/provider.py`
- Create: `phentrieve/llm/pipeline.py`
- Create: `phentrieve/llm/prompts/__init__.py`
- Create: `phentrieve/llm/prompts/loader.py`
- Create: `phentrieve/llm/prompts/templates/two_phase_system.txt`
- Create: `phentrieve/llm/prompts/templates/two_phase_user.txt`
- Create: `phentrieve/text_processing/full_text_service.py`
- Create: `phentrieve/benchmark/llm_benchmark.py`
- Create: `phentrieve/benchmark/llm_cli.py`
- Create: `api/llm_quota.py`
- Modify: `phentrieve/cli/text_commands.py`
- Modify: `phentrieve/cli/benchmark_commands.py`
- Modify: `api/config.py`
- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `api/api.yaml`
- Modify: `frontend/src/components/AdvancedOptionsPanel.vue`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Modify: `frontend/src/services/PhentrieveService.js`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/de.json`
- Modify: `frontend/src/locales/fr.json`
- Modify: `frontend/src/locales/nl.json`
- Modify: `frontend/nginx.conf`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `.env.docker.template`
- Modify: `docs/user-guide/api-usage.md`
- Modify: `docs/user-guide/frontend-usage.md`
- Modify: `docs/user-guide/benchmarking-guide.md`
- Test: `tests/unit/llm/test_pipeline.py`
- Test: `tests/unit/text_processing/test_full_text_service.py`
- Test: `tests/unit/cli/test_text_commands.py`
- Test: `tests/unit/api/test_schemas.py`
- Test: `tests/unit/api/test_text_processing_router.py`
- Test: `tests/unit/api/test_llm_quota.py`
- Test: `tests/unit/cli/test_benchmark_commands.py`
- Test: `tests/integration/test_benchmark_workflow.py`
- Test: `frontend/src/test/components/QueryInterface.test.js`
- Test: `frontend/src/test/components/ResultsDisplay.test.js`
- Test: `frontend/src/test/services/PhentrieveService.test.js`

### Task 1: Port the minimal LLM foundation

**Files:**
- Create: `phentrieve/llm/__init__.py`
- Create: `phentrieve/llm/types.py`
- Create: `phentrieve/llm/provider.py`
- Create: `phentrieve/llm/pipeline.py`
- Create: `phentrieve/llm/prompts/__init__.py`
- Create: `phentrieve/llm/prompts/loader.py`
- Create: `phentrieve/llm/prompts/templates/two_phase_system.txt`
- Create: `phentrieve/llm/prompts/templates/two_phase_user.txt`
- Modify: `pyproject.toml`
- Test: `tests/unit/llm/test_pipeline.py`

- [ ] **Step 1: Write the failing LLM foundation tests**

Create `tests/unit/llm/test_pipeline.py` with focused tests for prompt loading, provider invocation, and result normalization:

```python
from pathlib import Path

from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.provider import LLMProvider
from phentrieve.llm.types import LLMPhenotype, LLMPipelineConfig
from phentrieve.llm.prompts.loader import load_prompt_template


class FakeProvider(LLMProvider):
    def __init__(self, payload):
        self.payload = payload

    def run_structured_prompt(self, *, system_prompt: str, user_prompt: str, response_model):
        return response_model.model_validate(self.payload)


def test_load_prompt_template_reads_packaged_template():
    content = load_prompt_template("two_phase_system.txt")
    assert "HPO" in content


def test_two_phase_pipeline_normalizes_structured_terms():
    provider = FakeProvider(
        {
            "terms": [
                {
                    "term_id": "HP:0001250",
                    "label": "Seizure",
                    "evidence": "Patient had recurrent seizures",
                    "assertion": "present",
                }
            ]
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider)

    result = pipeline.run(
        text="Patient had recurrent seizures.",
        config=LLMPipelineConfig(model="gpt-5.4-mini", mode="two_phase"),
    )

    assert result.terms == [
        LLMPhenotype(
            term_id="HP:0001250",
            label="Seizure",
            evidence="Patient had recurrent seizures",
            assertion="present",
        )
    ]
    assert result.meta.llm_mode == "two_phase"
```

- [ ] **Step 2: Run the targeted tests to confirm the module is missing**

Run:

```bash
uv run pytest tests/unit/llm/test_pipeline.py -n 0 -v
```

Expected: FAIL with import errors for `phentrieve.llm`.

- [ ] **Step 3: Implement the minimal typed LLM subsystem**

Create the new modules with a narrow v1 surface:

```python
# phentrieve/llm/types.py
from pydantic import BaseModel, Field


class LLMPhenotype(BaseModel):
    term_id: str
    label: str
    evidence: str | None = None
    assertion: str = "present"


class LLMMeta(BaseModel):
    llm_model: str
    llm_mode: str
    prompt_version: str = "v1"
    token_input: int | None = None
    token_output: int | None = None


class LLMPipelineConfig(BaseModel):
    model: str
    mode: str = "two_phase"
    language: str | None = None


class LLMExtractionResult(BaseModel):
    terms: list[LLMPhenotype] = Field(default_factory=list)
    meta: LLMMeta
```

```python
# phentrieve/llm/provider.py
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model,
    ):
        raise NotImplementedError
```

```python
# phentrieve/llm/pipeline.py
from phentrieve.llm.prompts.loader import load_prompt_template
from phentrieve.llm.types import LLMExtractionResult, LLMMeta


class TwoPhaseLLMPipeline:
    def __init__(self, provider):
        self.provider = provider

    def run(self, *, text: str, config):
        system_prompt = load_prompt_template("two_phase_system.txt")
        user_prompt = load_prompt_template("two_phase_user.txt").format(text=text)
        parsed = self.provider.run_structured_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LLMExtractionResult,
        )
        return parsed.model_copy(
            update={
                "meta": LLMMeta(
                    llm_model=config.model,
                    llm_mode=config.mode,
                )
            }
        )
```

Add prompt loading with `importlib.resources`, and add a new optional dependency group entry in `pyproject.toml` for the initial OpenAI provider SDK instead of wiring provider calls directly into CLI or API code:

```toml
[project.optional-dependencies]
llm = [
    "openai>=2.0.0",
]
```

- [ ] **Step 4: Add the first prompt templates**

Create prompt templates with explicit structured-output intent and no benchmark-specific tuning:

```text
You map clinical text to Human Phenotype Ontology terms.
Return only supported HPO terms from the text.
Do not invent evidence.
```

```text
Clinical text:
{text}
```

Keep the first version intentionally small. Do not port `agentic_judge`, pricing tables, or branch-era benchmark artifacts.

- [ ] **Step 5: Run focused verification**

Run:

```bash
uv run pytest tests/unit/llm/test_pipeline.py -n 0 -v
uv run ruff check phentrieve/llm pyproject.toml tests/unit/llm/test_pipeline.py
```

Expected: PASS for the new LLM unit tests and no Ruff violations in touched files.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml phentrieve/llm tests/unit/llm/test_pipeline.py
git commit -m "feat: add minimal llm pipeline foundation"
```

### Task 2: Add the shared full-text service and stable response adapter

**Files:**
- Create: `phentrieve/text_processing/full_text_service.py`
- Modify: `phentrieve/text_processing/__init__.py`
- Test: `tests/unit/text_processing/test_full_text_service.py`

- [ ] **Step 1: Write the failing shared-service tests**

Create `tests/unit/text_processing/test_full_text_service.py` with one test for backend selection and one for the stable response shape:

```python
from phentrieve.text_processing.full_text_service import FullTextService


def test_full_text_service_uses_standard_backend(mocker):
    standard_backend = mocker.Mock(return_value={"meta": {"extraction_backend": "standard"}})
    llm_backend = mocker.Mock()
    service = FullTextService(
        standard_backend=standard_backend,
        llm_backend=llm_backend,
    )

    result = service.process(text="clinical text", extraction_backend="standard")

    assert result["meta"]["extraction_backend"] == "standard"
    standard_backend.assert_called_once()
    llm_backend.assert_not_called()


def test_full_text_service_llm_response_can_return_empty_chunks(mocker):
    llm_backend = mocker.Mock(
        return_value={
            "meta": {"extraction_backend": "llm", "llm_mode": "two_phase"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [{"id": "HP:0001250", "name": "Seizure"}],
        }
    )
    service = FullTextService(standard_backend=mocker.Mock(), llm_backend=llm_backend)

    result = service.process(text="clinical text", extraction_backend="llm")

    assert result["processed_chunks"] == []
    assert result["meta"]["extraction_backend"] == "llm"
```

- [ ] **Step 2: Run the new tests to verify the service does not exist yet**

Run:

```bash
uv run pytest tests/unit/text_processing/test_full_text_service.py -n 0 -v
```

Expected: FAIL because `full_text_service.py` is missing.

- [ ] **Step 3: Implement the service boundary**

Create `phentrieve/text_processing/full_text_service.py` with a thin orchestration layer and one response adapter:

```python
class FullTextService:
    def __init__(self, *, standard_backend, llm_backend):
        self._standard_backend = standard_backend
        self._llm_backend = llm_backend

    def process(self, *, text: str, extraction_backend: str, **kwargs) -> dict:
        if extraction_backend == "llm":
            return self._llm_backend(text=text, **kwargs)
        return self._standard_backend(text=text, **kwargs)
```

Add adapter helpers that always produce:

```python
{
    "meta": {"extraction_backend": "standard" | "llm"},
    "processed_chunks": [...],
    "aggregated_hpo_terms": [...],
}
```

Map current retrieval results into that shape without changing standard-mode semantics. For LLM mode, either pass through faithful chunks or return `[]`; do not invent synthetic chunks.

- [ ] **Step 4: Wire the current retrieval pipeline into the standard backend adapter**

In the service module, call the existing text-processing pipeline and HPO extraction orchestrator instead of duplicating that logic:

```python
def run_standard_backend(*, text: str, **kwargs) -> dict:
    pipeline_result = existing_text_pipeline(...)
    extraction_result = orchestrate_hpo_extraction(...)
    return adapt_standard_response(pipeline_result, extraction_result)
```

Keep this service as the only new place where backend selection happens.

- [ ] **Step 5: Run focused verification**

Run:

```bash
uv run pytest tests/unit/text_processing/test_full_text_service.py -n 0 -v
uv run ruff check phentrieve/text_processing/full_text_service.py tests/unit/text_processing/test_full_text_service.py
```

Expected: PASS and no Ruff violations in touched files.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/text_processing/__init__.py phentrieve/text_processing/full_text_service.py tests/unit/text_processing/test_full_text_service.py
git commit -m "feat: add shared full-text backend service"
```

### Task 3: Extend the CLI full-text command for `standard|llm`

**Files:**
- Modify: `phentrieve/cli/text_commands.py`
- Test: `tests/unit/cli/test_text_commands.py`

- [ ] **Step 1: Write the failing CLI tests**

Add tests to `tests/unit/cli/test_text_commands.py` that exercise the current Typer command surface with the new option:

```python
from typer.testing import CliRunner

from phentrieve.cli import app


def test_text_process_accepts_llm_backend(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Patient had recurrent seizures.",
            "--extraction-backend",
            "llm",
        ],
    )

    assert result.exit_code == 0
    assert "llm" in result.stdout.lower()
```

- [ ] **Step 2: Run the targeted CLI test**

Run:

```bash
uv run pytest tests/unit/cli/test_text_commands.py -n 0 -k llm_backend -v
```

Expected: FAIL because the new option and call path are not implemented.

- [ ] **Step 3: Add the backend selector to the existing command**

Extend `process_text_for_hpo_command()` in `phentrieve/cli/text_commands.py` with a narrow new option:

```python
extraction_backend: str = typer.Option(
    "standard",
    "--extraction-backend",
    help="Choose full-text extraction backend: standard or llm.",
)
```

Then route both variants through the shared service instead of branching the full pipeline inline:

```python
service_result = run_full_text_service(
    text=input_text,
    extraction_backend=extraction_backend,
    language=language,
    model_name=model_name,
)
```

Reject unsupported backend values with Typer validation. Do not add a separate top-level `phentrieve llm` command.

- [ ] **Step 4: Keep CLI output compatible**

Update the result rendering block to read from the stable response contract:

```python
meta = service_result["meta"]
terms = service_result["aggregated_hpo_terms"]
chunks = service_result["processed_chunks"]
```

For LLM mode, print a small metadata note such as model and mode only if those fields are present. If chunks are empty, skip the chunk preview instead of fabricating one.

- [ ] **Step 5: Run focused verification**

Run:

```bash
uv run pytest tests/unit/cli/test_text_commands.py -n 0 -k "llm_backend or text_process" -v
uv run ruff check phentrieve/cli/text_commands.py tests/unit/cli/test_text_commands.py
```

Expected: PASS for the new CLI coverage and no Ruff violations in touched files.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/cli/text_commands.py tests/unit/cli/test_text_commands.py
git commit -m "feat: add llm full-text backend to cli"
```

### Task 4: Extend the API request/response path without changing the route shape

**Files:**
- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `api/config.py`
- Modify: `api/api.yaml`
- Test: `tests/unit/api/test_schemas.py`
- Test: `tests/unit/api/test_text_processing_router.py`

- [ ] **Step 1: Write failing schema and router tests**

Add one schema test and one router test:

```python
from api.schemas.text_processing_schemas import TextProcessingRequest


def test_text_processing_request_accepts_llm_backend():
    request = TextProcessingRequest.model_validate(
        {
            "text": "Patient had recurrent seizures.",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
            "llm_mode": "two_phase",
        }
    )

    assert request.extraction_backend == "llm"
    assert request.llm_mode == "two_phase"
```

```python
def test_text_processing_router_returns_llm_meta(client, monkeypatch):
    monkeypatch.setattr(
        "api.routers.text_processing_router.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gpt-5.4-mini",
                "llm_mode": "two_phase",
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        },
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Patient had recurrent seizures.",
            "extraction_backend": "llm",
            "llm_model": "gpt-5.4-mini",
            "llm_mode": "two_phase",
        },
    )

    assert response.status_code == 200
    assert response.json()["meta"]["extraction_backend"] == "llm"
```

- [ ] **Step 2: Run the new API tests**

Run:

```bash
uv run pytest tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py -n 0 -k "llm or extraction_backend" -v
```

Expected: FAIL because the schema and router do not expose LLM fields yet.

- [ ] **Step 3: Extend the request schema**

Modify `api/schemas/text_processing_schemas.py` to add lean fields only:

```python
class TextProcessingRequest(BaseModel):
    text: str
    extraction_backend: Literal["standard", "llm"] = "standard"
    llm_model: str | None = None
    llm_mode: Literal["two_phase"] | None = None
```

Do not add auth, account, or billing fields. Keep defaults backward-compatible so old clients continue to work.

- [ ] **Step 4: Route the existing endpoint through the shared service**

In `api/routers/text_processing_router.py`, keep `POST /api/v1/text/process` as the only product endpoint and use the threadpool pattern already present in the file:

```python
service_result = await run_in_threadpool(
    run_full_text_service,
    text=request.text,
    extraction_backend=request.extraction_backend,
    llm_model=request.llm_model,
    llm_mode=request.llm_mode or "two_phase",
)
```

Extend the response metadata only when `extraction_backend == "llm"`. If `processed_chunks` is empty in LLM mode, return it unchanged; the frontend task will handle that gracefully.

- [ ] **Step 5: Name and document the environment controls**

In `api/config.py` and `api/api.yaml`, add the new config fields with defaults that match the approved spec:

```python
PHENTRIEVE_ENV = os.getenv("PHENTRIEVE_ENV", "development")
PHENTRIEVE_TRUSTED_PROXY_CIDRS = os.getenv("PHENTRIEVE_TRUSTED_PROXY_CIDRS", "")
PHENTRIEVE_LLM_DAILY_LIMIT = int(os.getenv("PHENTRIEVE_LLM_DAILY_LIMIT", "3"))
PHENTRIEVE_LLM_QUOTA_DB_PATH = os.getenv(
    "PHENTRIEVE_LLM_QUOTA_DB_PATH",
    "../data/app/llm_quota.db",
)
```

Keep config naming aligned with the spec and avoid inventing a second environment flag.

- [ ] **Step 6: Run focused verification**

Run:

```bash
uv run pytest tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py -n 0 -k "llm or extraction_backend" -v
uv run ruff check api/config.py api/schemas/text_processing_schemas.py api/routers/text_processing_router.py
```

Expected: PASS for the new API tests and no Ruff violations in touched files.

- [ ] **Step 7: Commit**

```bash
git add api/config.py api/api.yaml api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/unit/api/test_schemas.py tests/unit/api/test_text_processing_router.py
git commit -m "feat: add llm full-text support to api route"
```

### Task 5: Add production-only anonymous quota enforcement and proxy-aware subject resolution

**Files:**
- Create: `api/llm_quota.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `frontend/nginx.conf`
- Modify: `.env.example`
- Modify: `.env.docker.template`
- Test: `tests/unit/api/test_llm_quota.py`
- Test: `tests/unit/api/test_text_processing_router.py`

- [ ] **Step 1: Write failing quota tests**

Create `tests/unit/api/test_llm_quota.py` with coverage for the date-derived counter and fail-closed proxy resolution:

```python
from api.llm_quota import DailyQuotaStore, resolve_subject_ip


def test_daily_quota_store_counts_successful_requests_only(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    outcome = store.record_success(subject_key="hash1", usage_date_utc="2026-04-15")
    assert outcome.quota_used == 1
    assert outcome.quota_remaining == 2


def test_resolve_subject_ip_requires_trusted_proxy_for_forwarded_ip():
    subject_ip = resolve_subject_ip(
        client_host="172.18.0.10",
        x_forwarded_for="203.0.113.5",
        trusted_proxy_cidrs=[],
    )

    assert subject_ip is None
```

Extend `tests/unit/api/test_text_processing_router.py` with 429 and fail-closed coverage:

```python
def test_text_processing_router_returns_429_when_quota_exhausted(client, monkeypatch):
    monkeypatch.setattr(
        "api.routers.text_processing_router.check_llm_quota_or_raise",
        lambda *args, **kwargs: (_ for _ in ()).throw(QuotaExceededError(...)),
    )
    response = client.post(
        "/api/v1/text/process",
        json={"text": "note", "extraction_backend": "llm"},
    )
    assert response.status_code == 429
    assert response.json()["quota_remaining"] == 0
```

- [ ] **Step 2: Run the quota-focused tests**

Run:

```bash
uv run pytest tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py -n 0 -k "quota or 429 or trusted_proxy" -v
```

Expected: FAIL because the quota module and router hooks are not implemented.

- [ ] **Step 3: Implement the SQLite-backed quota store**

Create `api/llm_quota.py` with a tiny, self-contained store keyed by subject hash and UTC date:

```python
class DailyQuotaStore:
    def __init__(self, db_path: Path, daily_limit: int):
        self.db_path = db_path
        self.daily_limit = daily_limit

    def record_success(self, *, subject_key: str, usage_date_utc: str) -> QuotaStatus:
        ...

    def get_status(self, *, subject_key: str, usage_date_utc: str) -> QuotaStatus:
        ...
```

Use one SQLite table:

```sql
CREATE TABLE IF NOT EXISTS llm_daily_quota (
    subject_key TEXT NOT NULL,
    usage_date_utc TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (subject_key, usage_date_utc)
)
```

Do not add scheduled resets; derive reset from `usage_date_utc`.

- [ ] **Step 4: Implement trusted subject resolution and router enforcement**

Add helpers shaped like:

```python
def resolve_subject_ip(*, client_host: str | None, x_forwarded_for: str | None, trusted_proxy_cidrs: list[str]) -> str | None:
    ...


def hash_subject_key(subject_ip: str) -> str:
    ...
```

In the router:

```python
if request.extraction_backend == "llm" and settings.phentrieve_env == "production":
    quota_status = check_llm_quota_or_raise(http_request)
```

Apply quota checks only to public API LLM requests. Record usage only after a successful LLM response. When forwarded IP trust cannot be resolved safely, return `503`; this is an environment wiring failure, not malformed client JSON.

- [ ] **Step 5: Wire NGINX and env examples for the quota path**

Update `frontend/nginx.conf` to forward the client IP headers and add coarse rate limiting for the text-processing API:

```nginx
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
limit_req zone=text_process_limit burst=10 nodelay;
```

Document the new env vars in `.env.example` and `.env.docker.template`:

```dotenv
PHENTRIEVE_ENV=development
PHENTRIEVE_TRUSTED_PROXY_CIDRS=172.16.0.0/12,127.0.0.1/32
PHENTRIEVE_LLM_DAILY_LIMIT=3
PHENTRIEVE_LLM_QUOTA_DB_PATH=../data/app/llm_quota.db
```

- [ ] **Step 6: Run focused verification**

Run:

```bash
uv run pytest tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py -n 0 -k "quota or trusted_proxy or 429" -v
uv run ruff check api/llm_quota.py api/routers/text_processing_router.py
```

Expected: PASS for quota coverage and no Ruff violations in touched Python files.

- [ ] **Step 7: Commit**

```bash
git add api/llm_quota.py api/routers/text_processing_router.py frontend/nginx.conf .env.example .env.docker.template tests/unit/api/test_llm_quota.py tests/unit/api/test_text_processing_router.py
git commit -m "feat: enforce production llm full-text quota"
```

### Task 6: Extend the frontend full-text workflow without adding a new mode

**Files:**
- Modify: `frontend/src/components/AdvancedOptionsPanel.vue`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/ResultsDisplay.vue`
- Modify: `frontend/src/services/PhentrieveService.js`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/de.json`
- Modify: `frontend/src/locales/fr.json`
- Modify: `frontend/src/locales/nl.json`
- Test: `frontend/src/test/components/QueryInterface.test.js`
- Test: `frontend/src/test/components/ResultsDisplay.test.js`
- Test: `frontend/src/test/services/PhentrieveService.test.js`

- [ ] **Step 1: Write the failing frontend tests**

Add one UI test, one service test, and one rendering test:

```javascript
it('shows the LLM backend selector only in text-processing mode', async () => {
  const wrapper = mount(QueryInterface, { /* existing test helpers */ })
  await wrapper.setData({ forceTextProcessMode: true })
  expect(wrapper.text()).toContain('LLM extraction')
})
```

```javascript
it('sends extraction_backend=llm in text process requests', async () => {
  apiClient.post = vi.fn().mockResolvedValue({ data: { meta: { extraction_backend: 'llm' } } })
  await PhentrieveService.processText({
    text: 'Patient had recurrent seizures.',
    extractionBackend: 'llm',
    llmModel: 'gpt-5.4-mini',
    llmMode: 'two_phase',
  })
  expect(apiClient.post).toHaveBeenCalledWith(
    '/api/v1/text/process',
    expect.objectContaining({ extraction_backend: 'llm' }),
  )
})
```

```javascript
it('renders quota metadata and tolerates empty processed_chunks', async () => {
  const wrapper = mount(ResultsDisplay, {
    props: {
      results: {
        meta: { extraction_backend: 'llm', quota_remaining: 2, quota_limit: 3 },
        processed_chunks: [],
        aggregated_hpo_terms: [],
      },
    },
  })
  expect(wrapper.text()).toContain('2 / 3')
})
```

- [ ] **Step 2: Run the focused frontend tests**

Run:

```bash
make frontend-test
```

Expected: FAIL in the new LLM-specific assertions before the implementation lands.

- [ ] **Step 3: Add the advanced-options controls and request payload fields**

Modify the existing full-text controls instead of adding a new route or view. Add fields shaped like:

```javascript
textProcessOptions: {
  extractionBackend: 'standard',
  llmModel: 'gpt-5.4-mini',
  llmMode: 'two_phase',
}
```

Render the selector only when full-text mode is active, and only show model/mode controls when backend is `llm`.

- [ ] **Step 4: Keep results rendering compatible with both backends**

Update `ResultsDisplay.vue` so it reads:

```javascript
const extractionBackend = results?.meta?.extraction_backend ?? 'standard'
const processedChunks = results?.processed_chunks ?? []
const quotaRemaining = results?.meta?.quota_remaining
```

Show a concise quota notice only when those fields are present. If `processed_chunks` is empty in LLM mode, hide the chunk evidence section and keep the aggregated phenotype list visible.

- [ ] **Step 5: Add locale strings and structured 429 handling**

Add the new strings to all maintained locale files:

```json
"llmExtraction": "LLM extraction",
"llmLimitedNotice": "Limited production demo: 3 LLM analyses per day.",
"llmQuotaExceeded": "LLM full-text limit reached for today."
```

In `PhentrieveService.js`, preserve the structured quota error fields from `429` responses instead of flattening them away.

- [ ] **Step 6: Run focused verification**

Run:

```bash
make frontend-test
make frontend-lint
make frontend-i18n-check
```

Expected: PASS for the frontend test suite, no lint issues, and no locale-key drift.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/AdvancedOptionsPanel.vue frontend/src/components/QueryInterface.vue frontend/src/components/ResultsDisplay.vue frontend/src/services/PhentrieveService.js frontend/src/locales/en.json frontend/src/locales/de.json frontend/src/locales/fr.json frontend/src/locales/nl.json frontend/src/test/components/QueryInterface.test.js frontend/src/test/components/ResultsDisplay.test.js frontend/src/test/services/PhentrieveService.test.js
git commit -m "feat: add llm full-text controls to frontend"
```

### Task 7: Add the benchmark path, docs, and end-to-end verification

**Files:**
- Create: `phentrieve/benchmark/llm_benchmark.py`
- Create: `phentrieve/benchmark/llm_cli.py`
- Modify: `phentrieve/cli/benchmark_commands.py`
- Modify: `tests/unit/cli/test_benchmark_commands.py`
- Modify: `tests/integration/test_benchmark_workflow.py`
- Modify: `docs/user-guide/api-usage.md`
- Modify: `docs/user-guide/frontend-usage.md`
- Modify: `docs/user-guide/benchmarking-guide.md`

- [ ] **Step 1: Write the failing benchmark tests**

Add one CLI test and one integration-smoke test:

```python
def test_benchmark_group_exposes_llm_subcommand(runner):
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "llm" in result.stdout
```

```python
def test_llm_benchmark_smoke_writes_result_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "phentrieve.benchmark.llm_benchmark.run_llm_benchmark",
        lambda **kwargs: {"cases": 1, "output_path": str(tmp_path / "result.json")},
    )
    result = run_llm_benchmark_cli(...)
    assert result["cases"] == 1
```

- [ ] **Step 2: Run the benchmark-focused tests**

Run:

```bash
uv run pytest tests/unit/cli/test_benchmark_commands.py tests/integration/test_benchmark_workflow.py -n 0 -k "llm or smoke" -v
```

Expected: FAIL because the LLM benchmark command group is not implemented.

- [ ] **Step 3: Add the lean benchmark modules and CLI hook**

Create `phentrieve/benchmark/llm_benchmark.py` with a small entrypoint that accepts a test dataset, backend config, and output path:

```python
def run_llm_benchmark(*, test_file: str, llm_model: str, llm_mode: str = "two_phase") -> dict:
    ...
```

Create `phentrieve/benchmark/llm_cli.py` with a Typer sub-app, then register it in `phentrieve/cli/benchmark_commands.py`:

```python
from phentrieve.benchmark.llm_cli import app as llm_app

app.add_typer(llm_app, name="llm", help="Benchmark LLM full-text extraction.")
```

Make the benchmark code instantiate the LLM pipeline directly and explicitly bypass API quota logic.

- [ ] **Step 4: Document the new product and benchmark flows**

Update the user docs with concrete examples only:

```bash
phentrieve text process --extraction-backend llm --llm-model gpt-5.4-mini note.txt
phentrieve benchmark llm --test-file tests/data/benchmarks/german/tiny_v1.json --llm-model gpt-5.4-mini
```

Document:

- `PHENTRIEVE_ENV`
- `PHENTRIEVE_TRUSTED_PROXY_CIDRS`
- `PHENTRIEVE_LLM_DAILY_LIMIT`
- `PHENTRIEVE_LLM_QUOTA_DB_PATH`

Do not document auth or account setup because it is out of scope.

- [ ] **Step 5: Run the required project verification**

Run:

```bash
make check
make typecheck-fast
make test
make frontend-test
```

Expected: PASS. If any command fails, fix the code or the plan execution before claiming completion.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/benchmark/llm_benchmark.py phentrieve/benchmark/llm_cli.py phentrieve/cli/benchmark_commands.py tests/unit/cli/test_benchmark_commands.py tests/integration/test_benchmark_workflow.py docs/user-guide/api-usage.md docs/user-guide/frontend-usage.md docs/user-guide/benchmarking-guide.md
git commit -m "feat: add llm benchmark workflow and docs"
```
