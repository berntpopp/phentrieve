# Maintainability Quick Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve maintainability quickly by hardening API model-loading boundaries, making CI authoritative, correcting test metadata drift, and aligning developer docs with the current workflow.

**Architecture:** The work is split into four small workstreams with low file overlap. Each task restores a source of truth at a repository boundary: request schema, CI gate, pytest metadata, or developer docs. The plan avoids large refactors and adds only targeted tests around the intended changes.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, mypy, GitHub Actions, Markdown docs, `uv`, Makefile

---

## File Map

- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `api/dependencies.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `tests/unit/cli/test_benchmark_integration.py`
- Modify: `docs/development/running-tests.md`
- Modify: `docs/development/dev-environment.md`
- Modify: `docs/getting-started/installation.md`
- Create or modify: targeted API tests under `tests/unit/api/`

## Task 1: Harden text-processing model-loading inputs

**Files:**
- Modify: `api/schemas/text_processing_schemas.py`
- Modify: `api/routers/text_processing_router.py`
- Test: `tests/unit/api/test_text_processing_router.py`

- [ ] **Step 1: Read the current schema and router paths**

Run:

```bash
sed -n '1,140p' api/schemas/text_processing_schemas.py
sed -n '240,380p' api/routers/text_processing_router.py
```

Expected: `TextProcessingRequest` includes `trust_remote_code`, and `_process_text_internal()` passes `request.trust_remote_code or False` into `get_sbert_model_dependency()`.

- [ ] **Step 2: Write the failing tests for rejected model names and preserved zero values**

Create or update `tests/unit/api/test_text_processing_router.py` with tests shaped like:

```python
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_text_processing_rejects_non_allowlisted_retrieval_model():
    response = client.post(
        "/api/v1/process-text",
        json={
            "text_content": "Short clinical note",
            "retrieval_model_name": "attacker/custom-model",
        },
    )

    assert response.status_code == 400
    assert "retrieval_model_name" in response.text


def test_text_processing_preserves_explicit_zero_threshold(monkeypatch):
    captured: dict[str, float] = {}

    def fake_orchestrate_hpo_extraction(**kwargs):
        captured["chunk_retrieval_threshold"] = kwargs["chunk_retrieval_threshold"]
        captured["min_confidence_for_aggregated"] = kwargs[
            "min_confidence_for_aggregated"
        ]
        return [], []

    monkeypatch.setattr(
        "api.routers.text_processing_router.orchestrate_hpo_extraction",
        fake_orchestrate_hpo_extraction,
    )

    response = client.post(
        "/api/v1/process-text",
        json={
            "text_content": "Short clinical note",
            "chunk_retrieval_threshold": 0.0,
            "aggregated_term_confidence": 0.0,
        },
    )

    assert response.status_code == 200
    assert captured["chunk_retrieval_threshold"] == 0.0
    assert captured["min_confidence_for_aggregated"] == 0.0
```

- [ ] **Step 3: Run the targeted tests to confirm they fail**

Run:

```bash
uv run pytest tests/unit/api/test_text_processing_router.py -n 0 -v
```

Expected: FAIL because the route still accepts arbitrary model names and still uses `or` fallback semantics.

- [ ] **Step 4: Implement a server-owned allowlist and remove request-level trust control**

Edit `api/schemas/text_processing_schemas.py` so `TextProcessingRequest` no longer exposes:

```python
    trust_remote_code: bool | None = Field(
        default=False,
        description="Trust remote code when loading models from Hugging Face Hub (use with caution).",
    )
```

Edit `api/routers/text_processing_router.py` so the model boundary follows this shape:

```python
ALLOWED_TEXT_PROCESSING_MODELS = {
    DEFAULT_MODEL,
}


def _validate_model_name(field_name: str, model_name: str | None) -> str:
    effective_name = model_name or DEFAULT_MODEL
    if effective_name not in ALLOWED_TEXT_PROCESSING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {field_name}: {effective_name}",
        )
    return effective_name
```

Then replace:

```python
retrieval_model_name_to_load = request.retrieval_model_name or DEFAULT_MODEL
sbert_for_chunking_name_to_load = (
    request.semantic_model_name or retrieval_model_name_to_load
)
```

with explicit validation:

```python
retrieval_model_name_to_load = _validate_model_name(
    "retrieval_model_name", request.retrieval_model_name
)
sbert_for_chunking_name_to_load = _validate_model_name(
    "semantic_model_name",
    request.semantic_model_name or retrieval_model_name_to_load,
)
```

Also replace:

```python
trust_remote_code=request.trust_remote_code or False,
```

with:

```python
trust_remote_code=False,
```

- [ ] **Step 5: Replace `or` numeric fallback with explicit `is not None` handling**

In `api/routers/text_processing_router.py`, replace this pattern:

```python
chunk_retrieval_threshold=request.chunk_retrieval_threshold
or DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
min_confidence_for_aggregated=request.aggregated_term_confidence
or DEFAULT_MIN_CONFIDENCE_AGGREGATED,
```

with:

```python
chunk_retrieval_threshold=(
    request.chunk_retrieval_threshold
    if request.chunk_retrieval_threshold is not None
    else DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
),
min_confidence_for_aggregated=(
    request.aggregated_term_confidence
    if request.aggregated_term_confidence is not None
    else DEFAULT_MIN_CONFIDENCE_AGGREGATED
),
```

Use the same explicit pattern for any other threshold or count fields in the same path where `0` or `0.0` is schema-valid.

- [ ] **Step 6: Run targeted verification**

Run:

```bash
uv run pytest tests/unit/api/test_text_processing_router.py -n 0 -v
uv run ruff check api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/unit/api/test_text_processing_router.py
```

Expected: PASS for the targeted tests and no Ruff issues in the touched files.

- [ ] **Step 7: Commit**

```bash
git add api/schemas/text_processing_schemas.py api/routers/text_processing_router.py tests/unit/api/test_text_processing_router.py
git commit -m "fix(api): harden text processing model loading"
```

## Task 2: Make CI authoritative for typing and pytest collection

**Files:**
- Modify: `api/dependencies.py`
- Modify: `.github/workflows/ci.yml`
- Test: local `mypy` and `pytest --collect-only` runs

- [ ] **Step 1: Confirm the existing failures and advisory CI behavior**

Run:

```bash
uv run mypy phentrieve api
grep -n "Run mypy type checking" -A4 .github/workflows/ci.yml
```

Expected: mypy reports the unused ignore in `api/dependencies.py`, and the CI workflow shows `continue-on-error: true` on the mypy step.

- [ ] **Step 2: Remove the unused ignore and keep the import typed cleanly**

In `api/dependencies.py`, replace:

```python
from cachetools import TTLCache  # type: ignore[import-untyped]
```

with:

```python
from cachetools import TTLCache
```

- [ ] **Step 3: Add a collection-only pytest smoke step and make mypy blocking**

In `.github/workflows/ci.yml`, replace the mypy step block:

```yaml
      - name: Run mypy type checking
        run: |
          uv run mypy phentrieve/ api/
        continue-on-error: true
```

with:

```yaml
      - name: Run mypy type checking
        run: |
          uv run mypy phentrieve/ api/

      - name: Validate pytest collection
        run: |
          uv run pytest tests/ --collect-only -q
```

Keep the collection step before the full pytest execution so marker and import errors fail earlier.

- [ ] **Step 4: Run local verification before editing anything else**

Run:

```bash
uv run mypy phentrieve api
uv run pytest tests/ --collect-only -q
```

Expected: both commands pass locally.

- [ ] **Step 5: Commit**

```bash
git add api/dependencies.py .github/workflows/ci.yml
git commit -m "ci: enforce mypy and pytest collection checks"
```

## Task 3: Correct benchmark marker drift and contradictory test labeling

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/unit/cli/test_benchmark_integration.py`
- Test: targeted benchmark test collection

- [ ] **Step 1: Inspect current marker declarations and benchmark test metadata**

Run:

```bash
sed -n '175,220p' pyproject.toml
sed -n '1,40p' tests/unit/cli/test_benchmark_integration.py
```

Expected: `pytest.mark.benchmark` is used in the benchmark test file but `benchmark` is not declared in `pyproject.toml`, and the file is marked both `unit` and `integration`.

- [ ] **Step 2: Write the smallest truthful metadata fix**

Update the marker declaration block in `pyproject.toml` to include:

```toml
markers = [
    "unit: Fast unit tests (mocked dependencies)",
    "integration: Integration tests (real dependencies)",
    "benchmark: Benchmark dataset and evaluation tests",
    "e2e: End-to-end Docker tests",
    "slow: Slow tests (>5s)",
]
```

Update `tests/unit/cli/test_benchmark_integration.py` so the file-level marker matches its behavior. Replace:

```python
pytestmark = [pytest.mark.unit, pytest.mark.integration, pytest.mark.benchmark]
```

with:

```python
pytestmark = [pytest.mark.integration, pytest.mark.benchmark]
```

- [ ] **Step 3: Verify marker truthfulness through collection and targeted execution**

Run:

```bash
uv run pytest tests/unit/cli/test_benchmark_integration.py --collect-only -q
uv run pytest tests/unit/cli/test_benchmark_integration.py -n 0 -v
```

Expected: collection succeeds without unknown-marker errors, and the benchmark integration tests pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/unit/cli/test_benchmark_integration.py
git commit -m "test: fix benchmark marker metadata"
```

## Task 4: Align developer docs with the actual workflow

**Files:**
- Modify: `docs/development/running-tests.md`
- Modify: `docs/development/dev-environment.md`
- Modify: `docs/getting-started/installation.md`
- Optional modify: `README.md` if it duplicates outdated commands

- [ ] **Step 1: Confirm the stale instructions to replace**

Run:

```bash
sed -n '1,220p' docs/development/running-tests.md
sed -n '150,260p' docs/development/dev-environment.md
sed -n '60,140p' docs/getting-started/installation.md
```

Expected: docs still mention manual `venv` activation, direct `pytest` invocation as the primary path, `npm install`, and an outdated CI workflow filename.

- [ ] **Step 2: Rewrite `docs/development/running-tests.md` around current commands**

Replace old examples like:

```bash
source venv/bin/activate
pytest
pytest tests/unit/
pytest --cov=phentrieve
```

with a concise command surface built around the repository standard:

```bash
make test
make check
make typecheck-fast
uv run pytest tests/unit/api/ -n 0 -v
uv run pytest tests/ --collect-only -q
```

Also update the CI reference to:

```text
.github/workflows/ci.yml
```

- [ ] **Step 3: Rewrite the development environment and installation pages to match `uv` + Makefile**

In `docs/development/dev-environment.md`, replace `npm install` with:

```bash
make frontend-install
```

and prefer:

```bash
make install-dev
make dev-api
make dev-frontend
```

In `docs/getting-started/installation.md`, make the primary verification flow:

```bash
make install-dev
uv run phentrieve --version
```

Keep any `pip` path, if retained at all, as a short fallback note rather than a first-class workflow.

- [ ] **Step 4: Verify every command mentioned in the edited docs exists**

Run:

```bash
rg -n "frontend-install|install-dev|typecheck-fast|dev-api|dev-frontend|frontend-test|frontend-build" Makefile
uv run phentrieve --version
```

Expected: each documented command exists, and the CLI version command works through `uv run`.

- [ ] **Step 5: Lint the touched docs for obvious path drift**

Run:

```bash
rg -n "venv/bin/activate|pytest.ini|tests.yml|npm install|plan/" docs README.md
```

Expected: no remaining primary-path references to obsolete workflow commands or outdated planning paths in the files you updated.

- [ ] **Step 6: Commit**

```bash
git add docs/development/running-tests.md docs/development/dev-environment.md docs/getting-started/installation.md
git commit -m "docs: align developer workflow documentation"
```

## Final Verification

**Files:**
- Verify all files touched in Tasks 1-4

- [ ] **Step 1: Run the repository-standard checks**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: all three commands pass locally and reflect the same standards CI now enforces.

- [ ] **Step 2: Summarize remaining follow-up work**

Record the deferred items in the PR description or execution notes:

```text
- Broader test tree reclassification remains for a separate plan.
- Config centralization remains for a separate plan.
- Backend/frontend hotspot decomposition remains for a separate plan.
```

- [ ] **Step 3: Commit any final verification-only adjustments**

```bash
git status --short
```

Expected: clean working tree, or only intended uncommitted notes remain.
