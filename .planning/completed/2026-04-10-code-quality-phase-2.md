# Code Quality Phase 2 — PR #191 Extension

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all remaining open findings from the 2026-04-09 code quality review (critical findings 2, 4-residual, 6, plus the `ErrorResponse` schema + `visualization/` orphan) by extending PR #191's `improve/code-quality-2026-04` branch with 10 atomic commits.

**Architecture:** Follow the proven Stream A pattern from PR #191 — characterization tests first to lock current behavior, decomposition second, verification third. No new worktree: extend the existing branch directly. Commit atomically per concern so `git bisect` remains useful. Every behavior-changing task gets a characterization test *before* the change.

**Tech Stack:** Python 3.10+, pytest, Pydantic v2, FastAPI, mypy (daemon), ruff, cachetools.

**Branch:** Work directly on `improve/code-quality-2026-04` (already checked out). No worktree.

**Ground truth verified 2026-04-10:**
- `phentrieve/visualization/plot_utils.py` (357 LOC) has **zero callers** in live code (only referenced by the mypy cache).
- `api/main.py:11` still has `sys.path.append(os.path.dirname(...))`.
- `tests/unit/api/test_dependencies_char.py:16` and `tests/unit/api/test_main_char.py:15` mirror that hack.
- `phentrieve/text_processing/hpo_extraction_orchestrator.py` is 298 LOC, 9% coverage, untouched by PR #191.
- `phentrieve/text_processing/pipeline.py::_create_chunkers()` spans lines 91-249 and has a real duplicate: `sliding_window` handled at line 128 AND again at line 207-234 (unreachable for `sliding_window` because the first match wins — only `sliding_window_semantic` reaches the second block).
- `phentrieve/text_processing/chunkers.py::FinalChunkCleaner.__init__()` spans lines 80-214 with 3x repeated if-custom-else-load-resource patterns at lines 126-154, 172-185.
- No `ErrorResponse` Pydantic model exists in `api/schemas/`.

---

## File Structure

### Files to delete

| Path | Reason |
|---|---|
| `phentrieve/visualization/plot_utils.py` | Orphaned after reranker removal; no callers |
| `phentrieve/visualization/__init__.py` | Package is now empty |

### Files to create

| Path | Responsibility |
|---|---|
| `api/schemas/errors.py` | `ErrorResponse` Pydantic model — single source of truth for API error payload shape |
| `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py` | Characterization tests locking `orchestrate_hpo_extraction` behavior before refactor |
| `tests/unit/text_processing/test_pipeline_create_chunkers_char.py` | Characterization tests locking `TextProcessingPipeline._create_chunkers` behavior before refactor |
| `tests/unit/api/test_error_response_schema.py` | Tests verifying the new `ErrorResponse` schema + exception handler |
| `phentrieve/text_processing/_hpo_extraction_helpers.py` | Extracted helpers from `hpo_extraction_orchestrator.py` (chunk processing, batch DB load, evidence assembly, aggregation) |
| `phentrieve/text_processing/_chunker_registry.py` | Registry-based factory replacing the if/elif chain in `_create_chunkers` |

### Files to modify

| Path | Change |
|---|---|
| `api/main.py` | Remove `sys.path.append`; add global exception handler using `ErrorResponse` |
| `api/dependencies.py` | No change (already hardened in PR #191, but we exercise it) |
| `api/schemas/__init__.py` | Export `ErrorResponse` |
| `api/routers/query_router.py` | Add `responses={...}` metadata for OpenAPI error docs |
| `api/routers/text_processing_router.py` | Same |
| `api/routers/similarity_router.py` | Same |
| `api/routers/config_info_router.py` | Same |
| `api/routers/health.py` | Same (if HTTPException raising sites exist) |
| `tests/unit/api/test_dependencies_char.py` | Remove `sys.path.insert` |
| `tests/unit/api/test_main_char.py` | Remove `sys.path.insert` |
| `phentrieve/text_processing/hpo_extraction_orchestrator.py` | Replace god function body with calls into `_hpo_extraction_helpers` |
| `phentrieve/text_processing/pipeline.py` | Replace `_create_chunkers` with registry lookup |
| `phentrieve/text_processing/chunkers.py` | Extract `_load_language_resource_with_custom` helper in `FinalChunkCleaner.__init__` |
| `docs/architecture/caching.md` | New: document `@lru_cache` vs `TTLCache` decisions |

---

## Task 1: Delete orphaned `visualization/` module

**Rationale:** `plot_utils.py` (357 LOC) was the reranker-era benchmark plotting code. After PR #191 removed the reranker feature entirely, no code path references any function in this module. Verified via `grep -rn "plot_mrr_comparison\|plot_metric_at_k" phentrieve/ tests/ api/` → only `.mypy_cache` matches (not live code).

**Files:**
- Delete: `phentrieve/visualization/plot_utils.py`
- Delete: `phentrieve/visualization/__init__.py`

- [ ] **Step 1: Final verification that no live code imports from `phentrieve.visualization`**

Run:
```bash
grep -rn "from phentrieve.visualization\|import phentrieve.visualization\|from phentrieve\.visualization\|plot_mrr_comparison\|plot_metric_at_k_bars\|plot_metric_at_k_lines" phentrieve/ tests/ api/ 2>&1 | grep -v "__pycache__\|\.mypy_cache"
```

Expected: **no output** (empty grep result).

- [ ] **Step 2: Delete the files**

Run:
```bash
git rm phentrieve/visualization/plot_utils.py phentrieve/visualization/__init__.py
rmdir phentrieve/visualization
```

- [ ] **Step 3: Verify no mypy error about missing module**

Run:
```bash
make typecheck-fast
```

Expected: `Success: no issues found in 91 source files` (one fewer than before).

- [ ] **Step 4: Run the full Python test suite**

Run:
```bash
make test
```

Expected: `899 passed` (no change in test count — the module had no tests). Coverage goes up slightly because 149 uncovered statements disappear from the denominator.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: delete orphaned visualization/ module

The entire phentrieve/visualization/ package was reranker-era benchmark
plotting code. After PR #191 removed the cross-encoder reranker, no
code path references plot_mrr_comparison, plot_metric_at_k_bars, or
plot_metric_at_k_lines anywhere in phentrieve/, api/, or tests/.

Deletes:
- phentrieve/visualization/plot_utils.py (357 LOC, 0% coverage)
- phentrieve/visualization/__init__.py
- phentrieve/visualization/ (now empty)

Closes "visualization dead code" open finding from the updated
CODE-QUALITY-REVIEW-2026-04-09.md (Priority 1, item 1).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Remove `api/main.py` sys.path hack

**Rationale:** `api/main.py:11` mutates `sys.path` at import time to allow `api/` to find `phentrieve/` as a sibling. This is a legacy workaround from before the project was installable. The project is now always installed in editable mode via `uv sync` (`pyproject.toml` declares `include = ["phentrieve*", "api*"]`), so the path hack is dead weight that creates an import-time side effect (Critical Finding #4 residual).

**Files:**
- Modify: `api/main.py:10-12`
- Modify: `tests/unit/api/test_dependencies_char.py:10-16`
- Modify: `tests/unit/api/test_main_char.py:10-15`

- [ ] **Step 1: Verify `api/` can be imported without the sys.path hack**

Run:
```bash
uv run python -c "from api.main import create_app; app = create_app(); print('OK:', app.title)"
```

Expected: `OK: Phentrieve API`

If this fails, the package is not installed properly — fix with `uv sync --all-extras` before proceeding.

- [ ] **Step 2: Remove the sys.path mutation from `api/main.py`**

Current content at `api/main.py:1-13`:
```python
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path
# This needs to be before other project-specific imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import (  # noqa: E402
```

Replace with:
```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import (
```

Also remove every `# noqa: E402` in the subsequent import block — those were only needed because the imports used to come after a statement (the sys.path mutation).

- [ ] **Step 3: Remove the mirroring hack from `test_dependencies_char.py`**

Read `tests/unit/api/test_dependencies_char.py:1-20` first. Replace lines similar to:
```python
import sys
from pathlib import Path

# Ensure project root on sys.path so api.* imports work.
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
```

With simply removing those lines. The test can then `from api.dependencies import ...` directly.

- [ ] **Step 4: Remove the mirroring hack from `test_main_char.py`**

Same pattern: read the file, delete the `sys.path.insert` block at or around line 15.

- [ ] **Step 5: Run ruff to catch unused imports introduced by the removal**

Run:
```bash
make check
```

Expected: `All checks passed!` If `os` or `sys` are now unused in `api/main.py`, ruff will flag them — delete the imports.

- [ ] **Step 6: Run mypy to catch any path-resolution issues**

Run:
```bash
make typecheck-fast
```

Expected: `Success: no issues found in 91 source files`.

- [ ] **Step 7: Run the API test suite**

Run:
```bash
uv run pytest tests/unit/api/ -v
```

Expected: all tests in `tests/unit/api/` pass. Critical that `test_main_char.py::test_create_app_returns_fastapi_instance` (or equivalent) passes — that's the sanity check that the imports still work without the hack.

- [ ] **Step 8: Start the dev API briefly to catch runtime import failures**

Run:
```bash
timeout 8 uv run uvicorn api.main:app --host 127.0.0.1 --port 8935 2>&1 | tee /tmp/api-start.log
grep -E "Uvicorn running|ImportError|ModuleNotFoundError" /tmp/api-start.log
```

Expected: `Uvicorn running on http://127.0.0.1:8935`. No `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 9: Commit**

```bash
git add api/main.py tests/unit/api/test_dependencies_char.py tests/unit/api/test_main_char.py
git commit -m "$(cat <<'EOF'
refactor(api): remove sys.path hack from api/main.py and mirrors

api/main.py:11 mutated sys.path at import time to find phentrieve/
as a sibling. This was a legacy workaround from before the project
was installable. With setuptools.packages.find including both
phentrieve* and api*, the project is always installed via uv sync,
so the hack is dead weight that creates an import-time side effect.

Also removes the two test files that replicated the same pattern:
- tests/unit/api/test_dependencies_char.py
- tests/unit/api/test_main_char.py

Closes Critical Finding #4 residual from the updated
CODE-QUALITY-REVIEW-2026-04-09.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `ErrorResponse` Pydantic schema + global exception handler

**Rationale:** Finding #11 residual. Routers currently raise `HTTPException(status_code=..., detail=...)` with inconsistent `detail` payloads (some strings, some dicts). API consumers can't rely on a stable error shape. A Pydantic `ErrorResponse` schema plus a global exception handler normalizes every error to one shape and publishes it in OpenAPI.

**Design decision:** Use FastAPI's `app.exception_handler(HTTPException)` to intercept existing `HTTPException` raises without needing to rewrite every router. The handler renders an `ErrorResponse` from `exc.status_code` and `exc.detail`. Routers optionally declare `responses={...}` on their decorators so OpenAPI docs show the shape.

**Files:**
- Create: `api/schemas/errors.py`
- Create: `tests/unit/api/test_error_response_schema.py`
- Modify: `api/schemas/__init__.py` (export)
- Modify: `api/main.py` (register handler in `create_app`)

- [ ] **Step 1: Write the failing test for the schema shape**

Create `tests/unit/api/test_error_response_schema.py`:
```python
"""Tests for the ErrorResponse Pydantic schema and global handler."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas.errors import ErrorResponse

pytestmark = pytest.mark.unit


class TestErrorResponseSchema:
    def test_schema_fields(self):
        """ErrorResponse must have status_code, error, detail, and optional request_id."""
        resp = ErrorResponse(
            status_code=422,
            error="unprocessable_entity",
            detail="Field 'num_results' out of range",
        )
        assert resp.status_code == 422
        assert resp.error == "unprocessable_entity"
        assert resp.detail == "Field 'num_results' out of range"
        assert resp.request_id is None

    def test_schema_accepts_request_id(self):
        resp = ErrorResponse(
            status_code=500,
            error="internal_server_error",
            detail="Boom",
            request_id="abc-123",
        )
        assert resp.request_id == "abc-123"

    def test_schema_rejects_missing_fields(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ErrorResponse(status_code=500)  # type: ignore[call-arg]


class TestGlobalExceptionHandler:
    def test_http_exception_returns_error_response_shape(self):
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        # Hit an endpoint that definitely raises HTTPException.
        # Using the similarity endpoint with an invalid HPO id format.
        response = client.get("/api/v1/similarity/INVALID/HP:0001250")
        assert response.status_code in (400, 404, 422)
        body = response.json()
        # Must conform to ErrorResponse, not the raw FastAPI default shape.
        assert "status_code" in body
        assert "error" in body
        assert "detail" in body
        assert body["status_code"] == response.status_code
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
uv run pytest tests/unit/api/test_error_response_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'api.schemas.errors'`.

- [ ] **Step 3: Create the `ErrorResponse` schema**

Create `api/schemas/errors.py`:
```python
"""Standardized error response schema for the Phentrieve API.

All HTTPException responses are rendered through this schema via a
global exception handler registered in api.main.create_app(), giving
API consumers a single stable error shape.
"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response envelope.

    Every 4xx and 5xx response from the Phentrieve API conforms to this
    shape. The ``error`` field is a machine-readable slug derived from
    the HTTP status (e.g. "bad_request", "not_found"). The ``detail``
    field is the human-readable explanation from the HTTPException.
    """

    status_code: int = Field(
        ...,
        description="HTTP status code",
        examples=[422, 503],
    )
    error: str = Field(
        ...,
        description="Machine-readable error slug (snake_case)",
        examples=["unprocessable_entity", "service_unavailable"],
    )
    detail: str = Field(
        ...,
        description="Human-readable error description",
        examples=["Field 'num_results' must be between 1 and 50."],
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request correlation ID, if the server attaches one.",
    )
```

- [ ] **Step 4: Export it from `api/schemas/__init__.py`**

Read `api/schemas/__init__.py` first. Append:
```python
from api.schemas.errors import ErrorResponse  # noqa: F401
```

- [ ] **Step 5: Register the global exception handler in `api/main.py::create_app`**

Read the current `create_app()` body. Add these imports near the top (alongside `from api.schemas.errors import ErrorResponse`):

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from api.schemas.errors import ErrorResponse
```

Inside `create_app()`, immediately after the middleware setup and before the router includes, add:

```python
    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Render every HTTPException via ErrorResponse so API consumers
        see a single stable error shape regardless of which router raised."""
        # Slug from status phrase: "Unprocessable Entity" -> "unprocessable_entity"
        from http import HTTPStatus
        try:
            slug = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_")
        except ValueError:
            slug = "http_error"
        body = ErrorResponse(
            status_code=exc.status_code,
            error=slug,
            detail=str(exc.detail) if exc.detail is not None else slug,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(exclude_none=True),
            headers=getattr(exc, "headers", None) or None,
        )
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
uv run pytest tests/unit/api/test_error_response_schema.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 7: Run the full API test suite to catch regressions**

Run:
```bash
uv run pytest tests/unit/api/ -v 2>&1 | tail -20
```

Expected: all existing API tests still pass. If any test asserts on the old raw-`detail` shape, either update the assertion (preferred) or widen it to accept both the old and new shapes.

- [ ] **Step 8: Run typecheck + lint**

Run:
```bash
make check typecheck-fast
```

Expected: both clean.

- [ ] **Step 9: Commit**

```bash
git add api/schemas/errors.py api/schemas/__init__.py api/main.py tests/unit/api/test_error_response_schema.py
git commit -m "$(cat <<'EOF'
feat(api): add ErrorResponse schema + global exception handler

Every HTTPException raised by any router is now rendered through a
single ErrorResponse Pydantic model via a global exception handler
registered in create_app(). API consumers get a stable error shape
with status_code, error (machine-readable slug), detail, and optional
request_id, regardless of which router raised.

Addresses Finding #11 residual from the updated
CODE-QUALITY-REVIEW-2026-04-09.md (error response standardization).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Lock `hpo_extraction_orchestrator` behavior with characterization tests

**Rationale:** Critical Finding #2 — `phentrieve/text_processing/hpo_extraction_orchestrator.py::orchestrate_hpo_extraction()` is 298 LOC doing batch retrieval, threshold filtering, DB access, attribution, aggregation, and response shaping in one flow. Before refactoring in Task 5, lock the current behavior with characterization tests so the refactor is a pure mechanical split.

**Files:**
- Create: `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py`

- [ ] **Step 1: Write the characterization test file**

Create `tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py`:
```python
"""Characterization tests for orchestrate_hpo_extraction.

These tests lock the current behavior of the orchestrator before its
decomposition. They must continue to pass unchanged through the refactor.
"""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

pytestmark = pytest.mark.unit


def _make_mock_retriever(batch_results):
    """Build a mock DenseRetriever whose query_batch returns ``batch_results``."""
    retriever = MagicMock()
    retriever.query_batch.return_value = batch_results
    return retriever


def _chroma_batch_entry(items):
    """Build one ChromaDB-style batch entry from a list of
    (hpo_id, label, similarity) tuples."""
    metadatas = [[{"hpo_id": hpo_id, "label": label} for hpo_id, label, _ in items]]
    similarities = [[sim for _, _, sim in items]]
    return {"metadatas": metadatas, "similarities": similarities}


class TestEmptyAndSingleChunk:
    def test_empty_chunks_returns_empty_lists(self):
        retriever = _make_mock_retriever([])
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=[],
            retriever=retriever,
        )
        assert aggregated == []
        assert chunk_results == []

    def test_single_chunk_single_match(self):
        retriever = _make_mock_retriever(
            [_chroma_batch_entry([("HP:0001250", "Seizure", 0.9)])]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["Patient had a seizure."],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert len(chunk_results) == 1
        assert chunk_results[0]["chunk_idx"] == 0
        assert chunk_results[0]["chunk_text"] == "Patient had a seizure."
        assert len(chunk_results[0]["matches"]) == 1
        assert chunk_results[0]["matches"][0]["id"] == "HP:0001250"
        assert len(aggregated) == 1
        assert aggregated[0]["id"] == "HP:0001250"
        assert aggregated[0]["rank"] == 1
        assert aggregated[0]["count"] == 1


class TestThresholdFiltering:
    def test_matches_below_threshold_are_dropped(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry(
                    [
                        ("HP:0001250", "Seizure", 0.9),  # keep
                        ("HP:0000001", "All", 0.2),  # drop, below 0.5
                    ]
                )
            ]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert len(chunk_results[0]["matches"]) == 1
        assert chunk_results[0]["matches"][0]["id"] == "HP:0001250"

    def test_min_confidence_for_aggregated_filters_aggregated_only(self):
        retriever = _make_mock_retriever(
            [_chroma_batch_entry([("HP:0001250", "Seizure", 0.6)])]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
            min_confidence_for_aggregated=0.8,  # Above 0.6
        )
        # Chunk match survives chunk filter (0.6 >= 0.5)
        assert len(chunk_results[0]["matches"]) == 1
        # But aggregated is empty because avg_score 0.6 < 0.8
        assert aggregated == []


class TestTopTermPerChunk:
    def test_top_term_per_chunk_keeps_only_first(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry(
                    [
                        ("HP:0001250", "Seizure", 0.9),
                        ("HP:0001251", "Ataxia", 0.8),
                    ]
                )
            ]
        )
        _, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
            top_term_per_chunk=True,
        )
        assert len(chunk_results[0]["matches"]) == 1
        assert chunk_results[0]["matches"][0]["id"] == "HP:0001250"


class TestMultiChunkAggregation:
    def test_same_hpo_across_chunks_is_aggregated(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)]),
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.7)]),
            ]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["chunk a", "chunk b"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert len(chunk_results) == 2
        assert len(aggregated) == 1
        term = aggregated[0]
        assert term["id"] == "HP:0001250"
        assert term["count"] == 2
        assert term["score"] == pytest.approx(0.9)  # max
        assert term["avg_score"] == pytest.approx(0.8)  # (0.9 + 0.7) / 2
        assert sorted(term["chunks"]) == [0, 1]
        # Top evidence chunk should be the one with max_score (0.9)
        assert term["top_evidence_chunk_idx"] == 0


class TestAssertionStatuses:
    def test_assertion_status_propagated_to_matches_and_aggregated(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)]),
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.8)]),
            ]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["a", "b"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
            assertion_statuses=["affirmed", "negated"],
        )
        assert chunk_results[0]["matches"][0]["assertion_status"] == "affirmed"
        assert chunk_results[1]["matches"][0]["assertion_status"] == "negated"
        # Aggregated: most-common; here tied 1-1, so Counter returns the
        # first one encountered ("affirmed").
        assert aggregated[0]["assertion_status"] == "affirmed"
        assert aggregated[0]["status"] == "affirmed"  # alias


class TestRankingAndOrdering:
    def test_results_sorted_by_avg_score_then_count_desc(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry(
                    [
                        ("HP:0001250", "Seizure", 0.9),
                        ("HP:0001251", "Ataxia", 0.7),
                    ]
                )
            ]
        )
        aggregated, _ = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert [t["id"] for t in aggregated] == ["HP:0001250", "HP:0001251"]
        assert [t["rank"] for t in aggregated] == [1, 2]


class TestRetrieverInteraction:
    def test_retriever_query_batch_called_with_all_chunks(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)]),
                _chroma_batch_entry([]),
            ]
        )
        orchestrate_hpo_extraction(
            text_chunks=["a", "b"],
            retriever=retriever,
            num_results_per_chunk=7,
            chunk_retrieval_threshold=0.5,
        )
        retriever.query_batch.assert_called_once()
        kwargs = retriever.query_batch.call_args.kwargs
        assert kwargs["texts"] == ["a", "b"]
        assert kwargs["n_results"] == 7
        assert kwargs["include_similarities"] is True
```

- [ ] **Step 2: Run the characterization tests**

Run:
```bash
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py -v
```

Expected: all tests pass. The tests exercise the current god function against mock retrievers; since we're not mocking the HPO database lookup, the `HPODatabase` path fails quietly (`db_path.exists()` returns False in the test environment), which is fine — the tests assert on the core aggregation logic, not the synonym enrichment.

- [ ] **Step 3: Verify full suite still green**

Run:
```bash
make test
```

Expected: `907 passed` (899 prior + 8 new).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py
git commit -m "$(cat <<'EOF'
test(text_processing): lock hpo_extraction_orchestrator behavior

Adds characterization tests for orchestrate_hpo_extraction covering:
- empty input
- single chunk single/multi match
- chunk threshold filtering vs aggregated threshold filtering
- top_term_per_chunk
- multi-chunk aggregation (same HPO across chunks)
- assertion status propagation
- avg_score/count ranking
- retriever.query_batch call contract

These tests must pass unchanged through the Task 5 refactor.
Characterization test pattern inherited from PR #191 Stream A.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Decompose `hpo_extraction_orchestrator.py` into focused helpers

**Rationale:** With characterization tests in place, split the 298-LOC god function into four focused helpers and keep `orchestrate_hpo_extraction` as a thin coordinator. Each helper has one job and can be tested and reused independently.

**Target structure:**
- `_process_chunk_matches(text_chunks, batch_results, num_results_per_chunk, chunk_retrieval_threshold, top_term_per_chunk, assertion_statuses) -> list[dict]` — current lines 69-137
- `_load_term_details(all_hpo_ids, include_details) -> tuple[dict, dict]` — current lines 139-182
- `_build_evidence_map(chunk_results, hpo_synonyms_cache) -> dict[str, list[dict]]` — current lines 184-214
- `_aggregate_and_rank(evidence_map, min_confidence_for_aggregated, hpo_synonyms_cache, hpo_definitions_cache, include_details) -> list[dict]` — current lines 216-295

**Files:**
- Create: `phentrieve/text_processing/_hpo_extraction_helpers.py`
- Modify: `phentrieve/text_processing/hpo_extraction_orchestrator.py`

- [ ] **Step 1: Create the helpers module**

Create `phentrieve/text_processing/_hpo_extraction_helpers.py`:
```python
"""Focused helpers for hpo_extraction_orchestrator.

Split out from the 298-LOC god function orchestrate_hpo_extraction()
so each concern can be tested, read, and evolved independently.

The public API surface remains orchestrate_hpo_extraction(); these
helpers are a private implementation detail (leading underscore on
the module name). Tests exercise them via the public function.
"""

import logging
from collections import Counter, defaultdict
from typing import Any

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.retrieval.text_attribution import get_text_attributions
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)


def process_chunk_matches(
    text_chunks: list[str],
    all_query_results: list[dict[str, Any]],
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    top_term_per_chunk: bool,
    assertion_statuses: list[str | None] | None,
) -> list[dict[str, Any]]:
    """Convert batched retriever results into per-chunk match lists.

    Applies chunk_retrieval_threshold, the num_results_per_chunk cap,
    the top_term_per_chunk filter, and propagates assertion_statuses
    from the parallel list onto each match.

    Returns one dict per input chunk with keys: chunk_idx, chunk_text,
    matches (list of {id, name, score, assertion_status}).
    """
    chunk_results: list[dict[str, Any]] = []
    for chunk_idx, chunk_text in enumerate(text_chunks):
        try:
            query_results = all_query_results[chunk_idx]
            current_hpo_matches: list[dict[str, Any]] = []
            metadatas_entry = query_results.get("metadatas") or [[]]
            similarities_entry = query_results.get("similarities") or [[]]
            metadatas_list = metadatas_entry[0] if metadatas_entry else []
            similarities_list = similarities_entry[0] if similarities_entry else []

            matches_added = 0
            for i, metadata in enumerate(metadatas_list):
                if matches_added >= num_results_per_chunk:
                    break
                similarity = (
                    similarities_list[i] if i < len(similarities_list) else 0.0
                )
                if similarity < chunk_retrieval_threshold:
                    continue
                hpo_id = metadata.get("id") or metadata.get("hpo_id")
                name = metadata.get("label") or metadata.get("name")
                if not (hpo_id and name):
                    continue
                current_hpo_matches.append(
                    {
                        "id": hpo_id,
                        "name": name,
                        "score": similarity,
                        "assertion_status": (
                            assertion_statuses[chunk_idx]
                            if assertion_statuses
                            else None
                        ),
                    }
                )
                matches_added += 1

            logger.info(
                f"Found {len(current_hpo_matches)} matches for chunk {chunk_idx + 1}"
            )

            if top_term_per_chunk and current_hpo_matches:
                current_hpo_matches = [current_hpo_matches[0]]

            chunk_results.append(
                {
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_text,
                    "matches": current_hpo_matches,
                }
            )
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_idx + 1}: {e}")
            continue
    return chunk_results


def load_term_details(
    all_hpo_ids: set[str],
    include_details: bool,
) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    """Batch-load synonyms (and optionally definitions) for all HPO IDs.

    Returns (synonyms_cache, definitions_cache). definitions_cache is
    always returned but is only populated when include_details is True.
    Both caches are empty dicts if the HPO database file is not found
    or loading fails; the caller treats that as "no enrichment available".
    """
    hpo_synonyms_cache: dict[str, list[str]] = {}
    hpo_definitions_cache: dict[str, str | None] = {}
    if not all_hpo_ids:
        return hpo_synonyms_cache, hpo_definitions_cache

    try:
        data_dir = resolve_data_path(None, "data_dir", get_default_data_dir)
        db_path = data_dir / DEFAULT_HPO_DB_FILENAME
        if not db_path.exists():
            logger.warning(
                f"HPO database not found: {db_path}. Skipping synonym lookup."
            )
            return hpo_synonyms_cache, hpo_definitions_cache

        logger.debug(
            f"Loading {'synonyms and definitions' if include_details else 'synonyms'} "
            f"for {len(all_hpo_ids)} unique HPO terms"
        )
        db = HPODatabase(db_path)
        terms_map = db.get_terms_by_ids(list(all_hpo_ids))
        db.close()

        for hpo_id, term_data in terms_map.items():
            hpo_synonyms_cache[hpo_id] = term_data.get("synonyms", [])
            if include_details:
                hpo_definitions_cache[hpo_id] = term_data.get("definition")

        logger.info(
            f"Loaded {'synonyms and definitions' if include_details else 'synonyms'} "
            f"for {len(hpo_synonyms_cache)} HPO terms"
        )
    except Exception as e:
        logger.warning(f"Failed to batch-load HPO term data: {e}")

    return hpo_synonyms_cache, hpo_definitions_cache


def build_evidence_map(
    chunk_results: list[dict[str, Any]],
    hpo_synonyms_cache: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Group match evidence by HPO ID, attaching text attributions.

    One list per HPO ID, each element a dict with: score, chunk_idx,
    text, status, name, attributions_in_chunk.
    """
    evidence_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_result in chunk_results:
        chunk_idx: Any = chunk_result["chunk_idx"]
        chunk_text: Any = chunk_result["chunk_text"]
        matches: Any = chunk_result["matches"]
        for term in matches:
            hpo_id = term["id"]
            synonyms = hpo_synonyms_cache.get(hpo_id, [])
            attributions_in_chunk = get_text_attributions(
                source_chunk_text=chunk_text,
                hpo_term_label=term["name"],
                hpo_term_synonyms=synonyms,
                hpo_term_id=hpo_id,
            )
            evidence_map[hpo_id].append(
                {
                    "score": term["score"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "status": term.get("assertion_status"),
                    "name": term["name"],
                    "attributions_in_chunk": attributions_in_chunk,
                }
            )
    return evidence_map


def aggregate_and_rank(
    evidence_map: dict[str, list[dict[str, Any]]],
    min_confidence_for_aggregated: float,
    hpo_synonyms_cache: dict[str, list[str]],
    hpo_definitions_cache: dict[str, str | None],
    include_details: bool,
) -> list[dict[str, Any]]:
    """Collapse the evidence map into a ranked list of aggregated HPO terms.

    Each output dict mirrors the contract held by the existing API
    response shape: id, name, score (max), count, evidence_count,
    avg_score, confidence, chunks, top_evidence_chunk_idx,
    text_attributions, assertion_status, status, and optionally
    definition + synonyms when include_details is True. Ranks start at 1.
    """
    aggregated_list: list[dict[str, Any]] = []
    for hpo_id, evidence_list in evidence_map.items():
        if not evidence_list:
            continue
        total_score = sum(e["score"] for e in evidence_list)
        avg_score = total_score / len(evidence_list)
        if avg_score < min_confidence_for_aggregated:
            continue
        max_score = max(e["score"] for e in evidence_list)
        top_evidence_chunk_idx = next(
            e["chunk_idx"] for e in evidence_list if e["score"] == max_score
        )
        status_counter = Counter([e["status"] for e in evidence_list if e["status"]])
        assertion_status = (
            status_counter.most_common(1)[0][0] if status_counter else None
        )

        text_attributions: list[dict[str, Any]] = []
        for e in evidence_list:
            for attribution in e.get("attributions_in_chunk", []):
                enriched = attribution.copy()
                enriched["chunk_idx"] = e["chunk_idx"]
                text_attributions.append(enriched)

        term: dict[str, Any] = {
            "id": hpo_id,
            "name": evidence_list[0]["name"],
            "score": max_score,
            "count": len(evidence_list),
            "evidence_count": len(evidence_list),
            "avg_score": avg_score,
            "confidence": avg_score,
            "chunks": sorted({e["chunk_idx"] for e in evidence_list}),
            "top_evidence_chunk_idx": top_evidence_chunk_idx,
            "text_attributions": text_attributions,
            "assertion_status": assertion_status,
            "status": assertion_status,
        }
        if include_details:
            term["definition"] = hpo_definitions_cache.get(hpo_id)
            term["synonyms"] = hpo_synonyms_cache.get(hpo_id, [])
        aggregated_list.append(term)

    aggregated_list.sort(key=lambda x: (-x["avg_score"], -x["count"]))
    for idx, term in enumerate(aggregated_list):
        term["rank"] = idx + 1
    return aggregated_list
```

- [ ] **Step 2: Replace the body of `hpo_extraction_orchestrator.py`**

Read the current file first. Then replace the function body (lines 56-297) with a thin coordinator. Keep the docstring and signature unchanged.

New body of `orchestrate_hpo_extraction()`:
```python
    # Initialize results
    chunk_results = []  # filled below

    # Batch-query all chunks via the retriever in ONE call.
    logger.info(f"Batch querying {len(text_chunks)} chunks at once")
    all_query_results = retriever.query_batch(
        texts=text_chunks,
        n_results=num_results_per_chunk,
        include_similarities=True,
    )

    # Step 1 — Apply thresholds, top-term filter, assertion propagation.
    chunk_results = process_chunk_matches(
        text_chunks=text_chunks,
        all_query_results=all_query_results,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        top_term_per_chunk=top_term_per_chunk,
        assertion_statuses=assertion_statuses,
    )

    # Step 2 — Batch-load synonyms (and definitions if requested).
    all_hpo_ids: set[str] = {
        match["id"]
        for chunk in chunk_results
        for match in chunk.get("matches", [])
    }
    hpo_synonyms_cache, hpo_definitions_cache = load_term_details(
        all_hpo_ids=all_hpo_ids,
        include_details=include_details,
    )

    # Step 3 — Build evidence map with text attributions.
    evidence_map = build_evidence_map(
        chunk_results=chunk_results,
        hpo_synonyms_cache=hpo_synonyms_cache,
    )

    # Step 4 — Aggregate, rank, and render final output.
    aggregated_results_list = aggregate_and_rank(
        evidence_map=evidence_map,
        min_confidence_for_aggregated=min_confidence_for_aggregated,
        hpo_synonyms_cache=hpo_synonyms_cache,
        hpo_definitions_cache=hpo_definitions_cache,
        include_details=include_details,
    )

    logger.info(
        f"Found {len(aggregated_results_list)} unique HPO terms "
        f"above threshold {min_confidence_for_aggregated}"
    )
    return (aggregated_results_list, chunk_results)
```

Update imports at the top of `hpo_extraction_orchestrator.py`:
```python
import logging
from typing import Any

from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing._hpo_extraction_helpers import (
    aggregate_and_rank,
    build_evidence_map,
    load_term_details,
    process_chunk_matches,
)

logger = logging.getLogger(__name__)
```

The module-level imports of `Counter`, `defaultdict`, `HPODatabase`, `get_text_attributions`, `resolve_data_path`, `get_default_data_dir`, `DEFAULT_HPO_DB_FILENAME` all move to the helpers file and can be removed from here.

- [ ] **Step 3: Run the characterization tests — they MUST still pass**

Run:
```bash
uv run pytest tests/unit/text_processing/test_hpo_extraction_orchestrator_char.py -v
```

Expected: all 8 tests pass. If any fail, the refactor broke behavior — fix until green.

- [ ] **Step 4: Run ruff + mypy**

Run:
```bash
make check typecheck-fast
```

Expected: both clean. Unused imports will be flagged if any were missed.

- [ ] **Step 5: Run the full test suite**

Run:
```bash
make test
```

Expected: 907 passed (same count, no regressions).

- [ ] **Step 6: Commit**

```bash
git add phentrieve/text_processing/_hpo_extraction_helpers.py phentrieve/text_processing/hpo_extraction_orchestrator.py
git commit -m "$(cat <<'EOF'
refactor(text_processing): decompose hpo_extraction_orchestrator

Split the 298-LOC god function orchestrate_hpo_extraction() into four
focused helpers in a new _hpo_extraction_helpers.py module:

- process_chunk_matches:    retriever output -> per-chunk match lists
  (applies chunk_retrieval_threshold, num_results_per_chunk cap,
   top_term_per_chunk filter, assertion_status propagation)
- load_term_details:        batch-load synonyms/definitions from HPODatabase
- build_evidence_map:       group matches by HPO ID with text attributions
- aggregate_and_rank:       collapse evidence -> ranked aggregated terms

orchestrate_hpo_extraction() is now a thin 4-step coordinator with the
same public signature and behavior. All 8 characterization tests from
Task 4 pass unchanged, proving this is a pure mechanical split.

Closes Critical Finding #2 from the updated
CODE-QUALITY-REVIEW-2026-04-09.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Lock `TextProcessingPipeline._create_chunkers` behavior

**Rationale:** Critical Finding #6. Before simplifying the 158-line if/elif chain with its dead `sliding_window` duplicate branch, lock the current behavior.

**Files:**
- Create: `tests/unit/text_processing/test_pipeline_create_chunkers_char.py`

- [ ] **Step 1: Write the characterization test**

Create `tests/unit/text_processing/test_pipeline_create_chunkers_char.py`:
```python
"""Characterization tests for TextProcessingPipeline._create_chunkers.

Locks the factory behavior before the Task 7 refactor to a registry.
"""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FinalChunkCleaner,
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline

pytestmark = pytest.mark.unit


def _build(pipeline_config, model=None):
    return TextProcessingPipeline(
        language="en",
        chunking_pipeline_config=pipeline_config,
        assertion_config={"disable": True},
        sbert_model_for_semantic_chunking=model,
    )


class TestBasicChunkerTypes:
    def test_paragraph(self):
        pipe = _build([{"type": "paragraph"}])
        assert len(pipe.chunkers) == 1
        assert isinstance(pipe.chunkers[0], ParagraphChunker)

    def test_sentence(self):
        pipe = _build([{"type": "sentence"}])
        assert isinstance(pipe.chunkers[0], SentenceChunker)

    def test_fine_grained_punctuation(self):
        pipe = _build([{"type": "fine_grained_punctuation"}])
        assert isinstance(pipe.chunkers[0], FineGrainedPunctuationChunker)

    def test_conjunction(self):
        pipe = _build([{"type": "conjunction"}])
        assert isinstance(pipe.chunkers[0], ConjunctionChunker)

    def test_noop(self):
        pipe = _build([{"type": "noop"}])
        assert isinstance(pipe.chunkers[0], NoOpChunker)


class TestSlidingWindow:
    def test_sliding_window_requires_model(self):
        with pytest.raises(ValueError, match="SentenceTransformer model required"):
            _build([{"type": "sliding_window"}], model=None)

    def test_sliding_window_with_model(self):
        mock_model = MagicMock()
        pipe = _build(
            [
                {
                    "type": "sliding_window",
                    "config": {
                        "window_size_tokens": 4,
                        "step_size_tokens": 2,
                        "splitting_threshold": 0.5,
                        "min_split_segment_length_words": 50,
                    },
                }
            ],
            model=mock_model,
        )
        assert isinstance(pipe.chunkers[0], SlidingWindowSemanticSplitter)

    def test_sliding_window_semantic_alias(self):
        """sliding_window_semantic is a legacy alias for sliding_window."""
        mock_model = MagicMock()
        pipe = _build([{"type": "sliding_window_semantic"}], model=mock_model)
        assert isinstance(pipe.chunkers[0], SlidingWindowSemanticSplitter)


class TestFinalChunkCleaner:
    def test_default_final_chunk_cleaner(self):
        pipe = _build([{"type": "final_chunk_cleaner"}])
        assert isinstance(pipe.chunkers[0], FinalChunkCleaner)


class TestMultiStage:
    def test_full_pipeline(self):
        mock_model = MagicMock()
        pipe = _build(
            [
                {"type": "paragraph"},
                {"type": "sliding_window"},
                {"type": "fine_grained_punctuation"},
                {"type": "final_chunk_cleaner"},
            ],
            model=mock_model,
        )
        assert len(pipe.chunkers) == 4
        assert isinstance(pipe.chunkers[0], ParagraphChunker)
        assert isinstance(pipe.chunkers[1], SlidingWindowSemanticSplitter)
        assert isinstance(pipe.chunkers[2], FineGrainedPunctuationChunker)
        assert isinstance(pipe.chunkers[3], FinalChunkCleaner)


class TestFallback:
    def test_unknown_chunker_type_skipped(self):
        """Unknown types log warning and are skipped; empty result falls back to NoOp."""
        pipe = _build([{"type": "does_not_exist"}])
        assert len(pipe.chunkers) == 1
        assert isinstance(pipe.chunkers[0], NoOpChunker)

    def test_empty_config_falls_back_to_noop(self):
        pipe = _build([])
        assert len(pipe.chunkers) == 1
        assert isinstance(pipe.chunkers[0], NoOpChunker)
```

- [ ] **Step 2: Run the characterization tests**

Run:
```bash
uv run pytest tests/unit/text_processing/test_pipeline_create_chunkers_char.py -v
```

Expected: all 11 tests pass against the *current* `_create_chunkers`. If `sliding_window_semantic` fails, that's an important finding — current code at line 207-234 is only reachable for that specific alias value, and the test confirms the alias works.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/text_processing/test_pipeline_create_chunkers_char.py
git commit -m "$(cat <<'EOF'
test(text_processing): lock _create_chunkers factory behavior

Adds characterization tests for TextProcessingPipeline._create_chunkers
covering all supported chunker types (paragraph, sentence, fine-grained,
conjunction, noop, sliding_window, sliding_window_semantic alias,
final_chunk_cleaner), multi-stage pipelines, and fallback behavior
(unknown type and empty config both fall back to NoOpChunker).

These tests must pass unchanged through the Task 7 registry refactor.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Refactor `_create_chunkers` to registry-based factory

**Rationale:** Critical Finding #6. The current if/elif chain at lines 91-249 has a real duplicate: `sliding_window` is handled at line 128 AND again at line 207-234. The first match wins, so the second block is dead for `sliding_window` and only reachable for `sliding_window_semantic` (which is a legacy alias). A registry-based factory makes the alias explicit and eliminates the duplicate.

**Files:**
- Create: `phentrieve/text_processing/_chunker_registry.py`
- Modify: `phentrieve/text_processing/pipeline.py`

- [ ] **Step 1: Create the registry module**

Create `phentrieve/text_processing/_chunker_registry.py`:
```python
"""Registry-based factory for chunker construction.

Each entry in CHUNKER_FACTORIES maps a chunker_type string to a callable
``(language, chunker_config, sbert_model) -> TextChunker``. Legacy aliases
(e.g. "sliding_window_semantic" -> "sliding_window") are resolved via
CHUNKER_ALIASES before lookup.

Split out from pipeline.py::_create_chunkers() so the factory logic is
straight-line, table-driven, and testable, with no duplicate branches
(the prior if/elif chain handled "sliding_window" twice — once reachable
and once dead).
"""

from typing import Any, Callable

from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FinalChunkCleaner,
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
    TextChunker,
)

# Legacy aliases resolved before the main lookup
CHUNKER_ALIASES: dict[str, str] = {
    "sliding_window_semantic": "sliding_window",
}


def _make_simple(cls: type[TextChunker]) -> Callable[..., TextChunker]:
    """Return a factory that ignores config and instantiates ``cls(language=...)``."""

    def factory(
        language: str,
        chunker_config: dict[str, Any],
        sbert_model: SentenceTransformer | None,
    ) -> TextChunker:
        del chunker_config, sbert_model
        return cls(language=language)

    return factory


def _make_sliding_window(
    language: str,
    chunker_config: dict[str, Any],
    sbert_model: SentenceTransformer | None,
) -> TextChunker:
    if sbert_model is None:
        raise ValueError(
            "SentenceTransformer model required for sliding window semantic "
            "splitting but none was provided"
        )
    return SlidingWindowSemanticSplitter(
        language=language,
        model=sbert_model,
        window_size_tokens=chunker_config.get("window_size_tokens", 4),
        step_size_tokens=chunker_config.get("step_size_tokens", 2),
        splitting_threshold=chunker_config.get("splitting_threshold", 0.5),
        min_split_segment_length_words=chunker_config.get(
            "min_split_segment_length_words", 50
        ),
    )


def _make_final_chunk_cleaner(
    language: str,
    chunker_config: dict[str, Any],
    sbert_model: SentenceTransformer | None,
) -> TextChunker:
    del sbert_model
    params: dict[str, Any] = {
        "language": language,
        "min_cleaned_chunk_length_chars": chunker_config.get(
            "min_cleaned_chunk_length_chars", 1
        ),
        "filter_short_low_value_chunks_max_words": chunker_config.get(
            "filter_short_low_value_chunks_max_words", 2
        ),
        "max_cleanup_passes": chunker_config.get("max_cleanup_passes", 3),
    }
    # Optional custom lists — only forward when explicitly provided so the
    # FinalChunkCleaner falls back to loaded resources when absent.
    for key in (
        "custom_leading_words_to_remove",
        "custom_trailing_words_to_remove",
        "custom_leading_punctuation",
        "custom_trailing_punctuation",
        "custom_low_value_words",
    ):
        value = chunker_config.get(key)
        if value is not None:
            params[key] = value
    return FinalChunkCleaner(**params)


ChunkerFactory = Callable[
    [str, dict[str, Any], SentenceTransformer | None], TextChunker
]

CHUNKER_FACTORIES: dict[str, ChunkerFactory] = {
    "paragraph": _make_simple(ParagraphChunker),
    "sentence": _make_simple(SentenceChunker),
    "fine_grained_punctuation": _make_simple(FineGrainedPunctuationChunker),
    "conjunction": _make_simple(ConjunctionChunker),
    "noop": _make_simple(NoOpChunker),
    "sliding_window": _make_sliding_window,
    "final_chunk_cleaner": _make_final_chunk_cleaner,
}


def build_chunker(
    chunker_type: str,
    chunker_config: dict[str, Any],
    language: str,
    sbert_model: SentenceTransformer | None,
) -> TextChunker | None:
    """Build a chunker from its type string, or return None for unknown types.

    Aliases are resolved before lookup. The caller (pipeline.py) is
    responsible for logging the unknown-type warning and for the
    empty-pipeline NoOp fallback.
    """
    resolved_type = CHUNKER_ALIASES.get(chunker_type, chunker_type)
    factory = CHUNKER_FACTORIES.get(resolved_type)
    if factory is None:
        return None
    return factory(language, chunker_config, sbert_model)
```

- [ ] **Step 2: Replace `_create_chunkers` in `pipeline.py`**

Read `phentrieve/text_processing/pipeline.py` to confirm the current function spans lines 91-249. Replace the entire method body with:

```python
    def _create_chunkers(self) -> list[TextChunker]:
        """Create chunker instances based on configuration.

        Delegates the per-type construction to the registry in
        _chunker_registry.py so this method stays a thin loop.
        """
        from phentrieve.text_processing._chunker_registry import build_chunker

        chunkers: list[TextChunker] = []
        for stage_config in self.chunking_pipeline_config:
            if isinstance(stage_config, dict):
                chunker_type = stage_config.get("type", "unknown")
                chunker_config = stage_config.get("config", {})
            else:
                chunker_type = stage_config
                chunker_config = {}

            chunker = build_chunker(
                chunker_type=chunker_type,
                chunker_config=chunker_config,
                language=self.language,
                sbert_model=self.sbert_model,
            )
            if chunker is None:
                logger.warning(
                    "Unknown chunker type '%s' in config, skipping",
                    _sanitize(chunker_type),
                )
                continue
            chunkers.append(chunker)

        if not chunkers:
            logger.warning(
                "No valid chunkers specified in config, using NoOpChunker as fallback."
            )
            chunkers.append(NoOpChunker(language=self.language))
        return chunkers
```

- [ ] **Step 3: Run the characterization tests — they MUST still pass**

Run:
```bash
uv run pytest tests/unit/text_processing/test_pipeline_create_chunkers_char.py -v
```

Expected: all 11 tests pass. Special attention to `test_sliding_window_semantic_alias` — if it passes, the alias mechanism works. If it fails, the registry alias isn't being applied.

- [ ] **Step 4: Run the broader text_processing suite for safety**

Run:
```bash
uv run pytest tests/unit/text_processing/ -v 2>&1 | tail -20
```

Expected: all green.

- [ ] **Step 5: Run full suite + typecheck**

Run:
```bash
make check typecheck-fast test
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/text_processing/_chunker_registry.py phentrieve/text_processing/pipeline.py
git commit -m "$(cat <<'EOF'
refactor(text_processing): registry-based chunker factory

Replace the 158-line if/elif chain in TextProcessingPipeline._create_chunkers
with a table-driven registry in a new _chunker_registry.py module.

Benefits:
- Eliminates the dead duplicate branch: the old code handled
  "sliding_window" at line 128 AND again at line 207-234. The first
  match always won, making the second block dead for "sliding_window"
  and only reachable via the "sliding_window_semantic" alias.
- Aliases are now explicit (CHUNKER_ALIASES dict) instead of being
  accidentally encoded in the order of elif branches.
- Each type has a single factory callable; adding a new chunker type
  means adding one dict entry, not extending a 160-line conditional.
- pipeline.py::_create_chunkers() is now a 24-line thin loop.

All 11 characterization tests from Task 6 pass unchanged, including
the sliding_window_semantic alias test, proving behavior is preserved.

Closes Critical Finding #6 from the updated
CODE-QUALITY-REVIEW-2026-04-09.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Simplify `FinalChunkCleaner.__init__` language resource loading

**Rationale:** Finding #11 residual / Critical Finding #6 sub-item. The cleaner's `__init__` has three near-identical blocks (lines 126-139, 141-154, 172-185) that each do "if custom_list provided, use it; otherwise load from language resource file". Extract a helper.

**Files:**
- Modify: `phentrieve/text_processing/chunkers.py` (only `FinalChunkCleaner.__init__`)

- [ ] **Step 1: Read the current `FinalChunkCleaner.__init__`**

Run:
```bash
uv run pytest tests/unit/text_processing/ -k "FinalChunkCleaner or final_chunk_cleaner or cleaner" -v
```

Capture which tests currently exercise `FinalChunkCleaner` so you have a regression guardrail.

- [ ] **Step 2: Add the helper inside `chunkers.py` near `FinalChunkCleaner`**

Find the line immediately above `class FinalChunkCleaner(TextChunker):` and insert:

```python
def _load_language_word_list(
    custom: list[str] | None,
    default_resource_filename: str,
    config_key_for_custom_file: str,
    language_resources_section: dict[str, Any],
    language: str,
    *,
    lowercase: bool = True,
) -> list[str]:
    """Pick a language-specific word list, preferring a caller-provided custom list.

    Used by FinalChunkCleaner to DRY up the three identical leading/trailing/
    low-value list loading blocks. Returns the custom list (lowercased when
    ``lowercase`` is True) if provided; otherwise loads from the default
    resource file and returns the entry for ``language``, falling back to
    English.
    """
    if custom is not None:
        return [w.lower() for w in custom] if lowercase else list(custom)
    resource = load_language_resource(
        default_resource_filename=default_resource_filename,
        config_key_for_custom_file=config_key_for_custom_file,
        language_resources_config_section=language_resources_section,
    )
    return resource.get(language.lower(), resource.get("en", []))
```

Make sure `Any` is imported from `typing` at the top of `chunkers.py` (it probably already is).

- [ ] **Step 3: Replace the 3 blocks in `__init__`**

Replace the current:
```python
        if custom_leading_words_to_remove is not None:
            self.leading_words_to_strip = [
                w.lower() for w in custom_leading_words_to_remove
            ]
        else:
            leading_cleanup_resources = load_language_resource(...)
            self.leading_words_to_strip = leading_cleanup_resources.get(...)

        if custom_trailing_words_to_remove is not None:
            self.trailing_words_to_strip = [...]
        else:
            trailing_cleanup_resources = load_language_resource(...)
            self.trailing_words_to_strip = trailing_cleanup_resources.get(...)
```

and the low-value block:
```python
        if custom_low_value_words:
            self.low_value_words = {s.lower() for s in custom_low_value_words}
        else:
            low_value_resources = load_language_resource(...)
            self.low_value_words = set(low_value_resources.get(...))
```

With:
```python
        self.leading_words_to_strip = _load_language_word_list(
            custom=custom_leading_words_to_remove,
            default_resource_filename="leading_cleanup_words.json",
            config_key_for_custom_file="leading_cleanup_words_file",
            language_resources_section=language_resources_section,
            language=self.language,
        )
        self.trailing_words_to_strip = _load_language_word_list(
            custom=custom_trailing_words_to_remove,
            default_resource_filename="trailing_cleanup_words.json",
            config_key_for_custom_file="trailing_cleanup_words_file",
            language_resources_section=language_resources_section,
            language=self.language,
        )
        self.low_value_words = set(
            _load_language_word_list(
                custom=custom_low_value_words,
                default_resource_filename="low_semantic_value_words.json",
                config_key_for_custom_file="low_semantic_value_words_file",
                language_resources_section=language_resources_section,
                language=self.language,
            )
        )
```

- [ ] **Step 4: Run the focused test subset captured in Step 1**

Run:
```bash
uv run pytest tests/unit/text_processing/ -k "FinalChunkCleaner or final_chunk_cleaner or cleaner" -v
```

Expected: same pass/fail profile as before the change.

- [ ] **Step 5: Full suite + checks**

Run:
```bash
make check typecheck-fast test
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/text_processing/chunkers.py
git commit -m "$(cat <<'EOF'
refactor(text_processing): DRY language resource loading in FinalChunkCleaner

Extract _load_language_word_list() helper to replace 3 near-identical
if-custom-else-load-resource blocks in FinalChunkCleaner.__init__
(leading cleanup words, trailing cleanup words, low-value words).

Each block did:
  if custom is not None:
      use custom.lower()
  else:
      load_language_resource(...).get(lang, get("en", []))

Now a single parameterized helper does the work. __init__ drops from
~35 lines of cleanup-list logic to ~25 lines of helper calls.

Addresses Critical Finding #6 sub-item from the updated
CODE-QUALITY-REVIEW-2026-04-09.md.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Document `@lru_cache` decisions

**Rationale:** Finding #3 residual. `phentrieve/embeddings.py`, `phentrieve/retrieval/details_enrichment.py`, and `phentrieve/config.py` use `@lru_cache` without bounds. In practice they are load-once-and-pin patterns (one HPO graph, one config dict, one model), but the code quality review flagged them as unbounded globals. The correct response is documentation, not conversion — these caches are intentional singletons.

**Files:**
- Create: `docs/architecture/caching.md`

- [ ] **Step 1: Survey the actual `@lru_cache` sites**

Run:
```bash
grep -rn "@lru_cache\|@functools.lru_cache" phentrieve/ api/ --include="*.py"
```

Expected: a short list. Record each file + function name + what it caches.

- [ ] **Step 2: Create the documentation file**

Create `docs/architecture/caching.md`:
```markdown
# Caching Strategy

Phentrieve uses two distinct caching patterns, each for a different purpose. This document explains when to use which.

## Pattern 1 — `cachetools.TTLCache` with `_cache_lock` (API model cache)

**Where**: `api/dependencies.py`

**What it caches**:
- `LOADED_SBERT_MODELS` — loaded `SentenceTransformer` instances, keyed by model name
- `LOADED_RETRIEVERS` — loaded `DenseRetriever` instances, keyed by `f"retriever_for_{name}_multi={bool}"`
- `MODEL_LOADING_STATUS` — per-model load state machine (`loading` / `loaded` / `failed`)
- `MODEL_LOAD_LOCKS` — per-model asyncio locks for concurrent-request deduplication

**Why TTLCache**:
- Long-running API processes can see many distinct model names over their lifetime (e.g. a benchmark workload that rotates through models). Unbounded dicts would leak.
- TTL (3600s) ensures stale entries are reclaimed even under write pressure.
- `maxsize=10` for the model caches, `maxsize=50` for the tracking caches (bigger because the entries are tiny).
- `_cache_lock` (threading.Lock) serializes mutations — `TTLCache` itself is not thread-safe.
- All writes use `with _cache_lock:`; all reads use `.get(key)` which is safe even when an entry expires between check and read.

**Eviction**: Automatic on TTL expiry; LRU eviction when over `maxsize`.

**Lifecycle**: `cleanup_model_caches()` runs in the FastAPI lifespan shutdown. It first cancels any in-flight background loading tasks via `asyncio.shield` + `gather(return_exceptions=True)`, then clears all four caches under `_cache_lock`.

## Pattern 2 — `@functools.lru_cache` (process-wide singletons)

**Where**:
- `phentrieve/embeddings.py::load_embedding_model` (one model per (name, device, trust_remote_code) triple)
- `phentrieve/retrieval/details_enrichment.py::_get_hpo_database` (one SQLite connection per process)
- `phentrieve/config.py::load_user_config`, `get_default_index_dir`, `get_default_data_dir`, etc. (each returns a single pinned value)
- `similarity_router.py::load_hpo_graph_data` (~2M ontology graph, loaded once)

**Why `@lru_cache` and not TTL**:
- These are **true singletons**: the underlying data is immutable within a process run. Reloading would waste memory and time with no behavioral change.
- The key space is small and bounded: model names in phentrieve.yaml, DB filename, HPO ontology. Unlike the API model cache, a CLI run or benchmark pass does not rotate through dozens of distinct keys.
- Python's `@lru_cache` is thread-safe for reads.

**When to NOT use `@lru_cache`**:
- Anything where the key space is user-controlled and effectively unbounded (request parameters, query strings). Use `TTLCache` with a bounded `maxsize` instead.
- Anything with a lifecycle managed by an app factory (FastAPI startup/shutdown). Use module-level `TTLCache` + `cleanup_*` in lifespan hooks.
- Mutable state. `@lru_cache` caches **outputs**, not state — do not use it for anything that writes.

## Decision matrix

| Situation | Use |
|---|---|
| API request-scoped, key may grow unbounded | `TTLCache(maxsize=N, ttl=T)` + lock |
| Process-wide singleton, immutable data, bounded keys | `@lru_cache(maxsize=M)` |
| Per-request memoization | FastAPI dependency with `Depends()`, no caching |
| Coroutine/async safety across the event loop | `asyncio.Lock` per key (see `_get_lock_for_model`) |
| Tests need to reset caches | Both patterns expose `.cache_clear()` — call it in fixtures |

## Audit checklist

When adding a new cache:
- [ ] Is the key space bounded? (If no → TTLCache)
- [ ] Is the data mutable during the process run? (If yes → don't cache)
- [ ] Is there a lifecycle hook to clear it? (If it's module-level, add it to the relevant `cleanup_*` function)
- [ ] Are tests cleaning up? (`cache.cache_clear()` in `setup_method` / fixture teardown)
- [ ] Is there a concurrency concern? (If multi-threaded writes possible → wrap in `_cache_lock`)

## History

- **Pre-PR #191**: `api/dependencies.py` used **unbounded** module-level dicts (`LOADED_SBERT_MODELS`, `LOADED_RETRIEVERS`, `LOADED_CROSS_ENCODERS`). Cross-encoder has since been removed entirely with the reranker feature.
- **PR #191 (2026-04-10)**: Model caches converted to `TTLCache`, `_cache_lock` introduced, `cleanup_model_caches()` wired into lifespan shutdown.
- **PR #191 follow-up**: `MODEL_LOADING_STATUS` and `MODEL_LOAD_LOCKS` also converted to `TTLCache` to close the unbounded-tracking-dict gap.
```

- [ ] **Step 3: Verify the file is tracked and the doc tree is happy**

Run:
```bash
ls docs/architecture/ 2>/dev/null || mkdir -p docs/architecture
ls docs/architecture/caching.md
```

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/caching.md
git commit -m "$(cat <<'EOF'
docs(architecture): document TTLCache vs lru_cache decisions

Adds docs/architecture/caching.md explaining the two caching patterns
used in the codebase and when to pick each:

- cachetools.TTLCache + _cache_lock for API model caches where the
  key space is user-controlled and can grow unbounded over time
- @functools.lru_cache for process-wide singletons with immutable
  data and bounded key spaces (HPO graph, config, DB connection)

Includes a decision matrix, an audit checklist for new caches, and
a history section tracking the PR #191 conversion from unbounded
dicts to TTLCache.

Addresses Finding #3 residual from the updated
CODE-QUALITY-REVIEW-2026-04-09.md — the remaining @lru_cache sites
are intentional singletons, not a cleanup target.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification and push

- [ ] **Step 1: Update the status dashboard in the review document**

Read `plan/05-analysis/CODE-QUALITY-REVIEW-2026-04-09.md` first. Update the Status Dashboard rows for Findings 2, 4, 6, and 11 from ❌/⚠️ to ✅ with a brief note pointing at the Phase 2 commits. Leave Finding 8 status alone (still "Mostly fixed" — full coverage ratcheting is ongoing).

- [ ] **Step 2: Run the complete CI-equivalent check locally**

Run:
```bash
make check typecheck-fast test
```

Expected: all green. Full test count should be approximately 918 Python tests (899 baseline + 8 hpo_extraction char + 11 pipeline char + 4 error response = 922 ± minor dead test removal).

- [ ] **Step 3: Run the frontend tests too for symmetry**

Run:
```bash
make frontend-test frontend-build
```

Expected: 107 frontend tests pass; build succeeds.

- [ ] **Step 4: Run a benchmark smoke-test to confirm no retrieval regression**

Run:
```bash
uv run phentrieve benchmark run --test-file tiny_v1.json 2>&1 | tail -20
```

Expected: 9/9 test cases complete with MRR and Hit@k metrics in the same ballpark as before.

- [ ] **Step 5: Commit the status dashboard update**

```bash
git add plan/05-analysis/CODE-QUALITY-REVIEW-2026-04-09.md
git commit -m "$(cat <<'EOF'
docs(analysis): mark Phase 2 findings closed in status dashboard

Updates the CODE-QUALITY-REVIEW-2026-04-09.md status dashboard to
reflect the Phase 2 extension work on PR #191:

- Finding #2 (HPO extraction orchestration SRP) → closed by Task 5
- Finding #4 residual (sys.path hack) → closed by Task 2
- Finding #6 (chunker pipeline + FinalChunkCleaner) → closed by Tasks 7, 8
- Finding #11 residual (ErrorResponse schema) → closed by Task 3
- Visualization dead code → closed by Task 1
- @lru_cache audit → documented in docs/architecture/caching.md (Task 9)

Remaining open items: coverage ratcheting for untested modules
(ongoing), dual-package versioning documentation, config consolidation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Push to the PR branch**

Run:
```bash
git push origin improve/code-quality-2026-04
```

- [ ] **Step 7: Wait ~30 seconds, then verify CI**

Run:
```bash
sleep 30 && gh pr view 191 --json statusCheckRollup 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); [print(c['conclusion'] or c.get('status',''), c['name']) for c in d['statusCheckRollup']]"
```

Expected: CI starts re-running the new commits. Ideally all 17 checks green within 10 minutes.

---

## Expected outcome

After all 10 tasks land on `improve/code-quality-2026-04`:

| Metric | Before Phase 2 | After Phase 2 |
|---|---|---|
| Total commits on branch | 54 | ~64 |
| Python tests | 899 | ~918 (+19 char tests + 4 error response) |
| `phentrieve/text_processing/hpo_extraction_orchestrator.py` LOC | 298 | ~60 (thin coordinator) |
| `phentrieve/text_processing/pipeline.py::_create_chunkers` LOC | 158 | ~24 |
| `phentrieve/text_processing/chunkers.py::FinalChunkCleaner.__init__` LOC | ~130 | ~100 (helper extracted) |
| `phentrieve/visualization/` LOC | 358 | 0 (deleted) |
| `api/main.py` import-time side effects | 1 (sys.path) | 0 |
| API `ErrorResponse` schema | none | standardized |
| `@lru_cache` decisions | undocumented | `docs/architecture/caching.md` |
| Open Critical Findings from review | 2, 4-residual, 6 | **none** |

PR #191 scope expands from "Streams A/B/C" to "Streams A/B/C + Phase 2 cleanup". The breaking-change footer stays as-is (only the reranker removal is breaking; all Phase 2 work is additive or internal refactoring).

## Execution notes

- **Do not parallelize these tasks.** Unlike PR #191's three streams, this plan has real dependencies: Task 5 depends on Task 4 (characterization tests), Task 7 depends on Task 6, Task 8 touches the same file as Task 7 (sequential), and the verification task depends on everything. Running sequentially also keeps the commit log clean and bisectable.
- **If a characterization test fails after a refactor**, the refactor is wrong, not the test. Fix the refactor.
- **If a characterization test needed updating to match new behavior**, stop and escalate — that means the refactor is changing behavior, not just structure.
- **If mypy flags unreachable code in `_create_chunkers`** after Task 7, the fix is to delete the unreachable block, not to add `# type: ignore`.
- **Do not change the public function signatures** of `orchestrate_hpo_extraction`, `TextProcessingPipeline.__init__`, `TextProcessingPipeline.process`, or `FinalChunkCleaner.__init__`. These are external contracts used by CLI, API, and tests.
