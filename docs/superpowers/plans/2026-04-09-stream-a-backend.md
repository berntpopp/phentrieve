# Stream A: Backend Refactoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose backend orchestrators, unify API model loading, fix import-time side effects, and introduce bounded caching — all with zero behavioral regression.

**Architecture:** Characterization tests first (Tasks 1-3), then structural refactoring (Tasks 4-8), then caching improvements (Task 9). Each refactoring task must keep all characterization tests green.

**Tech Stack:** Python 3.10+, FastAPI, pytest, cachetools, pysbd, sentence-transformers

**Branch:** `improve/backend-refactor`

**Spec:** `docs/superpowers/specs/2026-04-09-code-quality-improvements-design.md` (Stream A)

---

## Phase 1: Characterization Tests

### Task 1: Characterization Tests for Query Orchestrator

**Files:**
- Create: `tests/unit/retrieval/test_query_orchestrator_char.py`
- Read: `phentrieve/retrieval/query_orchestrator.py`

These tests lock current behavior of `process_query()` across all 3 code paths. They mock at the boundary (retriever, reranker, cross-encoder) and assert on output structure.

- [ ] **Step 1: Write test fixtures and helpers**

Create `tests/unit/retrieval/test_query_orchestrator_char.py`:

```python
"""Characterization tests for query_orchestrator.process_query().

These tests lock current behavior before refactoring.
They must pass identically before AND after orchestrator decomposition.
"""
import pytest
from unittest.mock import MagicMock, patch

from phentrieve.retrieval.query_orchestrator import (
    process_query,
    convert_results_to_candidates,
    convert_multi_vector_to_chromadb_format,
    format_results,
    segment_text,
    _InteractiveState,
)

pytestmark = pytest.mark.unit


def _make_chromadb_results(ids, scores, docs=None, metadatas=None):
    """Helper to create ChromaDB-style result dicts."""
    n = len(ids)
    return {
        "ids": [ids],
        "distances": [[1.0 - s for s in scores]],
        "documents": [docs or [f"Document for {hid}" for hid in ids]],
        "metadatas": [metadatas or [{"source": "test"} for _ in ids]],
    }


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.query.return_value = _make_chromadb_results(
        ids=["HP:0001250", "HP:0001251", "HP:0001252"],
        scores=[0.95, 0.85, 0.75],
    )
    return retriever


@pytest.fixture
def mock_cross_encoder_model():
    """Mock cross-encoder model (not the reranker module)."""
    model = MagicMock()
    model.predict.return_value = [0.92, 0.88, 0.70]
    return model
```

- [ ] **Step 2: Write tests for sentence mode path (lines 554-606)**

Add to the same file:

```python
class TestProcessQuerySentenceMode:
    """Tests for process_query with sentence_mode=True."""

    def test_sentence_mode_returns_per_sentence_results(self, mock_retriever):
        """Each sentence gets its own retrieval results."""
        results = process_query(
            text="Patient has seizures. Also has ataxia.",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
        )
        # Two sentences should produce up to 2 result sets
        assert isinstance(results, list)
        assert len(results) >= 1
        # Retriever should be called at least once per sentence
        assert mock_retriever.query.call_count >= 1

    def test_sentence_mode_with_reranking(self, mock_retriever, mock_cross_encoder_model):
        """Sentence mode with cross-encoder reranking."""
        with patch("phentrieve.retrieval.query_orchestrator.reranker") as mock_reranker_mod:
            mock_reranker_mod.protected_dense_rerank.return_value = [
                {"hpo_id": "HP:0001250", "english_doc": "Seizure", "metadata": {"source": "test"},
                 "bi_encoder_score": 0.95, "rank": 1, "comparison_text": "Seizure"},
                {"hpo_id": "HP:0001251", "english_doc": "Ataxia", "metadata": {"source": "test"},
                 "bi_encoder_score": 0.85, "rank": 2, "comparison_text": "Ataxia"},
            ]
            results = process_query(
                text="Patient has seizures.",
                retriever=mock_retriever,
                cross_encoder=mock_cross_encoder_model,
                num_results=3,
                similarity_threshold=0.5,
                sentence_mode=True,
                rerank_count=10,
            )
            assert isinstance(results, list)
            assert len(results) >= 1
            # Reranker should have been called
            assert mock_reranker_mod.protected_dense_rerank.call_count >= 1

    def test_sentence_mode_empty_text_returns_empty(self, mock_retriever):
        """Empty text returns empty results."""
        mock_retriever.query.return_value = _make_chromadb_results(ids=[], scores=[])
        results = process_query(
            text="",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
        )
        assert isinstance(results, list)
```

- [ ] **Step 3: Write tests for full-text mode path (lines 737-799)**

```python
class TestProcessQueryFullTextMode:
    """Tests for process_query with sentence_mode=False."""

    def test_full_text_without_reranking(self, mock_retriever):
        """Full text mode returns formatted results."""
        results = process_query(
            text="Patient presents with severe intellectual disability",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=False,
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        # Should have results with expected structure
        for result_set in results:
            assert "results" in result_set

    def test_full_text_with_reranking(self, mock_retriever, mock_cross_encoder_model):
        """Full text with cross-encoder reranking."""
        with patch("phentrieve.retrieval.query_orchestrator.reranker") as mock_reranker_mod:
            mock_reranker_mod.protected_dense_rerank.return_value = [
                {"hpo_id": "HP:0001250", "english_doc": "Seizure", "metadata": {"source": "test"},
                 "bi_encoder_score": 0.95, "rank": 1, "comparison_text": "Seizure"},
            ]
            results = process_query(
                text="Patient presents with severe intellectual disability",
                retriever=mock_retriever,
                cross_encoder=mock_cross_encoder_model,
                num_results=3,
                similarity_threshold=0.5,
                sentence_mode=False,
                rerank_count=10,
            )
            assert isinstance(results, list)
            assert mock_reranker_mod.protected_dense_rerank.call_count == 1
```

- [ ] **Step 4: Write tests for fallback path (sentence mode -> full text, lines 608-682)**

```python
class TestProcessQueryFallback:
    """Tests for sentence mode falling back to full text when no results."""

    def test_fallback_to_full_text_when_sentences_empty(self, mock_retriever):
        """When sentence mode yields no results, falls back to full text."""
        call_count = 0
        def query_side_effect(text, n_results=10):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First calls (per-sentence) return nothing
                return _make_chromadb_results(ids=[], scores=[])
            # Fallback call returns results
            return _make_chromadb_results(
                ids=["HP:0001250"], scores=[0.9]
            )

        mock_retriever.query.side_effect = query_side_effect
        results = process_query(
            text="Short text. Another sentence.",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
        )
        assert isinstance(results, list)
        # Should have called retriever for sentences + fallback
        assert mock_retriever.query.call_count >= 2
```

- [ ] **Step 5: Write tests for helper functions**

```python
class TestConvertResultsToCandidates:
    """Tests for convert_results_to_candidates."""

    def test_converts_valid_results(self):
        results = _make_chromadb_results(
            ids=["HP:0001250", "HP:0001251"],
            scores=[0.95, 0.85],
        )
        candidates = convert_results_to_candidates(results)
        assert len(candidates) == 2
        assert candidates[0]["hpo_id"] == "HP:0001250"
        assert candidates[0]["rank"] == 1
        assert abs(candidates[0]["bi_encoder_score"] - 0.95) < 0.01

    def test_empty_results_returns_empty_list(self):
        results = _make_chromadb_results(ids=[], scores=[])
        candidates = convert_results_to_candidates(results)
        assert candidates == []

    def test_none_results_returns_empty_list(self):
        candidates = convert_results_to_candidates(None)
        assert candidates == []


class TestSegmentText:
    """Tests for segment_text."""

    def test_segments_english(self):
        sentences = segment_text("Patient has seizures. Also has ataxia.")
        assert len(sentences) == 2

    def test_single_sentence(self):
        sentences = segment_text("Patient has seizures")
        assert len(sentences) == 1


class TestInteractiveState:
    """Tests for _InteractiveState class."""

    def test_default_values(self):
        state = _InteractiveState()
        assert state.model is None
        assert state.retriever is None
        assert state.cross_encoder is None
        assert state.multi_vector is False
        assert state.aggregation_strategy == "label_synonyms_max"
```

- [ ] **Step 6: Run tests to verify they pass against current code**

Run:
```bash
pytest tests/unit/retrieval/test_query_orchestrator_char.py -v
```
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/retrieval/test_query_orchestrator_char.py && git commit -m "test: add characterization tests for query_orchestrator

Lock current behavior of process_query() across all 3 code paths
(sentence mode, full-text mode, sentence-with-fallback) plus helper
functions. These tests must remain green through refactoring."
```

---

### Task 2: Characterization Tests for API Dependencies

**Files:**
- Create: `tests/unit/api/test_dependencies_char.py`
- Read: `api/dependencies.py`

- [ ] **Step 1: Write characterization tests for model loading**

Create `tests/unit/api/test_dependencies_char.py`:

```python
"""Characterization tests for api/dependencies.py model loading.

Tests the status tracking, double-check locking, timeout behavior, and cache
hit paths. Must pass identically before AND after dependency unification.
"""
import sys
from pathlib import Path

# API import path workaround (see tests/unit/api/README.md)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.dependencies import (
    LOADED_SBERT_MODELS,
    LOADED_CROSS_ENCODERS,
    MODEL_LOADING_STATUS,
    MODEL_LOAD_LOCKS,
    MODEL_LOADING_TASKS,
    get_sbert_model_dependency,
    get_cross_encoder_dependency,
    _get_lock_for_model,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_global_state():
    """Reset all module-level state between tests."""
    LOADED_SBERT_MODELS.clear()
    LOADED_CROSS_ENCODERS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_LOCKS.clear()
    MODEL_LOADING_TASKS.clear()
    yield
    LOADED_SBERT_MODELS.clear()
    LOADED_CROSS_ENCODERS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_LOCKS.clear()
    MODEL_LOADING_TASKS.clear()


class TestGetLockForModel:
    def test_creates_lock_for_new_model(self):
        lock = _get_lock_for_model("test-model")
        assert isinstance(lock, asyncio.Lock)

    def test_returns_same_lock_for_same_model(self):
        lock1 = _get_lock_for_model("test-model")
        lock2 = _get_lock_for_model("test-model")
        assert lock1 is lock2

    def test_different_models_get_different_locks(self):
        lock1 = _get_lock_for_model("model-a")
        lock2 = _get_lock_for_model("model-b")
        assert lock1 is not lock2


class TestSbertModelCacheHit:
    @pytest.mark.asyncio
    async def test_returns_cached_model(self):
        mock_model = MagicMock()
        LOADED_SBERT_MODELS["test-model"] = mock_model
        result = await get_sbert_model_dependency("test-model")
        assert result is mock_model


class TestSbertModelStatusFailed:
    @pytest.mark.asyncio
    async def test_failed_status_raises_503(self):
        MODEL_LOADING_STATUS["test-model"] = "failed"
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="failed to load"):
            await get_sbert_model_dependency("test-model")


class TestCrossEncoderCacheHit:
    @pytest.mark.asyncio
    async def test_returns_cached_cross_encoder(self):
        mock_ce = MagicMock()
        LOADED_CROSS_ENCODERS["test-ce"] = mock_ce
        result = await get_cross_encoder_dependency("test-ce")
        assert result is mock_ce

    @pytest.mark.asyncio
    async def test_none_model_returns_none(self):
        result = await get_cross_encoder_dependency(None)
        assert result is None


class TestCrossEncoderStatusFailed:
    @pytest.mark.asyncio
    async def test_failed_status_raises_503(self):
        MODEL_LOADING_STATUS["test-ce"] = "failed"
        from fastapi import HTTPException
        with pytest.raises(HTTPException, match="failed to load"):
            await get_cross_encoder_dependency("test-ce")
```

- [ ] **Step 2: Run tests**

Run:
```bash
PYTHONPATH=$PWD pytest tests/unit/api/test_dependencies_char.py -v
```
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/api/test_dependencies_char.py && git commit -m "test: add characterization tests for API dependency loading

Lock current behavior of get_sbert_model_dependency and
get_cross_encoder_dependency: cache hits, status transitions,
lock management. Must remain green through unification refactor."
```

---

### Task 3: Characterization Tests for API Main

**Files:**
- Create: `tests/unit/api/test_main_char.py`
- Read: `api/main.py`

- [ ] **Step 1: Write tests for app structure and endpoints**

Create `tests/unit/api/test_main_char.py`:

```python
"""Characterization tests for api/main.py.

Tests app creation, router mounting, CORS config, and root endpoint.
Must pass before AND after factory/lifespan refactoring.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Create test client. Import app here to isolate import-time effects."""
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestAppStructure:
    def test_app_has_title(self, client):
        assert client.app.title == "Phentrieve API"

    def test_app_has_lifespan(self, client):
        assert client.app.router.lifespan_context is not None


class TestRootEndpoint:
    def test_root_returns_api_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["api"] == "Phentrieve API"
        assert "version" in data
        assert "endpoints" in data

    def test_root_lists_expected_endpoint_categories(self, client):
        response = client.get("/")
        data = response.json()
        endpoints = data["endpoints"]
        assert "HPO Term Query" in endpoints
        assert "Text Processing" in endpoints
        assert "HPO Term Similarity" in endpoints
        assert "Health Check" in endpoints


class TestRouterMounting:
    def test_health_endpoint_exists(self, client):
        response = client.get("/api/v1/health/")
        # Should return 200 (health check doesn't depend on models)
        assert response.status_code == 200

    def test_docs_endpoint_exists(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
```

- [ ] **Step 2: Run tests**

Run:
```bash
PYTHONPATH=$PWD pytest tests/unit/api/test_main_char.py -v
```
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/api/test_main_char.py && git commit -m "test: add characterization tests for API main app structure

Lock current behavior of app creation, router mounting, root endpoint
response structure. Must remain green through factory/lifespan refactor."
```

---

## Phase 2: Structural Refactoring

### Task 4: Extract Shared Utility (`convert_multi_vector_to_chromadb_format`)

**Files:**
- Create: `phentrieve/retrieval/utils.py`
- Modify: `phentrieve/retrieval/query_orchestrator.py:61`
- Modify: `phentrieve/evaluation/runner.py:52`

- [ ] **Step 1: Create the shared utility module**

Create `phentrieve/retrieval/utils.py`:

```python
"""Shared retrieval utilities.

Functions used across multiple retrieval and evaluation modules.
"""
from typing import Any


def convert_multi_vector_to_chromadb_format(
    multi_vector_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert multi-vector aggregated results to ChromaDB-style format.

    This allows reusing existing format_results() and output formatters.

    Args:
        multi_vector_results: List of aggregated results from query_multi_vector()

    Returns:
        Dictionary in ChromaDB query result format
    """
    if not multi_vector_results:
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}

    ids = [r["hpo_id"] for r in multi_vector_results]
    distances = [1.0 - r.get("combined_score", 0.0) for r in multi_vector_results]
    documents = [r.get("english_label", "") for r in multi_vector_results]
    metadatas = [r.get("metadata", {}) for r in multi_vector_results]

    return {
        "ids": [ids],
        "distances": [distances],
        "documents": [documents],
        "metadatas": [metadatas],
    }
```

- [ ] **Step 2: Update query_orchestrator.py to import from utils**

In `phentrieve/retrieval/query_orchestrator.py`, replace the local function definition (around line 61) with an import:

```python
# Remove the local convert_multi_vector_to_chromadb_format function definition.
# Add this import at the top with the other imports:
from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format
```

- [ ] **Step 3: Update evaluation/runner.py to import from utils**

In `phentrieve/evaluation/runner.py`, find the duplicated function (around line 52) and replace with the same import:

```python
from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format
```

Remove the local definition.

- [ ] **Step 4: Verify all tests pass**

Run:
```bash
make test
```
Expected: All pass — behavior is identical, only the import location changed.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/retrieval/utils.py phentrieve/retrieval/query_orchestrator.py phentrieve/evaluation/runner.py && git commit -m "refactor: extract convert_multi_vector_to_chromadb_format to retrieval/utils.py

DRY fix — this function was duplicated identically in
query_orchestrator.py and evaluation/runner.py. Now shared
from a single location."
```

---

### Task 5: Extract Interactive State

**Files:**
- Create: `phentrieve/retrieval/interactive_state.py`
- Modify: `phentrieve/retrieval/query_orchestrator.py`

- [ ] **Step 1: Create the interactive state module**

Create `phentrieve/retrieval/interactive_state.py`:

```python
"""Interactive query session state management.

Encapsulates state needed across interactive query sessions in the CLI.
"""
from typing import Any, Optional

from phentrieve.retrieval.dense_retriever import DenseRetriever


class InteractiveState:
    """Container for interactive mode state across query sessions."""

    model: Optional[Any] = None
    retriever: Optional[DenseRetriever] = None
    cross_encoder: Optional[Any] = None
    query_assertion_detector: Optional[Any] = None
    multi_vector: bool = False
    aggregation_strategy: str = "label_synonyms_max"
    component_weights: Optional[dict[str, float]] = None
    custom_formula: Optional[str] = None


# Singleton instance for interactive mode
interactive_state = InteractiveState()
```

- [ ] **Step 2: Update query_orchestrator.py to import from interactive_state**

In `query_orchestrator.py`:
- Remove `_InteractiveState` class definition (lines 43-58) and `_interactive_state` singleton (line 58)
- Add import: `from phentrieve.retrieval.interactive_state import InteractiveState, interactive_state`
- Replace all references to `_InteractiveState` with `InteractiveState` and `_interactive_state` with `interactive_state`

- [ ] **Step 3: Verify all tests pass**

Run:
```bash
make test && pytest tests/unit/retrieval/test_query_orchestrator_char.py -v
```
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add phentrieve/retrieval/interactive_state.py phentrieve/retrieval/query_orchestrator.py && git commit -m "refactor: extract InteractiveState to its own module

Separates CLI interactive session state from query execution logic.
First step in query_orchestrator decomposition."
```

---

### Task 6: Extract Retrieval Pipeline (Eliminate 3x Duplication)

**Files:**
- Create: `phentrieve/retrieval/pipeline.py`
- Modify: `phentrieve/retrieval/query_orchestrator.py`

This is the highest-leverage refactoring — extracting the duplicated retrieve→convert→rerank→format sequence.

- [ ] **Step 1: Create the pipeline module**

Create `phentrieve/retrieval/pipeline.py`:

```python
"""Single-vector retrieval pipeline.

Extracts the retrieve -> convert -> rerank -> format_to_chromadb sequence
that was duplicated 3x in query_orchestrator.py (lines 554-606, 637-679, 737-799).
"""
from typing import Any, Optional

from phentrieve.config import DEFAULT_DENSE_TRUST_THRESHOLD
from phentrieve.retrieval import reranker as reranker_module
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.query_orchestrator import convert_results_to_candidates


def execute_single_vector_pipeline(
    retriever: DenseRetriever,
    text: str,
    num_results: int,
    cross_encoder: Optional[Any] = None,
    rerank_count: Optional[int] = None,
    debug: bool = False,
    output_func=print,
) -> dict[str, Any]:
    """Execute single-vector retrieval with optional reranking.

    Returns results in ChromaDB format (reranked if cross-encoder provided).

    Args:
        retriever: Dense retriever instance
        text: Query text
        num_results: Number of results to return
        cross_encoder: Optional cross-encoder model for reranking
        rerank_count: Number of candidates to rerank (None = no reranking)
        debug: Enable debug output
        output_func: Function for debug output

    Returns:
        ChromaDB-format result dict (possibly reranked)
    """
    # Determine query count
    if cross_encoder and rerank_count is not None:
        query_count = rerank_count * 2
    else:
        query_count = num_results * 2

    # Query the retriever
    results = retriever.query(text, n_results=query_count)

    # Rerank with cross-encoder if available
    if cross_encoder and rerank_count is not None:
        if debug:
            output_func("[DEBUG] Reranking with protected dense retrieval")
        try:
            candidates = convert_results_to_candidates(results)
            reranked_candidates = reranker_module.protected_dense_rerank(
                text,
                candidates,
                cross_encoder,
                trust_threshold=DEFAULT_DENSE_TRUST_THRESHOLD,
            )
            # Convert back to ChromaDB format
            results = {
                "ids": [[c["hpo_id"] for c in reranked_candidates]],
                "metadatas": [[c["metadata"] for c in reranked_candidates]],
                "documents": [[c["english_doc"] for c in reranked_candidates]],
                "distances": [
                    [1.0 - c.get("bi_encoder_score", 0.0) for c in reranked_candidates]
                ],
            }
        except Exception as e:
            if debug:
                output_func(f"[DEBUG] Error during re-ranking: {str(e)}")
            # Fall through to return unranked results

    return results
```

- [ ] **Step 2: Write a test for the pipeline**

Add to `tests/unit/retrieval/test_query_orchestrator_char.py`:

```python
from phentrieve.retrieval.pipeline import execute_single_vector_pipeline


class TestExecuteSingleVectorPipeline:
    def test_without_reranking(self, mock_retriever):
        results = execute_single_vector_pipeline(
            retriever=mock_retriever,
            text="seizures",
            num_results=3,
        )
        assert "ids" in results
        assert len(results["ids"][0]) == 3
        mock_retriever.query.assert_called_once()

    def test_with_reranking(self, mock_retriever, mock_cross_encoder_model):
        with patch("phentrieve.retrieval.pipeline.reranker_module") as mock_reranker:
            mock_reranker.protected_dense_rerank.return_value = [
                {"hpo_id": "HP:0001250", "english_doc": "Seizure",
                 "metadata": {"source": "test"}, "bi_encoder_score": 0.95,
                 "rank": 1, "comparison_text": "Seizure"},
            ]
            results = execute_single_vector_pipeline(
                retriever=mock_retriever,
                text="seizures",
                num_results=3,
                cross_encoder=mock_cross_encoder_model,
                rerank_count=10,
            )
            assert "ids" in results
            mock_reranker.protected_dense_rerank.assert_called_once()
```

- [ ] **Step 3: Run the pipeline test**

Run:
```bash
pytest tests/unit/retrieval/test_query_orchestrator_char.py::TestExecuteSingleVectorPipeline -v
```
Expected: All PASS.

- [ ] **Step 4: Refactor query_orchestrator to use the pipeline**

Replace the 3 duplicated sequences in `process_query()` with calls to `execute_single_vector_pipeline()`. The key change: each of the 3 locations (sentence mode ~line 554, fallback ~line 637, full-text ~line 737) should call the pipeline function then pass results to `format_results()`.

Example for the sentence mode path (around line 547-602):

```python
# Before: ~55 lines of inline retrieve+rerank+convert
# After:
from phentrieve.retrieval.pipeline import execute_single_vector_pipeline

# In the sentence loop:
results = execute_single_vector_pipeline(
    retriever=retriever,
    text=sentence,
    num_results=num_results,
    cross_encoder=cross_encoder,
    rerank_count=rerank_count,
    debug=debug,
    output_func=output_func,
)
reranked = cross_encoder is not None and rerank_count is not None
formatted = format_results(
    results=results,
    threshold=similarity_threshold,
    max_results=num_results,
    query=sentence,
    reranked=reranked,
    original_query_assertion_status=original_query_assertion_status,
    original_query_assertion_details=original_query_assertion_details,
)
```

Apply the same transformation to the other two locations.

- [ ] **Step 5: Verify all characterization tests still pass**

Run:
```bash
pytest tests/unit/retrieval/test_query_orchestrator_char.py -v && make test
```
Expected: All PASS. This is the critical gate — if any characterization test fails, the refactoring changed behavior.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/retrieval/pipeline.py phentrieve/retrieval/query_orchestrator.py tests/unit/retrieval/test_query_orchestrator_char.py && git commit -m "refactor: extract single-vector pipeline, eliminate 3x duplication

The retrieve->convert->rerank->format sequence was duplicated at
3 locations in query_orchestrator.py. Extracted to pipeline.py
execute_single_vector_pipeline(). All characterization tests pass."
```

---

### Task 7: Unify API Model Loading

**Files:**
- Modify: `api/dependencies.py`

- [ ] **Step 1: Extract the shared loading function**

Add this function to `api/dependencies.py` before `get_sbert_model_dependency`:

```python
async def _load_model_with_status_tracking(
    model_name: str,
    cache_dict: dict,
    is_sbert: bool,
    trust_remote_code: bool,
    device: str | None,
    timeout: float,
    model_type_label: str,
) -> Any:
    """Shared model loading with status tracking, locking, and timeout.

    Replaces duplicated logic in get_sbert_model_dependency and
    get_cross_encoder_dependency.
    """
    # Cache hit (fast path)
    if model_name in cache_dict and cache_dict[model_name] is not None:
        logger.debug("API: Returning cached %s: %s", model_type_label, _sanitize(model_name))
        return cache_dict[model_name]

    lock = _get_lock_for_model(model_name)
    async with lock:
        # Re-check after acquiring lock
        if model_name in cache_dict and cache_dict[model_name] is not None:
            logger.debug(
                "API: Returning cached %s (post-lock): %s",
                model_type_label, _sanitize(model_name),
            )
            return cache_dict[model_name]

        current_status = MODEL_LOADING_STATUS.get(model_name, "not_loaded")

        if current_status == "loading":
            logger.info(
                "API: %s '%s' is loading. Waiting up to %ss...",
                model_type_label, _sanitize(model_name), timeout,
            )
            if model_name in MODEL_LOADING_TASKS:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(MODEL_LOADING_TASKS[model_name]),
                        timeout=timeout,
                    )
                    if model_name in cache_dict and cache_dict[model_name] is not None:
                        logger.info(
                            "API: %s '%s' finished loading, returning it.",
                            model_type_label, _sanitize(model_name),
                        )
                        return cache_dict[model_name]
                except asyncio.TimeoutError:
                    logger.warning(
                        "API: %s '%s' loading timeout (%ss).",
                        model_type_label, _sanitize(model_name), timeout,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"{model_type_label} '{model_name}' is taking longer than expected to load. Please try again.",
                        headers={"Retry-After": "30"},
                    )
            raise HTTPException(
                status_code=503,
                detail=f"{model_type_label} '{model_name}' is currently being prepared. Please try again.",
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error("API: %s '%s' failed to load.", model_type_label, _sanitize(model_name))
            raise HTTPException(
                status_code=503,
                detail=f"{model_type_label} '{model_name}' failed to load and is unavailable.",
            )

        # not_loaded: initiate loading
        logger.info("API: Initiating load for %s: %s", model_type_label, _sanitize(model_name))
        MODEL_LOADING_STATUS[model_name] = "loading"
        task = asyncio.create_task(
            _load_model_in_background(model_name, is_sbert, trust_remote_code, device)
        )
        MODEL_LOADING_TASKS[model_name] = task

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            if model_name in cache_dict and cache_dict[model_name] is not None:
                logger.info("API: %s '%s' loaded successfully.", model_type_label, _sanitize(model_name))
                return cache_dict[model_name]
            raise HTTPException(
                status_code=500,
                detail=f"{model_type_label} '{model_name}' failed to load due to an internal error.",
            )
        except asyncio.TimeoutError:
            logger.warning(
                "API: %s '%s' loading timeout (%ss) on first request.",
                model_type_label, _sanitize(model_name), timeout,
            )
            raise HTTPException(
                status_code=503,
                detail=f"{model_type_label} '{model_name}' is taking longer than expected to load. Please try again.",
                headers={"Retry-After": "30"},
            )
```

- [ ] **Step 2: Simplify get_sbert_model_dependency to use the shared function**

```python
async def get_sbert_model_dependency(
    model_name_requested: str | None = None,
    trust_remote_code: bool = False,
    device_override: str | None = None,
) -> SentenceTransformer:
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE
    return await _load_model_with_status_tracking(
        model_name=model_name,
        cache_dict=LOADED_SBERT_MODELS,
        is_sbert=True,
        trust_remote_code=trust_remote_code,
        device=device,
        timeout=SBERT_LOAD_TIMEOUT,
        model_type_label="SBERT model",
    )
```

- [ ] **Step 3: Simplify get_cross_encoder_dependency to use the shared function**

```python
async def get_cross_encoder_dependency(
    reranker_model_name: str | None = None,
    device_override: str | None = None,
) -> CrossEncoder | None:
    if not reranker_model_name:
        return None
    device = device_override or DEFAULT_DEVICE
    return await _load_model_with_status_tracking(
        model_name=reranker_model_name,
        cache_dict=LOADED_CROSS_ENCODERS,
        is_sbert=False,
        trust_remote_code=False,
        device=device,
        timeout=CROSS_ENCODER_LOAD_TIMEOUT,
        model_type_label="CrossEncoder",
    )
```

- [ ] **Step 4: Verify characterization tests pass**

Run:
```bash
PYTHONPATH=$PWD pytest tests/unit/api/test_dependencies_char.py -v && make test
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add api/dependencies.py && git commit -m "refactor: unify model loading into _load_model_with_status_tracking

Replaces ~140 lines of duplicated logic between
get_sbert_model_dependency and get_cross_encoder_dependency.
Both now delegate to a single shared function."
```

---

### Task 8: API Lifecycle — Factory, Version Fix, Import-Time Cleanup

**Files:**
- Modify: `api/main.py`
- Modify: `api/routers/similarity_router.py`

- [ ] **Step 1: Fix version hardcoding in main.py**

In `api/main.py`, add import and fix version:

```python
from api.version import get_api_version

# Line 106 - change:
app = FastAPI(title="Phentrieve API", version=get_api_version(), lifespan=lifespan)

# Line 188 - change the root endpoint:
# Replace hardcoded "0.1.0" with:
"version": get_api_version(),
```

- [ ] **Step 2: Move MCP mounting into lifespan**

In `api/main.py`, move the `_try_mount_mcp()` call from module-level (line 171) into the lifespan, right before `yield`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing model pre-loading code ...

    # Mount MCP if enabled (moved from module-level)
    _try_mount_mcp()

    yield

    # Shutdown
    logger.info("Shutting down Phentrieve API...")
```

Remove the module-level `_try_mount_mcp()` call at line 171.

- [ ] **Step 3: Move import-time graph loading from similarity_router**

In `api/routers/similarity_router.py`, remove the import-time call at line 61:

```python
# Remove this line:
# load_hpo_graph_data()
```

The `load_hpo_graph_data()` function already uses `@lru_cache`, so it will load lazily on first use. No lifespan hook needed — the cache handles it.

- [ ] **Step 4: Wrap create_app factory**

In `api/main.py`, wrap the module-level app creation in a factory function while preserving the module-level `app` variable for compatibility:

```python
def create_app() -> FastAPI:
    """Application factory for the Phentrieve API."""
    application = FastAPI(
        title="Phentrieve API",
        version=get_api_version(),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
    )

    application.include_router(query_router.router, prefix="/api/v1/query", tags=["HPO Term Query"])
    application.include_router(health.router, prefix="/api/v1/health", tags=["Health Check"])
    application.include_router(system.router)
    application.include_router(similarity_router.router, prefix="/api/v1/similarity", tags=["HPO Term Similarity"])
    application.include_router(config_info_router.router, prefix="/api/v1")
    application.include_router(text_processing_router.router, tags=["Text Processing and HPO Extraction"])

    return application


# Module-level app for uvicorn and existing imports
app = create_app()
```

Move the `@app.get("/")` root endpoint into the factory before the return, using `@application.get("/")`.

- [ ] **Step 5: Verify characterization tests pass**

Run:
```bash
PYTHONPATH=$PWD pytest tests/unit/api/test_main_char.py -v && make test
```
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add api/main.py api/routers/similarity_router.py && git commit -m "refactor: add create_app factory, fix version, remove import-time side effects

- Add create_app() factory function for testability
- Fix hardcoded '0.1.0' version -> get_api_version()
- Move MCP mounting from module-level to lifespan
- Remove eager load_hpo_graph_data() from similarity_router import time
All characterization tests pass."
```

---

### Task 9: Bounded Caching with cachetools

**Files:**
- Modify: `api/dependencies.py`
- Modify: `pyproject.toml` (add dependency)

- [ ] **Step 1: Add cachetools dependency**

Run:
```bash
uv add cachetools
```

- [ ] **Step 2: Replace unbounded dicts with TTLCache**

In `api/dependencies.py`, replace the global cache dicts:

```python
# Before:
LOADED_SBERT_MODELS: dict[str, SentenceTransformer] = {}
LOADED_RETRIEVERS: dict[str, DenseRetriever] = {}
LOADED_CROSS_ENCODERS: dict[str, Optional[CrossEncoder]] = {}

# After:
import threading
from cachetools import TTLCache

# Bounded model caches with 1-hour TTL and max 10 models
# TTL prevents stale models in long-running processes
_cache_lock = threading.Lock()
LOADED_SBERT_MODELS: TTLCache = TTLCache(maxsize=10, ttl=3600)
LOADED_RETRIEVERS: TTLCache = TTLCache(maxsize=10, ttl=3600)
LOADED_CROSS_ENCODERS: TTLCache = TTLCache(maxsize=10, ttl=3600)
```

- [ ] **Step 3: Add cleanup callable for lifespan shutdown**

Add to `api/dependencies.py`:

```python
def cleanup_model_caches() -> None:
    """Clear all model caches. Called during app shutdown."""
    with _cache_lock:
        LOADED_SBERT_MODELS.clear()
        LOADED_RETRIEVERS.clear()
        LOADED_CROSS_ENCODERS.clear()
        MODEL_LOADING_STATUS.clear()
        MODEL_LOAD_LOCKS.clear()
        MODEL_LOADING_TASKS.clear()
    logger.info("API: All model caches cleared.")
```

- [ ] **Step 4: Wire cleanup into lifespan shutdown**

In `api/main.py`, update the lifespan shutdown:

```python
from api.dependencies import cleanup_model_caches

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... startup code ...
    yield
    # Shutdown
    cleanup_model_caches()
    logger.info("Shutting down Phentrieve API...")
```

- [ ] **Step 5: Update characterization test fixtures for TTLCache**

In `tests/unit/api/test_dependencies_char.py`, the `clean_global_state` fixture needs no changes — `TTLCache.clear()` works the same as `dict.clear()`. Verify by running tests.

- [ ] **Step 6: Verify all tests pass**

Run:
```bash
PYTHONPATH=$PWD pytest tests/unit/api/test_dependencies_char.py -v && make test
```
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add api/dependencies.py api/main.py pyproject.toml uv.lock && git commit -m "refactor: replace unbounded model caches with cachetools.TTLCache

LOADED_SBERT_MODELS, LOADED_RETRIEVERS, LOADED_CROSS_ENCODERS
are now bounded (maxsize=10, ttl=3600). Cleanup callable wired
into app lifespan shutdown. Same access patterns, just bounded."
```

---

### Task 10: Run Verification Gates

- [ ] **Step 1: Gate 1 — Lint and type check**

Run:
```bash
make check && make typecheck-fast
```
Expected: Zero errors.

- [ ] **Step 2: Gate 2 — All existing tests pass**

Run:
```bash
make test
```
Expected: All pass.

- [ ] **Step 3: Gate 3 — All characterization tests pass**

Run:
```bash
pytest tests/unit/retrieval/test_query_orchestrator_char.py tests/unit/api/test_dependencies_char.py tests/unit/api/test_main_char.py -v
```
Expected: All PASS.

- [ ] **Step 4: Gate 4 — Bounded cache verification**

Run:
```bash
python3 -c "
from cachetools import TTLCache
cache = TTLCache(maxsize=2, ttl=3600)
cache['a'] = 1
cache['b'] = 2
cache['c'] = 3  # Should evict 'a'
assert 'a' not in cache, 'Eviction failed'
assert 'b' in cache
assert 'c' in cache
print('TTLCache eviction works correctly')
"
```
Expected: "TTLCache eviction works correctly"

- [ ] **Step 5: Gate 5 — query_orchestrator.py LOC check**

Run:
```bash
wc -l phentrieve/retrieval/query_orchestrator.py
```
Expected: Significantly less than 1057 (target: <350 for the coordinator).

- [ ] **Step 6: Gate 6 — No import-time side effects in API**

Run:
```bash
python3 -c "
import time
start = time.time()
# Import should NOT trigger heavy operations
import api.main
elapsed = time.time() - start
print(f'Import took {elapsed:.2f}s')
assert elapsed < 2.0, f'Import too slow ({elapsed:.2f}s) — likely has import-time side effects'
print('No import-time side effects detected')
" 2>&1 | head -5
```
Expected: Import under 2s, no side effects message.
