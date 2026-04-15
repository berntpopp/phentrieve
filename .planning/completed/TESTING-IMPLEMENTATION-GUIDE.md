# Testing Coverage Expansion - Implementation Guide (REVISED)

**Status:** ✅ Ready to Execute (Revised after senior review)
**Created:** 2025-11-17
**Revised:** 2025-11-17 (KISS + Incremental approach)
**Parent Plan:** [TESTING-COVERAGE-EXPANSION-PLAN.md](./TESTING-COVERAGE-EXPANSION-PLAN.md)
**Priority:** CRITICAL - Zero coverage on main API endpoints

---

## Executive Summary

**Revised approach** based on critical review findings. This guide now follows **KISS principle** with **incremental validation** and **existing patterns**.

### Current Coverage (As of 2025-11-17)

**✅ EXCELLENT (Already at 98-100%):**
- `phentrieve/retrieval/dense_retriever.py` - **98%**
- `phentrieve/retrieval/output_formatters.py` - **100%**
- `phentrieve/retrieval/reranker.py` - **100%**

**❌ CRITICAL GAPS (0% coverage):**
- `api/routers/query_router.py` - **0%** (63 statements) ⚠️ **HIGHEST PRIORITY**
- `api/routers/similarity_router.py` - **0%** (67 statements)
- `api/routers/text_processing_router.py` - **0%** (145 statements)
- `phentrieve/retrieval/query_orchestrator.py` - **8%** (258/279 uncovered)

### Revised Strategy: KISS + Incremental

```
Day 1 (2h): Validate TestClient approach (1-2 minimal tests)
Day 2 (4h): Query Router - Basic tests (5 tests)
Day 3 (4h): Query Router - Expand (10 more tests)
Day 4 (4h): Query Orchestrator tests (8 tests)
Day 5 (2h): Verify coverage, iterate

Total: 16 hours, incremental validation at each step
```

---

## Key Principles (Lessons from Review)

### ✅ DO:
1. **Mock ONLY external dependencies** (ChromaDB, ML models, network)
2. **Follow existing patterns** (pytestmark, Arrange/Act/Assert)
3. **Extend existing fixtures** (don't duplicate)
4. **Validate incrementally** (1 test → 5 tests → 10 tests)
5. **Keep it simple** (parametrize > factories)

### ❌ DON'T:
1. **Over-mock** - Don't mock our own business logic!
2. **Skip markers** - Always add `pytestmark`
3. **Duplicate fixtures** - Check what exists first
4. **Create god fixtures** - Keep fixtures focused
5. **Write 500 lines before testing** - Validate first!

---

## Day 1: Validate Approach (2 hours)

### Task 1.1: Create Minimal Test File (30 min)

**File:** `tests/unit/api/test_query_router.py` (CREATE NEW)

```python
"""Tests for API query router (main endpoint).

Following existing patterns:
- pytestmark for test categorization
- Arrange/Act/Assert structure
- Direct testing without over-mocking
"""

import pytest
from fastapi.testclient import TestClient

# ✅ CRITICAL: Add marker (follows existing pattern)
pytestmark = pytest.mark.integration


class TestQueryRouterSetup:
    """Validate TestClient approach works."""

    def test_can_import_app(self):
        """Verify we can import the FastAPI app."""
        from api.main import app

        assert app is not None
        assert hasattr(app, 'routes')

    def test_query_endpoint_exists(self):
        """Verify query endpoint responds (smoke test)."""
        from api.main import app

        client = TestClient(app)

        # Smoke test - endpoint should exist
        response = client.post("/api/v1/query", json={})

        # Should be 422 (validation error), not 404 (not found)
        assert response.status_code in [422, 400], \
            f"Expected validation error (422/400), got {response.status_code}"
```

**Run it:**
```bash
pytest tests/unit/api/test_query_router.py -v
```

**Expected output:**
```
tests/unit/api/test_query_router.py::TestQueryRouterSetup::test_can_import_app PASSED
tests/unit/api/test_query_router.py::TestQueryRouterSetup::test_query_endpoint_exists PASSED
```

**If tests PASS** → Approach validated, proceed to Day 2
**If tests FAIL** → Fix imports/approach before continuing

---

### Task 1.2: Add pytest-mock if Needed (15 min)

**Check if installed:**
```bash
uv pip list | grep pytest-mock
```

**If not found:**
```bash
uv add --dev pytest-mock
```

---

## Day 2: Query Router Basic Tests (4 hours)

### Task 2.1: Extend Existing Fixtures (1 hour)

**File:** `tests/conftest.py` (EXTEND, don't replace)

**Add ONLY what's needed:**
```python
# ADD to existing tests/conftest.py (line ~43)

@pytest.fixture
def sample_query_payload():
    """Valid query request payload for API tests.

    Reusable across all API endpoint tests.
    """
    return {
        "text": "Patient presents with seizures",
        "model_name": "FremyCompany/BioLORD-2023-M",
        "language": "en",
        "num_results": 10,
        "similarity_threshold": 0.3,
        "enable_reranker": False,
    }


@pytest.fixture
def mock_dense_retriever_for_api(mocker):
    """Mock DenseRetriever for API tests.

    ✅ CORRECT: Mock external dependency (retriever)
    ❌ DON'T: Mock execute_hpo_retrieval_for_api (our code!)
    """
    mock = mocker.MagicMock()
    mock.model_name = "FremyCompany/BioLORD-2023-M"
    mock.query.return_value = [
        {
            "hpo_id": "HP:0001250",
            "label": "Seizure",
            "score": 0.85,
            "definition": "Abnormal excessive neuronal activity",
        }
    ]
    return mock
```

**Why minimal?** Follow KISS - add fixtures only when needed (not upfront).

---

### Task 2.2: Basic Query Tests (3 hours)

**File:** `tests/unit/api/test_query_router.py` (EXTEND)

**Add after TestQueryRouterSetup class:**

```python
class TestQueryRouterPOST:
    """Tests for POST /api/v1/query endpoint."""

    def test_query_with_valid_payload_returns_200(
        self, mocker, sample_query_payload, mock_dense_retriever_for_api
    ):
        """Test successful query with valid payload."""
        from api.main import app

        # ✅ CORRECT: Mock ONLY external dependency
        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )

        # ❌ DON'T mock execute_hpo_retrieval_for_api - let it run!

        client = TestClient(app)
        response = client.post("/api/v1/query", json=sample_query_payload)

        assert response.status_code == 200
        data = response.json()
        assert "query_text_received" in data
        assert "results" in data
        assert data["query_text_received"] == sample_query_payload["text"]

    def test_query_without_text_returns_422(self):
        """Test query without required field returns validation error."""
        from api.main import app

        client = TestClient(app)
        response = client.post("/api/v1/query", json={})

        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data

    @pytest.mark.parametrize("invalid_payload,expected_field", [
        ({"text": ""}, "text"),
        ({"text": "query", "num_results": -1}, "num_results"),
        ({"text": "query", "similarity_threshold": 1.5}, "similarity_threshold"),
    ])
    def test_query_with_invalid_params_returns_422(
        self, invalid_payload, expected_field
    ):
        """Test query with invalid parameters returns validation error."""
        from api.main import app

        client = TestClient(app)
        response = client.post("/api/v1/query", json=invalid_payload)

        assert response.status_code == 422
        error_str = str(response.json())
        assert expected_field in error_str.lower()

    def test_query_uses_provided_language(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test explicit language parameter is used."""
        from api.main import app

        mock_detect = mocker.patch("api.routers.query_router.detect_language")
        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )

        client = TestClient(app)
        response = client.post(
            "/api/v1/query",
            json={"text": "seizures", "language": "de"}
        )

        assert response.status_code == 200
        # Should NOT call detect_language if language provided
        assert not mock_detect.called

    def test_query_auto_detects_language_when_not_provided(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test language is auto-detected when not in request."""
        from api.main import app

        mock_detect = mocker.patch(
            "api.routers.query_router.detect_language",
            return_value="en"
        )
        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )

        client = TestClient(app)
        response = client.post("/api/v1/query", json={"text": "seizures"})

        assert response.status_code == 200
        assert mock_detect.called
        data = response.json()
        assert data["language_detected"] == "en"
```

**Run and verify:**
```bash
pytest tests/unit/api/test_query_router.py::TestQueryRouterPOST -v

# Check coverage
pytest tests/unit/api/test_query_router.py \
  --cov=api/routers/query_router.py \
  --cov-report=term-missing
```

**Expected:** ~30-40% coverage after Day 2

---

## Day 3: Expand Query Router Tests (4 hours)

### Task 3.1: Add Reranker and Error Tests

**File:** `tests/unit/api/test_query_router.py` (EXTEND)

**Add new test class:**

```python
class TestQueryRouterReranking:
    """Tests for reranker functionality."""

    def test_query_with_reranker_enabled(
        self, mocker, sample_query_payload, mock_dense_retriever_for_api
    ):
        """Test enabling reranker calls cross-encoder dependency."""
        from api.main import app

        mock_cross_encoder = mocker.MagicMock()

        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )
        mocker.patch(
            "api.dependencies.get_cross_encoder_dependency",
            return_value=mock_cross_encoder
        )

        client = TestClient(app)
        payload = {**sample_query_payload, "enable_reranker": True}
        response = client.post("/api/v1/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["reranker_used"] is not None

    def test_query_continues_without_reranker_if_fails(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test query continues if reranker fails to load."""
        from api.main import app

        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )
        mocker.patch(
            "api.dependencies.get_cross_encoder_dependency",
            return_value=None  # Reranker fails
        )

        client = TestClient(app)
        payload = {"text": "seizures", "enable_reranker": True}
        response = client.post("/api/v1/query", json=payload)

        # Should succeed without reranker
        assert response.status_code == 200
        data = response.json()
        assert data["reranker_used"] is None


class TestQueryRouterGET:
    """Tests for GET /api/v1/query endpoint."""

    def test_get_query_with_text_param(self, mocker, mock_dense_retriever_for_api):
        """Test GET query with text parameter."""
        from api.main import app

        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )

        client = TestClient(app)
        response = client.get("/api/v1/query?text=seizures")

        assert response.status_code == 200
        data = response.json()
        assert data["query_text_received"] == "seizures"

    def test_get_query_without_text_returns_422(self):
        """Test GET query without text parameter."""
        from api.main import app

        client = TestClient(app)
        response = client.get("/api/v1/query")

        assert response.status_code == 422


class TestQueryRouterErrors:
    """Tests for error handling."""

    def test_query_returns_503_when_retriever_unavailable(self, mocker):
        """Test 503 when retriever fails to initialize."""
        from api.main import app

        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=None  # Retriever failed
        )

        client = TestClient(app)
        response = client.post("/api/v1/query", json={"text": "seizures"})

        assert response.status_code == 503

    def test_query_handles_language_detection_failure(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test graceful fallback when language detection fails."""
        from api.main import app

        mocker.patch(
            "api.routers.query_router.detect_language",
            side_effect=Exception("Detection failed")
        )
        mocker.patch(
            "api.dependencies.get_dense_retriever_dependency",
            return_value=mock_dense_retriever_for_api
        )

        client = TestClient(app)
        response = client.post("/api/v1/query", json={"text": "seizures"})

        # Should fallback to default language
        assert response.status_code == 200
        data = response.json()
        assert data["language_detected"] == "en"  # DEFAULT_LANGUAGE
```

**Run and verify:**
```bash
pytest tests/unit/api/test_query_router.py -v

# Check coverage increase
pytest tests/unit/api/test_query_router.py \
  --cov=api/routers/query_router.py \
  --cov-report=term-missing
```

**Expected:** ~60-70% coverage after Day 3

---

## Day 4: Query Orchestrator Tests (4 hours)

### Task 4.1: Create Orchestrator Tests

**File:** `tests/unit/retrieval/test_query_orchestrator.py` (CREATE NEW)

```python
"""Tests for query orchestrator (core query execution logic)."""

import pytest

pytestmark = pytest.mark.unit


class TestQueryOrchestratorBasic:
    """Basic functionality tests."""

    @pytest.mark.asyncio
    async def test_execute_returns_results_dict(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test basic query execution returns formatted results."""
        from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api

        result = await execute_hpo_retrieval_for_api(
            text="patient has seizures",
            language="en",
            retriever=mock_dense_retriever_for_api,
            num_results=10,
            similarity_threshold=0.3,
        )

        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_execute_respects_num_results(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test num_results parameter is passed to retriever."""
        from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api

        await execute_hpo_retrieval_for_api(
            text="seizures",
            language="en",
            retriever=mock_dense_retriever_for_api,
            num_results=5,
            similarity_threshold=0.0,
        )

        # Verify retriever called with correct top_k
        mock_dense_retriever_for_api.query.assert_called_once()
        call_kwargs = mock_dense_retriever_for_api.query.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_execute_with_reranker_uses_cross_encoder(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test reranker is used when enabled."""
        from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api

        mock_cross_encoder = mocker.MagicMock()

        await execute_hpo_retrieval_for_api(
            text="seizures",
            language="en",
            retriever=mock_dense_retriever_for_api,
            enable_reranker=True,
            cross_encoder=mock_cross_encoder,
        )

        # Cross-encoder should be involved (implementation-dependent)
        assert mock_cross_encoder is not None

    @pytest.mark.asyncio
    async def test_execute_detects_query_assertion(
        self, mocker, mock_dense_retriever_for_api
    ):
        """Test assertion detection works."""
        from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api

        mocker.patch(
            "phentrieve.retrieval.api_helpers.detect_assertion",
            return_value="negated"
        )

        result = await execute_hpo_retrieval_for_api(
            text="no seizures",
            language="en",
            retriever=mock_dense_retriever_for_api,
            detect_query_assertion=True,
        )

        assert result.get("original_query_assertion_status") == "negated"
```

**Run and verify:**
```bash
pytest tests/unit/retrieval/test_query_orchestrator.py -v

# Check coverage
pytest tests/unit/retrieval/test_query_orchestrator.py \
  --cov=phentrieve/retrieval/query_orchestrator.py \
  --cov-report=term-missing
```

---

## Day 5: Verify and Iterate (2 hours)

### Task 5.1: Check Overall Coverage

```bash
# Run all tests
make test

# Check coverage for all critical modules
pytest tests/ \
  --cov=api/routers/query_router.py \
  --cov=phentrieve/retrieval/query_orchestrator.py \
  --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Task 5.2: Identify Gaps

**Look for uncovered lines in critical paths:**
- Error handling blocks
- Edge cases
- Parameter validation

**Add targeted tests** to cover gaps (2-3 tests per gap).

---

## Success Metrics

**After Week 1:**
- ✅ `api/routers/query_router.py`: 0% → **60-70%**
- ✅ `phentrieve/retrieval/query_orchestrator.py`: 8% → **50-60%**
- ✅ All tests pass
- ✅ No over-mocking (business logic tested)
- ✅ Following existing patterns
- ✅ Incremental validation successful

**Total:** ~15-18 well-focused tests covering critical paths

---

## Quick Reference

### Running Tests

```bash
# All tests
make test

# Specific file
pytest tests/unit/api/test_query_router.py -v

# With coverage
pytest tests/unit/api/test_query_router.py \
  --cov=api/routers/query_router.py \
  --cov-report=term-missing

# Single test
pytest tests/unit/api/test_query_router.py::TestQueryRouterPOST::test_query_with_valid_payload_returns_200 -v
```

### Test Template

```python
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration  # or pytest.mark.unit


class TestFeatureName:
    """Tests for feature."""

    def test_behavior_description(self, mocker):
        """Test what this verifies."""
        # Arrange
        from api.main import app
        mock_dep = mocker.MagicMock()
        mocker.patch("path.to.dependency", return_value=mock_dep)

        # Act
        client = TestClient(app)
        response = client.post("/endpoint", json={...})

        # Assert
        assert response.status_code == 200
```

---

## Principles Recap

1. ✅ **KISS:** Simple tests, parametrize > factories
2. ✅ **DRY:** Extend existing fixtures, don't duplicate
3. ✅ **SOLID:** Focused fixtures (SRP), mock composition
4. ✅ **Incremental:** 1 test → 5 tests → 10 tests → validate
5. ✅ **Strategic mocking:** External deps only, NOT business logic

---

**Status:** ✅ Ready to implement incrementally
**Time Estimate:** Week 1 = 16 hours
**Coverage Goal:** 8% → 30-35% (critical paths tested)
**Approach:** Validate → Build → Verify → Iterate
