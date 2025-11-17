# Testing Coverage Expansion Plan

**Status:** Active - Ready for Implementation
**Created:** 2025-11-17
**Priority:** High
**Related Issue:** [#37: Implement comprehensive testing suite](https://github.com/berntpopp/phentrieve/issues/37)

---

## Executive Summary

Issue #37 requested a comprehensive testing suite. **Testing infrastructure is complete** (pytest, coverage, CI/CD), but **coverage is only 8%** vs. the **30% baseline target**. This plan focuses on expanding test coverage to meet the original acceptance criteria while following best practices (DRY, KISS, SOLID).

### Current State ✅
- ✅ 263 tests (206 unit/integration + 57 E2E)
- ✅ pytest + pytest-cov configured
- ✅ Test structure (unit/, integration/, e2e/)
- ✅ CI/CD integration (GitHub Actions)
- ✅ Documentation (CLAUDE.md)
- ✅ Make commands for testing

### Gap Analysis ❌
- ❌ Coverage: 8% (need 30% minimum)
- ❌ 33 modules with 0% coverage
- ❌ No API route tests (all API modules 0%)
- ❌ No CLI command tests (most at 0%)
- ❌ No evaluation/benchmark tests
- ❌ No indexing tests

---

## Objectives

### Primary Goal
Increase test coverage from **8% to 30%+** by adding strategic tests for high-value, critical-path modules.

### Secondary Goals
1. Maintain fast test execution (<2 minutes for unit tests)
2. Follow pytest best practices (fixtures, parametrization, mocking)
3. Focus on meaningful tests over coverage percentage
4. Keep tests maintainable and well-documented

---

## Best Practices Framework

### Testing Philosophy (Based on 2024 Research)

**Quality > Quantity**
- Focus on critical paths and edge cases
- 30% meaningful coverage > 100% shallow coverage
- Test behavior, not implementation details

**Test Pyramid**
```
       /\
      /E2\      E2E Tests (Slow, Comprehensive) - 10%
     /----\
    / Int  \    Integration Tests (Medium) - 20%
   /--------\
  /   Unit   \  Unit Tests (Fast, Focused) - 70%
 /------------\
```

**KISS Principle**
- Simple, focused tests (1 test = 1 behavior)
- Descriptive names: `test_function_name_when_condition_then_expected_behavior`
- Minimal setup/teardown
- Avoid test interdependencies

**DRY Principle**
- Use pytest fixtures for common setup
- Parametrize tests for multiple inputs
- Share test utilities in conftest.py
- Extract test data factories

**SOLID for Tests**
- Single Responsibility: One assertion per test (when practical)
- Open/Closed: Extend fixtures, don't modify tests
- Dependency Injection: Use fixtures for dependencies
- Interface Segregation: Focused fixtures
- Dependency Inversion: Mock external dependencies

---

## Strategic Coverage Plan

### Phase 1: Critical Path Coverage (Target: 15% → 20%)

**Rationale:** Cover the most-used, highest-risk code paths first.

#### 1.1 API Route Tests (Priority: HIGH)
**Current:** 0% coverage on all API routers
**Target:** 60% coverage on critical routes
**Effort:** 2-3 days

**Modules to Test:**
- `api/routers/query_router.py` (0% → 60%)
  - `/api/v1/query` endpoint (most critical)
  - Error handling (invalid input, timeout)
  - Response format validation

- `api/routers/text_processing_router.py` (0% → 60%)
  - `/api/v1/text/process` endpoint
  - All 7 chunking strategies
  - Edge cases (already have some tests, expand)

- `api/routers/similarity_router.py` (0% → 60%)
  - `/api/v1/similarity` endpoint
  - Different similarity formulas
  - Invalid HPO IDs

**Test Strategy:**
```python
# Use FastAPI TestClient (NOT Docker - too slow)
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_retriever(monkeypatch):
    # Mock heavy dependencies (ChromaDB, models)
    # This makes tests fast (<100ms each)
    pass

def test_query_endpoint_returns_valid_response(client, mock_retriever):
    response = client.post("/api/v1/query", json={
        "query_text": "seizures",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 5
```

**Key Insight:** Mock ChromaDB and ML models for fast tests. Real integration happens in E2E tests.

#### 1.2 Core Retrieval Tests (Priority: HIGH)
**Current:** dense_retriever.py at 13%, reranker.py at 22%
**Target:** 70% coverage
**Effort:** 1-2 days

**Modules:**
- `phentrieve/retrieval/dense_retriever.py` (13% → 70%)
  - Query processing
  - Result filtering
  - Error handling

- `phentrieve/retrieval/reranker.py` (22% → 70%)
  - Cross-encoder scoring
  - NLI mode
  - Score normalization

**Test Strategy:**
```python
# Test with minimal fixtures (use sample data, not real models)
@pytest.fixture
def sample_embeddings():
    # Return pre-computed embeddings (no model loading)
    return np.random.rand(10, 384)

def test_dense_retriever_filters_by_threshold():
    retriever = DenseRetriever(threshold=0.5)
    results = retriever.filter_results([
        {"id": "HP:0000001", "score": 0.7},  # Keep
        {"id": "HP:0000002", "score": 0.3},  # Filter
    ])
    assert len(results) == 1
    assert results[0]["score"] >= 0.5
```

#### 1.3 Text Processing Pipeline (Priority: HIGH)
**Current:** pipeline.py at 10%
**Target:** 50% coverage
**Effort:** 1 day

**Focus Areas:**
- Pipeline initialization
- Component chaining
- Error propagation
- Language handling

---

### Phase 2: Utility & Helper Coverage (Target: 20% → 25%)

**Rationale:** Utils are widely used - high leverage for coverage.

#### 2.1 Utils Module (Priority: MEDIUM)
**Current:** utils.py at 30%
**Target:** 70% coverage
**Effort:** 1 day

**Focus:**
- HPO ID normalization
- Similarity calculations
- Model slug generation
- Translation loading

**Test Strategy:**
```python
# Utils are perfect for parametrized tests
@pytest.mark.parametrize("input_id,expected", [
    ("http://purl.obolibrary.org/obo/HP_0000001", "HP:0000001"),
    ("HP:0000001", "HP:0000001"),
    ("HP_0000001", "HP:0000001"),
])
def test_normalize_id(input_id, expected):
    assert normalize_id(input_id) == expected
```

#### 2.2 Config Module (Priority: MEDIUM)
**Current:** config.py at 87%
**Target:** 95% coverage
**Effort:** 0.5 day

**Focus:**
- Config loading edge cases
- Default value handling
- Validation errors

---

### Phase 3: CLI Command Coverage (Target: 25% → 30%)

**Rationale:** CLI is primary user interface - needs solid tests.

#### 3.1 CLI Commands (Priority: MEDIUM)
**Current:** Most CLI commands at 0%
**Target:** 50% coverage
**Effort:** 2 days

**Modules:**
- `phentrieve/cli/query_commands.py` (0% → 50%)
- `phentrieve/cli/data_commands.py` (0% → 50%)
- `phentrieve/cli/index_commands.py` (0% → 50%)

**Test Strategy:**
```python
# Use Typer's CliRunner for CLI testing
from typer.testing import CliRunner
from phentrieve.cli import app

runner = CliRunner()

def test_query_command_basic():
    result = runner.invoke(app, ["query", "seizures", "--top-k", "5"])
    assert result.exit_code == 0
    assert "HP:" in result.stdout

def test_query_command_invalid_input():
    result = runner.invoke(app, ["query", "", "--top-k", "5"])
    assert result.exit_code != 0
    assert "Error" in result.stdout
```

**Note:** We already have some CLI tests (test_query_commands.py, test_similarity_commands.py) - expand those.

---

## Implementation Strategy

### Test Development Workflow

```bash
# 1. Identify module to test
# 2. Check current coverage
pytest tests/unit/core/test_utils.py --cov=phentrieve/utils.py --cov-report=term-missing

# 3. Write tests for uncovered lines
# 4. Verify coverage increase
pytest tests/unit/core/test_utils.py --cov=phentrieve/utils.py --cov-report=term-missing

# 5. Run full suite to catch regressions
make test

# 6. Commit with coverage increase in message
git commit -m "test(utils): Add tests for normalize_id function (coverage: 30% → 45%)"
```

### Mocking Strategy

**When to Mock:**
- ✅ External APIs (ChromaDB, Hugging Face)
- ✅ ML model inference (slow, non-deterministic)
- ✅ File I/O for most unit tests
- ✅ Network requests

**When NOT to Mock:**
- ❌ Simple utility functions (test real implementation)
- ❌ Integration tests (test real interactions)
- ❌ E2E tests (test real system)

**Example:**
```python
# Mock ChromaDB for fast API tests
@pytest.fixture
def mock_chroma(monkeypatch):
    class MockCollection:
        def query(self, query_embeddings, n_results):
            return {
                "ids": [["HP:0000001", "HP:0000002"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[{"name": "Seizure"}, {"name": "Tremor"}]]
            }

    def mock_connect(*args, **kwargs):
        client = MagicMock()
        client.get_collection.return_value = MockCollection()
        return client

    monkeypatch.setattr("chromadb.Client", mock_connect)
```

### Fixture Organization

**conftest.py Structure:**
```
tests/
├── conftest.py              # Shared fixtures (all tests)
├── unit/
│   ├── conftest.py          # Unit test fixtures
│   ├── api/
│   │   ├── conftest.py      # API test fixtures
│   │   └── test_*.py
│   └── core/
│       ├── conftest.py      # Core test fixtures
│       └── test_*.py
└── integration/
    ├── conftest.py          # Integration fixtures
    └── test_*.py
```

**Common Fixtures:**
```python
# tests/conftest.py
@pytest.fixture
def sample_hpo_data():
    """Minimal HPO data for testing."""
    return {
        "HP:0000001": {
            "name": "Seizure",
            "definition": "An abnormal electrical discharge in the brain"
        }
    }

@pytest.fixture
def temp_config_file(tmp_path):
    """Create temporary config file."""
    config = tmp_path / "phentrieve.yaml"
    config.write_text("model: paraphrase-multilingual-MiniLM-L12-v2")
    return config
```

---

## Test Structure & Naming

### Directory Structure
```
tests/
├── conftest.py                     # Shared fixtures
├── unit/                           # Fast, isolated tests (<1s each)
│   ├── api/                        # API unit tests
│   │   ├── test_query_router.py
│   │   ├── test_text_processing_router.py   # ✅ Already exists
│   │   └── test_similarity_router.py
│   ├── cli/                        # CLI unit tests
│   │   ├── test_query_commands.py  # ✅ Already exists (expand)
│   │   ├── test_data_commands.py
│   │   └── test_index_commands.py
│   ├── core/                       # Core library tests
│   │   ├── test_utils.py           # ✅ Partially exists (expand)
│   │   ├── test_embeddings.py      # ✅ Already exists
│   │   └── test_config.py
│   └── retrieval/                  # Retrieval tests
│       ├── test_dense_retriever.py # ✅ Already exists (expand)
│       ├── test_reranker.py        # ✅ Already exists (expand)
│       └── test_output_formatters.py # ✅ Already exists
├── integration/                    # Component interaction tests
│   ├── test_chunking_pipeline_integration.py  # ✅ Already exists
│   └── test_api_workflows.py       # NEW - End-to-end API flows
└── e2e/                            # Docker-based E2E tests
    ├── test_api_e2e.py             # ✅ Already exists
    ├── test_docker_security.py     # ✅ Already exists
    └── test_docker_health.py       # ✅ Already exists
```

### Naming Conventions

**Test Functions:**
```python
# Pattern: test_<function>_<condition>_<expected_behavior>
def test_normalize_id_with_uri_returns_hpo_format()
def test_query_endpoint_with_invalid_input_returns_400()
def test_dense_retriever_with_empty_query_raises_error()
```

**Test Classes:**
```python
# Pattern: Test<ComponentName>
class TestDenseRetriever:
    """Tests for dense retriever functionality."""

    def test_query_returns_results(self):
        pass

    def test_filter_by_threshold(self):
        pass
```

---

## Modules Prioritized by Impact

### High Priority (Critical Path - Phase 1)
| Module | Current | Target | Tests | Impact |
|--------|---------|--------|-------|--------|
| `api/routers/query_router.py` | 0% | 60% | NEW | Most-used API endpoint |
| `api/routers/text_processing_router.py` | 0% | 60% | EXPAND | Core text processing |
| `phentrieve/retrieval/dense_retriever.py` | 13% | 70% | EXPAND | Core retrieval |
| `phentrieve/retrieval/reranker.py` | 22% | 70% | EXPAND | Result quality |
| `phentrieve/text_processing/pipeline.py` | 10% | 50% | NEW | Core pipeline |

### Medium Priority (Utilities - Phase 2)
| Module | Current | Target | Tests | Impact |
|--------|---------|--------|-------|--------|
| `phentrieve/utils.py` | 30% | 70% | EXPAND | Widely used |
| `phentrieve/config.py` | 87% | 95% | EXPAND | Almost done |
| `phentrieve/embeddings.py` | 19% | 60% | EXPAND | Model loading |

### Medium Priority (CLI - Phase 3)
| Module | Current | Target | Tests | Impact |
|--------|---------|--------|-------|--------|
| `phentrieve/cli/query_commands.py` | 0% | 50% | EXPAND | Primary CLI |
| `phentrieve/cli/data_commands.py` | 0% | 50% | NEW | Data management |
| `phentrieve/cli/index_commands.py` | 0% | 50% | NEW | Index building |

### Lower Priority (Defer to Future)
- Evaluation modules (used for benchmarking, not core functionality)
- Visualization modules (plotting, not core)
- Data processing (HPO parser - complex, low ROI)

---

## Anti-Patterns to Avoid

### ❌ Don't Do This

**1. Testing Implementation Details**
```python
# BAD - brittle, breaks on refactoring
def test_query_calls_chromadb():
    with patch("chromadb.Client") as mock:
        query("seizures")
        mock.assert_called_once()  # Implementation detail!
```

**2. Over-Mocking**
```python
# BAD - mocking everything makes test meaningless
def test_addition():
    with patch("operator.add", return_value=4):
        assert add(2, 2) == 4  # What are we testing?
```

**3. Test Interdependencies**
```python
# BAD - tests depend on execution order
class TestCounter:
    counter = 0

    def test_increment(self):
        self.counter += 1
        assert self.counter == 1

    def test_value(self):  # Fails if run in isolation!
        assert self.counter == 1
```

**4. Assertion Roulette**
```python
# BAD - which assertion failed?
def test_user():
    user = create_user()
    assert user.name == "John"
    assert user.age == 30
    assert user.email == "john@example.com"
    # Better: One test per property or use pytest.fail with message
```

### ✅ Do This Instead

**1. Test Behavior**
```python
# GOOD - tests observable behavior
def test_query_returns_relevant_results():
    results = query("seizures")
    assert len(results) > 0
    assert any("seizure" in r.name.lower() for r in results)
```

**2. Strategic Mocking**
```python
# GOOD - mock slow/external dependencies only
@pytest.fixture
def mock_model():
    # Mock ML model (slow, non-deterministic)
    return MagicMock(embed=lambda x: [0.1, 0.2, 0.3])

def test_retriever_with_mock_model(mock_model):
    retriever = DenseRetriever(model=mock_model)
    results = retriever.query("test")
    # Test retriever logic, not model
```

**3. Isolated Tests**
```python
# GOOD - each test is independent
def test_increment():
    counter = Counter()
    counter.increment()
    assert counter.value == 1

def test_decrement():
    counter = Counter()  # Fresh instance
    counter.decrement()
    assert counter.value == -1
```

**4. Clear Assertions**
```python
# GOOD - descriptive, focused tests
def test_user_has_valid_name():
    user = create_user(name="John")
    assert user.name == "John", "User name should match input"

def test_user_has_valid_email():
    user = create_user(email="john@example.com")
    assert "@" in user.email, "Email should contain @"
```

---

## Acceptance Criteria (Issue #37)

| Criterion | Status | Notes |
|-----------|--------|-------|
| ✅ pytest framework | DONE | Configured in pyproject.toml |
| ✅ pytest-cov plugin | DONE | Coverage reporting works |
| ✅ tests/ directory structure | DONE | unit/, integration/, e2e/ |
| ⚠️ Unit tests for utilities | PARTIAL | Have some, need more |
| ✅ Integration tests | DONE | Good integration test coverage |
| ⚠️ CLI tests with CliRunner | PARTIAL | Have some, need more |
| ❌ Coverage >30% baseline | **8%** | **PRIMARY GAP** |
| ✅ Simple `pytest` command | DONE | Works via Make commands |
| ✅ CI/CD integration | DONE | GitHub Actions configured |
| ⚠️ Documentation | PARTIAL | In CLAUDE.md, need README |

---

## Timeline & Effort Estimate

### Phase 1: Critical Path (Target: 20% coverage)
**Effort:** 4-6 days
- API route tests: 2-3 days
- Retrieval tests: 1-2 days
- Pipeline tests: 1 day

### Phase 2: Utilities (Target: 25% coverage)
**Effort:** 1.5 days
- Utils module: 1 day
- Config module: 0.5 day

### Phase 3: CLI (Target: 30% coverage)
**Effort:** 2 days
- CLI command tests: 2 days

### Total Effort
**8-10 days** of focused development to reach 30% coverage target.

**Note:** Can be parallelized if multiple developers work on different modules.

---

## Success Metrics

### Quantitative
- ✅ **Coverage:** 8% → 30%+ (minimum target from issue #37)
- ✅ **Test Count:** 263 → 400+ tests
- ✅ **Test Speed:** Unit tests <2 minutes total
- ✅ **CI Pass Rate:** >95% (stable tests)

### Qualitative
- ✅ Tests are **maintainable** (follow DRY, clear naming)
- ✅ Tests are **meaningful** (catch real bugs, not implementation details)
- ✅ Tests are **fast** (enable TDD workflow)
- ✅ Tests are **documented** (docstrings explain purpose)

---

## Documentation Updates

### 1. Add Testing Section to README.md

```markdown
## Testing

Phentrieve uses pytest for comprehensive testing across unit, integration, and E2E layers.

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/unit/api/test_query_router.py -v

# Run E2E tests (requires Docker)
make test-e2e
```

### Test Structure

- `tests/unit/` - Fast, isolated tests for individual functions/classes
- `tests/integration/` - Tests for component interactions
- `tests/e2e/` - Docker-based end-to-end tests

### Writing Tests

See [CONTRIBUTING.md](CONTRIBUTING.md#testing) for testing guidelines and best practices.

### Coverage Goals

- Overall: 30%+ (current: 8%)
- Critical paths: 60%+
- Utilities: 70%+
```

### 2. Create CONTRIBUTING.md Testing Section

```markdown
## Testing Guidelines

### Test Structure

Follow the test pyramid:
- 70% unit tests (fast, focused)
- 20% integration tests (component interaction)
- 10% E2E tests (full system)

### Writing Good Tests

**DO:**
- ✅ Use descriptive names: `test_function_when_condition_then_expected`
- ✅ Test behavior, not implementation
- ✅ Keep tests simple and focused
- ✅ Use fixtures for common setup
- ✅ Parametrize for multiple inputs

**DON'T:**
- ❌ Test implementation details
- ❌ Create test interdependencies
- ❌ Over-mock (mock only slow/external deps)
- ❌ Write brittle tests

### Example

```python
@pytest.mark.parametrize("input_text,expected_count", [
    ("Patient has seizures", 1),
    ("Seizures and tremor", 2),
    ("No symptoms", 0),
])
def test_extract_hpo_terms_counts(input_text, expected_count):
    results = extract_hpo_terms(input_text)
    assert len(results) == expected_count
```
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tests slow down development | Medium | High | Keep unit tests <1s each, use mocking |
| Brittle tests (break on refactoring) | Medium | Medium | Test behavior not implementation |
| Low-quality tests (high coverage, low value) | Medium | High | Code review, focus on critical paths |
| Test maintenance burden | Low | Medium | Follow DRY, use fixtures, clear naming |
| CI pipeline too slow | Low | Medium | Parallelize tests, cache dependencies |

---

## Next Steps

1. **Review & Approve Plan** - Get stakeholder buy-in
2. **Create Feature Branch** - `feature/testing-coverage-expansion`
3. **Phase 1: Critical Path** - Start with API route tests (highest impact)
4. **Incremental PRs** - Small PRs per module (easier review)
5. **Track Progress** - Update this plan with completion status
6. **Celebrate 30%** - Close issue #37 when target reached! 🎉

---

## References

### Best Practices
- [pytest Best Practices (2024)](https://pytest-with-eric.com/pytest-best-practices/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python Test Coverage Best Practices](https://medium.com/@keployio/mastering-python-test-coverage-tools-tips-and-best-practices-11daf699d79b)

### Internal Docs
- `CLAUDE.md` - Development commands and pre-commit checklist
- `plan/02-completed/TESTING-MODERNIZATION-PLAN.md` - Previous testing work
- Issue #37 - Original testing requirements

---

**Status:** Ready for Implementation
**Next Action:** Review plan, then start Phase 1 (API route tests)
**Target:** Close issue #37 with 30%+ coverage and comprehensive test suite
