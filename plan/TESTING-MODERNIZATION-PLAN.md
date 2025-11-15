# Testing Modernization Plan (Medical-Grade Software)

**Status**: Planning
**Priority**: **CRITICAL** (Patient Safety)
**Estimated Effort**: 2-3 weeks
**Regulatory Context**: Clinical Decision Support (CDS) Software

## ⚕️ Medical-Grade Software Considerations

Phentrieve is a **Clinical Decision Support (CDS) system** that maps clinical text to HPO terms for diagnostic assistance. This classification requires:

- **IEC 62304 Compliance**: Medical device software life cycle processes
- **FDA CDS Guidelines**: Clinical decision support software requirements
- **Patient Safety Focus**: Errors in HPO term mapping could impact clinical decisions
- **High Reliability Standards**: Medical-grade testing and validation

### Regulatory Classification

- **Software Type**: Clinical Decision Support (CDS)
- **IEC 62304 Safety Class**: Class B (probable Class C for diagnostic use)
  - Class A: No injury or damage to health possible
  - Class B: Non-serious injury possible
  - Class C: Death or serious injury possible
- **Coverage Requirement**: **95-100% code coverage** (Class C standard)
- **Testing Burden**: 50-60% of total development effort (industry standard)

## Overview

Modernize Phentrieve's test suite to achieve **medical-grade quality standards** with comprehensive, maintainable coverage across all components (CLI, API, Core). Focus on patient safety, clinical accuracy, and regulatory compliance.

## Current State Analysis

### What We Have (87 tests)
- ✅ Core functionality tests (6 files): chunking, assertion detection, semantic metrics
- ✅ CLI tests (2 files): query commands, similarity commands
- ✅ API tests (2 files): config info, text processing routers
- ✅ Basic pytest setup in `pyproject.toml`

### Critical Gaps (Medical-Grade Perspective)

**Testing Infrastructure**:
- ❌ **Mixed testing styles**: unittest.TestCase + pytest (inconsistent)
- ❌ **No conftest.py**: No shared fixtures, setup duplication
- ❌ **No test markers**: Cannot filter unit/integration/slow tests
- ❌ **No coverage reporting**: Unknown actual test coverage (<95% unacceptable)
- ❌ **Incomplete API coverage**: Missing routers, schemas, dependencies

**Medical-Grade Requirements**:
- ❌ **No clinical validation tests**: HPO term mapping accuracy not validated
- ❌ **No assertion detection validation**: Negation/normality detection accuracy untested
- ❌ **No edge case coverage**: Missing tests for malformed input, boundary conditions
- ❌ **No safety testing**: Error handling, graceful degradation untested
- ❌ **No data quality tests**: Input validation, sanitization not verified
- ❌ **No requirements traceability**: Cannot map tests to requirements
- ❌ **No branch/MC/DC coverage**: Only statement coverage (insufficient for Class C)
- ❌ **No regression testing**: No baseline for clinical accuracy metrics
- ❌ **No security testing**: OWASP vulnerabilities, PHI handling untested
- ❌ **No integration tests**: Components tested in isolation only
- ❌ **No Docker/E2E tests**: Production environment not tested

## Goals & Principles

### Goals (Medical-Grade Standards)

**Code Quality**:
1. **100% pytest**: Eliminate unittest.TestCase, pure pytest style
2. **95%+ code coverage**: Meet IEC 62304 Class C requirements
   - Statement coverage: 95%+
   - Branch coverage: 90%+ (pytest-cov --cov-branch)
   - Critical paths: 100% (assertion detection, HPO mapping)
3. **Requirements traceability**: Every requirement → test mapping
4. **Fast feedback**: Unit tests <10s, integration <60s, full suite <5min

**Clinical Safety**:
5. **Clinical validation**: HPO term mapping accuracy benchmarks
6. **Assertion accuracy**: Negation/normality detection validation (F1 scores)
7. **Edge case coverage**: Malformed input, boundary conditions, adversarial cases
8. **Error resilience**: Graceful degradation, error handling, fallback mechanisms
9. **Data quality**: Input validation, sanitization, PHI handling

**Infrastructure**:
10. **Clear categorization**: Markers for unit/integration/clinical/security/e2e
11. **Reusable fixtures**: Shared setup via conftest.py hierarchy
12. **Regression prevention**: Automated clinical accuracy baseline checks
13. **Security testing**: OWASP Top 10, dependency scanning, PHI protection

### Principles (DRY, KISS, SOLID)
- **DRY**: Shared fixtures in conftest.py, parametrization for similar tests
- **KISS**: Simple, focused tests doing one thing well
- **SOLID**:
  - Single Responsibility: One assertion per test where practical
  - Open/Closed: Easy to add new tests without changing existing ones
  - Dependency Inversion: Mock external dependencies, test abstractions

## Test Architecture

### Test Categories (Pytest Markers)

```python
# pytest.ini
[pytest]
markers =
    # Standard categories
    unit: Fast unit tests (no I/O, mocked dependencies)
    integration: Integration tests (real database, embeddings)
    api: FastAPI endpoint tests
    cli: CLI command tests
    slow: Slow tests (>5s, run in CI only)

    # Medical-grade categories
    clinical: Clinical validation tests (accuracy, precision, recall)
    safety: Safety-critical tests (error handling, graceful degradation)
    security: Security tests (OWASP, input validation, PHI protection)
    regression: Regression tests (baseline accuracy preservation)
    edge_case: Edge case and boundary condition tests

    # Requirements traceability
    req_REQ001: Tests for requirement REQ-001 (HPO term retrieval)
    req_REQ002: Tests for requirement REQ-002 (assertion detection)
    # ... add all requirements

    # Resource requirements
    requires_data: Tests requiring HPO data
    requires_models: Tests requiring ML models
    requires_internet: Tests requiring internet (model downloads)
```

### Directory Structure

```
tests/
├── conftest.py                      # Root fixtures (session-scoped resources)
├── pytest.ini                       # Pytest configuration
├── requirements_traceability.json   # Requirement → Test mapping
│
├── unit/                            # Unit tests (fast, mocked)
│   ├── conftest.py                  # Unit test fixtures
│   ├── test_chunking.py            # Text chunking (migrated)
│   ├── test_assertion_detection.py # Assertion detection (migrated)
│   ├── test_semantic_metrics.py    # Semantic metrics (migrated)
│   ├── test_embeddings.py          # Embedding utilities
│   ├── test_resource_loader.py     # Resource loading (migrated)
│   └── test_input_validation.py    # Input validation and sanitization
│
├── integration/                     # Integration tests (real dependencies)
│   ├── conftest.py                  # Integration fixtures
│   ├── test_retrieval_pipeline.py  # End-to-end retrieval
│   ├── test_indexing_flow.py       # ChromaDB indexing
│   └── test_query_orchestrator.py  # Query orchestration
│
├── clinical/                        # ⚕️ Clinical validation tests
│   ├── conftest.py                  # Clinical test fixtures
│   ├── test_hpo_mapping_accuracy.py # HPO term mapping validation
│   ├── test_assertion_accuracy.py   # Assertion detection F1 scores
│   ├── test_clinical_scenarios.py   # Real-world clinical text
│   ├── test_edge_cases.py          # Boundary conditions, adversarial
│   ├── test_multilingual_accuracy.py # Language-specific validation
│   └── baseline_metrics.json       # Regression baseline (F1, precision, recall)
│
├── safety/                          # ⚕️ Safety-critical tests
│   ├── conftest.py                  # Safety test fixtures
│   ├── test_error_handling.py      # Error handling, exceptions
│   ├── test_graceful_degradation.py # Fallback mechanisms
│   ├── test_malformed_input.py     # Invalid/malformed input handling
│   ├── test_resource_limits.py     # Memory/CPU limits, OOM handling
│   └── test_data_quality.py        # Data validation, sanitization
│
├── security/                        # 🔒 Security tests
│   ├── conftest.py                  # Security test fixtures
│   ├── test_input_injection.py     # SQL/NoSQL/Command injection
│   ├── test_xss_prevention.py      # Cross-site scripting prevention
│   ├── test_phi_protection.py      # PHI/PII handling (if applicable)
│   ├── test_dependency_scanning.py # Known vulnerabilities (Safety, Bandit)
│   └── test_api_security.py        # OWASP API Security Top 10
│
├── api/                             # FastAPI tests
│   ├── conftest.py                  # API-specific fixtures
│   ├── test_health.py               # Health endpoints
│   ├── test_query_router.py        # Query endpoints
│   ├── test_similarity_router.py   # Similarity endpoints
│   ├── test_text_processing_router.py  # (migrate existing)
│   ├── test_config_info_router.py  # (migrate existing)
│   ├── test_dependencies.py        # Dependency injection
│   └── test_error_responses.py     # Error handling (4xx, 5xx)
│
├── cli/                             # CLI tests
│   ├── conftest.py                  # CLI-specific fixtures
│   ├── test_query_commands.py      # (migrate existing)
│   ├── test_similarity_commands.py # (migrate existing)
│   ├── test_data_commands.py       # Data preparation
│   ├── test_index_commands.py      # Index building
│   ├── test_benchmark_commands.py  # Benchmark commands
│   └── test_cli_error_handling.py  # CLI error messages, exit codes
│
└── e2e/                             # End-to-end tests (Phase 5)
    ├── conftest.py                  # E2E fixtures (Docker, etc.)
    ├── test_docker_containers.py   # Container health, runtime
    └── test_full_workflow.py       # Complete user workflows
```

## Key Fixtures (conftest.py)

### Root Fixtures (`tests/conftest.py`)

```python
"""Shared fixtures for all tests."""
import pytest
from pathlib import Path

# Session-scoped resources
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def sample_clinical_texts() -> list[str]:
    """Sample clinical texts for testing."""
    return [
        "Patient presents with seizures and developmental delay",
        "No evidence of heart disease or abnormalities",
        "Normal blood pressure, within normal limits",
    ]

# Function-scoped mocks
@pytest.fixture
def mock_embedding_model(mocker):
    """Mock sentence transformer model."""
    mock = mocker.MagicMock()
    mock.encode.return_value = [[0.1] * 384]  # Mock embedding
    return mock

@pytest.fixture
def mock_chromadb_collection(mocker):
    """Mock ChromaDB collection."""
    mock = mocker.MagicMock()
    mock.query.return_value = {
        "ids": [["HP:0001250"]],
        "distances": [[0.15]],
        "metadatas": [[{"label": "Seizure"}]],
    }
    return mock
```

### API Fixtures (`tests/api/conftest.py`)

```python
"""FastAPI test fixtures."""
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.fixture
def test_client() -> TestClient:
    """Synchronous test client."""
    return TestClient(app)

@pytest.fixture
async def async_client() -> AsyncClient:
    """Async test client for async endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

@pytest.fixture
def override_dependencies():
    """Override FastAPI dependencies for testing."""
    # Override expensive model loading
    app.dependency_overrides[get_embedding_model] = lambda: mock_model
    yield
    app.dependency_overrides.clear()
```

### Clinical Fixtures (`tests/clinical/conftest.py`)

```python
"""Clinical validation test fixtures."""
import pytest
import json
from pathlib import Path

@pytest.fixture(scope="session")
def clinical_test_corpus() -> list[dict]:
    """Validated clinical text corpus with gold-standard HPO annotations."""
    corpus_path = Path(__file__).parent / "clinical_corpus.json"
    with open(corpus_path) as f:
        return json.load(f)

@pytest.fixture(scope="session")
def baseline_metrics() -> dict:
    """Baseline clinical accuracy metrics (regression prevention)."""
    baseline_path = Path(__file__).parent / "baseline_metrics.json"
    with open(baseline_path) as f:
        return json.load(f)

@pytest.fixture
def assertion_test_cases() -> list[dict]:
    """Test cases for assertion detection (negation, normality)."""
    return [
        {
            "text": "Patient denies fever",
            "expected_status": "NEGATED",
            "clinical_note": "Negative assertion (denies)",
        },
        {
            "text": "No evidence of seizures",
            "expected_status": "NEGATED",
            "clinical_note": "Negative assertion (no evidence)",
        },
        {
            "text": "Normal heart rhythm",
            "expected_status": "NORMAL",
            "clinical_note": "Normal finding",
        },
        {
            "text": "Patient has developmental delay",
            "expected_status": "AFFIRMED",
            "clinical_note": "Positive assertion",
        },
    ]
```

### Safety Fixtures (`tests/safety/conftest.py`)

```python
"""Safety-critical test fixtures."""
import pytest

@pytest.fixture
def malformed_inputs() -> list[dict]:
    """Malformed/adversarial inputs for safety testing."""
    return [
        {"input": "", "expected": "empty_input_error"},
        {"input": "x" * 1000000, "expected": "input_too_long_error"},  # 1MB text
        {"input": "SELECT * FROM users;", "expected": "safe_handling"},  # SQL-like
        {"input": "<script>alert(1)</script>", "expected": "safe_handling"},  # XSS
        {"input": "../../../etc/passwd", "expected": "safe_handling"},  # Path traversal
        {"input": "\x00\x01\x02", "expected": "safe_handling"},  # Binary data
        {"input": "null\nNone\nundefined", "expected": "safe_handling"},  # Edge values
    ]

@pytest.fixture
def resource_limits() -> dict:
    """Resource limits for safety testing."""
    return {
        "max_text_length": 100000,  # 100KB
        "max_chunk_count": 1000,
        "max_results": 100,
        "timeout_seconds": 30,
    }
```

### CLI Fixtures (`tests/cli/conftest.py`)

```python
"""CLI test fixtures."""
import pytest
from typer.testing import CliRunner
from phentrieve.cli import app

@pytest.fixture
def cli_runner() -> CliRunner:
    """Typer CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_query_orchestrator(mocker):
    """Mock query orchestrator for CLI tests."""
    return mocker.patch(
        "phentrieve.retrieval.query_orchestrator.orchestrate_query"
    )
```

## Migration Strategy

### Phase 1: Foundation (Week 1, Days 1-3)

**Goals**: Setup infrastructure, migrate existing tests

**Tasks**:
- [ ] Create new directory structure (`tests/{unit,integration,api,cli}`)
- [ ] Write pytest.ini with markers and configuration
- [ ] Create conftest.py hierarchy with shared fixtures
- [ ] Configure pytest-cov for coverage reporting
- [ ] Migrate existing tests to new structure:
  - [ ] `test_assertion_detection.py` → `unit/test_assertion_detection.py` (convert to pytest)
  - [ ] `test_basic_chunkers.py` → `unit/test_chunking.py` (convert to pytest)
  - [ ] `test_semantic_metrics.py` → `unit/test_semantic_metrics.py` (convert to pytest)
  - [ ] `test_sliding_window_chunker.py` → `unit/test_chunking.py` (merge)
  - [ ] `test_chunking_pipeline_integration.py` → `integration/` (evaluate)
  - [ ] `test_resource_loader.py` → `unit/test_resource_loader.py`
  - [ ] `cli/test_query_commands.py` → `cli/test_query_commands.py` (update fixtures)
  - [ ] `cli/test_similarity_commands.py` → `cli/test_similarity_commands.py` (update fixtures)
  - [ ] `api/test_config_info_router.py` → `api/test_config_info_router.py` (async support)
  - [ ] `api/test_text_processing_router.py` → `api/test_text_processing_router.py` (async support)

**Success Criteria**:
- All 87 existing tests pass in new structure
- Coverage baseline established
- No unittest.TestCase classes remaining

### Phase 2: Coverage Expansion (Week 1, Days 4-5)

**Goals**: Fill critical gaps in test coverage

**Unit Tests** (New):
- [ ] `unit/test_embeddings.py`: Embedding model utilities
- [ ] `unit/test_output_formatters.py`: Output formatting (text, JSON, JSONL)
- [ ] `unit/test_config.py`: Configuration management

**API Tests** (New):
- [ ] `api/test_health.py`: Health check endpoints
- [ ] `api/test_query_router.py`: Query endpoints (comprehensive)
- [ ] `api/test_similarity_router.py`: Similarity calculation endpoints
- [ ] `api/test_dependencies.py`: Dependency injection
- [ ] `api/test_schemas.py`: Pydantic schema validation

**CLI Tests** (New):
- [ ] `cli/test_data_commands.py`: Data preparation commands
- [ ] `cli/test_index_commands.py`: Index building commands
- [ ] `cli/test_benchmark_commands.py`: Benchmark commands

**Integration Tests** (New):
- [ ] `integration/test_retrieval_pipeline.py`: End-to-end retrieval
- [ ] `integration/test_indexing_flow.py`: ChromaDB indexing workflow
- [ ] `integration/test_query_orchestrator.py`: Query orchestration

**Success Criteria**:
- >80% code coverage for phentrieve/ package
- >70% code coverage for api/ package
- All major user workflows tested

### Phase 3: Async & Performance (Week 2, Days 1-2)

**Goals**: Add async testing, parametrization, performance tests

**Tasks**:
- [ ] Add pytest-asyncio for async test support
- [ ] Convert API tests to use AsyncClient where appropriate
- [ ] Add parametrized tests for:
  - [ ] Multiple chunking strategies
  - [ ] Multiple embedding models
  - [ ] Multiple languages
  - [ ] Edge cases (empty input, very long text, special characters)
- [ ] Add performance benchmarks:
  - [ ] Chunking speed
  - [ ] Embedding generation speed
  - [ ] Retrieval latency

**Success Criteria**:
- All async endpoints tested with AsyncClient
- Parametrized tests cover major variations
- Performance baselines established

### Phase 4: CI/CD Integration (Week 2, Days 3-4)

**Goals**: Optimize test execution, CI/CD pipeline

**Tasks**:
- [ ] Update `.github/workflows/ci.yml`:
  - [ ] Add coverage reporting (Codecov)
  - [ ] Run unit tests on every commit (fast)
  - [ ] Run integration tests on PR (requires data)
  - [ ] Run slow tests on merge to main
- [ ] Add pytest-xdist for parallel test execution
- [ ] Add pytest-timeout to prevent hanging tests
- [ ] Configure coverage thresholds (fail if <80%)
- [ ] Add coverage badge to README.md

**Success Criteria**:
- Unit tests complete in <10s
- Full test suite completes in <60s
- Coverage report published to Codecov
- CI fails if coverage drops below 80%

### Phase 5: Docker/E2E Foundation (Week 2, Day 5)

**Goals**: Prepare for Docker integration testing (future work)

**Tasks**:
- [ ] Create `tests/e2e/` directory structure
- [ ] Add pytest-docker or testcontainers-python
- [ ] Write sample Docker container tests:
  - [ ] Container builds successfully
  - [ ] Container starts and responds to health checks
  - [ ] API accessible via container
- [ ] Document E2E testing approach in plan/

**Success Criteria**:
- E2E test framework configured
- Sample Docker tests demonstrate approach
- Ready for DOCKER-TEST-SUITE-PLAN.md implementation

## Updated pyproject.toml

```toml
[project.optional-dependencies]
dev = [
    # Existing
    "mypy>=1.18.2",
    "ruff>=0.8.4",

    # Testing (Medical-Grade Requirements)
    "pytest>=8.0.0",
    "pytest-asyncio>=0.21.0",      # Async test support
    "pytest-cov>=6.0.0",            # Coverage reporting (statement + branch)
    "pytest-xdist>=3.5.0",          # Parallel test execution
    "pytest-timeout>=2.2.0",        # Prevent hanging tests
    "pytest-mock>=3.12.0",          # Mocking utilities
    "httpx>=0.27.0",                # AsyncClient for FastAPI

    # Clinical/Safety Testing
    "hypothesis>=6.90.0",           # Property-based testing (edge cases)
    "faker>=22.0.0",                # Generate test data (clinical text)

    # Security Testing
    "safety>=3.0.0",                # Dependency vulnerability scanning
    "bandit>=1.7.5",                # Security linting (AST analysis)

    # Performance/Regression
    "pytest-benchmark>=4.0.0",      # Performance regression testing

    # Requirements Traceability
    "pytest-html>=4.0.0",           # HTML test reports
    "pytest-json-report>=1.5.0",    # JSON test reports (traceability)

    # Docker/E2E (Phase 5)
    "pytest-docker>=3.1.0",         # Docker integration
    # OR "testcontainers>=4.0.0",   # Alternative to pytest-docker
]

[tool.pytest.ini_options]
python_files = ["test_*.py"]
testpaths = ["tests"]
addopts = [
    "-v",                           # Verbose output
    "--strict-markers",             # Fail on unknown markers
    "--tb=short",                   # Short traceback format

    # Coverage (Medical-Grade: 95%+ required)
    "--cov=phentrieve",             # Coverage for phentrieve package
    "--cov=api",                    # Coverage for API package
    "--cov-branch",                 # ⚕️ Branch coverage (IEC 62304 requirement)
    "--cov-report=term-missing",    # Show missing lines
    "--cov-report=html:htmlcov",    # HTML coverage report
    "--cov-report=xml:coverage.xml", # XML for CI (Codecov)
    "--cov-fail-under=95",          # ⚕️ Medical-grade: Fail if coverage < 95%

    # Test execution
    "-m", "not slow",               # Skip slow tests by default
    "--maxfail=5",                  # Stop after 5 failures

    # Reporting
    "--html=htmlreports/report.html",  # HTML test report
    "--json-report",                # JSON report (requirements traceability)
    "--json-report-file=testreports/report.json",
]

markers = [
    # Standard categories
    "unit: Fast unit tests (no I/O, mocked dependencies)",
    "integration: Integration tests (real database, embeddings)",
    "api: FastAPI endpoint tests",
    "cli: CLI command tests",
    "slow: Slow tests (>5s, run in CI only)",

    # Medical-grade categories
    "clinical: Clinical validation tests (accuracy, precision, recall)",
    "safety: Safety-critical tests (error handling, graceful degradation)",
    "security: Security tests (OWASP, input validation, PHI protection)",
    "regression: Regression tests (baseline accuracy preservation)",
    "edge_case: Edge case and boundary condition tests",

    # Requirements traceability (add all requirements)
    "req_REQ001: HPO term retrieval from clinical text",
    "req_REQ002: Assertion detection (negation, normality, uncertainty)",
    "req_REQ003: Multilingual support (EN, DE, ES, FR, NL)",
    "req_REQ004: Semantic chunking strategies",
    "req_REQ005: Vector search with ChromaDB",
    "req_REQ006: Cross-encoder reranking",
    "req_REQ007: Input validation and sanitization",
    "req_REQ008: Error handling and logging",
    "req_REQ009: API endpoint functionality",
    "req_REQ010: CLI command functionality",
    # ... add more requirements as needed

    # Resource requirements
    "requires_data: Tests requiring HPO data",
    "requires_models: Tests requiring ML models",
    "requires_internet: Tests requiring internet (model downloads)",
    "docker: Docker container tests (Phase 5)",
    "e2e: End-to-end tests (Phase 5)",
]

asyncio_mode = "auto"              # Automatic async test detection
timeout = 300                      # 5 minute global timeout
log_cli = true                     # Show logs during test execution
log_cli_level = "INFO"             # Log level for CLI output

norecursedirs = [
    ".*",
    "build",
    "dist",
    "docs",
    "data",
    "venv",
    "env",
    "*.egg-info",
]
```

## Running Tests

### Local Development

```bash
# Fast unit tests only (default)
make test                          # or: pytest

# All tests including integration
pytest -m "unit or integration"

# Specific categories
pytest -m unit                     # Unit tests only
pytest -m api                      # API tests only
pytest -m cli                      # CLI tests only
pytest -m integration              # Integration tests only

# Include slow tests
pytest -m "slow"                   # Slow tests only
pytest                             # All tests (including slow)

# Parallel execution (8 workers)
pytest -n 8

# With coverage
pytest --cov=phentrieve --cov-report=html
open htmlcov/index.html            # View coverage report

# Specific test file
pytest tests/unit/test_chunking.py

# Pattern matching
pytest -k "assertion"              # Tests with 'assertion' in name
```

### CI/CD

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip uv
          uv sync --all-extras

      # Unit tests (fast, always run)
      - name: Run unit tests
        run: pytest -m unit --cov --cov-report=xml

      # Integration tests (requires data, run on PR)
      - name: Run integration tests
        if: github.event_name == 'pull_request'
        run: pytest -m integration --cov --cov-append

      # Upload coverage
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

## Success Metrics (Medical-Grade Standards)

### Phase 1 (Foundation)
- ✅ All 87 existing tests migrated and passing
- ✅ No unittest.TestCase classes remaining
- ✅ Coverage baseline established (statement + branch)
- ✅ pytest.ini configured with medical-grade markers

### Phase 2 (Coverage Expansion)
- ✅ **95%+ statement coverage** for phentrieve/ (IEC 62304 Class C)
- ✅ **90%+ branch coverage** for phentrieve/ (safety-critical paths)
- ✅ **100% coverage** for safety-critical modules (assertion detection, HPO mapping)
- ✅ **90%+ coverage** for api/ package
- ✅ All critical user workflows tested
- ✅ Requirements traceability matrix complete

### Phase 3 (Clinical & Safety)
- ✅ **Clinical validation complete**:
  - HPO mapping F1 score ≥ 0.85 (baseline established)
  - Assertion detection F1 score ≥ 0.90 (negation/normality)
  - Multilingual accuracy validated (EN, DE, ES, FR, NL)
- ✅ **Safety testing complete**:
  - All edge cases covered (malformed input, resource limits)
  - Error handling validated (graceful degradation)
  - Data quality tests passing (validation, sanitization)
- ✅ **Security testing complete**:
  - OWASP Top 10 vulnerabilities tested
  - Dependency scanning clean (no high/critical CVEs)
  - Input injection attacks prevented

### Phase 4 (CI/CD Integration)
- ✅ Unit tests <10s
- ✅ Full suite <5min (medical-grade thorough testing)
- ✅ Coverage reports in CI (Codecov)
- ✅ **Coverage thresholds enforced** (fail if <95%)
- ✅ **Regression tests automated** (F1 score baselines)
- ✅ Coverage badge in README (95%+ displayed)
- ✅ Security scans automated (Safety, Bandit)

### Phase 5 (E2E Foundation)
- ✅ E2E framework configured (pytest-docker)
- ✅ Sample Docker tests working
- ✅ Production environment validated
- ✅ Ready for full Docker test suite (DOCKER-TEST-SUITE-PLAN.md)

## Antipatterns to Avoid

### ❌ Don't Do This

1. **Mixing pytest and unittest styles**
   ```python
   # BAD: unittest.TestCase
   class TestChunking(unittest.TestCase):
       def setUp(self): ...

   # GOOD: Pure pytest
   @pytest.fixture
   def chunker():
       return SlidingWindowChunker()
   ```

2. **Duplicated setup code**
   ```python
   # BAD: Duplicate setup in every test
   def test_a():
       model = load_model()  # Expensive
       ...

   def test_b():
       model = load_model()  # Expensive, again!
       ...

   # GOOD: Shared fixture
   @pytest.fixture(scope="module")
   def model():
       return load_model()  # Once per module
   ```

3. **Testing implementation details**
   ```python
   # BAD: Testing internal state
   assert chunker._internal_buffer == [...]

   # GOOD: Testing behavior
   assert len(chunker.chunk(text)) == 3
   ```

4. **Overly complex tests**
   ```python
   # BAD: Testing multiple things
   def test_everything():
       assert chunker.chunk(text1) == expected1
       assert chunker.chunk(text2) == expected2
       assert chunker.chunk(text3) == expected3

   # GOOD: Parametrized test
   @pytest.mark.parametrize("text,expected", [
       (text1, expected1),
       (text2, expected2),
       (text3, expected3),
   ])
   def test_chunking(text, expected):
       assert chunker.chunk(text) == expected
   ```

5. **Slow tests without markers**
   ```python
   # BAD: Unmarked slow test
   def test_full_indexing():
       build_index()  # Takes 30s

   # GOOD: Marked as slow
   @pytest.mark.slow
   @pytest.mark.requires_models
   def test_full_indexing():
       build_index()
   ```

## References

- [Pytest Documentation](https://docs.pytest.org/en/stable/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Typer Testing](https://typer.tiangolo.com/tutorial/testing/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-docker](https://github.com/avast/pytest-docker)
- [Testcontainers Python](https://testcontainers-python.readthedocs.io/)
- VNtyper CI/CD: `/tmp/VNtyper/tests/conftest.py` (session hooks, fixtures)

## Medical-Grade Testing Summary

### Coverage Requirements Comparison

| Aspect | General Software | **Medical-Grade (Phentrieve)** |
|--------|-----------------|--------------------------------|
| **Statement Coverage** | 70-80% acceptable | **95%+ required** (IEC 62304 Class C) |
| **Branch Coverage** | Optional | **90%+ required** (safety-critical) |
| **Critical Path Coverage** | Not specified | **100% required** (assertion, HPO mapping) |
| **Testing Burden** | 20-30% of development | **50-60% of development** (industry standard) |
| **Clinical Validation** | N/A | **Required** (F1 ≥ 0.85 for HPO mapping) |
| **Assertion Accuracy** | N/A | **Required** (F1 ≥ 0.90 for negation/normality) |
| **Edge Case Testing** | Nice to have | **Mandatory** (malformed input, adversarial) |
| **Security Testing** | Recommended | **Mandatory** (OWASP Top 10, PHI protection) |
| **Regression Testing** | Nice to have | **Automated** (baseline F1 scores) |
| **Requirements Traceability** | Optional | **Required** (every req → test mapping) |
| **Safety Testing** | Optional | **Mandatory** (error handling, graceful degradation) |

### Why Medical-Grade Matters for Phentrieve

1. **Patient Safety Impact**: Incorrect HPO term mapping could lead to misdiagnosis
2. **Clinical Decision Support**: CDS systems require FDA/IEC 62304 compliance
3. **High Reliability**: Medical-grade software requires 95-100% test coverage
4. **Regulatory Compliance**: IEC 62304 Class C standards for diagnostic use
5. **Legal Liability**: Medical software errors can result in patient harm and litigation
6. **Professional Trust**: Healthcare professionals need confidence in accuracy

### Testing Investment Justification

**Testing Burden**: Medical-grade software requires **50-60% of total development effort** dedicated to testing (FDA/IEC 62304 industry standard).

**For Phentrieve (2-3 week sprint)**:
- Week 1: Foundation + Coverage (Phase 1-2)
- Week 2: Clinical/Safety Testing (Phase 3)
- Week 3: CI/CD + E2E Foundation (Phase 4-5)

**Return on Investment**:
- ✅ Prevent clinical errors (patient safety)
- ✅ Regulatory compliance (FDA CDS guidelines)
- ✅ Professional credibility (95%+ coverage badge)
- ✅ Reduced liability (documented validation)
- ✅ Faster debugging (comprehensive test suite)
- ✅ Confident deployment (regression prevention)

## Notes

- **Start small**: Migrate existing tests first, then expand
- **Incremental adoption**: Don't rewrite everything at once
- **Coverage is a guide**: 95% coverage + clinical validation = medical-grade quality
- **Fast feedback**: Keep unit tests fast (<10s) for developer workflow
- **CI optimization**: Use markers to run subsets of tests strategically
- **Clinical accuracy first**: Prioritize HPO mapping and assertion detection validation
- **Safety-critical focus**: 100% coverage for assertion detection and HPO mapping modules
- **Requirements traceability**: Every requirement must map to at least one test
- **Docker later**: Foundation first, Docker/E2E in Phase 5 (or separate sprint)
