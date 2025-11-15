# Testing Modernization Plan

**Status**: Ready for Implementation
**Priority**: High
**Estimated Effort**: 1-2 weeks
**Coverage Target**: 80% statement (pragmatic quality standard)

## Overview

Modernize Phentrieve's test suite with **practical 80% coverage** and strong e2e testing. Focus on **not blocking development** while ensuring quality.

### Guiding Principles

- **80% coverage is pragmatic**: Not too strict, not too loose
- **Fast feedback**: Don't slow down development
- **E2E tests matter most**: Production environment validation is critical
- **Flexible**: Tests help development, not hinder it
- **Simple**: No over-engineering, clear structure

---

## Current State

### What We Have
- ✅ **87 tests**: Core functionality, CLI, API
- ✅ **Basic pytest setup** in pyproject.toml
- ✅ **Mixed style**: unittest.TestCase + pytest (needs migration)

### What We Need
- ❌ **Unknown coverage**: No baseline measured
- ❌ **No Docker tests**: Production environment untested
- ❌ **Mixed test styles**: Need pytest migration
- ❌ **No clear structure**: Tests not organized by type

---

## Goals

1. **80% statement coverage** for phentrieve/ and api/
2. **Strong Docker E2E tests** for production validation
3. **Fast test suite**: Unit <10s, integration <60s, full <5min
4. **Developer-friendly**: Easy to run, debug, and extend
5. **CI-ready**: Automated testing without blocking merges

---

## Test Architecture

### Directory Structure (Simple 3-Tier)

```
tests/
├── conftest.py                      # Root fixtures
├── test_data/                       # Test data files
│   └── sample_cases.json            # Test cases for validation
│
├── unit/                            # Fast unit tests (mocked)
│   ├── conftest.py
│   ├── core/                        # phentrieve/ package
│   │   ├── test_chunking.py
│   │   ├── test_assertion_detection.py
│   │   ├── test_semantic_metrics.py
│   │   └── test_embeddings.py
│   ├── api/                         # API unit tests
│   │   ├── test_schemas.py
│   │   └── test_dependencies.py
│   └── cli/                         # CLI unit tests
│       └── test_commands.py
│
├── integration/                     # Integration tests (real deps)
│   ├── conftest.py
│   ├── test_retrieval_pipeline.py  # End-to-end retrieval
│   ├── test_indexing_flow.py       # ChromaDB indexing
│   └── test_api_endpoints.py       # API integration
│
└── e2e/                             # Docker + End-to-End tests
    ├── conftest.py
    ├── test_docker_build.py         # Container builds
    ├── test_docker_runtime.py       # Container health, startup
    ├── test_docker_security.py      # Non-root, read-only FS
    └── test_production_workflow.py  # Full user workflows
```

### Pytest Markers (Minimal)

```ini
[pytest]
markers =
    unit: Fast unit tests (mocked, no I/O)
    integration: Integration tests (real ChromaDB, embeddings)
    e2e: End-to-end Docker tests (slow)
    slow: Slow tests (>5s, skip in local dev)
```

**Usage**:
```python
@pytest.mark.integration
def test_retrieval_pipeline():
    """Integration test with real ChromaDB"""
    ...

@pytest.mark.e2e
@pytest.mark.slow
def test_docker_api_health(api_service):
    """Docker E2E test"""
    ...
```

---

## Key Fixtures

### Root Fixtures (`tests/conftest.py`)

```python
"""Shared fixtures for all tests."""
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_clinical_texts() -> list[str]:
    """Sample clinical texts for testing."""
    return [
        "Patient presents with seizures and developmental delay",
        "No evidence of heart disease",
        "Family history of autism spectrum disorder",
    ]

# Mocks (function-scoped for isolation)
@pytest.fixture
def mock_embedding_model(mocker):
    """Mock sentence transformer model."""
    mock = mocker.MagicMock()
    mock.encode.return_value = [[0.1] * 384]  # Mock 384-dim embedding
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

### Integration Fixtures (`tests/integration/conftest.py`)

```python
"""Integration test fixtures (real dependencies)."""
import pytest

@pytest.fixture(scope="module")
def real_embedding_model():
    """Real embedding model (cached at module scope)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@pytest.fixture(scope="module")
def real_chromadb_collection(tmp_path_factory):
    """Real ChromaDB collection (test isolation)."""
    import chromadb

    persist_dir = tmp_path_factory.mktemp("chromadb")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection("test_hpo_terms")

    yield collection

    # Cleanup
    client.delete_collection("test_hpo_terms")
```

### E2E Fixtures (`tests/e2e/conftest.py`)

```python
"""Docker E2E test fixtures."""
import pytest
import time
import requests

@pytest.fixture(scope="session")
def docker_compose_file():
    """Point to docker-compose files for testing."""
    from pathlib import Path
    return [
        Path.cwd() / "docker-compose.yml",
        Path.cwd() / "docker-compose.test.yml",  # Test overrides
    ]

@pytest.fixture(scope="session")
def docker_compose_project_name():
    """Unique project name for test isolation."""
    return "phentrieve_test"

@pytest.fixture(scope="session")
def api_service(docker_services):
    """Wait for API service health check."""
    url = "http://localhost:8001/api/v1/health"

    # Wait up to 180s for health check
    for _ in range(60):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return "http://localhost:8001"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(3)

    raise RuntimeError("API service failed to become healthy")
```

---

## Implementation Plan

### Phase 0: Baseline Measurement (Day 0)

**Measure current state before ANY changes.**

```bash
# Measure current coverage
pytest --cov=phentrieve --cov=api \
       --cov-report=json:coverage-baseline.json \
       --cov-report=term-missing

# Document baseline
echo "Baseline coverage: $(jq '.totals.percent_covered' coverage-baseline.json)%"
```

**Commit baseline**:
```
docs: establish testing baseline

Current coverage: XX%
Total tests: 87
Test duration: ZZ seconds
```

---

### Phase 1: Foundation (Days 1-2)

**Goal**: Safe migration to new structure.

**Tasks**:
1. **Create new structure** (parallel to existing):
   ```bash
   mkdir -p tests_new/{unit/{core,api,cli},integration,e2e,test_data}
   touch tests_new/conftest.py
   touch tests_new/{unit,integration,e2e}/conftest.py
   ```

2. **Migrate tests** (one category at a time):
   - [ ] Unit tests → `tests_new/unit/`
   - [ ] Convert unittest.TestCase → pure pytest
   - [ ] Update imports and fixtures

3. **Dual-path CI** (safety period - 1 week):
   ```yaml
   # Run BOTH old and new tests
   - name: Run old tests (baseline)
     run: pytest tests/ --cov

   - name: Run new tests (migrated)
     run: pytest tests_new/ --cov
   ```

**Success Criteria**:
- ✅ New structure created
- ✅ 87 tests migrated and passing in `tests_new/`
- ✅ Old tests still passing in `tests/`
- ✅ Coverage ≥ baseline (no regression)

---

### Phase 2: Coverage Expansion (Days 3-5)

**Goal**: Achieve 80% statement coverage.

**New Tests**:
- [ ] `unit/core/test_embeddings.py`: Embedding utilities
- [ ] `unit/api/test_schemas.py`: Pydantic models
- [ ] `integration/test_retrieval_pipeline.py`: Full retrieval flow
- [ ] `integration/test_api_endpoints.py`: API integration

**Success Criteria**:
- ✅ **80%+ coverage** for phentrieve/ and api/
- ✅ All critical workflows tested
- ✅ Test suite <5min

---

### Phase 3: Docker E2E Testing (Days 6-8)

**Goal**: Validate production Docker environment.

**Setup**:

1. **Create docker-compose.test.yml**:
   ```yaml
   # docker-compose.test.yml
   version: '3.9'

   services:
     phentrieve_api:
       environment:
         - LOG_LEVEL=DEBUG
         - TESTING=true
       ports:
         - "8001:8000"  # Expose for pytest

     phentrieve_frontend:
       ports:
         - "8081:8080"  # Expose for pytest
   ```

2. **Install pytest-docker**:
   ```bash
   uv add --group dev pytest-docker
   ```

**Docker Tests**:

```python
# tests/e2e/test_docker_build.py

@pytest.mark.e2e
def test_api_dockerfile_builds():
    """Docker: API image builds without errors"""
    import subprocess
    result = subprocess.run(
        ["docker", "build", "-f", "api/Dockerfile", "-t", "phentrieve-api:test", "."],
        capture_output=True,
        timeout=300
    )
    assert result.returncode == 0

# tests/e2e/test_docker_runtime.py

@pytest.mark.e2e
def test_api_health_check(api_service):
    """Docker: API responds to health check"""
    response = requests.get(f"{api_service}/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.e2e
def test_api_runs_as_nonroot(docker_services):
    """Docker: API container runs as UID 10001 (non-root)"""
    import subprocess
    result = subprocess.run(
        ["docker", "exec", "phentrieve_test_phentrieve_api_1", "id", "-u"],
        capture_output=True,
        text=True
    )
    assert result.stdout.strip() == "10001"

# tests/e2e/test_docker_security.py

@pytest.mark.e2e
def test_api_readonly_filesystem(docker_services):
    """Docker: Root filesystem is read-only"""
    import subprocess
    result = subprocess.run(
        ["docker", "exec", "phentrieve_test_phentrieve_api_1", "touch", "/app/test.txt"],
        capture_output=True
    )
    assert result.returncode != 0  # Should fail (read-only)

# tests/e2e/test_production_workflow.py

@pytest.mark.e2e
@pytest.mark.slow
def test_full_query_workflow(api_service):
    """E2E: Complete query workflow through Docker"""
    response = requests.post(
        f"{api_service}/api/v1/query",
        json={"text": "Patient has seizures", "top_k": 5}
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results["results"]) > 0
    assert results["results"][0]["hpo_id"].startswith("HP:")
```

**Success Criteria**:
- ✅ API + Frontend containers build successfully
- ✅ Containers start and pass health checks
- ✅ Non-root user verified
- ✅ Read-only filesystem validated
- ✅ Full E2E workflow tested

---

### Phase 4: CI/CD Integration & Cleanup (Days 9-10)

**Goal**: Automate testing in CI, remove old structure.

**Tasks**:
1. **Update CI workflow**:
   ```yaml
   # .github/workflows/ci.yml
   jobs:
     test-unit:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5

         - name: Install dependencies
           run: |
             pip install uv
             uv sync --all-extras

         - name: Run unit tests
           run: pytest tests/unit -m unit --cov

         - name: Upload coverage
           uses: codecov/codecov-action@v4

     test-integration:
       if: github.event_name == 'pull_request'
       steps:
         - name: Run integration tests
           run: pytest tests/integration -m integration

     test-e2e:
       if: github.ref == 'refs/heads/main'
       steps:
         - name: Run E2E Docker tests
           run: pytest tests/e2e -m e2e
   ```

2. **Remove old test structure** (after 1-week safety period):
   ```bash
   # Promote new tests
   git mv tests_new tests
   git commit -m "refactor: promote new test structure"
   ```

**Success Criteria**:
- ✅ Unit tests on every commit (<10s)
- ✅ Integration tests on PRs (<60s)
- ✅ E2E tests on main merges (<5min)
- ✅ Coverage badge in README
- ✅ Old structure removed

---

## Updated pyproject.toml

```toml
[project.optional-dependencies]
dev = [
    # Code quality
    "mypy>=1.18.2",
    "ruff>=0.8.4",

    # Testing core
    "pytest>=8.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=6.0.0",
    "pytest-xdist>=3.5.0",          # Parallel execution
    "pytest-timeout>=2.2.0",        # Prevent hanging
    "pytest-mock>=3.12.0",          # Mocking
    "httpx>=0.27.0",                # FastAPI testing

    # Docker E2E
    "pytest-docker>=3.1.0",
    "requests>=2.31.0",
]

[tool.pytest.ini_options]
python_files = ["test_*.py"]
testpaths = ["tests"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",

    # Coverage (80% target)
    "--cov=phentrieve",
    "--cov=api",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=80",          # Enforce 80% threshold

    # Execution
    "-m", "not slow",               # Skip slow by default
]

markers = [
    "unit: Fast unit tests (mocked)",
    "integration: Integration tests (real deps)",
    "e2e: End-to-end Docker tests",
    "slow: Slow tests (>5s)",
]

asyncio_mode = "auto"
timeout = 300
log_cli = true
log_cli_level = "INFO"
```

---

## Running Tests

### Local Development

```bash
# Fast unit tests (default)
pytest                                 # Runs unit tests, skips slow

# All tests including slow
pytest -m "unit or integration or e2e"

# Specific categories
pytest -m unit                         # Unit only
pytest -m integration                  # Integration only
pytest -m e2e                          # Docker E2E only

# Parallel execution (fast!)
pytest -n auto                         # Use all CPU cores

# With coverage
pytest --cov --cov-report=html
open htmlcov/index.html
```

### Docker E2E Tests

```bash
# Build images first
docker-compose -f docker-compose.yml -f docker-compose.test.yml build

# Run E2E tests
pytest tests/e2e -m e2e

# Cleanup
docker-compose -f docker-compose.yml -f docker-compose.test.yml down -v
```

---

## Success Metrics

### Phase 1: Foundation
- ✅ Baseline measured
- ✅ New structure created
- ✅ 87 tests migrated
- ✅ Dual-path CI running

### Phase 2: Coverage
- ✅ **80%+ coverage** for phentrieve/ and api/
- ✅ Test suite <5min

### Phase 3: Docker E2E
- ✅ Container builds verified
- ✅ Health checks passing
- ✅ Security validated
- ✅ Production workflows tested

### Phase 4: CI/CD
- ✅ Automated CI
- ✅ Coverage badge in README
- ✅ Old structure removed

---

## Developer Experience

### Tests Won't Block You

- **Fast by default**: `pytest` runs only fast unit tests
- **Parallel execution**: Use `-n auto` for speed
- **Skip slow tests**: Marked with `@pytest.mark.slow`
- **Clear failures**: Short tracebacks, clear error messages
- **Easy debugging**: Run specific tests easily

### Flexible Coverage

- **80% is pragmatic**: Not perfect, not lax
- **Fails CI at 80%**: Prevents regressions
- **Easy to see gaps**: `--cov-report=html` shows uncovered lines
- **Not blocking**: Can commit with >80%, improve gradually

---

## Notes

- **Start with baseline**: Measure before changing anything
- **Safety first**: Keep old tests during migration
- **Incremental**: Foundation → Coverage → Docker → CI
- **Pragmatic**: 80% is the target, not 100%
- **Fast feedback**: Keep unit tests <10s
- **E2E matters**: Docker tests validate production

---

## References

- [Pytest Documentation](https://docs.pytest.org/en/stable/)
- [pytest-docker Examples](https://github.com/avast/pytest-docker/tree/master/tests)
- [Docker Python SDK](https://docker-py.readthedocs.io/)
