# Docker Test Suite Implementation Plan

**Status**: Planning
**Priority**: High
**Estimated Effort**: 2-3 days

## Overview

Implement a comprehensive, maintainable Docker testing strategy using pytest that works seamlessly in both local development and CI/CD environments. Tests should verify container functionality, security, and integration points without unnecessary complexity.

## Goals

1. ✅ **Unified Testing**: Same tests run locally and in GitHub Actions
2. ✅ **Fast Feedback**: Quick tests for common issues, thorough tests for CI
3. ✅ **Simple Setup**: Minimal dependencies, clear documentation
4. ✅ **Practical Coverage**: Test what matters, skip what doesn't

## Test Categories

### 1. Container Build Tests (Fast)
**Purpose**: Verify images build successfully
**Scope**: Both API and Frontend
**Runtime**: ~2-5 minutes

```python
# tests/docker/test_build.py
def test_api_dockerfile_builds():
    """Verify API Dockerfile builds without errors"""

def test_frontend_dockerfile_builds():
    """Verify Frontend Dockerfile builds without errors"""

def test_build_layers_optimized():
    """Check layer count is reasonable (multi-stage worked)"""
```

### 2. Container Runtime Tests (Medium)
**Purpose**: Verify containers start and respond
**Scope**: Health checks, port exposure, user permissions
**Runtime**: ~30-60 seconds

```python
# tests/docker/test_runtime.py
def test_api_container_starts():
    """API container starts and stays running"""

def test_api_health_endpoint():
    """Health endpoint returns 200 OK"""

def test_api_runs_as_nonroot():
    """Container runs as user phentrieve (UID 10001)"""

def test_frontend_container_starts():
    """Frontend container starts and stays running"""

def test_frontend_health_endpoint():
    """Health endpoint returns 'OK'"""

def test_frontend_runs_as_nonroot():
    """Container runs as nginx user (UID 101)"""
```

### 3. Security Verification Tests (Fast)
**Purpose**: Verify security hardening is applied
**Scope**: User permissions, exposed ports, capabilities
**Runtime**: ~10-20 seconds

```python
# tests/docker/test_security.py
def test_api_user_is_nonroot():
    """Inspect image confirms User: phentrieve"""

def test_api_exposed_port_is_8000():
    """Only port 8000 is exposed"""

def test_frontend_user_is_nonroot():
    """Inspect image confirms User: nginx"""

def test_frontend_exposed_port_is_8080():
    """Only port 8080 is exposed (non-privileged)"""
```

### 4. Integration Tests (Slow - CI Only)
**Purpose**: Verify API + Frontend work together
**Scope**: End-to-end request flow
**Runtime**: ~2-5 minutes

```python
# tests/docker/test_integration.py
@pytest.mark.integration
def test_frontend_can_reach_api():
    """Frontend can query API health endpoint"""

@pytest.mark.integration
def test_api_query_flow():
    """Complete query flow: frontend → API → database → response"""
```

## Test Infrastructure

### Directory Structure
```
tests/
├── docker/
│   ├── __init__.py
│   ├── conftest.py              # Docker-specific fixtures
│   ├── test_build.py            # Build verification
│   ├── test_runtime.py          # Runtime verification
│   ├── test_security.py         # Security checks
│   └── test_integration.py      # E2E tests
├── conftest.py                  # Global fixtures
└── pytest.ini                   # Pytest configuration
```

### Key Dependencies
```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-docker>=3.1.0",      # Container orchestration
    "pytest-timeout>=2.2.0",     # Prevent hanging tests
    "requests>=2.31.0",          # HTTP testing
]
```

### Pytest Configuration
```ini
# pytest.ini
[pytest]
markers =
    docker: Docker container tests
    integration: Integration tests (slow)
    unit: Fast unit tests

timeout = 300
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Don't run integration tests by default
addopts = -v --tb=short -m "not integration"
```

### Shared Fixtures (conftest.py)

```python
# tests/docker/conftest.py
import pytest
import docker
import time

@pytest.fixture(scope="session")
def docker_client():
    """Docker client for all tests"""
    return docker.from_env()

@pytest.fixture(scope="module")
def api_image(docker_client):
    """Build API image once per module"""
    image, logs = docker_client.images.build(
        path=".",
        dockerfile="api/Dockerfile",
        tag="phentrieve-api:test"
    )
    yield image
    # Cleanup handled by session teardown

@pytest.fixture(scope="module")
def frontend_image(docker_client):
    """Build Frontend image once per module"""
    image, logs = docker_client.images.build(
        path="./frontend",
        dockerfile="Dockerfile",
        tag="phentrieve-frontend:test",
        buildargs={"VITE_API_URL": "/api/v1"}
    )
    yield image

@pytest.fixture
def api_container(docker_client, api_image):
    """Start API container for test, cleanup after"""
    container = docker_client.containers.run(
        image="phentrieve-api:test",
        detach=True,
        ports={"8000/tcp": 8001},
        volumes={"/tmp/test-data": {"bind": "/phentrieve_data_mount", "mode": "ro"}},
        remove=True
    )
    time.sleep(5)  # Wait for startup
    yield container
    container.stop()

@pytest.fixture
def frontend_container(docker_client, frontend_image):
    """Start Frontend container for test, cleanup after"""
    container = docker_client.containers.run(
        image="phentrieve-frontend:test",
        detach=True,
        ports={"8080/tcp": 8082},
        remove=True
    )
    time.sleep(2)  # Wait for nginx
    yield container
    container.stop()
```

## GitHub Actions Integration

### Workflow Updates

```yaml
# .github/workflows/docker-publish.yml
jobs:
  test-docker:
    name: Test Docker Images
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install test dependencies
        run: pip install pytest pytest-docker pytest-timeout requests

      - name: Run Docker tests (unit + build)
        run: pytest tests/docker -v -m "docker and not integration"

      - name: Run integration tests
        run: pytest tests/docker -v -m "integration"
```

## Local Development Workflow

### Running Tests Locally

```bash
# Install test dependencies
pip install -e .[test]

# Run fast tests (build + runtime)
pytest tests/docker -v -m "docker and not integration"

# Run all tests including integration
pytest tests/docker -v

# Run specific test file
pytest tests/docker/test_runtime.py -v

# Run with detailed output
pytest tests/docker -vv --tb=long
```

### Make Targets (Optional)

```makefile
# Makefile additions
.PHONY: test-docker test-docker-all

test-docker:
	pytest tests/docker -v -m "docker and not integration"

test-docker-all:
	pytest tests/docker -v

test-docker-build:
	pytest tests/docker/test_build.py -v
```

## Implementation Phases

### Phase 1: Basic Infrastructure (Week 1)
- [ ] Create test directory structure
- [ ] Add pytest dependencies to pyproject.toml
- [ ] Create pytest.ini configuration
- [ ] Implement conftest.py with basic fixtures
- [ ] Write build tests (test_build.py)

### Phase 2: Runtime Tests (Week 1-2)
- [ ] Implement runtime tests (test_runtime.py)
- [ ] Implement security tests (test_security.py)
- [ ] Add timeout handling
- [ ] Test locally

### Phase 3: Integration Tests (Week 2)
- [ ] Implement integration tests (test_integration.py)
- [ ] Add docker-compose orchestration
- [ ] Test multi-container scenarios

### Phase 4: CI/CD Integration (Week 2-3)
- [ ] Update GitHub Actions workflow
- [ ] Add test job to docker-publish.yml
- [ ] Configure caching for faster builds
- [ ] Document in README.md

## Success Criteria

✅ **Build Tests**: Images build successfully in <5 minutes
✅ **Runtime Tests**: Containers start and respond to health checks
✅ **Security Tests**: All security configurations verified
✅ **Integration Tests**: Frontend can communicate with API
✅ **CI/CD**: Tests run automatically on PR and pass consistently
✅ **Documentation**: Clear instructions for running tests locally

## Non-Goals (Keeping It Simple)

❌ **NOT** testing every API endpoint (use API-specific tests for that)
❌ **NOT** performance benchmarking (separate concern)
❌ **NOT** load testing (use dedicated tools)
❌ **NOT** testing third-party images (trust base images)
❌ **NOT** complex mocking (use real containers where possible)

## References

- [pytest-docker documentation](https://github.com/avast/pytest-docker)
- [Docker Python SDK](https://docker-py.readthedocs.io/)
- [VNtyper CI/CD implementation](https://github.com/hassansaei/VNtyper)
- Best Practices: Testcontainers, pytest fixtures, CI/CD optimization

## Notes

- **Keep tests fast**: Cache built images, reuse containers when safe
- **Keep tests simple**: One assertion per test where practical
- **Keep tests maintainable**: Clear names, good fixtures, minimal duplication
- **Keep tests practical**: Test real scenarios, not hypotheticals
