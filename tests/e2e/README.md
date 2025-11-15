# End-to-End (E2E) Docker Tests

Comprehensive E2E test suite that validates the complete Phentrieve system running in Docker containers with production-like security hardening.

## Overview

The E2E test suite validates three critical aspects:

1. **Security Hardening** (12 tests) - `test_docker_security.py`
   - Non-root user execution (UID 10001)
   - Read-only root filesystem
   - Capability dropping (ALL removed)
   - Memory and CPU limits
   - Security options (no-new-privileges, seccomp)
   - Network isolation
   - Tmpfs mount permissions

2. **Health & Readiness** (14 tests) - `test_docker_health.py`
   - Health endpoint functionality
   - Service readiness checks
   - Container health status
   - Uptime monitoring
   - Restart behavior
   - OOM protection

3. **API Functionality** (17 tests) - `test_api_e2e.py`
   - Query endpoint workflows
   - Clinical text processing
   - Error handling and validation
   - Response format compliance
   - Performance benchmarks
   - Multiple query scenarios

**Total: 43 E2E tests** validating production deployment readiness.

## Quick Start

### Prerequisites

- Docker Engine installed and running
- docker-compose V2 installed
- HPO data prepared: `phentrieve data prepare && phentrieve index build`
- At least 6GB free disk space (Docker images + volumes)
- At least 4GB RAM available for containers

### Running Tests

```bash
# Run all E2E tests (recommended)
make test-e2e

# Run specific test categories
make test-e2e-security    # Security tests only
make test-e2e-health      # Health tests only
make test-e2e-api         # API workflow tests only

# Run with verbose output
pytest tests/e2e/ -v -s

# Run specific test
pytest tests/e2e/test_docker_security.py::TestDockerSecurity::test_container_runs_as_non_root_user -v
```

### Cleanup

```bash
# Clean up Docker resources after tests
make test-e2e-clean

# View container logs
make test-e2e-logs

# Open shell in test container
make test-e2e-shell
```

## Test Architecture

### Fixtures (conftest.py)

E2E tests use **session-scoped fixtures** from pytest-docker for optimal performance:

- `docker_compose_file` - Points to docker-compose.test.yml
- `docker_compose_project_name` - Fixed project name for test isolation
- `docker_setup` - Container lifecycle commands
- `api_service` - API base URL with health check wait (up to 180s)
- `api_container` - Container object for inspection
- `api_health_endpoint`, `api_config_endpoint`, `api_query_endpoint` - Convenience URLs

### Container Lifecycle

```
Test Session Start
    ↓
Build Docker Images (first run: ~5-10 min, cached: ~30s)
    ↓
Start Containers (docker-compose up -d)
    ↓
Wait for Health Check (up to 180s for model loading)
    ↓
Run All Tests (containers reused across all tests)
    ↓
Stop Containers (docker-compose down -v)
    ↓
Test Session End
```

### Docker Configuration

**docker-compose.test.yml** - Optimized for testing:
- Reduced resource limits (2 CPU, 4GB RAM vs production 4 CPU, 8GB)
- Shorter health check timeouts (60s start period vs 180s)
- Test-specific port mapping (8001:8000 to avoid conflicts)
- Same security hardening as production
- Minimal data volumes (no persistent storage needed)

## Test Details

### Security Tests (test_docker_security.py)

Validates Docker security best practices are correctly applied:

```python
@pytest.mark.e2e
class TestDockerSecurity:
    def test_container_runs_as_non_root_user(...)
    def test_root_filesystem_is_read_only(...)
    def test_all_capabilities_dropped(...)
    def test_no_new_privileges_enabled(...)
    def test_memory_limit_enforced(...)
    def test_cpu_limit_enforced(...)
    # ... 12 tests total
```

**Why These Tests Matter:**
- Prevents container escape exploits (non-root user)
- Limits lateral movement (read-only filesystem)
- Reduces attack surface (capability dropping)
- Prevents DoS attacks (resource limits)
- Enforces least privilege (security options)

### Health Tests (test_docker_health.py)

Validates service readiness and availability:

```python
@pytest.mark.e2e
class TestDockerHealth:
    def test_health_endpoint_accessible(...)
    def test_health_endpoint_returns_valid_json(...)
    def test_container_is_healthy(...)
    def test_container_has_no_health_failures(...)
    def test_api_responds_within_timeout(...)
    # ... 14 tests total
```

**Why These Tests Matter:**
- Ensures service is ready before accepting traffic
- Validates monitoring integration points
- Detects startup issues early
- Verifies graceful degradation

### API Workflow Tests (test_api_e2e.py)

Validates complete API functionality through real HTTP requests:

```python
@pytest.mark.e2e
class TestQueryWorkflow:
    def test_query_with_simple_text(...)
    def test_query_with_medical_terminology(...)
    def test_query_with_top_k_parameter(...)
    def test_query_with_empty_text_fails(...)
    def test_query_performance_acceptable(...)
    # ... 17 tests total
```

**Why These Tests Matter:**
- Validates end-to-end integration (API → embeddings → ChromaDB)
- Ensures error handling works correctly
- Verifies response schemas are correct
- Benchmarks performance under realistic conditions

## Performance Expectations

### First Run (Cold Start)
- **Docker build**: 5-10 minutes (downloads models, builds images)
- **Container startup**: 2-3 minutes (model loading, index initialization)
- **Total first run**: ~15 minutes for full suite

### Subsequent Runs (Cached)
- **Docker build**: ~30 seconds (layer cache hit)
- **Container startup**: 2-3 minutes (model loading)
- **Total cached run**: ~3-4 minutes for full suite

### Per-Test Performance
- **Security tests**: <1 second each (container inspection)
- **Health tests**: 1-2 seconds each (HTTP requests)
- **API workflow tests**: 2-10 seconds each (model inference)

## Troubleshooting

### Tests Fail to Start

**Problem**: `docker-compose` command not found

**Solution**:
```bash
# Install Docker Compose V2
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker compose version
```

**Problem**: Permission denied accessing Docker socket

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

### Container Fails Health Check

**Problem**: API container never becomes healthy

**Solution**:
```bash
# View container logs
make test-e2e-logs

# Common issues:
# 1. HPO data not prepared → Run: phentrieve data prepare && phentrieve index build
# 2. Insufficient memory → Increase Docker memory limit to 4GB+
# 3. Model download timeout → Check internet connection, increase timeout

# Manually verify health endpoint
docker-compose -f docker-compose.test.yml -p phentrieve_e2e_test up -d
sleep 60
curl http://localhost:8001/api/v1/health
```

### Tests Pass Locally But Fail in CI

**Problem**: Different Docker versions or resource constraints

**Solution**:
```bash
# Check Docker version consistency
docker --version
docker compose version

# CI may need explicit resource allocation
# Add to docker-compose.test.yml:
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

### Slow Test Execution

**Problem**: Tests take too long to run

**Solution**:
```bash
# Use test markers to run subsets
pytest tests/e2e/ -v -m e2e -k "security"  # Only security tests

# Reuse existing containers (skip rebuild)
make test-e2e-fast

# Run tests in parallel (requires pytest-xdist)
pytest tests/e2e/ -v -n auto
```

## Best Practices

### Writing New E2E Tests

1. **Use `@pytest.mark.e2e` decorator** for all E2E tests
2. **Reuse session-scoped fixtures** (don't restart containers per test)
3. **Test one thing per test** (single assertion principle)
4. **Use descriptive test names** following pattern: `test_<component>_<scenario>_<expected>`
5. **Include docstrings** explaining security rationale or expected behavior
6. **Validate both success and failure cases**

Example:
```python
@pytest.mark.e2e
def test_query_with_invalid_top_k_fails(api_query_endpoint: str):
    """
    Verify query endpoint validates top_k parameter.

    Expected:
        Invalid top_k (negative) returns 422 validation error
    """
    payload = {"query_text": "test", "top_k": -1}
    response = requests.post(api_query_endpoint, json=payload, timeout=10)
    assert response.status_code in [422, 400]
```

### CI/CD Integration

Add to `.github/workflows/ci.yml`:

```yaml
jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Prepare HPO Data
        run: |
          # Download minimal test dataset
          phentrieve data prepare --minimal

      - name: Run E2E Tests
        run: make test-e2e

      - name: Upload Logs on Failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: docker-logs
          path: /tmp/docker-logs/
```

## Related Documentation

- **Docker Configuration**: `docker-compose.test.yml`
- **Production Docker Setup**: `docker-compose.yml`
- **Security Hardening**: `api/Dockerfile`
- **Testing Modernization Plan**: `TESTING-MODERNIZATION-PLAN.md`
- **Local Development**: `plan/LOCAL-DEV-ENVIRONMENT.md`

## Future Enhancements

### Phase 4 - Performance Testing
- Load testing with concurrent requests
- Stress testing resource limits
- Benchmark model inference latency
- Memory leak detection

### Phase 5 - Security Scanning
- SAST with Bandit
- Dependency scanning with Safety
- Docker image scanning with Trivy
- Secret detection with TruffleHog

### Phase 6 - Integration Tests
- Multi-container orchestration (API + frontend)
- Database persistence tests
- Network isolation validation
- Volume mount security

## Support

For issues or questions:
1. Check troubleshooting section above
2. View container logs: `make test-e2e-logs`
3. Open shell in container: `make test-e2e-shell`
4. File issue with logs attached

---

**Status**: ✅ Phase 3 Complete - 43 E2E tests validating production deployment
**Coverage**: Security, Health, API Workflows
**Maintainer**: Phentrieve Testing Team
**Last Updated**: 2025-11-15
