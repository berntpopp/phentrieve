"""
End-to-End (E2E) Docker Tests.

This package contains E2E tests that validate the complete Phentrieve system
running in Docker containers. Tests are organized into three categories:

1. **Security Tests** (test_docker_security.py):
   - Non-root user enforcement
   - Read-only filesystem validation
   - Capability dropping verification
   - Resource limits enforcement
   - Security options validation

2. **Health Tests** (test_docker_health.py):
   - Health endpoint functionality
   - Service readiness and stability
   - Container health checks
   - Uptime and availability monitoring

3. **E2E Workflow Tests** (test_api_e2e.py):
   - API endpoint functionality
   - Query workflow end-to-end
   - Error handling and validation
   - Response format validation
   - Performance benchmarks

Running E2E Tests
-----------------
E2E tests require Docker and docker-compose. They use pytest-docker to
manage container lifecycle automatically.

Basic usage:
    # Run all E2E tests (requires Docker)
    make test-e2e

    # Run specific test file
    pytest tests/e2e/test_docker_security.py -v

    # Run specific test
    pytest tests/e2e::TestDockerSecurity::test_container_runs_as_non_root_user -v

    # Run with verbose output and show logs
    pytest tests/e2e/ -v -s

Test Markers
------------
All E2E tests are marked with @pytest.mark.e2e for easy filtering:

    # Run only E2E tests
    pytest -m e2e

    # Run everything except E2E tests
    pytest -m "not e2e"

Prerequisites
-------------
- Docker Engine installed and running
- docker-compose V2 installed
- HPO data prepared (data/hpo_core_data/)
- At least 6GB free disk space for images and volumes
- At least 4GB RAM available for containers

Performance Notes
-----------------
- First run: ~5-10 minutes (builds Docker images, downloads models)
- Subsequent runs: ~2-3 minutes (uses cached images)
- Session-scoped fixtures reuse containers across all tests
- Health check wait: up to 180 seconds for model loading

Troubleshooting
---------------
If tests fail:
    1. Check Docker is running: `docker ps`
    2. Check compose file: `docker-compose -f docker-compose.test.yml config`
    3. View container logs: `docker logs phentrieve_e2e_test-phentrieve_api_test-1`
    4. Clean up: `docker-compose -f docker-compose.test.yml down -v`
    5. Rebuild: `docker-compose -f docker-compose.test.yml build --no-cache`
"""
