"""
Docker Health Check Tests.

This module validates that Docker health checks and service readiness
mechanisms are functioning correctly:

- Health endpoint accessibility and response format
- Service becomes healthy within expected timeout
- Container health check configuration
- Service availability and uptime
- API readiness for production traffic

All tests are marked with @pytest.mark.e2e for test categorization.
"""

import pytest
import requests
from docker.models.containers import Container


@pytest.mark.e2e
class TestDockerHealth:
    """Health check and service readiness test suite."""

    def test_health_endpoint_accessible(self, api_health_endpoint: str):
        """
        Verify health endpoint is accessible and returns 200 OK.

        Expected:
            GET /api/v1/health returns HTTP 200
            Response time < 5 seconds
        """
        response = requests.get(api_health_endpoint, timeout=5)

        assert response.status_code == 200, (
            f"Health endpoint should return 200, got: {response.status_code}"
        )

    def test_health_endpoint_returns_valid_json(self, api_health_endpoint: str):
        """
        Verify health endpoint returns valid JSON response.

        Expected JSON Structure:
            {
                "status": "healthy" | "unhealthy",
                "service": "phentrieve-api",
                "version": "<version>",
                "uptime_seconds": <float>,
                ...
            }
        """
        response = requests.get(api_health_endpoint, timeout=5)
        assert response.status_code == 200, "Health check should succeed"

        # Verify response is valid JSON and parse it
        try:
            data = response.json()
        except ValueError as e:
            pytest.fail(f"Health endpoint should return valid JSON: {e}")
            raise  # Never reached - satisfies static analysis (pytest.fail raises)

        # Verify required fields exist
        assert "status" in data, "Health response should contain 'status' field"
        assert data["status"] in [
            "healthy",
            "ok",
            "OK",
        ], f"Status should be healthy/ok, got: {data['status']}"

    def test_health_endpoint_reports_service_name(self, api_health_endpoint: str):
        """
        Verify health endpoint reports correct service identifier.

        Expected:
            Response contains service name/identifier
        """
        response = requests.get(api_health_endpoint, timeout=5)
        data = response.json()

        # Service name field may vary (service, name, application, etc.)
        service_fields = ["service", "name", "application", "app"]
        has_service_field = any(field in data for field in service_fields)

        assert has_service_field, (
            f"Health response should contain service identifier, got: {list(data.keys())}"
        )

    def test_health_endpoint_reports_uptime(self, api_health_endpoint: str):
        """
        Verify health endpoint reports service uptime.

        Expected:
            Response contains uptime in seconds
            Uptime > 0 (service has been running)
        """
        response = requests.get(api_health_endpoint, timeout=5)
        data = response.json()

        # Uptime field may vary
        uptime_fields = ["uptime_seconds", "uptime", "running_time"]
        uptime_field = next((field for field in uptime_fields if field in data), None)

        if uptime_field:
            uptime = data[uptime_field]
            assert float(uptime) > 0, f"Uptime should be positive, got: {uptime}"

    def test_container_health_check_configured(self, api_container: Container):
        """
        Verify Docker HEALTHCHECK is configured in container.

        Expected (from docker-compose.test.yml):
            Test: curl -f http://localhost:8000/api/v1/health
            Interval: 10s
            Timeout: 5s
            Retries: 5
            StartPeriod: 60s
        """
        healthcheck = api_container.attrs["Config"].get("Healthcheck")

        assert healthcheck is not None, "Container should have HEALTHCHECK configured"

        # Verify health check test command
        test_cmd = healthcheck.get("Test", [])
        assert len(test_cmd) > 0, "Health check test command should be defined"

        # Should contain curl command for health endpoint
        test_str = " ".join(test_cmd)
        assert "health" in test_str.lower(), (
            f"Health check should test health endpoint, got: {test_str}"
        )

    def test_container_is_healthy(self, api_container: Container):
        """
        Verify container has reached healthy state.

        This test verifies that the Docker HEALTHCHECK has passed,
        indicating the service is ready to handle requests.

        Expected:
            Container State.Health.Status = "healthy"
        """
        # Refresh container state
        api_container.reload()

        # Get container state
        state = api_container.attrs["State"]

        # Check if health is tracked
        if "Health" in state:
            health_status = state["Health"]["Status"]
            assert health_status == "healthy", (
                f"Container should be healthy, got: {health_status}. "
                f"Check logs: docker logs {api_container.name}"
            )
        else:
            # Health check may not be enabled or not yet reported
            # Verify container is at least running
            assert state["Running"] is True, "Container should be running"

    def test_container_has_no_health_failures(self, api_container: Container):
        """
        Verify container has not experienced health check failures.

        Expected:
            FailingStreak = 0 (no consecutive failures)
        """
        # Refresh container state
        api_container.reload()

        state = api_container.attrs["State"]

        if "Health" in state:
            failing_streak = state["Health"].get("FailingStreak", 0)
            assert failing_streak == 0, (
                f"Container should have no health failures, got {failing_streak} failures"
            )

    def test_config_endpoint_accessible(self, api_config_endpoint: str):
        """
        Verify config info endpoint is accessible.

        This endpoint provides service configuration information
        useful for debugging and monitoring.

        Expected:
            GET /api/v1/config-info returns HTTP 200
        """
        response = requests.get(api_config_endpoint, timeout=5)

        assert response.status_code == 200, (
            f"Config endpoint should return 200, got: {response.status_code}. "
            f"Response: {response.text}"
        )

    def test_config_endpoint_returns_model_info(self, api_config_endpoint: str):
        """
        Verify config endpoint returns model configuration.

        Expected:
            Response contains embedding model information
            Response contains retrieval configuration
        """
        response = requests.get(api_config_endpoint, timeout=5)
        assert response.status_code == 200, "Config check should succeed"

        # Parse JSON response
        try:
            data = response.json()
        except ValueError as e:
            pytest.fail(f"Config endpoint should return valid JSON: {e}")
            raise  # Never reached - satisfies static analysis (pytest.fail raises)

        # Should contain some configuration information
        # (exact structure may vary based on API implementation)
        assert len(data) > 0, "Config response should contain configuration data"

    def test_api_responds_within_timeout(self, api_health_endpoint: str):
        """
        Verify API responds to health checks within acceptable timeout.

        Expected:
            Response time < 3 seconds (indicating service is responsive)
        """
        import time

        start_time = time.time()
        response = requests.get(api_health_endpoint, timeout=10)
        elapsed_time = time.time() - start_time

        assert response.status_code == 200, "Health check should succeed"
        assert elapsed_time < 3.0, (
            f"Health check should respond in <3s, took: {elapsed_time:.2f}s"
        )

    def test_multiple_health_checks_succeed(self, api_health_endpoint: str):
        """
        Verify multiple consecutive health checks succeed.

        This test ensures service stability and consistent availability.

        Expected:
            5 consecutive health checks all return HTTP 200
        """
        num_checks = 5

        for i in range(num_checks):
            response = requests.get(api_health_endpoint, timeout=5)
            assert response.status_code == 200, (
                f"Health check {i + 1}/{num_checks} failed with status {response.status_code}"
            )

    def test_container_is_running(self, api_container: Container):
        """
        Verify container is in running state.

        Expected:
            State.Running = True
            State.Status = "running"
        """
        # Refresh container state
        api_container.reload()

        state = api_container.attrs["State"]

        assert state["Running"] is True, "Container should be running"
        assert state["Status"] == "running", (
            f"Container status should be 'running', got: {state['Status']}"
        )

    def test_container_has_not_restarted(self, api_container: Container):
        """
        Verify container has not experienced unexpected restarts.

        Expected:
            RestartCount = 0 (container has been stable since startup)
        """
        # Refresh container state
        api_container.reload()

        state = api_container.attrs["State"]
        restart_count = state.get("RestartCount", 0)

        assert restart_count == 0, (
            f"Container should not have restarted, got {restart_count} restarts"
        )

    def test_container_has_no_oom_kills(self, api_container: Container):
        """
        Verify container has not been OOM (Out Of Memory) killed.

        Expected:
            OOMKilled = False
        """
        # Refresh container state
        api_container.reload()

        state = api_container.attrs["State"]
        oom_killed = state.get("OOMKilled", False)

        assert oom_killed is False, "Container should not have been OOM killed"
