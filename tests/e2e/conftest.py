"""
Pytest fixtures for Docker E2E tests.

This module provides session-scoped fixtures for Docker container lifecycle
management using pytest-docker plugin. Containers are started once per test
session and reused across all E2E tests for performance.

Fixtures:
    - docker_compose_file: Path to docker-compose.test.yml
    - docker_compose_project_name: Fixed project name for test isolation
    - docker_setup: Container lifecycle commands
    - api_service: API base URL with health check wait
    - api_container: Docker container object for inspection
"""

import os

import pytest
import requests
from docker import DockerClient
from docker.models.containers import Container


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig) -> str:
    """
    Specify the docker-compose file for E2E tests.

    Returns:
        str: Absolute path to docker-compose.test.yml
    """
    return os.path.join(str(pytestconfig.rootdir), "docker-compose.test.yml")


@pytest.fixture(scope="session")
def docker_compose_project_name() -> str:
    """
    Pin Docker Compose project name for test isolation.

    Using a fixed project name ensures:
    - Predictable container names for inspection
    - No conflicts with development/production stacks
    - Easy cleanup with `docker compose -p phentrieve_e2e_test down -v`

    Returns:
        str: Fixed project name for E2E tests
    """
    return "phentrieve_e2e_test"


@pytest.fixture(scope="session")
def docker_setup() -> list[str]:
    """
    Define Docker stack lifecycle for pytest-docker.

    Commands executed in order:
    1. down -v: Stop and remove containers, networks, volumes
    2. up --build -d: Build images and start containers in detached mode

    Returns:
        list[str]: Docker Compose commands for setup
    """
    return ["down -v", "up --build -d"]


def is_api_responsive(url: str, timeout: int = 1) -> bool:
    """
    Check if API health endpoint is responding.

    Args:
        url: API health endpoint URL
        timeout: Request timeout in seconds

    Returns:
        bool: True if API returns 200 status, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def api_service(docker_ip: str, docker_services) -> str:
    """
    Ensure API service is up and responsive, return base URL.

    This fixture uses pytest-docker's wait_until_responsive to poll the
    health endpoint until the API is ready or timeout is reached.

    The health check accounts for:
    - Container startup time
    - Model loading time (sentence transformers, embeddings)
    - ChromaDB initialization
    - FastAPI application startup

    Args:
        docker_ip: Docker host IP from pytest-docker
        docker_services: Service manager from pytest-docker

    Returns:
        str: API base URL (e.g., "http://localhost:8001")

    Raises:
        TimeoutError: If API doesn't become healthy within 180 seconds
    """
    # Port from docker-compose.test.yml (host:container = 8001:8000)
    port = docker_services.port_for("phentrieve_api_test", 8000)
    api_url = f"http://{docker_ip}:{port}"
    health_url = f"{api_url}/api/v1/health"

    # Wait up to 180 seconds for API to become healthy
    # (matches start_period in docker-compose.test.yml)
    docker_services.wait_until_responsive(
        timeout=180.0, pause=2.0, check=lambda: is_api_responsive(health_url)
    )

    return api_url


@pytest.fixture(scope="session")
def api_container(
    docker_client: DockerClient, docker_compose_project_name: str
) -> Container:
    """
    Get API container object for inspection and security tests.

    This fixture provides access to the Docker container object for:
    - Security validation (user, capabilities, read-only FS)
    - Resource limit verification
    - Log inspection
    - Process inspection

    Args:
        docker_client: Docker client from pytest-docker
        docker_compose_project_name: Project name for container lookup

    Returns:
        Container: Docker container object for phentrieve_api_test service

    Raises:
        AssertionError: If API container not found
    """
    # Container name format: {project_name}_{service_name}_{replica_number}
    # Example: phentrieve_e2e_test_phentrieve_api_test_1
    container_name = f"{docker_compose_project_name}_phentrieve_api_test_1"

    # Alternative: phentrieve_e2e_test-phentrieve_api_test-1 (with hyphens)
    # Docker Compose V2 uses hyphens instead of underscores
    container_name_v2 = f"{docker_compose_project_name}-phentrieve_api_test-1"

    # Try both naming conventions (v1 and v2)
    for name in [container_name_v2, container_name]:
        try:
            container = docker_client.containers.get(name)
            return container
        except Exception:
            continue

    # If both fail, list all containers for debugging
    all_containers = docker_client.containers.list()
    container_names = [c.name for c in all_containers]

    raise AssertionError(
        f"API container not found. Expected '{container_name_v2}' or '{container_name}'. "
        f"Available containers: {container_names}"
    )


@pytest.fixture(scope="session")
def api_health_endpoint(api_service: str) -> str:
    """
    Convenience fixture for API health endpoint URL.

    Args:
        api_service: API base URL

    Returns:
        str: Health endpoint URL
    """
    return f"{api_service}/api/v1/health"


@pytest.fixture(scope="session")
def api_config_endpoint(api_service: str) -> str:
    """
    Convenience fixture for API config endpoint URL.

    Args:
        api_service: API base URL

    Returns:
        str: Config info endpoint URL
    """
    return f"{api_service}/api/v1/config-info"


@pytest.fixture(scope="session")
def api_query_endpoint(api_service: str) -> str:
    """
    Convenience fixture for API query endpoint URL.

    Args:
        api_service: API base URL

    Returns:
        str: Query endpoint URL
    """
    return f"{api_service}/api/v1/query"
