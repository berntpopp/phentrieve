"""Characterization tests for api/main.py.

Tests app creation, router mounting, CORS config, and root endpoint.
Must pass before AND after factory/lifespan refactoring.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# API import path workaround (see tests/unit/api/README.md)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Create test client with lifespan context manager.

    Using context manager ensures lifespan startup/shutdown events fire,
    per FastAPI testing docs: https://fastapi.tiangolo.com/advanced/testing-events/
    """
    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


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


class TestAppVersion:
    """Tests for version information in the API."""

    def test_root_version_is_string(self, client):
        """Version field should be a non-empty string."""
        response = client.get("/")
        data = response.json()
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_root_has_description(self, client):
        """Root endpoint should include a description."""
        response = client.get("/")
        data = response.json()
        assert "description" in data
        assert "HPO" in data["description"]
