"""
Tests for system router endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.unit

client = TestClient(app)


def test_version_endpoint():
    """Test /api/v1/system/version endpoint."""
    response = client.get("/api/v1/system/version")

    assert response.status_code == 200

    data = response.json()
    assert "cli" in data
    assert "api" in data
    assert "environment" in data
    assert "timestamp" in data

    # Verify structure of version objects
    assert "version" in data["cli"]
    assert "name" in data["cli"]
    assert "type" in data["cli"]

    assert "version" in data["api"]
    assert "name" in data["api"]
    assert "type" in data["api"]


def test_health_endpoint():
    """Test /api/v1/system/health endpoint."""
    response = client.get("/api/v1/system/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "service" in data
    assert "timestamp" in data
    assert data["service"] == "phentrieve-api"


def test_version_endpoint_returns_valid_semver():
    """Test that version endpoint returns valid semantic version."""
    response = client.get("/api/v1/system/version")

    assert response.status_code == 200

    data = response.json()
    api_version = data["api"]["version"]

    # Check semantic versioning format (x.y.z)
    assert api_version != "unknown"
    parts = api_version.split(".")
    assert len(parts) >= 2  # At least MAJOR.MINOR


def test_health_endpoint_performance():
    """Test that health endpoint responds quickly."""
    import time

    start = time.time()
    response = client.get("/api/v1/system/health")
    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 1.0  # Should respond in less than 1 second
