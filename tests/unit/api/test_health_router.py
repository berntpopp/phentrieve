"""Health routes: one canonical route plus thin aliases (#433)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.unit

client = TestClient(app)

HEALTH_KEYS = {"status", "service", "version", "timestamp"}


def test_canonical_health_route_answers_without_redirect():
    """Docker probes call /api/v1/health; it must not 307 to the slash form."""
    response = client.get("/api/v1/health", follow_redirects=False)

    assert response.status_code == 200
    data = response.json()
    assert set(data) == HEALTH_KEYS
    assert data["status"] == "healthy"
    assert data["service"] == "phentrieve-api"


@pytest.mark.parametrize("alias", ["/api/v1/health/", "/api/v1/system/health"])
def test_health_aliases_return_the_canonical_payload(alias):
    response = client.get(alias, follow_redirects=False)

    assert response.status_code == 200
    data = response.json()
    assert set(data) == HEALTH_KEYS
    assert data["status"] == "healthy"


def test_only_the_canonical_health_route_is_documented():
    paths = app.openapi()["paths"]
    health_paths = sorted(p for p in paths if p.endswith("/health") or "/health/" in p)
    assert health_paths == ["/api/v1/health"]
