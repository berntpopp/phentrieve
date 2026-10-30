"""Health routes: one canonical route plus thin aliases (#433)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import health
from phentrieve.config import DEFAULT_MODEL

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


def test_ready_route_stays_503_until_the_default_model_is_loaded(monkeypatch):
    monkeypatch.setattr(health, "LOADED_SBERT_MODELS", {})
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "loading"
    assert response.headers["Retry-After"] == "30"
    # Liveness does not depend on the model.
    assert client.get("/api/v1/health").status_code == 200

    monkeypatch.setattr(health, "LOADED_SBERT_MODELS", {DEFAULT_MODEL: object()})
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_only_the_canonical_health_route_is_documented():
    paths = app.openapi()["paths"]
    health_paths = sorted(p for p in paths if p.endswith("/health") or "/health/" in p)
    assert health_paths == ["/api/v1/health"]
