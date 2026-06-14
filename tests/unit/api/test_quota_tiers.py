"""Tests for auth-aware LLM quota tiers and the quota status endpoint."""

import pytest
from fastapi.testclient import TestClient

import api.config as api_config
from api.auth import tokens
from api.auth.store import UserStore
from api.main import create_app

pytestmark = pytest.mark.unit

_SECRET = "q" * 48


@pytest.fixture
def quota_env(tmp_path, monkeypatch):
    """Enable auth + quota enforcement against temp SQLite DBs."""
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_JWT_SECRET", _SECRET)
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_AUTH_DB_PATH", str(tmp_path / "users.db")
    )
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_LLM_QUOTA_DB_PATH", str(tmp_path / "quota.db")
    )
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_QUOTA_ENFORCE", "true")
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_DAILY_LIMIT", 5)
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT", 10)
    return tmp_path


@pytest.fixture
def client(quota_env):
    # Provide a real client IP so the anonymous (IP-keyed) tier can resolve a
    # trusted subject (TestClient defaults to the non-IP host "testclient").
    return TestClient(create_app(), client=("127.0.0.1", 5555))


def _make_user(tmp_path, *, verified: bool) -> str:
    """Create a user directly in the store, return a valid access token."""
    store = UserStore(api_config.PHENTRIEVE_AUTH_DB_PATH)
    user = store.create_user(email="u@ex.com", password_hash="x")
    if verified:
        store.mark_verified(user.id)
    return tokens.create_access_token(user_id=user.id, secret=_SECRET, ttl_seconds=600)


def test_quota_status_anonymous(client):
    r = client.get("/api/v1/text/quota")
    assert r.status_code == 200
    body = r.json()
    assert body["quota_limit"] == 5
    assert body["authenticated"] is False
    assert body["verified"] is False
    assert body["enforced"] is True


def test_quota_status_verified_user_gets_higher_limit(client, quota_env):
    token = _make_user(quota_env, verified=True)
    r = client.get("/api/v1/text/quota", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["quota_limit"] == 10
    assert body["authenticated"] is True
    assert body["verified"] is True


def test_quota_status_unverified_user_stays_anonymous_limit(client, quota_env):
    token = _make_user(quota_env, verified=False)
    r = client.get("/api/v1/text/quota", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["quota_limit"] == 5
    assert body["verified"] is False


def test_quota_status_does_not_consume(client):
    first = client.get("/api/v1/text/quota").json()
    second = client.get("/api/v1/text/quota").json()
    assert first["quota_used"] == 0
    assert second["quota_used"] == 0


def test_quota_enforce_override_false(client, monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_LLM_QUOTA_ENFORCE", "false")
    r = client.get("/api/v1/text/quota")
    assert r.json()["enforced"] is False
