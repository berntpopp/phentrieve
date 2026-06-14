"""Integration tests for the auth router (register/verify/login/refresh/reset)."""

import logging
import re
import smtplib

import pytest
from fastapi.testclient import TestClient

import api.config as api_config
from api.main import create_app

pytestmark = pytest.mark.unit

_PASSWORD = "Sup3rSecret9"


@pytest.fixture
def auth_client(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_JWT_SECRET", "t" * 48)
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_AUTH_DB_PATH", str(tmp_path / "users.db")
    )
    monkeypatch.setattr(api_config, "PHENTRIEVE_EMAIL_BACKEND", "console")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_COOKIE_SECURE", False)
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_COOKIE_SAMESITE", "lax")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS", 3)
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_PUBLIC_BASE_URL", "http://localhost:5734"
    )
    caplog.set_level(logging.INFO, logger="api.auth.email")
    return TestClient(create_app())


def _link_token(caplog, kind: str) -> str:
    match = re.search(rf"/{kind}\?token=(\S+)", caplog.text)
    assert match is not None, f"no {kind} link logged"
    return match.group(1)


def _register(client, caplog, email="a@ex.com"):
    r = client.post(
        "/api/v1/auth/register", json={"email": email, "password": _PASSWORD}
    )
    assert r.status_code == 201
    return _link_token(caplog, "verify")


def test_register_verify_login_me_flow(auth_client, caplog):
    token = _register(auth_client, caplog)

    assert (
        auth_client.post("/api/v1/auth/verify", json={"token": token}).status_code
        == 200
    )

    r = auth_client.post(
        "/api/v1/auth/login", json={"email": "a@ex.com", "password": _PASSWORD}
    )
    assert r.status_code == 200
    access = r.json()["access_token"]
    assert r.json()["user"]["is_verified"] is True
    assert auth_client.cookies.get("refresh_token")

    me = auth_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access}"}
    )
    assert me.status_code == 200
    assert me.json() == {"email": "a@ex.com", "is_verified": True}


def test_me_requires_token(auth_client):
    assert auth_client.get("/api/v1/auth/me").status_code == 401


def test_verify_invalid_token(auth_client):
    assert (
        auth_client.post("/api/v1/auth/verify", json={"token": "bogus"}).status_code
        == 400
    )


def test_login_wrong_password_then_lockout(auth_client, caplog):
    _register(auth_client, caplog)
    for _ in range(3):
        r = auth_client.post(
            "/api/v1/auth/login", json={"email": "a@ex.com", "password": "wrongpass1"}
        )
        assert r.status_code == 401
    # Account now locked: even the correct password is rejected with 429.
    r = auth_client.post(
        "/api/v1/auth/login", json={"email": "a@ex.com", "password": _PASSWORD}
    )
    assert r.status_code == 429


def test_no_user_enumeration_on_register_existing(auth_client, caplog):
    _register(auth_client, caplog)
    r = auth_client.post(
        "/api/v1/auth/register", json={"email": "a@ex.com", "password": _PASSWORD}
    )
    assert r.status_code == 201  # same generic response


def test_refresh_rotates_and_logout_revokes(auth_client, caplog):
    token = _register(auth_client, caplog)
    auth_client.post("/api/v1/auth/verify", json={"token": token})
    auth_client.post(
        "/api/v1/auth/login", json={"email": "a@ex.com", "password": _PASSWORD}
    )
    csrf = auth_client.cookies.get("csrf_token")

    # Refresh without CSRF header -> 403
    assert auth_client.post("/api/v1/auth/refresh").status_code == 403

    # Refresh with CSRF header -> new access token
    r = auth_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf})
    assert r.status_code == 200
    assert r.json()["access_token"]

    # Logout revokes; subsequent refresh fails
    csrf2 = auth_client.cookies.get("csrf_token")
    assert (
        auth_client.post(
            "/api/v1/auth/logout", headers={"X-CSRF-Token": csrf2}
        ).status_code
        == 200
    )
    csrf3 = auth_client.cookies.get("csrf_token")
    r = auth_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf3 or "x"})
    assert r.status_code in (401, 403)


def test_password_reset_flow(auth_client, caplog):
    token = _register(auth_client, caplog)
    auth_client.post("/api/v1/auth/verify", json={"token": token})

    r = auth_client.post(
        "/api/v1/auth/password-reset/request", json={"email": "a@ex.com"}
    )
    assert r.status_code == 200
    reset_token = _link_token(caplog, "reset-password")

    new_password = "Br4ndNewPass"
    r = auth_client.post(
        "/api/v1/auth/password-reset/confirm",
        json={"token": reset_token, "new_password": new_password},
    )
    assert r.status_code == 200

    # Old password fails, new password works.
    assert (
        auth_client.post(
            "/api/v1/auth/login", json={"email": "a@ex.com", "password": _PASSWORD}
        ).status_code
        == 401
    )
    assert (
        auth_client.post(
            "/api/v1/auth/login",
            json={"email": "a@ex.com", "password": new_password},
        ).status_code
        == 200
    )


def test_weak_password_rejected(auth_client):
    r = auth_client.post(
        "/api/v1/auth/register", json={"email": "b@ex.com", "password": "short"}
    )
    assert r.status_code == 422


def test_register_survives_email_send_failure(auth_client, monkeypatch, caplog):
    """A delivery failure must not turn registration into a 500.

    Account endpoints are non-enumerating: a refused recipient / SMTP outage
    must be logged for operators but still return the generic 201, otherwise the
    SPA shows a hard error and the verification email is never (re)sendable.
    """
    import api.auth.router as auth_router

    class _BoomSender:
        async def send(self, *, to, subject, text):
            raise smtplib.SMTPRecipientsRefused({to: (550, b"blocked (MBL-R)")})

    monkeypatch.setattr(auth_router, "get_email_sender", lambda: _BoomSender())
    caplog.set_level(logging.ERROR, logger="api.auth.router")

    r = auth_client.post(
        "/api/v1/auth/register",
        json={"email": "boom@ex.com", "password": _PASSWORD},
    )
    assert r.status_code == 201
    # Operators still get a signal in the logs.
    assert "boom@ex.com" in caplog.text


def test_password_reset_survives_email_send_failure(auth_client, monkeypatch, caplog):
    """Password-reset request must also tolerate a delivery failure."""
    import api.auth.router as auth_router

    _register(auth_client, caplog)

    class _BoomSender:
        async def send(self, *, to, subject, text):
            raise smtplib.SMTPException("smtp down")

    monkeypatch.setattr(auth_router, "get_email_sender", lambda: _BoomSender())
    caplog.set_level(logging.ERROR, logger="api.auth.router")

    r = auth_client.post(
        "/api/v1/auth/password-reset/request", json={"email": "a@ex.com"}
    )
    assert r.status_code == 200
    assert "a@ex.com" in caplog.text
