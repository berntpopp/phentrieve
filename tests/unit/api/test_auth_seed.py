"""Tests for the optional dev-account seeder."""

import api.config as api_config
from api.auth.passwords import verify_password
from api.auth.seed import seed_user, seed_user_from_config
from api.auth.store import UserStore


def test_seed_user_creates_verified_account(tmp_path):
    store = UserStore(tmp_path / "users.db")
    seed_user(store, email="Dev@Ex.com", password="DevPassw0rd")
    user = store.get_by_email("dev@ex.com")
    assert user is not None
    assert user.is_verified is True
    assert verify_password("DevPassw0rd", user.password_hash)


def test_seed_user_is_idempotent_and_authoritative(tmp_path):
    store = UserStore(tmp_path / "users.db")
    seed_user(store, email="dev@ex.com", password="FirstPassw0rd")
    first = store.get_by_email("dev@ex.com")
    # Re-seed with a new password: same account, password updated.
    seed_user(store, email="dev@ex.com", password="SecondPassw0rd")
    second = store.get_by_email("dev@ex.com")
    assert second.id == first.id
    assert verify_password("SecondPassw0rd", second.password_hash)
    assert not verify_password("FirstPassw0rd", second.password_hash)


def test_seed_from_config_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_ENABLED", False)
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_SEED_EMAIL", "dev@ex.com")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_SEED_PASSWORD", "DevPassw0rd")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_DB_PATH", str(tmp_path / "u.db"))
    seed_user_from_config()
    assert UserStore(tmp_path / "u.db").get_by_email("dev@ex.com") is None


def test_seed_from_config_noop_when_creds_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_SEED_EMAIL", "")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_SEED_PASSWORD", "")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_DB_PATH", str(tmp_path / "u.db"))
    seed_user_from_config()  # should not raise
    assert UserStore(tmp_path / "u.db").get_by_email("dev@ex.com") is None


def test_seed_from_config_creates_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_ENABLED", True)
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_SEED_EMAIL", "dev@ex.com")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_SEED_PASSWORD", "DevPassw0rd")
    monkeypatch.setattr(api_config, "PHENTRIEVE_AUTH_DB_PATH", str(tmp_path / "u.db"))
    seed_user_from_config()
    user = UserStore(tmp_path / "u.db").get_by_email("dev@ex.com")
    assert user is not None and user.is_verified is True
