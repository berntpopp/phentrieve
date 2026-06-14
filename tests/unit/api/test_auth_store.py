"""Tests for the SQLite-backed UserStore."""

from datetime import UTC, datetime, timedelta

import pytest

from api.auth.store import EmailExistsError, UserStore


@pytest.fixture
def store(tmp_path):
    return UserStore(tmp_path / "users.db")


def test_create_and_get_normalizes_email(store):
    u = store.create_user(email="A@Ex.com", password_hash="h")
    assert u.id > 0
    assert u.email == "a@ex.com"
    assert u.is_verified is False
    got = store.get_by_email("  a@ex.com ")
    assert got is not None and got.id == u.id
    assert store.get_by_id(u.id).email == "a@ex.com"


def test_duplicate_email_rejected(store):
    store.create_user(email="a@ex.com", password_hash="h")
    with pytest.raises(EmailExistsError):
        store.create_user(email="A@EX.COM", password_hash="h2")


def test_get_missing_returns_none(store):
    assert store.get_by_email("nobody@ex.com") is None
    assert store.get_by_id(999) is None


def test_mark_verified_and_update_password(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    store.mark_verified(u.id)
    assert store.get_by_id(u.id).is_verified is True
    store.update_password(u.id, "h2")
    assert store.get_by_id(u.id).password_hash == "h2"


def test_token_single_use_and_purpose(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    exp = datetime.now(UTC) + timedelta(hours=1)
    store.put_token("tokhash", purpose="verify", user_id=u.id, expires_at=exp)
    # wrong purpose -> None
    assert store.consume_token("tokhash", purpose="reset") is None
    # correct purpose -> user id, then single-use
    assert store.consume_token("tokhash", purpose="verify") == u.id
    assert store.consume_token("tokhash", purpose="verify") is None


def test_expired_token_not_consumable(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    past = datetime.now(UTC) - timedelta(seconds=1)
    store.put_token("old", purpose="verify", user_id=u.id, expires_at=past)
    assert store.consume_token("old", purpose="verify") is None


def test_lockout_then_reset(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    for _ in range(5):
        store.record_failed_login(u.id, max_attempts=5, lockout_seconds=900)
    assert store.is_locked(store.get_by_id(u.id)) is True
    store.reset_failed_login(u.id)
    refreshed = store.get_by_id(u.id)
    assert store.is_locked(refreshed) is False
    assert refreshed.failed_attempts == 0


def test_refresh_session_rotation_and_revoke(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    exp = datetime.now(UTC) + timedelta(days=1)
    store.create_refresh_session("rh1", user_id=u.id, expires_at=exp)
    assert store.get_active_refresh_user("rh1") == u.id
    store.revoke_refresh_session("rh1")
    assert store.get_active_refresh_user("rh1") is None


def test_revoke_all_for_user(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    exp = datetime.now(UTC) + timedelta(days=1)
    store.create_refresh_session("r1", user_id=u.id, expires_at=exp)
    store.create_refresh_session("r2", user_id=u.id, expires_at=exp)
    store.revoke_all_for_user(u.id)
    assert store.get_active_refresh_user("r1") is None
    assert store.get_active_refresh_user("r2") is None


def test_expired_refresh_session_inactive(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    past = datetime.now(UTC) - timedelta(seconds=1)
    store.create_refresh_session("old", user_id=u.id, expires_at=past)
    assert store.get_active_refresh_user("old") is None
