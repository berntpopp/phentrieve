"""Tests for JWT access tokens and opaque refresh/CSRF token helpers."""

import pytest

from api.auth import tokens

_SECRET = "x" * 48
_OTHER_SECRET = "y" * 48


def test_access_token_roundtrip():
    tok = tokens.create_access_token(user_id=42, secret=_SECRET, ttl_seconds=60)
    claims = tokens.decode_access_token(tok, secret=_SECRET)
    assert claims["sub"] == "42"
    assert claims["typ"] == "access"


def test_access_token_expired():
    tok = tokens.create_access_token(user_id=1, secret=_SECRET, ttl_seconds=-1)
    with pytest.raises(tokens.TokenError):
        tokens.decode_access_token(tok, secret=_SECRET)


def test_access_token_bad_secret():
    tok = tokens.create_access_token(user_id=1, secret=_SECRET, ttl_seconds=60)
    with pytest.raises(tokens.TokenError):
        tokens.decode_access_token(tok, secret=_OTHER_SECRET)


def test_access_token_rejects_garbage():
    with pytest.raises(tokens.TokenError):
        tokens.decode_access_token("not.a.jwt", secret=_SECRET)


def test_refresh_token_hash_is_stable_and_opaque():
    raw = tokens.generate_refresh_token()
    assert len(raw) >= 32
    assert tokens.hash_token(raw) == tokens.hash_token(raw)
    assert tokens.hash_token(raw) != raw


def test_distinct_refresh_tokens():
    assert tokens.generate_refresh_token() != tokens.generate_refresh_token()
    assert tokens.generate_csrf_token() != tokens.generate_csrf_token()
