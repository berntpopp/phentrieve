"""Token helpers: short-lived JWT access tokens + opaque refresh tokens.

Access tokens are stateless JWTs (HS256) carried by the SPA in memory and sent
as ``Authorization: Bearer``. Refresh tokens are opaque random strings stored
server-side as SHA-256 hashes (in ``refresh_sessions``) and delivered to the
client as an HttpOnly cookie. CSRF tokens use the same opaque generation.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

_ALGO = "HS256"


class TokenError(Exception):
    """Raised when an access token is invalid, expired, or of the wrong type."""


def create_access_token(*, user_id: int, secret: str, ttl_seconds: int) -> str:
    """Return a signed access JWT for ``user_id`` valid for ``ttl_seconds``."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "typ": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    token: str = jwt.encode(payload, secret, algorithm=_ALGO)
    return token


def decode_access_token(token: str, *, secret: str) -> dict[str, Any]:
    """Decode and validate an access JWT, raising ``TokenError`` on failure."""
    try:
        claims: dict[str, Any] = jwt.decode(token, secret, algorithms=[_ALGO])
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    if claims.get("typ") != "access":
        raise TokenError("wrong token type")
    return claims


def generate_refresh_token() -> str:
    """Return a new opaque refresh token (URL-safe, high entropy)."""
    return secrets.token_urlsafe(48)


def hash_token(raw: str) -> str:
    """Return the SHA-256 hex digest used to store a token server-side."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    """Return a new opaque CSRF token for the double-submit cookie pattern."""
    return secrets.token_urlsafe(24)
