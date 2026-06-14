"""FastAPI dependencies for authentication.

``get_optional_user`` resolves the bearer access token if present and returns
the user, or ``None`` (used by the quota path to pick the tier).
``get_current_user`` is the same but raises 401 when no valid user is present.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request, status

import api.config as api_config
from api.auth import tokens
from api.auth.store import User, UserStore


def get_user_store() -> UserStore:
    """Return a UserStore bound to the configured DB path.

    Constructed per call (sqlite connections are per-call anyway); schema
    creation is idempotent. Reading the path at call time keeps tests able to
    repoint the DB via config monkeypatching.
    """
    return UserStore(Path(api_config.PHENTRIEVE_AUTH_DB_PATH))


def _user_from_bearer(request: Request) -> User | None:
    auth_header = request.headers.get("authorization") or ""
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    secret = api_config.PHENTRIEVE_AUTH_JWT_SECRET
    if not secret:
        return None
    try:
        claims = tokens.decode_access_token(token.strip(), secret=secret)
        user_id = int(claims["sub"])
    except (tokens.TokenError, KeyError, ValueError):
        return None
    return get_user_store().get_by_id(user_id)


def get_optional_user(request: Request) -> User | None:
    """Return the authenticated user, or None if unauthenticated/invalid."""
    return _user_from_bearer(request)


def get_current_user(request: Request) -> User:
    """Return the authenticated user or raise 401."""
    user = _user_from_bearer(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
