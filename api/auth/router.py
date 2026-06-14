"""Auth endpoints: register, verify, login, refresh, logout, me, password reset.

Token strategy: a short-lived access JWT (returned in the body, held in memory
by the SPA) plus a rotating opaque refresh token in an HttpOnly cookie. The
cookie-bearing endpoints (``/refresh``, ``/logout``) require a double-submit
CSRF token. Register and password-reset responses are intentionally generic to
avoid account enumeration.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response, status

import api.config as api_config
from api.auth import tokens
from api.auth.deps import get_current_user, get_user_store
from api.auth.email import build_reset_email, build_verify_email, get_email_sender
from api.auth.passwords import hash_password, verify_password
from api.auth.schemas import (
    EmailRequest,
    LoginRequest,
    LoginResponse,
    MessageResponse,
    PasswordResetConfirm,
    RegisterRequest,
    TokenRequest,
    UserOut,
)
from api.auth.store import EmailExistsError, User, UserStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

REFRESH_COOKIE = "refresh_token"
CSRF_COOKIE = "csrf_token"
CSRF_HEADER = "x-csrf-token"
AUTH_COOKIE_PATH = "/api/v1/auth"

_VERIFY_TTL = timedelta(hours=24)
_RESET_TTL = timedelta(hours=1)

# Generic, non-enumerating messages.
_REGISTER_MSG = "If the address is valid, a verification email has been sent."
_RESET_MSG = "If the address is valid, a password reset email has been sent."
_VERIFY_OK = "Email verified. You now have the higher daily quota."
_RESET_OK = "Password updated. Please sign in with your new password."


def _samesite() -> Literal["lax", "strict", "none"]:
    value = api_config.PHENTRIEVE_AUTH_COOKIE_SAMESITE
    if value in {"lax", "strict", "none"}:
        return value  # type: ignore[return-value]
    return "lax"


def _set_session_cookies(
    response: Response, *, refresh_raw: str, csrf_raw: str
) -> None:
    secure = api_config.PHENTRIEVE_AUTH_COOKIE_SECURE
    samesite = _samesite()
    max_age = api_config.PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_raw,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        path=AUTH_COOKIE_PATH,
    )
    # CSRF cookie is readable by JS (double-submit pattern), not HttpOnly.
    response.set_cookie(
        CSRF_COOKIE,
        csrf_raw,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite=samesite,
        path=AUTH_COOKIE_PATH,
    )


def _clear_session_cookies(response: Response) -> None:
    response.delete_cookie(REFRESH_COOKIE, path=AUTH_COOKIE_PATH)
    response.delete_cookie(CSRF_COOKIE, path=AUTH_COOKIE_PATH)


def _require_csrf(request: Request) -> None:
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if not cookie or not header or cookie != header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed."
        )


def _issue_session(response: Response, store: UserStore, user: User) -> str:
    """Create a refresh session + CSRF token, set cookies, return access JWT."""
    refresh_raw = tokens.generate_refresh_token()
    csrf_raw = tokens.generate_csrf_token()
    expires_at = datetime.now(UTC) + timedelta(
        seconds=api_config.PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS
    )
    store.create_refresh_session(
        tokens.hash_token(refresh_raw), user_id=user.id, expires_at=expires_at
    )
    _set_session_cookies(response, refresh_raw=refresh_raw, csrf_raw=csrf_raw)
    return tokens.create_access_token(
        user_id=user.id,
        secret=api_config.PHENTRIEVE_AUTH_JWT_SECRET,
        ttl_seconds=api_config.PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS,
    )


async def _send_verify(store: UserStore, user: User) -> None:
    raw = tokens.generate_refresh_token()
    store.put_token(
        tokens.hash_token(raw),
        purpose="verify",
        user_id=user.id,
        expires_at=datetime.now(UTC) + _VERIFY_TTL,
    )
    link = f"{api_config.PHENTRIEVE_PUBLIC_BASE_URL}/verify?token={raw}"
    subject, text = build_verify_email(link)
    await get_email_sender().send(to=user.email, subject=subject, text=text)


@router.post("/register", response_model=MessageResponse, status_code=201)
async def register(body: RegisterRequest) -> MessageResponse:
    store = get_user_store()
    existing = store.get_by_email(body.email)
    if existing is not None:
        # No enumeration: behave identically. Resend verification if pending.
        if not existing.is_verified:
            await _send_verify(store, existing)
        return MessageResponse(message=_REGISTER_MSG)
    try:
        user = store.create_user(
            email=body.email, password_hash=hash_password(body.password)
        )
    except EmailExistsError:
        return MessageResponse(message=_REGISTER_MSG)
    await _send_verify(store, user)
    return MessageResponse(message=_REGISTER_MSG)


@router.post("/verify", response_model=MessageResponse)
async def verify(body: TokenRequest) -> MessageResponse:
    store = get_user_store()
    user_id = store.consume_token(tokens.hash_token(body.token), purpose="verify")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )
    store.mark_verified(user_id)
    return MessageResponse(message=_VERIFY_OK)


@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification(body: EmailRequest) -> MessageResponse:
    store = get_user_store()
    user = store.get_by_email(body.email)
    if user is not None and not user.is_verified:
        await _send_verify(store, user)
    return MessageResponse(message=_REGISTER_MSG)


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response) -> LoginResponse:
    store = get_user_store()
    user = store.get_by_email(body.email)
    if user is not None and store.is_locked(user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed attempts. Please try again later.",
        )
    if user is None or not verify_password(body.password, user.password_hash):
        if user is not None:
            store.record_failed_login(
                user.id,
                max_attempts=api_config.PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS,
                lockout_seconds=api_config.PHENTRIEVE_AUTH_LOCKOUT_SECONDS,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    store.reset_failed_login(user.id)
    access_token = _issue_session(response, store, user)
    return LoginResponse(
        access_token=access_token,
        user=UserOut(email=user.email, is_verified=user.is_verified),
    )


@router.post("/refresh", response_model=LoginResponse)
async def refresh(request: Request, response: Response) -> LoginResponse:
    _require_csrf(request)
    store = get_user_store()
    raw = request.cookies.get(REFRESH_COOKIE)
    user_id = store.get_active_refresh_user(tokens.hash_token(raw)) if raw else None
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
        )
    user = store.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown user."
        )
    # Rotate: revoke the presented token, issue a fresh session.
    store.revoke_refresh_session(tokens.hash_token(raw))  # type: ignore[arg-type]
    access_token = _issue_session(response, store, user)
    return LoginResponse(
        access_token=access_token,
        user=UserOut(email=user.email, is_verified=user.is_verified),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, response: Response) -> MessageResponse:
    _require_csrf(request)
    store = get_user_store()
    raw = request.cookies.get(REFRESH_COOKIE)
    if raw:
        store.revoke_refresh_session(tokens.hash_token(raw))
    _clear_session_cookies(response)
    return MessageResponse(message="Signed out.")


@router.get("/me", response_model=UserOut)
async def me(request: Request) -> UserOut:
    user = get_current_user(request)
    return UserOut(email=user.email, is_verified=user.is_verified)


@router.post("/password-reset/request", response_model=MessageResponse)
async def password_reset_request(body: EmailRequest) -> MessageResponse:
    store = get_user_store()
    user = store.get_by_email(body.email)
    if user is not None:
        raw = tokens.generate_refresh_token()
        store.put_token(
            tokens.hash_token(raw),
            purpose="reset",
            user_id=user.id,
            expires_at=datetime.now(UTC) + _RESET_TTL,
        )
        link = f"{api_config.PHENTRIEVE_PUBLIC_BASE_URL}/reset-password?token={raw}"
        subject, text = build_reset_email(link)
        await get_email_sender().send(to=user.email, subject=subject, text=text)
    return MessageResponse(message=_RESET_MSG)


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def password_reset_confirm(body: PasswordResetConfirm) -> MessageResponse:
    store = get_user_store()
    user_id = store.consume_token(tokens.hash_token(body.token), purpose="reset")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token.",
        )
    store.update_password(user_id, hash_password(body.new_password))
    store.revoke_all_for_user(user_id)
    return MessageResponse(message=_RESET_OK)
