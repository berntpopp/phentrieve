"""Pydantic request/response models for the auth endpoints."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator

# bcrypt rejects (bcrypt>=5) or truncates inputs beyond 72 bytes, so cap the
# password there. ``max_length`` counts characters; the byte cap is enforced in
# ``_validate_password_strength`` because non-ASCII characters take 2-4 bytes.
_MIN_PASSWORD = 10
_MAX_PASSWORD = 72
_MAX_PASSWORD_BYTES = 72


def _validate_password_strength(value: str) -> str:
    if len(value.encode("utf-8")) > _MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password is too long: at most {_MAX_PASSWORD_BYTES} bytes when "
            "UTF-8 encoded (accented and other non-ASCII characters count as "
            "2-4 bytes each)."
        )
    if not any(c.isalpha() for c in value):
        raise ValueError("Password must contain at least one letter.")
    if not any(c.isdigit() for c in value):
        raise ValueError("Password must contain at least one digit.")
    return value


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=_MIN_PASSWORD, max_length=_MAX_PASSWORD)

    @field_validator("password")
    @classmethod
    def _strength(cls, value: str) -> str:
        return _validate_password_strength(value)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenRequest(BaseModel):
    token: str = Field(min_length=1)


class EmailRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str = Field(min_length=1)
    new_password: str = Field(min_length=_MIN_PASSWORD, max_length=_MAX_PASSWORD)

    @field_validator("new_password")
    @classmethod
    def _strength(cls, value: str) -> str:
        return _validate_password_strength(value)


class UserOut(BaseModel):
    email: EmailStr
    is_verified: bool


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth2 token type label, not a secret
    user: UserOut


class MessageResponse(BaseModel):
    message: str
