# Accounts + Higher Full-Text Quota Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a verified, logged-in user get a 10/day full-text LLM quota (vs the anonymous IP limit), via an unobtrusive top-right login/register control.

**Architecture:** New `api/auth/` package (raw sqlite3, mirroring `api/llm_quota.py`) provides bcrypt password hashing, PyJWT access tokens, rotating opaque refresh sessions in an HttpOnly cookie, and a console/SMTP email sender. The existing IP-keyed quota becomes auth-aware: a verified user's quota is keyed on `user:<id>` with a higher limit. Frontend adds a Pinia auth store, axios refresh-retry interceptor, a top-right `AccountButton` opening an `AuthDialog`, and email-link landing routes.

**Tech Stack:** FastAPI, sqlite3, PyJWT, bcrypt (all already in `uv.lock`), stdlib smtplib; Vue 3 + Vuetify 3 + Pinia + Vue Router + Vue I18n + axios; pytest, Vitest.

**Spec:** `.planning/specs/2026-06-14-auth-and-quota-uplift-design.md`

**Conventions:** modern typing (`str | None`); Ruff format/lint; mypy (py3.11); tests under `tests/`; commit after each green task. Run `make check && make typecheck-fast && make test` periodically; `make ci-local` + `make security-python` before pushing.

---

## Phase A - Backend auth core

### Task A1: Config additions

**Files:**
- Modify: `api/config.py` (add constants + `__all__` entries, after the existing LLM block ~line 243-251)

- [ ] **Step 1:** Add a `_env_int` helper next to `_env_bool`, and append auth/email/quota constants:

```python
def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid int for %s; using default %d", name, default)
        return default


# Auth
PHENTRIEVE_AUTH_ENABLED: bool = _env_bool("PHENTRIEVE_AUTH_ENABLED", False)
PHENTRIEVE_AUTH_JWT_SECRET: str = os.getenv("PHENTRIEVE_AUTH_JWT_SECRET", "")
PHENTRIEVE_AUTH_DB_PATH: str = os.getenv(
    "PHENTRIEVE_AUTH_DB_PATH", "../data/app/users.db"
)
PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS: int = _env_int(
    "PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS", 1800
)
PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS: int = _env_int(
    "PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS", 1_209_600
)
PHENTRIEVE_AUTH_COOKIE_SECURE: bool = _env_bool("PHENTRIEVE_AUTH_COOKIE_SECURE", True)
PHENTRIEVE_AUTH_COOKIE_SAMESITE: str = os.getenv(
    "PHENTRIEVE_AUTH_COOKIE_SAMESITE", "lax"
).lower()
PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS: int = _env_int(
    "PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS", 5
)
PHENTRIEVE_AUTH_LOCKOUT_SECONDS: int = _env_int("PHENTRIEVE_AUTH_LOCKOUT_SECONDS", 900)

# Email
PHENTRIEVE_EMAIL_BACKEND: str = os.getenv("PHENTRIEVE_EMAIL_BACKEND", "console").lower()
PHENTRIEVE_EMAIL_FROM: str = os.getenv("PHENTRIEVE_EMAIL_FROM", "noreply@phentrieve.org")
PHENTRIEVE_SMTP_HOST: str = os.getenv("PHENTRIEVE_SMTP_HOST", "")
PHENTRIEVE_SMTP_PORT: int = _env_int("PHENTRIEVE_SMTP_PORT", 587)
PHENTRIEVE_SMTP_USERNAME: str = os.getenv("PHENTRIEVE_SMTP_USERNAME", "")
PHENTRIEVE_SMTP_PASSWORD: str = os.getenv("PHENTRIEVE_SMTP_PASSWORD", "")
PHENTRIEVE_SMTP_TLS: str = os.getenv("PHENTRIEVE_SMTP_TLS", "starttls").lower()
PHENTRIEVE_PUBLIC_BASE_URL: str = os.getenv(
    "PHENTRIEVE_PUBLIC_BASE_URL", "http://localhost:5734"
)

# Quota (authenticated tier + enforce override)
PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT: int = _env_int(
    "PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT", 10
)
# tri-state: "" (unset) -> production-only; "true"/"false" -> explicit
PHENTRIEVE_LLM_QUOTA_ENFORCE: str = os.getenv("PHENTRIEVE_LLM_QUOTA_ENFORCE", "").lower()
```

Add each new public name to `__all__`.

- [ ] **Step 2:** `make typecheck-fast` -> PASS. Commit: `feat(api): add auth/email/quota config`.

---

### Task A2: Password hashing (`api/auth/passwords.py`)

**Files:**
- Create: `api/auth/__init__.py` (empty)
- Create: `api/auth/passwords.py`
- Test: `tests/unit/api/auth/test_passwords.py`

- [ ] **Step 1: Write failing test**

```python
from api.auth.passwords import hash_password, verify_password

def test_hash_and_verify_roundtrip():
    h = hash_password("Sup3r$ecret")
    assert h != "Sup3r$ecret"
    assert verify_password("Sup3r$ecret", h) is True

def test_verify_rejects_wrong_password():
    h = hash_password("Sup3r$ecret")
    assert verify_password("nope", h) is False

def test_verify_handles_garbage_hash():
    assert verify_password("x", "not-a-hash") is False
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/api/auth/test_passwords.py -n 0` -> FAIL (import error). Create `tests/unit/api/auth/__init__.py` if the package layout needs it (mirror existing `tests/unit/api/`).

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

import bcrypt

_ROUNDS = 12


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), password_hash.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        cost = int(password_hash.split("$")[2])
    except (IndexError, ValueError):
        return True
    return cost < _ROUNDS
```

Note: bcrypt truncates at 72 bytes; enforce a max password length in schema validation (Task A6).

- [ ] **Step 4:** Run tests -> PASS. **Step 5:** Commit `feat(api): bcrypt password hashing`.

---

### Task A3: Tokens (`api/auth/tokens.py`)

**Files:**
- Create: `api/auth/tokens.py`
- Test: `tests/unit/api/auth/test_tokens.py`

- [ ] **Step 1: Failing test**

```python
import time
import pytest
from api.auth import tokens

def test_access_token_roundtrip():
    tok = tokens.create_access_token(user_id=42, secret="s3cr3t", ttl_seconds=60)
    claims = tokens.decode_access_token(tok, secret="s3cr3t")
    assert claims["sub"] == "42"
    assert claims["typ"] == "access"

def test_access_token_expired():
    tok = tokens.create_access_token(user_id=1, secret="s", ttl_seconds=-1)
    with pytest.raises(tokens.TokenError):
        tokens.decode_access_token(tok, secret="s")

def test_access_token_bad_secret():
    tok = tokens.create_access_token(user_id=1, secret="a", ttl_seconds=60)
    with pytest.raises(tokens.TokenError):
        tokens.decode_access_token(tok, secret="b")

def test_refresh_token_hash_is_stable_and_opaque():
    raw = tokens.generate_refresh_token()
    assert len(raw) >= 32
    assert tokens.hash_token(raw) == tokens.hash_token(raw)
    assert tokens.hash_token(raw) != raw
```

- [ ] **Step 2:** Run -> FAIL. **Step 3: Implement**

```python
from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

_ALGO = "HS256"


class TokenError(Exception):
    pass


def create_access_token(*, user_id: int, secret: str, ttl_seconds: int) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "typ": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=_ALGO)


def decode_access_token(token: str, *, secret: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(token, secret, algorithms=[_ALGO])
    except jwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc
    if claims.get("typ") != "access":
        raise TokenError("wrong token type")
    return claims


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(24)
```

- [ ] **Step 4:** Tests PASS. **Step 5:** Commit `feat(api): jwt access tokens + opaque refresh helpers`.

---

### Task A4: User store (`api/auth/store.py`)

**Files:**
- Create: `api/auth/store.py`
- Test: `tests/unit/api/auth/test_store.py`

Schema and `UserStore` mirror `DailyQuotaStore` (raw sqlite3, WAL, per-call connection, `closing`). Dataclass `User(id, email, is_verified, ...)`.

- [ ] **Step 1: Failing tests** (use `tmp_path` for the DB):

```python
from datetime import UTC, datetime, timedelta
import pytest
from api.auth.store import UserStore, EmailExistsError

@pytest.fixture
def store(tmp_path):
    return UserStore(tmp_path / "users.db")

def test_create_and_get(store):
    u = store.create_user(email="A@Ex.com", password_hash="h")
    assert u.id > 0
    assert u.email == "a@ex.com"        # normalized lower
    assert u.is_verified is False
    got = store.get_by_email("a@ex.com")
    assert got is not None and got.id == u.id

def test_duplicate_email_rejected(store):
    store.create_user(email="a@ex.com", password_hash="h")
    with pytest.raises(EmailExistsError):
        store.create_user(email="A@EX.COM", password_hash="h2")

def test_verify_flow(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    exp = datetime.now(UTC) + timedelta(hours=1)
    store.put_token("tokhash", purpose="verify", user_id=u.id, expires_at=exp)
    consumed = store.consume_token("tokhash", purpose="verify")
    assert consumed == u.id
    assert store.consume_token("tokhash", purpose="verify") is None  # single use
    store.mark_verified(u.id)
    assert store.get_by_id(u.id).is_verified is True

def test_lockout(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    for _ in range(5):
        store.record_failed_login(u.id, max_attempts=5, lockout_seconds=900)
    assert store.is_locked(store.get_by_id(u.id)) is True
    store.reset_failed_login(u.id)
    assert store.is_locked(store.get_by_id(u.id)) is False

def test_refresh_rotation(store):
    u = store.create_user(email="a@ex.com", password_hash="h")
    exp = datetime.now(UTC) + timedelta(days=1)
    store.create_refresh_session("rh1", user_id=u.id, expires_at=exp)
    assert store.get_active_refresh_user("rh1") == u.id
    store.revoke_refresh_session("rh1")
    assert store.get_active_refresh_user("rh1") is None
```

- [ ] **Step 2:** Run -> FAIL. **Step 3: Implement** `UserStore` with:
  - `CREATE TABLE` statements for `users`, `auth_tokens`, `refresh_sessions` (see spec §3) run in `_ensure_schema()`.
  - `_connect()` identical PRAGMA setup to `DailyQuotaStore`.
  - `create_user` lower-cases email, `INSERT`, catches `sqlite3.IntegrityError` -> `EmailExistsError`.
  - `get_by_email`/`get_by_id` -> `User | None`.
  - `mark_verified`, `update_password` (also bumps `updated_at`).
  - `put_token(token_hash, purpose, user_id, expires_at)`, `consume_token(token_hash, purpose) -> int | None` (atomic `UPDATE ... SET consumed_at WHERE consumed_at IS NULL AND expires_at > now` then return user_id; single-use + expiry enforced in SQL).
  - `record_failed_login(user_id, max_attempts, lockout_seconds)` increments and sets `locked_until` when threshold hit; `reset_failed_login`; `is_locked(user)` compares `locked_until` to now.
  - `create_refresh_session`, `get_active_refresh_user(token_hash)` (not revoked, not expired), `revoke_refresh_session`, `revoke_all_for_user(user_id)`.
  Store times as ISO-8601 UTC strings (consistent with `llm_quota.py`).

- [ ] **Step 4:** Tests PASS. **Step 5:** Commit `feat(api): sqlite UserStore (users, tokens, refresh sessions)`.

---

### Task A5: Email sender (`api/auth/email.py`)

**Files:**
- Create: `api/auth/email.py`
- Test: `tests/unit/api/auth/test_email.py`

- [ ] **Step 1: Failing test**

```python
from api.auth.email import ConsoleEmailSender, build_verify_email, build_reset_email

def test_console_sender_captures(caplog):
    sender = ConsoleEmailSender()
    import asyncio
    asyncio.run(sender.send(to="a@ex.com", subject="Hi", text="body http://x/y"))
    assert "a@ex.com" in caplog.text

def test_build_verify_email_contains_link():
    subject, text = build_verify_email("https://phentrieve.org/verify?token=abc")
    assert "verify?token=abc" in text and subject
```

- [ ] **Step 2:** FAIL. **Step 3: Implement** `EmailSender` Protocol with async `send(to, subject, text)`; `ConsoleEmailSender` logs via module logger (info level, includes recipient + body so dev can copy the link); `SmtpEmailSender` builds an `email.message.EmailMessage` and sends through `smtplib` (`SMTP`+`starttls` or `SMTP_SSL`) inside `await run_in_threadpool(...)`; `get_email_sender()` returns console/smtp per `api_config.PHENTRIEVE_EMAIL_BACKEND`. `build_verify_email`/`build_reset_email` return `(subject, text)` plain-text bodies referencing the link and `noreply@phentrieve.org` sender context.

- [ ] **Step 4:** PASS. **Step 5:** Commit `feat(api): console + smtp email senders`.

---

### Task A6: Auth schemas (`api/auth/schemas.py`)

**Files:**
- Create: `api/auth/schemas.py`

- [ ] **Step 1:** Implement Pydantic models (no test file needed beyond router tests):
  - `RegisterRequest(email: EmailStr, password: constr(min_length=10, max_length=72))`
  - `LoginRequest(email: EmailStr, password: str)`
  - `TokenRequest(token: str)`, `EmailRequest(email: EmailStr)`
  - `PasswordResetConfirm(token: str, new_password: constr(min_length=10, max_length=72))`
  - `UserOut(email: EmailStr, is_verified: bool)`
  - `LoginResponse(access_token: str, token_type="bearer", user: UserOut)`
  - `MessageResponse(message: str)`
  Password policy: min 10, max 72 (bcrypt limit), require at least letters+digits via a validator.
- [ ] **Step 2:** `make typecheck-fast` PASS. Commit `feat(api): auth pydantic schemas`.

---

### Task A7: Auth dependencies (`api/auth/deps.py`)

**Files:**
- Create: `api/auth/deps.py`
- Test: covered via router tests (A8)

- [ ] **Step 1:** Implement:
  - `get_user_store()` -> cached `UserStore(Path(api_config.PHENTRIEVE_AUTH_DB_PATH))` (module-level singleton like a lazy global; fine because sqlite connections are per-call).
  - `_bearer_user(request) -> User | None`: read `Authorization: Bearer`, decode via `tokens.decode_access_token` with `PHENTRIEVE_AUTH_JWT_SECRET`, load user; return None on any failure.
  - `get_optional_user(request) -> User | None` (FastAPI dependency wrapper).
  - `get_current_user(request) -> User` raising `HTTPException(401)` when absent.
- [ ] **Step 2:** Commit with A8 (deps are exercised by router tests).

---

### Task A8: Auth router (`api/auth/router.py`) + mount

**Files:**
- Create: `api/auth/router.py`
- Modify: `api/main.py` (conditionally include router)
- Test: `tests/integration/api/test_auth_router.py` (mirror existing integration test setup; force `PHENTRIEVE_AUTH_ENABLED=true`, `console` email, `tmp_path` DBs, `PHENTRIEVE_AUTH_JWT_SECRET` set, via monkeypatch + `TestClient`)

- [ ] **Step 1: Failing integration test** covering the happy path and guards:

```python
def test_register_verify_login_me_flow(client, captured_emails):
    r = client.post("/api/v1/auth/register",
                    json={"email": "a@ex.com", "password": "Sup3rSecret9"})
    assert r.status_code == 201
    token = extract_token_from_email(captured_emails)      # parse console log
    assert client.post("/api/v1/auth/verify", json={"token": token}).status_code == 200
    r = client.post("/api/v1/auth/login",
                    json={"email": "a@ex.com", "password": "Sup3rSecret9"})
    assert r.status_code == 200
    access = r.json()["access_token"]
    assert "refresh_token" in r.cookies
    me = client.get("/api/v1/auth/me",
                    headers={"Authorization": f"Bearer {access}"})
    assert me.json()["email"] == "a@ex.com" and me.json()["is_verified"] is True

def test_login_wrong_password_then_lockout(client): ...
def test_password_reset_flow(client, captured_emails): ...
def test_no_user_enumeration_on_register_existing(client): ...
def test_refresh_rotates_and_logout_revokes(client): ...
```

- [ ] **Step 2:** FAIL. **Step 3: Implement** the router endpoints from spec §3. Key details:
  - Set refresh cookie: `response.set_cookie("refresh_token", raw, httponly=True, secure=cfg.SECURE, samesite=cfg.SAMESITE, max_age=REFRESH_TTL, path="/api/v1/auth")` and a readable `csrf_token` cookie (not httponly).
  - `/refresh` and `/logout` require the `csrf_token` cookie value to equal the `X-CSRF-Token` header (double-submit); 403 otherwise.
  - `/register` and `/password-reset/request` always return a generic success body (no enumeration); only differ internally.
  - `/login` checks lockout first, verifies password, on failure `record_failed_login`, on success `reset_failed_login` + issue tokens.
  - Token TTLs from config; email links use `PHENTRIEVE_PUBLIC_BASE_URL` + `/verify?token=` / `/reset-password?token=`.
  - In `api/main.py`, after the existing `include_router` calls: `if api_config.PHENTRIEVE_AUTH_ENABLED: application.include_router(auth_router.router)`. If enabled and `PHENTRIEVE_AUTH_JWT_SECRET` is empty, log a clear error and (in production) raise at startup.
- [ ] **Step 4:** Tests PASS. **Step 5:** Commit `feat(api): auth router (register/verify/login/refresh/reset)`.

---

## Phase B - Quota integration

### Task B1: Auth-aware quota subject + tiered limit

**Files:**
- Modify: `api/routers/text_processing_router.py:66-133` and the call sites (~207-263)
- Test: `tests/integration/api/test_quota_tiers.py`

- [ ] **Step 1: Failing test**: with quota enforced, a verified bearer user gets `quota_limit == 10`; anonymous gets `5`; unverified gets `5`.
- [ ] **Step 2:** FAIL. **Step 3: Implement**
  - Add `_resolve_quota_subject(http_request) -> tuple[str, int]`: try `_bearer_user`; if present and `is_verified`, return `(hash_subject_key(f"user:{user.id}"), AUTHENTICATED_DAILY_LIMIT)`; else fall back to the existing IP resolution returning `(hash_subject_key(ip), DAILY_LIMIT)`.
  - `_get_llm_quota_store(limit: int)` takes the resolved limit.
  - `check_llm_quota_or_raise` and `_record_llm_quota_success` use the resolved `(subject_key, limit)`. Keep the 503 "untrusted subject" path only for the anonymous branch.
  - Add `_quota_enforced()` helper honoring `PHENTRIEVE_LLM_QUOTA_ENFORCE` tri-state (override `true`/`false`, else production-only). Replace the `_is_production_environment()` check at line 213.
- [ ] **Step 4:** PASS. **Step 5:** Commit `feat(api): tiered LLM quota for verified users (10/day)`.

### Task B2: Quota status endpoint

**Files:**
- Modify: `api/routers/text_processing_router.py` (add `GET /api/v1/text/quota`)
- Test: extend `test_quota_tiers.py`

- [ ] **Step 1:** Failing test: `GET /api/v1/text/quota` (anon) returns `quota_limit=5, authenticated=false`; with verified bearer returns `quota_limit=10, authenticated=true, verified=true`. Does not increment usage.
- [ ] **Step 2-4:** Implement endpoint that resolves subject + limit, reads `get_status`, returns the dict from spec §4 (plus `quota_reset_at`). PASS.
- [ ] **Step 5:** Commit `feat(api): GET /api/v1/text/quota status endpoint`.

---

## Phase C - Frontend

### Task C1: Auth store (`frontend/src/stores/auth.js`)

**Files:**
- Create: `frontend/src/stores/auth.js`
- Test: `frontend/src/test/stores/auth.test.js`

- [ ] **Step 1: Failing tests** (mirror `disclaimer.test.js`): initial `isAuthenticated=false`; `setSession(token,user)` -> `isAuthenticated=true`, `isVerified` reflects user; `clearSession()` resets; token is NOT in persisted keys (`pick` excludes `accessToken`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement setup store: refs `accessToken` (memory), `user` (persisted via `pick:['user']`), getters `isAuthenticated`/`isVerified`, actions calling the API service (register/login/logout/refresh/fetchMe/verify/requestPasswordReset/confirmPasswordReset/resendVerification) and updating state. Persist key `phentrieve-auth`, `pick: ['user']` only.
- [ ] **Step 4:** PASS. **Step 5:** Commit `feat(frontend): pinia auth store`.

### Task C2: Auth API service + interceptor

**Files:**
- Create: `frontend/src/services/AuthService.js`
- Modify: `frontend/src/services/PhentrieveService.js` (shared axios instance + interceptors) OR create `frontend/src/services/apiClient.js` and re-use it
- Test: `frontend/src/test/services/AuthService.test.js`

- [ ] **Step 1: Failing test:** mock axios; a 401 from a protected call triggers exactly one `/auth/refresh`, then retries the original; concurrent 401s share a single refresh (single-flight).
- [ ] **Step 2:** FAIL. **Step 3:** Implement a shared axios instance with `baseURL = API_URL`, request interceptor adding `Authorization` from the store, and a response interceptor implementing single-flight refresh-retry; `AuthService` wraps the `/auth/*` endpoints (with `withCredentials:true` and the CSRF header for refresh/logout). Refactor `PhentrieveService` to use the shared instance.
- [ ] **Step 4:** PASS. **Step 5:** Commit `feat(frontend): auth service + refresh-retry interceptor`.

### Task C3: AccountButton + AuthDialog

**Files:**
- Create: `frontend/src/components/auth/AccountButton.vue`, `frontend/src/components/auth/AuthDialog.vue`
- Modify: `frontend/src/App.vue` (mount `AccountButton` top-right via a fixed overlay / teleport)
- Test: `frontend/src/test/components/AuthDialog.test.js`

- [ ] **Step 1: Failing test:** mounting `AuthDialog` shows email/password fields; toggling to Register shows the confirm/register CTA; invalid email blocks submit.
- [ ] **Step 2:** FAIL. **Step 3:** Implement `AccountButton` (subtle `v-btn variant="text"` icon, `position:fixed; top:8px; right:8px; z-index` overlay; logged-in -> `v-menu` with email, verification chip, quota chip, logout). `AuthDialog` (`v-dialog`, login/register toggle, forgot-password link, validation, calls store actions, shows verification-sent + resend states). Mount `<AccountButton/>` in `App.vue`.
- [ ] **Step 4:** PASS + manual check. **Step 5:** Commit `feat(frontend): top-right account button + auth dialog`.

### Task C4: Email-link views + routes + quota nudge

**Files:**
- Create: `frontend/src/views/VerifyEmailView.vue`, `frontend/src/views/ResetPasswordView.vue`
- Modify: `frontend/src/router/index.js` (add `/verify`, `/reset-password`; handle `?auth=login`)
- Modify: `frontend/src/components/ResultsDisplay.vue` (anon near-limit nudge -> open dialog)
- Modify: `frontend/src/App.vue` (on mount: silent `authStore.refresh()`)

- [ ] **Step 1:** Add routes + views that read `token` from the query, call `verify`/`confirmPasswordReset`, show success/error with a link back. Add the nudge to the quota notice (only when anonymous and `quota_remaining<=1`). Trigger silent refresh on app mount.
- [ ] **Step 2:** `make frontend-test` PASS. **Step 3:** Commit `feat(frontend): verify/reset views, routes, quota login nudge`.

### Task C5: i18n

**Files:**
- Modify: `frontend/src/locales/{en,de,fr,es,nl}.json` (add `auth.*` block)

- [ ] **Step 1:** Add a complete `auth` key block to `en.json` (labels: login, register, email, password, confirmPassword, forgotPassword, logout, verifyPending, resendVerification, loggedInAs, quotaToday, loginForMore, errors.*). Translate into de/fr/es/nl (German quality matters - maintainer is DE).
- [ ] **Step 2:** `make frontend-i18n-check` PASS. **Step 3:** Commit `feat(frontend): auth i18n for 5 locales`.

---

## Phase D - Config, docs, gates

### Task D1: Env templates + docs

**Files:**
- Modify: `.env.example`, `.env.docker.template`, `docker-compose.yml` (pass new env to api), relevant `docs/`
- Create/Modify: `api/local_api_config.env` example entries for local auth testing

- [ ] **Step 1:** Document every new env var from spec §6 with safe dev defaults (auth on locally, console email). Note production setup (SMTP, JWT secret, secure cookies, base URL). Commit `docs: document auth/email/quota configuration`.

### Task D2: Full gate run

- [ ] **Step 1:** `make ci-local` -> PASS (fix lint/type/test).
- [ ] **Step 2:** `make security-python` -> PASS.
- [ ] **Step 3:** `make frontend-test-ci` + `make frontend-build-ci` -> PASS.
- [ ] **Step 4:** Manual smoke: start API (`make dev-api`) + frontend (`make dev-frontend`), register -> grab link from API logs -> verify -> login -> confirm quota shows 10.
- [ ] **Step 5:** Commit any fixes; update CHANGELOG (per release process) is deferred to release time.

---

## Self-review notes

- Spec coverage: §3 auth pkg -> A2-A8; §4 quota -> B1-B2; §5 frontend -> C1-C5; §6 config -> A1+D1; §7 testing -> tests in each task + D2; §8 security -> A2/A3/A8/B1. All covered.
- Type consistency: `UserStore` method names match between A4 (definition) and A7/A8/B1 (use); `tokens.*` names consistent A3->A7/A8; store getter `get_active_refresh_user` used in A8.
- No placeholders: tricky code shown inline; repetitive UI/i18n described by exact files + key list.
