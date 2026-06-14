# Accounts and Higher Full-Text Quota - Design

- Status: Approved (design), pending implementation
- Date: 2026-06-14
- Area: `api/` (FastAPI), `frontend/` (Vue 3 / Vuetify / Pinia), config, docs, tests.
- Related: extends the IP-keyed LLM daily quota (`api/llm_quota.py`,
  `api/routers/text_processing_router.py`). Auth design mirrors the sibling
  project `../hnf1b-db` (FastAPI + Vue, the cleanest reusable reference).

## 1. Problem and Goal

The full-text / LLM extraction path is rate-limited by a daily quota keyed on a
hashed client IP (default 5/day, `PHENTRIEVE_LLM_DAILY_LIMIT`). Anonymous-only
access means there is no way to give trusted users more headroom, and an IP can
be shared (NAT) or rotated.

Goal: add an unobtrusive registration/login capability so that a **verified,
logged-in user** gets a higher full-text quota of **10/day**, while anonymous
users keep the existing IP-based limit. The feature must be:

1. **Unobtrusive** - a subtle login/register control in the top-right corner.
2. **Best-practice secure** - proper password hashing, email verification,
   password reset, login lockout, XSS-safe token handling.
3. **Testable locally** - no Postgres, no real SMTP required for local dev/CI.
4. **Configurable for production** - real SMTP (`noreply@phentrieve.org`),
   secrets, and secure cookies via environment variables.
5. **Backward compatible** - with auth disabled the app behaves exactly as today.

## 2. Decisions (locked)

- **User store: SQLite** (raw `sqlite3`, same style as `DailyQuotaStore`). No
  SQLAlchemy/Alembic. Default DB `../data/app/users.db`, configurable.
- **Email policy: verify-to-upgrade.** The higher 10/day quota applies only to
  accounts with `is_verified = 1`. Unverified accounts fall back to the
  anonymous limit.
- **UI: minimal top-right control + modal dialog.** The app has no top bar
  today (controls live in the footer); we add a small fixed top-right button
  that opens a login/register `v-dialog`. No full-page auth navigation except
  the email-link landing routes.
- **Scope: lean + security essentials.** Register, login, email verify,
  password reset, login rate-limit/lockout, higher quota. No roles, no admin UI.
- **Tokens (recommended, hnf1b-db-aligned):** short-lived **access JWT** held in
  memory by the SPA + **rotating opaque refresh token** in an HttpOnly, Secure,
  SameSite cookie with server-side revocation. CSRF double-submit token on the
  cookie-bearing endpoints (`refresh`, `logout`). No long-lived token in
  `localStorage`.
- **Zero new Python runtime dependencies.** `pyjwt[crypto]`, `bcrypt` (5.0),
  `email-validator`, `python-multipart` are already in `uv.lock`. SMTP uses
  stdlib `smtplib` via `run_in_threadpool`.
- **Config style:** plain `os.getenv` in `api/config.py`, matching the existing
  module (not pydantic-settings).

## 3. Architecture - backend (`api/auth/` package)

New package mirrors the raw-sqlite, per-request-connection style of
`api/llm_quota.py`.

- `passwords.py` - `hash_password` / `verify_password` via bcrypt (cost 12);
  `needs_rehash` for future migration.
- `tokens.py` - PyJWT HS256 access tokens (`sub=user_id`, `exp`, `iat`, `typ`);
  opaque refresh tokens (`secrets.token_urlsafe`) stored as SHA-256 hashes.
- `store.py` - `UserStore` (raw sqlite3, WAL, `busy_timeout`) with tables:
  - `users(id, email_lower UNIQUE, password_hash, is_verified, created_at,
    failed_attempts, locked_until, updated_at)`
  - `auth_tokens(token_hash PK, purpose, user_id, expires_at, consumed_at)` -
    single-use tokens for `verify` and `password_reset`.
  - `refresh_sessions(token_hash PK, user_id, expires_at, revoked_at,
    created_at)` - rotation + logout revocation.
  Methods: create_user, get_by_email/id, mark_verified, update_password,
  register/consume auth token, create/rotate/revoke refresh session, record
  failed login + lockout, reset failed counter.
- `email.py` - `EmailSender` protocol; `ConsoleEmailSender` (logs the link,
  default for dev/test) and `SmtpEmailSender` (stdlib smtplib in threadpool).
  `get_email_sender()` factory selects on `PHENTRIEVE_EMAIL_BACKEND`. From
  address `noreply@phentrieve.org`; links built from `PHENTRIEVE_PUBLIC_BASE_URL`.
- `schemas.py` - Pydantic request/response models (`EmailStr`-validated).
- `deps.py` - `get_current_user` (required, 401 on missing/invalid) and
  `get_optional_user` (returns `None`) FastAPI dependencies. A shared
  `get_user_store()` provider.
- `router.py` - `APIRouter(prefix="/api/v1/auth")`:
  - `POST /register` -> 201, creates unverified user, sends verify email.
    Generic response (no enumeration).
  - `POST /verify` `{token}` -> marks verified.
  - `POST /resend-verification` `{email}` -> generic 200.
  - `POST /login` `{email,password}` -> sets refresh cookie + CSRF cookie,
    returns access token + user. Rate-limit + lockout via `users` counters.
  - `POST /refresh` -> rotates refresh session, returns new access token.
    Requires refresh cookie + matching CSRF header.
  - `POST /logout` -> revokes refresh session, clears cookies.
  - `GET /me` -> current user (requires access token).
  - `POST /password-reset/request` `{email}` -> generic 200, sends reset email.
  - `POST /password-reset/confirm` `{token,new_password}` -> sets new password,
    revokes all refresh sessions.
- Router mounted in `api/main.py` only when `PHENTRIEVE_AUTH_ENABLED` is true.

## 4. Quota integration

Surgical change to `api/routers/text_processing_router.py` (helpers at
lines ~78-133):

- Add an auth-aware subject resolver: if a valid access token is present and the
  user `is_verified`, use `subject_key = hash_subject_key(f"user:{id}")` and
  `limit = PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT` (10). Otherwise the
  existing IP path with `PHENTRIEVE_LLM_DAILY_LIMIT` (5).
- `_get_llm_quota_store(limit)` takes the resolved limit so one store serves
  both tiers (the SQLite table is already keyed by `subject_key`).
- New `GET /api/v1/text/quota` returns the current tier's
  `{quota_used, quota_limit, quota_remaining, usage_date_utc, quota_reset_at,
  authenticated, verified}` without consuming quota, so the UI can display it.
- New `PHENTRIEVE_LLM_QUOTA_ENFORCE` (tri-state: unset = current behavior
  [enforce only when `PHENTRIEVE_ENV=production`], `true`/`false` = explicit
  override) so the quota can be exercised locally.
- The `user:<id>` keyspace cannot collide with IP keys (prefix-namespaced before
  hashing).

## 5. Architecture - frontend

- `stores/auth.js` - Pinia setup store. `accessToken` in memory (NOT persisted);
  lightweight `user` (`{email, isVerified}`) persisted. Getters
  `isAuthenticated`, `isVerified`. Actions: register, verify, resendVerification,
  login, logout, fetchMe, refresh, requestPasswordReset, confirmPasswordReset.
  On app mount: silent `refresh()` to restore session.
- API client (`services/PhentrieveService.js` or a small `apiClient.js`):
  axios request interceptor attaches `Authorization: Bearer <accessToken>`;
  auth calls use `withCredentials: true`; response interceptor does a
  single-flight 401 -> `/refresh` -> retry queue (hnf1b-db pattern). CSRF header
  read from the non-HttpOnly CSRF cookie for `refresh`/`logout`.
- `components/auth/AccountButton.vue` - subtle top-right control (fixed/teleport
  overlay, `v-btn variant="text"` density compact). Logged out: person-outline
  icon -> opens `AuthDialog`. Logged in: avatar initial -> menu with email,
  verification status, today's quota chip, logout.
- `components/auth/AuthDialog.vue` - single `v-dialog` with Login/Register
  toggle, "Forgot password?" link, verification-sent / resend state, Vuetify
  validation.
- Views + routes for email links: `/verify` (`VerifyEmailView.vue`) and
  `/reset-password` (`ResetPasswordView.vue`); `?auth=login` deep-link opens the
  dialog. Added to `router/index.js`.
- `ResultsDisplay.vue` - when an anonymous user nears/hits the limit, the quota
  notice gains a subtle "Log in for 10/day" affordance (opens the dialog).
- i18n: new `auth.*` block in all five locales (`en, de, fr, es, nl`);
  `make frontend-i18n-check` must pass.
- Reuse existing Vuetify patterns (`v-dialog`, `v-btn variant="tonal/text"`,
  `v-menu`) and the persisted-store pattern from `stores/disclaimer.js`.

## 6. Configuration

New env vars (safe dev defaults), surfaced in `.env.example` and
`.env.docker.template`:

- `PHENTRIEVE_AUTH_ENABLED` (default `false`; local dev/test `true`)
- `PHENTRIEVE_AUTH_JWT_SECRET` (required when auth enabled; fail-fast in prod)
- `PHENTRIEVE_AUTH_DB_PATH` (default `../data/app/users.db`)
- `PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS` (default 1800)
- `PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS` (default 1209600 / 14d)
- `PHENTRIEVE_AUTH_COOKIE_SECURE` (default `true`; `false` for local http)
- `PHENTRIEVE_AUTH_COOKIE_SAMESITE` (default `lax`; `none` for cross-site)
- `PHENTRIEVE_EMAIL_BACKEND` (`console` | `smtp`; default `console`)
- `PHENTRIEVE_EMAIL_FROM` (default `noreply@phentrieve.org`)
- `PHENTRIEVE_SMTP_HOST/PORT/USERNAME/PASSWORD/TLS` (prod)
- `PHENTRIEVE_PUBLIC_BASE_URL` (default `http://localhost:5734`) - email links
- `PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT` (default 10)
- `PHENTRIEVE_LLM_QUOTA_ENFORCE` (unset | `true` | `false`)

Local: auth on, console email (links in logs), SQLite under `data/app/`, quota
enforce-on. Production: real JWT secret, smtp backend, secure cookies, real base
URL.

## 7. Testing

Backend (`tests/unit/api/` + integration, pytest, follows existing structure):

- `passwords` hash/verify/needs_rehash.
- `tokens` access issue/verify/expiry; refresh hash round-trip.
- `UserStore` CRUD, token single-use, lockout, refresh rotation/revocation.
- `email` console sender captures link; smtp sender builds correct message
  (monkeypatched transport).
- auth router flow: register -> verify -> login -> me -> refresh -> logout;
  password reset; resend; rate-limit/lockout; no-enumeration responses.
- quota tiering: verified user = 10, anonymous = 5, unverified = 5;
  `GET /quota` for each; `PHENTRIEVE_LLM_QUOTA_ENFORCE` override.

Frontend (Vitest):

- auth store actions/getters and persistence boundary (token not persisted).
- `AuthDialog` validation and mode toggle.
- axios interceptor 401 -> refresh -> retry single-flight.

Gates: coverage added for all touched code; `make ci-local` and
`make security-python` before any push.

## 8. Security model

bcrypt (cost 12) hashing; verify-to-upgrade blocks fake-account quota farming;
no user enumeration on register/login/reset; login lockout after N failures;
access JWT short-lived; refresh rotation with server-side revocation; HttpOnly +
Secure + SameSite cookie; double-submit CSRF token on cookie-bearing endpoints;
all secrets via env; bounded authenticated limit (10) caps verified-account
spend; auth endpoints behind `PHENTRIEVE_AUTH_ENABLED`.

## 9. Out of scope (YAGNI)

Roles/permissions, admin user management UI, OAuth/social login, 2FA, account
deletion/GDPR export flows, organization/team accounts, per-user configurable
limits, PostgreSQL. The quota keyspace and config leave room for these later.
