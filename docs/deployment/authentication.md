# Accounts and the Higher Full-Text Quota

Phentrieve can optionally let users **register and sign in** to receive a
higher daily full-text (LLM) quota than anonymous visitors. The feature is
**off by default**: with `PHENTRIEVE_AUTH_ENABLED=false` the API behaves exactly
as before (anonymous, IP-keyed quota only).

## Full-text quota: anonymous vs signed in

Full-text (LLM) analyses are capped per UTC day. Who you are decides the cap and
how it is counted:

| Caller | Counted by | Daily limit | Setting |
| --- | --- | --- | --- |
| Anonymous | Client IP | 5 | `PHENTRIEVE_LLM_DAILY_LIMIT` |
| Signed in, **unverified** | Client IP | 5 | (same as anonymous) |
| Signed in, **verified** | Account | 10 | `PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT` |

The higher cap is **verify-to-upgrade**: an account only earns it after
confirming its email, which keeps throwaway accounts on the anonymous limit.
Limits reset at 00:00 UTC. Standard (non-LLM) extraction is never quota-limited.

`GET /api/v1/text/quota` reports your current tier and remaining count without
spending any. Accounts live in a small SQLite database
(`PHENTRIEVE_AUTH_DB_PATH`), alongside the existing quota DB — no PostgreSQL.

### Token model

- A short-lived **access token** (JWT) is returned to the SPA and held in
  memory; it is sent as `Authorization: Bearer`.
- A rotating, opaque **refresh token** is stored server-side (hashed) and
  delivered as an `HttpOnly`, `Secure`, `SameSite` cookie. The SPA silently
  refreshes on load and on `401`.
- Cookie-bearing endpoints (`/auth/refresh`, `/auth/logout`) require a
  double-submit **CSRF token**.

## Endpoints

All under `/api/v1/auth` (mounted only when auth is enabled):

`register`, `verify`, `resend-verification`, `login`, `refresh`, `logout`,
`me`, `password-reset/request`, `password-reset/confirm`.

The quota status is available without consuming quota at
`GET /api/v1/text/quota`.

## Local testing

Set these (e.g. in `.env` / `api/local_api_config.env`) and start the API +
frontend:

```bash
PHENTRIEVE_AUTH_ENABLED=true
PHENTRIEVE_AUTH_JWT_SECRET=$(openssl rand -hex 32)
PHENTRIEVE_AUTH_COOKIE_SECURE=false      # local HTTP
PHENTRIEVE_EMAIL_BACKEND=console         # links printed to the API log/stdout
PHENTRIEVE_LLM_QUOTA_ENFORCE=true        # exercise the quota locally
```

Register in the UI (top-right account button), copy the verification link from
the API log, open it to verify, then sign in. The account menu shows your
remaining full-text quota for the day.

### Pre-seeded dev account

To skip the register/verify step, seed a ready-to-use, already-verified account
at startup:

```bash
PHENTRIEVE_AUTH_SEED_EMAIL=dev@phentrieve.org
PHENTRIEVE_AUTH_SEED_PASSWORD=DevPassw0rd!
```

The account is created (or refreshed to match these credentials) on each boot,
so you can sign in immediately. Leave both empty in production.

## Production

```bash
PHENTRIEVE_AUTH_ENABLED=true
PHENTRIEVE_AUTH_JWT_SECRET=<32+ byte secret>   # required; API refuses to start without it
PHENTRIEVE_AUTH_COOKIE_SECURE=true             # HTTPS
PHENTRIEVE_AUTH_COOKIE_SAMESITE=lax            # "none" if frontend/API are cross-site
PHENTRIEVE_EMAIL_BACKEND=smtp
PHENTRIEVE_EMAIL_FROM=noreply@phentrieve.org
PHENTRIEVE_PUBLIC_BASE_URL=https://phentrieve.org
PHENTRIEVE_SMTP_HOST=...
PHENTRIEVE_SMTP_PORT=587
PHENTRIEVE_SMTP_USERNAME=...
PHENTRIEVE_SMTP_PASSWORD=...
PHENTRIEVE_SMTP_TLS=starttls                   # or ssl / none
```

Security notes: passwords are hashed with bcrypt; register and password-reset
responses are intentionally generic (no account enumeration); repeated failed
logins lock the account for `PHENTRIEVE_AUTH_LOCKOUT_SECONDS`. See
`.env.docker.template` for the full list of variables.
