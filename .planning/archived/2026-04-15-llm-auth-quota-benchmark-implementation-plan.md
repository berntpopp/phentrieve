# LLM Auth Quota Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM-based full-text analysis to `phentrieve`, protect it with anonymous and registered-user quotas, introduce registration and authentication, and integrate LLM benchmarking into the existing scriptable benchmark framework with automated benchmark-data acquisition.

**Architecture:** The implementation is split into seven workstreams: app-data persistence, auth/session flows, quota enforcement, LLM runtime port, benchmark integration, benchmark-data management, and edge/deployment updates. The design reuses the existing FastAPI, Vue, Docker, and benchmark structure instead of creating parallel systems. Public product requests go through quota checks and proxy throttling; benchmark commands bypass public quota middleware and instantiate the pipeline directly.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, Typer, Vue 3, Axios, SQLite for initial app data, NGINX, Docker Compose, pytest, Ruff, `uv`, Makefile, optional LiteLLM

---

## File Map

- Create: `phentrieve/llm/__init__.py`
- Create: `phentrieve/llm/types.py`
- Create: `phentrieve/llm/utils.py`
- Create: `phentrieve/llm/provider.py`
- Create: `phentrieve/llm/pipeline.py`
- Create: `phentrieve/llm/pricing.py`
- Create: `phentrieve/llm/pricing.yaml`
- Create: `phentrieve/llm/prompts/__init__.py`
- Create: `phentrieve/llm/prompts/loader.py`
- Create: `phentrieve/llm/prompts/templates/**`
- Create: `phentrieve/llm/annotation/__init__.py`
- Create: `phentrieve/llm/annotation/base.py`
- Create: `phentrieve/llm/annotation/direct_text.py`
- Create: `phentrieve/llm/annotation/retrieval_only.py`
- Create: `phentrieve/llm/annotation/tool_guided.py`
- Create: `phentrieve/llm/annotation/two_phase.py`
- Create: `phentrieve/llm/annotation/agentic_judge.py`
- Create: `phentrieve/llm/postprocess/__init__.py`
- Create: `phentrieve/llm/postprocess/base.py`
- Create: `phentrieve/llm/postprocess/validation.py`
- Create: `phentrieve/llm/postprocess/refinement.py`
- Create: `phentrieve/llm/postprocess/assertion_review.py`
- Create: `phentrieve/llm/postprocess/combined.py`
- Create: `phentrieve/appdata/__init__.py`
- Create: `phentrieve/appdata/db.py`
- Create: `phentrieve/appdata/schema.py`
- Create: `phentrieve/appdata/migrations.py`
- Create: `phentrieve/auth/__init__.py`
- Create: `phentrieve/auth/passwords.py`
- Create: `phentrieve/auth/sessions.py`
- Create: `phentrieve/auth/tokens.py`
- Create: `phentrieve/auth/service.py`
- Create: `phentrieve/quota/__init__.py`
- Create: `phentrieve/quota/service.py`
- Create: `phentrieve/quota/models.py`
- Create: `phentrieve/benchmark/data_loader.py`
- Create: `phentrieve/benchmark/llm_benchmark.py`
- Create: `phentrieve/benchmark/llm_cli.py`
- Create: `phentrieve/benchmark/data_sync.py`
- Modify: `phentrieve/cli/__init__.py`
- Modify: `phentrieve/cli/benchmark_commands.py`
- Modify: `phentrieve/cli/data_commands.py`
- Modify: `api/main.py`
- Create: `api/schemas/auth_schemas.py`
- Create: `api/schemas/llm_annotation_schemas.py`
- Create: `api/schemas/account_schemas.py`
- Create: `api/dependencies_auth.py`
- Create: `api/routers/auth_router.py`
- Create: `api/routers/account_router.py`
- Create: `api/routers/llm_annotation_router.py`
- Modify: `api/config.py`
- Modify: `api/api.yaml`
- Modify: `frontend/src/router/index.js`
- Modify: `frontend/src/services/PhentrieveService.js`
- Create: `frontend/src/services/AuthService.js`
- Create: `frontend/src/stores/auth.js`
- Create: `frontend/src/views/LoginView.vue`
- Create: `frontend/src/views/RegisterView.vue`
- Create: `frontend/src/views/AccountView.vue`
- Modify: `frontend/src/views/HomeView.vue`
- Modify: `frontend/nginx.conf`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `.env.docker.template`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Modify: `docs/user-guide/benchmarking-guide.md`
- Modify: `docs/user-guide/frontend-usage.md`
- Modify: `docs/user-guide/api-usage.md`
- Modify: `docs/deployment/security.md`
- Create or modify: targeted tests under `tests/unit/llm/`, `tests/unit/api/`, `tests/unit/auth/`, `tests/unit/quota/`, `tests/integration/`

## Task 1: Add application data persistence for auth, sessions, quotas, and usage

**Files:**
- Create: `phentrieve/appdata/db.py`
- Create: `phentrieve/appdata/schema.py`
- Create: `phentrieve/appdata/migrations.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/auth/test_appdata_schema.py`

- [ ] **Step 1: Confirm there is no existing app-data persistence layer**

Run:

```bash
rg -n "alembic|sqlalchemy|users table|sessions table|analysis_usage_daily|billing" phentrieve api tests
```

Expected: no existing app-level auth/session/quota database layer.

- [ ] **Step 2: Write the failing schema tests**

Create `tests/unit/auth/test_appdata_schema.py` with tests shaped like:

```python
from pathlib import Path

from phentrieve.appdata.db import connect_app_db
from phentrieve.appdata.migrations import ensure_app_schema


def test_ensure_app_schema_creates_core_tables(tmp_path: Path):
    db_path = tmp_path / "app.db"
    conn = connect_app_db(db_path)
    ensure_app_schema(conn)

    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    assert "users" in table_names
    assert "sessions" in table_names
    assert "analysis_usage_daily" in table_names
    assert "analysis_events" in table_names


def test_analysis_usage_daily_enforces_unique_subject_per_day(tmp_path: Path):
    db_path = tmp_path / "app.db"
    conn = connect_app_db(db_path)
    ensure_app_schema(conn)

    conn.execute(
        """
        INSERT INTO analysis_usage_daily (
            subject_type, subject_key, usage_date_utc, count_used
        ) VALUES (?, ?, ?, ?)
        """,
        ("anonymous_ip", "hash1", "2026-04-15", 1),
    )

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO analysis_usage_daily (
                subject_type, subject_key, usage_date_utc, count_used
            ) VALUES (?, ?, ?, ?)
            """,
            ("anonymous_ip", "hash1", "2026-04-15", 1),
        )
```

- [ ] **Step 3: Run the tests to verify the persistence layer is missing**

Run:

```bash
uv run pytest tests/unit/auth/test_appdata_schema.py -n 0 -v
```

Expected: FAIL because `phentrieve.appdata` does not yet exist.

- [ ] **Step 4: Implement the SQLite-first app-data layer**

Create `phentrieve/appdata/db.py` with a small connection helper shaped like:

```python
import sqlite3
from pathlib import Path


def connect_app_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
```

Create `phentrieve/appdata/schema.py` with table definitions for:

- `users`
- `sessions`
- `email_verification_tokens`
- `password_reset_tokens`
- `analysis_usage_daily`
- `analysis_events`

Use:

- `TEXT` primary keys where convenient
- `created_at` / `updated_at` timestamps
- unique email on `users.email`
- unique `(subject_type, subject_key, usage_date_utc)` on `analysis_usage_daily`

Create `phentrieve/appdata/migrations.py` with an `ensure_app_schema()` function that applies `CREATE TABLE IF NOT EXISTS` statements in dependency order.

- [ ] **Step 5: Add configuration for the app database path**

Add a new environment/config value in `api/config.py` and `api/api.yaml`:

```python
APP_DB_PATH: str = os.getenv("PHENTRIEVE_APP_DB_PATH", "../data/app/app.db")
```

Keep it separate from `hpo_data.db`.

- [ ] **Step 6: Run targeted verification**

Run:

```bash
uv run pytest tests/unit/auth/test_appdata_schema.py -n 0 -v
uv run ruff check phentrieve/appdata api/config.py tests/unit/auth/test_appdata_schema.py
```

Expected: PASS for the new schema tests and no Ruff issues in touched files.

## Task 2: Implement registration, login, email verification, and secure sessions

**Files:**
- Create: `phentrieve/auth/passwords.py`
- Create: `phentrieve/auth/sessions.py`
- Create: `phentrieve/auth/tokens.py`
- Create: `phentrieve/auth/service.py`
- Create: `api/schemas/auth_schemas.py`
- Create: `api/dependencies_auth.py`
- Create: `api/routers/auth_router.py`
- Create: `api/routers/account_router.py`
- Modify: `api/main.py`
- Test: `tests/unit/auth/test_auth_service.py`
- Test: `tests/unit/api/test_auth_router.py`

- [ ] **Step 1: Write failing tests for auth core behavior**

Create `tests/unit/auth/test_auth_service.py` with tests shaped like:

```python
from pathlib import Path

from phentrieve.appdata.db import connect_app_db
from phentrieve.appdata.migrations import ensure_app_schema
from phentrieve.auth.service import AuthService


def test_register_user_hashes_password(tmp_path: Path):
    conn = connect_app_db(tmp_path / "app.db")
    ensure_app_schema(conn)
    service = AuthService(conn)

    user = service.register_user("user@example.com", "correct horse battery staple")

    assert user["email"] == "user@example.com"
    assert "password_hash" in user.keys()
    assert "correct horse" not in user["password_hash"]


def test_login_rejects_unverified_user(tmp_path: Path):
    conn = connect_app_db(tmp_path / "app.db")
    ensure_app_schema(conn)
    service = AuthService(conn)
    service.register_user("user@example.com", "secret123456")

    assert service.authenticate_user("user@example.com", "secret123456") is None
```

Create `tests/unit/api/test_auth_router.py` with tests shaped like:

```python
from fastapi.testclient import TestClient

from api.main import create_app


client = TestClient(create_app())


def test_register_endpoint_sets_pending_verification_state():
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "user@example.com", "password": "secret123456"},
    )

    assert response.status_code == 201
    assert response.json()["email"] == "user@example.com"
    assert response.json()["email_verified"] is False
```

- [ ] **Step 2: Run the tests to confirm auth is not implemented**

Run:

```bash
uv run pytest tests/unit/auth/test_auth_service.py tests/unit/api/test_auth_router.py -n 0 -v
```

Expected: FAIL because the auth modules and routes do not exist.

- [ ] **Step 3: Implement password hashing with Argon2id**

Add an optional dependency in `pyproject.toml`:

```toml
"argon2-cffi>=23.1.0",
```

Create `phentrieve/auth/passwords.py` with helpers shaped like:

```python
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    return _HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False
```

- [ ] **Step 4: Implement auth service and server-side sessions**

Create `phentrieve/auth/service.py` with methods for:

- `register_user(email, password)`
- `verify_email(token)`
- `authenticate_user(email, password)`
- `create_session(user_id, expires_at)`
- `get_user_for_session(session_id)`
- `logout_session(session_id)`

Create `phentrieve/auth/sessions.py` to issue opaque session IDs with `secrets.token_urlsafe()`.

Create `phentrieve/auth/tokens.py` to issue email-verification and password-reset tokens.

Use generic failure behavior:

```python
return None
```

for failed login, rather than leaking whether a user exists.

- [ ] **Step 5: Add auth API routes and secure session cookies**

Create `api/routers/auth_router.py` with:

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/logout`
- `POST /api/v1/auth/verify-email`
- `POST /api/v1/auth/request-password-reset`
- `POST /api/v1/auth/reset-password`

Create `api/routers/account_router.py` with:

- `GET /api/v1/account/me`

Set cookies in the login route like:

```python
response.set_cookie(
    key="phentrieve_session",
    value=session_id,
    httponly=True,
    secure=True,
    samesite="lax",
    path="/",
)
```

Add `api/dependencies_auth.py` with:

- `get_current_session()`
- `get_current_user()`
- `get_optional_user()`

Register the routers in `api/main.py`.

- [ ] **Step 6: Run targeted verification**

Run:

```bash
uv run pytest tests/unit/auth/test_auth_service.py tests/unit/api/test_auth_router.py -n 0 -v
uv run ruff check phentrieve/auth api/routers/auth_router.py api/routers/account_router.py api/dependencies_auth.py tests/unit/auth/test_auth_service.py tests/unit/api/test_auth_router.py
```

Expected: PASS for targeted auth tests and no Ruff issues in touched files.

## Task 3: Implement anonymous and registered-user daily quota enforcement

**Files:**
- Create: `phentrieve/quota/service.py`
- Create: `phentrieve/quota/models.py`
- Modify: `api/dependencies_auth.py`
- Modify: `api/routers/llm_annotation_router.py`
- Modify: `api/main.py`
- Test: `tests/unit/quota/test_quota_service.py`
- Test: `tests/unit/api/test_llm_quota_enforcement.py`

- [ ] **Step 1: Write failing quota tests**

Create `tests/unit/quota/test_quota_service.py` with tests shaped like:

```python
from pathlib import Path

from phentrieve.appdata.db import connect_app_db
from phentrieve.appdata.migrations import ensure_app_schema
from phentrieve.quota.service import QuotaService


def test_anonymous_user_gets_one_daily_analysis(tmp_path: Path):
    conn = connect_app_db(tmp_path / "app.db")
    ensure_app_schema(conn)
    service = QuotaService(conn)

    assert service.check_and_consume_anonymous_analysis("iphash1", "2026-04-15") is True
    assert service.check_and_consume_anonymous_analysis("iphash1", "2026-04-15") is False


def test_verified_user_gets_ten_daily_analyses(tmp_path: Path):
    conn = connect_app_db(tmp_path / "app.db")
    ensure_app_schema(conn)
    service = QuotaService(conn)

    for _ in range(10):
        assert service.check_and_consume_user_analysis("user-1", "2026-04-15") is True
    assert service.check_and_consume_user_analysis("user-1", "2026-04-15") is False
```

- [ ] **Step 2: Run the tests to verify quota logic is missing**

Run:

```bash
uv run pytest tests/unit/quota/test_quota_service.py -n 0 -v
```

Expected: FAIL because the quota service does not exist.

- [ ] **Step 3: Implement the quota service**

Create `phentrieve/quota/service.py` with:

- UTC-date bucketing
- anonymous quota = 1/day
- verified user quota = 10/day
- atomic increment semantics using `INSERT ... ON CONFLICT DO UPDATE`

Shape the core function like:

```python
def consume_daily_quota(
    self,
    subject_type: str,
    subject_key: str,
    usage_date_utc: str,
    daily_limit: int,
) -> bool:
    ...
```

Return `True` only when the increment succeeds within the allowed limit.

- [ ] **Step 4: Implement trusted-IP-based anonymous identity**

Add a helper in `api/dependencies_auth.py` shaped like:

```python
import hashlib


def get_client_ip_hash(request: Request) -> str:
    ip = request.headers.get("x-real-ip") or request.client.host or "unknown"
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()
```

Use the trusted forwarded IP only after proxy configuration is fixed in Task 7.

- [ ] **Step 5: Enforce quota in the LLM annotation route**

In `api/routers/llm_annotation_router.py`, before running expensive work:

- if request has authenticated, verified user: consume user quota
- else: consume anonymous quota
- if quota exceeded, return `429`

Use a clear response shape:

```python
raise HTTPException(
    status_code=429,
    detail="Daily analysis quota exceeded.",
)
```

Count only successful analysis attempts that start pipeline execution, not arbitrary page loads.

- [ ] **Step 6: Run targeted verification**

Run:

```bash
uv run pytest tests/unit/quota/test_quota_service.py tests/unit/api/test_llm_quota_enforcement.py -n 0 -v
uv run ruff check phentrieve/quota api/dependencies_auth.py api/routers/llm_annotation_router.py tests/unit/quota/test_quota_service.py tests/unit/api/test_llm_quota_enforcement.py
```

Expected: PASS for quota tests and no Ruff issues in touched files.

## Task 4: Port the LLM runtime subsystem and wire it into API and frontend

**Files:**
- Create: `phentrieve/llm/**`
- Create: `api/schemas/llm_annotation_schemas.py`
- Create: `api/routers/llm_annotation_router.py`
- Modify: `api/main.py`
- Modify: `phentrieve/cli/__init__.py`
- Create or modify: frontend auth/account and analysis UI files
- Test: `tests/unit/llm/*`
- Test: `tests/unit/api/test_llm_annotation_router.py`

- [ ] **Step 1: Port the tested LLM foundation from the benchmark repo**

Copy and adapt the runtime modules from the reviewed branches into:

- `phentrieve/llm/provider.py`
- `phentrieve/llm/types.py`
- `phentrieve/llm/utils.py`
- `phentrieve/llm/pipeline.py`
- prompts and templates
- base annotation and postprocess modules

Start with:

- `direct_text`
- `retrieval_only`
- `tool_guided`
- `two_phase`

Then add `agentic_judge` after the baseline modes pass tests.

- [ ] **Step 2: Add LiteLLM optional dependency and package data wiring**

Update `pyproject.toml`:

```toml
llm = [
    "litellm>=1.30.0,<2.0.0",
]
```

Ensure package-data includes prompt YAML files.

- [ ] **Step 3: Add API router for full-text LLM analysis**

Create `api/schemas/llm_annotation_schemas.py` and `api/routers/llm_annotation_router.py` with:

- request schema for full-text analysis
- response schema including annotations, processing time, token usage, and estimated cost
- default cheap mode for public/free use
- optional advanced mode behind auth/plan rules later

Register the router in `api/main.py`.

- [ ] **Step 4: Add frontend routes and account-aware UI**

Create:

- `frontend/src/services/AuthService.js`
- `frontend/src/stores/auth.js`
- `frontend/src/views/LoginView.vue`
- `frontend/src/views/RegisterView.vue`
- `frontend/src/views/AccountView.vue`

Modify:

- `frontend/src/router/index.js`
- `frontend/src/services/PhentrieveService.js`
- `frontend/src/views/HomeView.vue`

The home view should:

- show remaining quota summary when available
- surface login/register CTA after anonymous analysis is used
- keep existing text-processing behavior intact

- [ ] **Step 5: Add targeted verification**

Run:

```bash
uv run pytest tests/unit/llm -n 0 -v
uv run pytest tests/unit/api/test_llm_annotation_router.py -n 0 -v
cd frontend && npm test -- --runInBand
```

Expected: unit tests for core LLM behavior and frontend auth/account wiring pass.

## Task 5: Integrate LLM benchmarking into the existing benchmark namespace

**Files:**
- Create: `phentrieve/benchmark/data_loader.py`
- Create: `phentrieve/benchmark/llm_benchmark.py`
- Create: `phentrieve/benchmark/llm_cli.py`
- Modify: `phentrieve/cli/benchmark_commands.py`
- Modify: `Makefile`
- Modify: docs under `docs/user-guide/` and `docs/advanced-topics/`
- Test: `tests/unit/test_llm_benchmark.py`
- Test: `tests/integration/test_llm_benchmark_workflow.py`

- [ ] **Step 1: Write failing benchmark CLI integration tests**

Create `tests/integration/test_llm_benchmark_workflow.py` with tests shaped like:

```python
from pathlib import Path

from typer.testing import CliRunner

from phentrieve.cli import app


runner = CliRunner()


def test_llm_benchmark_cli_namespace_exists():
    result = runner.invoke(
        app,
        ["benchmark", "llm", "--help"],
    )

    assert result.exit_code == 0
    assert "llm" in result.output.lower()
```

- [ ] **Step 2: Run tests to confirm the namespace does not yet exist**

Run:

```bash
uv run pytest tests/integration/test_llm_benchmark_workflow.py -n 0 -v
```

Expected: FAIL because `benchmark llm` is not registered.

- [ ] **Step 3: Port and adapt the LLM benchmark runner**

Create:

- `phentrieve/benchmark/data_loader.py`
- `phentrieve/benchmark/llm_benchmark.py`
- `phentrieve/benchmark/llm_cli.py`

Adapt them to follow the repo’s benchmark style:

- CLI namespace under `phentrieve benchmark llm`
- shared use of `ExtractionResult` and `CorpusExtractionMetrics`
- stable output files:
  - `llm_results.json`
  - `llm_predictions.json`
  - `llm_summary.json`

- [ ] **Step 4: Register LLM benchmarking under the existing benchmark family**

In `phentrieve/cli/benchmark_commands.py`:

```python
from phentrieve.benchmark.llm_cli import app as llm_app

app.add_typer(
    llm_app,
    name="llm",
    help="LLM annotation benchmarking.",
)
```

Do not make benchmarking depend on product login or public quota middleware.

- [ ] **Step 5: Add scriptable Make targets**

Add targets in `Makefile` such as:

```make
benchmark-llm-smoke:
	uv run phentrieve benchmark llm run data/benchmarks/llm/tiny --limit 1

benchmark-llm-genereviews:
	uv run phentrieve benchmark llm run data/benchmarks/llm/phenobert --dataset GeneReviews
```

- [ ] **Step 6: Run targeted verification**

Run:

```bash
uv run pytest tests/unit/test_llm_benchmark.py tests/integration/test_llm_benchmark_workflow.py -n 0 -v
uv run phentrieve benchmark llm --help
```

Expected: benchmark namespace is registered and tests pass against tiny fixture data.

## Task 6: Add benchmark-data acquisition, layout, and manifest tracking

**Files:**
- Create: `phentrieve/benchmark/data_sync.py`
- Modify: `phentrieve/cli/data_commands.py` or `phentrieve/benchmark/llm_cli.py`
- Create: tiny fixture data under `tests/data/benchmarks/llm/`
- Modify: docs and Makefile
- Test: `tests/unit/benchmark/test_data_sync.py`

- [ ] **Step 1: Write failing tests for benchmark-data manifest creation**

Create `tests/unit/benchmark/test_data_sync.py` with tests shaped like:

```python
from pathlib import Path

from phentrieve.benchmark.data_sync import write_benchmark_manifest


def test_write_benchmark_manifest(tmp_path: Path):
    target_dir = tmp_path / "benchmarks"
    target_dir.mkdir()

    write_benchmark_manifest(
        target_dir=target_dir,
        bundle_version="2026-04-15",
        source_url="https://example.com/benchmarks.tar.gz",
        datasets=["GeneReviews", "ID_68", "GSC_plus"],
    )

    manifest = target_dir / "manifest.json"
    assert manifest.exists()
```

- [ ] **Step 2: Implement benchmark-data sync helpers**

Create `phentrieve/benchmark/data_sync.py` with functions for:

- download bundle
- verify checksum
- unpack bundle
- write manifest
- skip when manifest already matches target version

Keep the manifest shape like:

```json
{
  "bundle_version": "2026-04-15",
  "source_url": "https://...",
  "datasets": ["GeneReviews", "ID_68", "GSC_plus"]
}
```

- [ ] **Step 3: Add CLI entry point for benchmark data acquisition**

Recommended command:

```bash
phentrieve benchmark llm fetch-data
```

Alternative acceptable command:

```bash
phentrieve data sync --include-benchmarks
```

The command must:

- fetch into `data/benchmarks/`
- preserve separation from `data/results/`
- emit a clear message when data already exists

- [ ] **Step 4: Add tiny in-repo fixture data**

Create a minimal fixture under:

```text
tests/data/benchmarks/llm/tiny/
```

Use it for:

- smoke tests
- CI benchmark namespace checks
- compare/report tests without needing the full corpus

- [ ] **Step 5: Run targeted verification**

Run:

```bash
uv run pytest tests/unit/benchmark/test_data_sync.py -n 0 -v
uv run phentrieve benchmark llm fetch-data --help
```

Expected: data sync helpers and CLI path are available and tested.

## Task 7: Update NGINX, Docker, and docs for real IP handling, throttling, and operations

**Files:**
- Modify: `frontend/nginx.conf`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `.env.docker.template`
- Modify: docs under `docs/deployment/` and `docs/user-guide/`
- Test: `tests/e2e/` or configuration-focused unit checks if available

- [ ] **Step 1: Add configurable trusted-proxy and rate-limit settings**

In `.env.example` and `.env.docker.template`, add variables such as:

```env
PHENTRIEVE_TRUSTED_PROXY_CIDRS=127.0.0.1/32,172.25.0.0/16
PHENTRIEVE_LIMIT_REQ_RATE=5r/m
PHENTRIEVE_LIMIT_REQ_BURST=3
```

- [ ] **Step 2: Extend frontend NGINX config for real IP and limit_req**

Update `frontend/nginx.conf` with sections shaped like:

```nginx
real_ip_header X-Forwarded-For;
real_ip_recursive on;

set_real_ip_from 127.0.0.1;
set_real_ip_from 172.25.0.0/16;

limit_req_zone $binary_remote_addr zone=llm_zone:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=auth_zone:10m rate=10r/m;
```

Apply them conservatively to:

- `/api/v1/llm/`
- `/api/v1/auth/`

Leave ordinary health and static routes alone.

- [ ] **Step 3: Ensure proxy headers remain intact**

Keep and verify:

```nginx
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

Document clearly whether trusted real-IP resolution is expected at this NGINX layer or at the external reverse proxy in front of it.

- [ ] **Step 4: Update docs**

Update:

- `docs/deployment/security.md`
- `docs/user-guide/frontend-usage.md`
- `docs/user-guide/api-usage.md`
- `docs/user-guide/benchmarking-guide.md`

Cover:

- anonymous vs registered quotas
- auth flow
- benchmark data fetch flow
- benchmark command namespace
- rate-limiting behavior

- [ ] **Step 5: Run targeted verification**

Run:

```bash
docker compose config > /tmp/phentrieve-compose.rendered.yml
nginx -t -c $(pwd)/frontend/nginx.conf
```

If local `nginx -t` is not viable because of image-specific paths, validate via containerized nginx config test instead.

Expected: rendered Compose is valid and NGINX config parses successfully.

## Task 8: Final verification before claiming completion

**Files:**
- All touched files

- [ ] **Step 1: Run focused backend verification**

Run:

```bash
uv run pytest tests/unit/auth tests/unit/quota tests/unit/llm tests/unit/api -n 0 -v
```

Expected: targeted backend tests pass.

- [ ] **Step 2: Run benchmark verification**

Run:

```bash
uv run pytest tests/integration/test_llm_benchmark_workflow.py -n 0 -v
uv run phentrieve benchmark llm run tests/data/benchmarks/llm/tiny --limit 1 --output-dir /tmp/phentrieve-llm-bench
```

Expected: smoke benchmark run succeeds and writes stable output files.

- [ ] **Step 3: Run frontend verification**

Run:

```bash
cd frontend && npm test -- --runInBand
```

Expected: frontend tests pass, including any new auth/account views and stores.

- [ ] **Step 4: Run static checks**

Run:

```bash
uv run ruff check phentrieve api tests
uv run mypy phentrieve api
```

Expected: no new Ruff issues and mypy remains green.

- [ ] **Step 5: Commit in small slices**

Recommended commit sequence:

```bash
git add phentrieve/appdata phentrieve/auth api/schemas/auth_schemas.py api/routers/auth_router.py api/routers/account_router.py api/dependencies_auth.py
git commit -m "feat(auth): add app data, registration, and sessions"

git add phentrieve/quota api/routers/llm_annotation_router.py api/main.py frontend/src/services/AuthService.js frontend/src/stores/auth.js frontend/src/views/LoginView.vue frontend/src/views/RegisterView.vue frontend/src/views/AccountView.vue frontend/src/router/index.js frontend/src/views/HomeView.vue
git commit -m "feat(llm): add quotas and account-aware analysis flow"

git add phentrieve/llm pyproject.toml
git commit -m "feat(llm): port annotation runtime and prompts"

git add phentrieve/benchmark phentrieve/cli/benchmark_commands.py Makefile tests/unit/test_llm_benchmark.py tests/integration/test_llm_benchmark_workflow.py
git commit -m "feat(benchmark): add scriptable llm benchmark workflow"

git add frontend/nginx.conf docker-compose.yml .env.example .env.docker.template docs/user-guide/benchmarking-guide.md docs/user-guide/frontend-usage.md docs/user-guide/api-usage.md docs/deployment/security.md
git commit -m "feat(deploy): add rate limiting and benchmark data operations"
```
