# CI/CD Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Reduce GitHub Actions PR runtime and compute while preserving code
quality, local parity, Python compatibility signal, security scanning, and
frontend build confidence.

**Architecture:** Split Python CI into one full quality gate on Python 3.11 and
lightweight compatibility test jobs on Python 3.12/3.13. Use `setup-uv` built-in
caching instead of manual `.venv` caching, keep frontend/security/CodeQL
separate, and align Makefile targets with CI command groups.

**Tech Stack:** GitHub Actions, uv, pytest, Ruff, mypy, npm, Vitest, Vite,
pip-audit, Bandit, CodeQL.

---

## Task 1: Optimize Python CI Jobs

**Files:**

- Modify: `.github/workflows/ci.yml`

- [x] **Step 1: Replace the Python matrix job with two jobs**

In `.github/workflows/ci.yml`, replace `python-ci` with:

- `python-quality` on Python 3.11:
  - `uv sync --locked --all-extras --dev`
  - `make format-check`
  - `make lint`
  - `make typecheck`
  - `make test-ci`
  - upload coverage
- `python-compat` on Python 3.12 and 3.13:
  - path-gated to Python/dependency/workflow changes
  - `uv sync --locked --all-extras --dev`
  - pytest only, no lint/type/coverage upload

Use `astral-sh/setup-uv@v7` with:

```yaml
with:
  version: "latest"
  enable-cache: true
  cache-dependency-glob: uv.lock
```

- [x] **Step 2: Update CI summary needs and labels**

Change `ci-summary` from `python-ci` to `python-quality` and
`python-compat`, and fail if either job fails.

- [x] **Step 3: Validate workflow syntax**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml
for path in Path(".github/workflows").glob("*.yml"):
    yaml.safe_load(path.read_text())
print("workflow yaml ok")
PY
```

Expected: `workflow yaml ok`.

## Task 2: Align Security Workflow Caching

**Files:**

- Modify: `.github/workflows/security.yml`

- [x] **Step 1: Replace manual uv caches**

In `pip-audit` and `bandit`, remove the manual `actions/cache` steps for
`~/.cache/uv` and `.venv`. Configure `astral-sh/setup-uv@v7` with built-in
cache:

```yaml
with:
  version: "latest"
  enable-cache: true
  cache-dependency-glob: uv.lock
```

- [x] **Step 2: Use locked sync**

Change `uv sync --all-extras --dev` to:

```bash
uv sync --locked --all-extras --dev
```

- [x] **Step 3: Validate workflow syntax**

Run the same YAML validation command from Task 1.

Expected: `workflow yaml ok`.

## Task 3: Align Local Makefile With CI

**Files:**

- Modify: `Makefile`

- [x] **Step 1: Add CI parity targets**

Add/update these targets:

- `ci-python-quality`: full Python PR gate
- `ci-python-compat`: pytest-only compatibility gate for the current local
  Python interpreter
- `ci-frontend`: already exists; align to PR CI by using `frontend-test-ci`
  semantics instead of coverage
- `ci-local`: call `ci-python-quality ci-frontend`

- [x] **Step 2: Keep existing convenience aliases**

Keep `precommit: ci-local` and keep existing `check`, `test-ci`,
`frontend-test-ci`, and `frontend-build-ci` unchanged.

- [x] **Step 3: Run Makefile smoke checks**

Run:

```bash
make check
make typecheck-fast
make frontend-test-ci
```

Expected: all pass.

## Task 4: Final Verification And Commit

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/security.yml`
- Modify: `Makefile`

- [x] **Step 1: Run focused CI verification**

Run:

```bash
make check
make typecheck-fast
make test
make frontend-test-ci
make frontend-build-ci
uv run python - <<'PY'
from pathlib import Path
import yaml
for path in Path(".github/workflows").glob("*.yml"):
    yaml.safe_load(path.read_text())
print("workflow yaml ok")
PY
```

Expected: all commands pass.

- [x] **Step 2: Commit**

Commit:

```bash
git add .github/workflows/ci.yml .github/workflows/security.yml Makefile .planning/active/2026-05-22-ci-cd-optimization-plan.md
git commit -m "ci: optimize pull request quality gates"
```
