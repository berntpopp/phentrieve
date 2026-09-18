# Consolidated Dependabot Updates Implementation Plan

**Goal:** Consolidate open Dependabot PRs (#351 through #358) and Dependabot security alert #132 into a single verified branch and PR, resolving peer dependency deadlocks and security advisories while maintaining full local CI parity.

**Branch:** `chore/consolidate-dependabot-updates`

---

## 1. Scope & Mapping

| PR / Alert | Target Package | From | To | Resolution / Impact |
|---|---|---|---|---|
| #358 | frontend-minor-patch group | multiple | multiple | `libphonenumber-js` ^1.13.13, `vue-router` ^5.3.1, `autoprefixer` ^10.5.6, `eslint` ^10.10.0, `eslint-plugin-vue` ^10.11.0, `happy-dom` ^20.14.3, `postcss` ^8.5.28 |
| #357 | `vuetify` | 3.13.3 | 3.13.4 | Clean minor bump in `frontend/package.json` |
| #356 | `vite` | 8.2.2 | 8.3.0 | Added `.js` extensions to local imports in `vite.config.js` and `vite-icon-optimizer.js` to satisfy Vite 8.3 native loader |
| #355 | `hadolint/hadolint-action` | 3.4.0 | 3.5.0 | Updated action ref in `.github/workflows/docker-publish.yml` |
| #354 | `fastmcp` | `<4.0.0,>=3.2.0` | `>=3.2.0,<5.0.0` | Broadened requirement in `pyproject.toml` |
| #353 | `@vitest/coverage-v8` | 4.1.11 | 5.0.0 (5.0.1) | Consolidated with vitest and @vitest/ui to eliminate peer-dependency conflicts |
| #352 | `vitest` | 4.1.11 | 5.0.0 (5.0.1) | Upgraded to 5.0.1; tested clean across 326 Vitest unit tests |
| #351 | `@vitest/ui` | 4.1.11 | 5.0.0 (5.0.1) | Upgraded to 5.0.1 alongside core vitest packages |
| Alert #132 | `mkdocs-material` | 9.7.6 | >=9.7.7 (9.7.7) | Resolves CVE-2026-73295 |
| Advisory | `anyio` (transitive) | 4.13.0 | 4.14.2 | Added constraint `anyio>=4.14.2` in `tool.uv.constraint-dependencies` to resolve CVE-2026-63374 / CVE-2026-64847 reported during pip-audit |
| Tooling | `.github/dependabot.yml` | - | - | Grouped `vitest`, `@vitest/*` into `vitest-updates` group to prevent future split-PR peer dependency deadlocks |
| Typing | `types-cachetools` | - | >=5.5.0 | Added to dev dependencies to provide clean types for `cachetools.TTLCache` |

---

## 2. Verification Tasks

- [x] Consolidate frontend dependencies in `frontend/package.json` and `package-lock.json`
- [x] Fix Vite 8.3 config import warnings by specifying `.js` extensions
- [x] Add Vitest grouping rule in `.github/dependabot.yml`
- [x] Update GitHub Actions hadolint action to v3.5.0 in `.github/workflows/docker-publish.yml`
- [x] Update Python dependencies in `pyproject.toml` (`fastmcp`, `mkdocs-material`, `anyio` constraint, `types-cachetools`)
- [x] Update test policy expectation in `tests/unit/test_dependency_security_policy.py`
- [x] Regenerate `uv.lock`
- [x] Run `make check` (Ruff format + lint)
- [x] Run `make typecheck-fast` and `make typecheck-fresh`
- [x] Run `tests/unit/test_dependency_security_policy.py` and `tests/unit/test_frontend_toolchain_policy.py`
- [x] Run `make ci-frontend` (audit, lint, format, 326 unit tests, build)
- [x] Run `make ci-python-quality` (Ruff, mypy, pytest with coverage)
- [x] Run `uv run pip-audit` (0 vulnerabilities found)
- [x] Run `make ci-python-compat PYTHON=3.12`
- [x] Run `make ci-python-compat PYTHON=3.13`
- [x] Commit changes with conventional commit
- [x] Push branch to remote and create draft PR
