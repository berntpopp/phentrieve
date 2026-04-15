# AGENTS.md

Shared repository instructions for coding agents working in this project.

## Project

Phentrieve maps clinical text to Human Phenotype Ontology (HPO) terms with a
retrieval-augmented workflow.

Primary areas:

- `phentrieve/` - Python CLI and library
- `api/` - FastAPI backend and MCP server
- `frontend/` - Vue 3 application
- `tests/` - unit, integration, and E2E coverage
- `.planning/` - specs, plans, analysis, and historical execution records

## Source Of Truth

- Use this file for repo-wide agent guidance.
- Keep `CLAUDE.md` concise and Claude-specific.
- Keep all planning artifacts under `.planning/`. Do not create new plan files
  in `plan/` or `docs/superpowers/`.
- Use `.planning/README.md` as the planning index.

## Working Rules

- Do not revert or overwrite changes you did not make unless explicitly asked.
- Keep edits scoped to the task; avoid unrelated refactors.
- Prefer existing patterns over inventing new abstractions.
- Keep all tests under `tests/`. Do not create `tests_new/`.
- Use ASCII unless a file already requires non-ASCII content.

## Commands

Required checks before claiming completion:

- `make check`
- `make typecheck-fast`
- `make test`

Combined shortcut:

- `make all`

Useful Python and CLI commands:

- `make install`
- `make install-dev`
- `make format`
- `make lint`
- `make lint-fix`
- `make typecheck`
- `make typecheck-fresh`
- `pytest tests/unit/api/`
- `phentrieve --help`

Frontend commands:

- `make frontend-install`
- `make frontend-lint`
- `make frontend-format`
- `make frontend-dev`
- `make frontend-build`
- `make frontend-test`
- `make frontend-test-cov`
- Run `make frontend-i18n-check` for locale changes

Local services:

- `make dev-api` starts the API on `http://localhost:8734`
- `make dev-frontend` starts the frontend on `http://localhost:5734`
- `make mcp-serve` starts the MCP server over stdio
- `make mcp-serve-http` starts the MCP server over HTTP

Docker and E2E:

- `make docker-build`
- `make docker-up`
- `make docker-down`
- `make docker-dev`
- `make test-e2e`
- `make test-e2e-fast`
- `make test-e2e-clean`

## Coding Standards

- Use `uv` for Python dependency management; do not use `pip`.
- Use modern Python typing: `list[str]`, `dict[str, int]`, `str | None`.
- Python formatting and linting use Ruff.
- Type checking uses mypy targeting Python 3.10.
- Frontend uses Vue 3, Vuetify, Pinia, Vue I18n, ESLint 9, and Prettier.

## Testing Notes

- `make test` excludes `slow` and `e2e` tests by default.
- Available pytest markers: `unit`, `integration`, `e2e`, `slow`.
- Tests run in parallel via `pytest-xdist`.
- For single-threaded debugging, use `uv run pytest tests/ -n 0 ...`.
- Treat failing checks as real issues unless you have clear evidence otherwise.

## Repo Notes

- The API runs from `api/`, so `PHENTRIEVE_DATA_ROOT_DIR=../data` in
  `api/local_api_config.env` points back to the repository root `data/`
  directory.
- Planning lifecycle:
  - active work -> `.planning/active/`
  - completed plans -> `.planning/completed/`
  - archived/superseded plans -> `.planning/archived/`
  - design specs -> `.planning/specs/`
  - analysis and reviews -> `.planning/analysis/`
  - rough or not-yet-activated plans -> `.planning/drafts/`
