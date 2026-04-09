# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phentrieve is an AI-powered system for mapping clinical text to Human Phenotype Ontology (HPO) terms using a Retrieval-Augmented Generation (RAG) approach. It consists of four main components:

1. **Python CLI/Library** (`phentrieve/`) - Core functionality for text processing, HPO term retrieval, and benchmarking
2. **FastAPI Backend** (`api/`) - REST API exposing Phentrieve functionality
3. **Vue.js Frontend** (`frontend/`) - Web interface for interactive HPO term mapping
4. **MCP Server** (`api/mcp/`) - Model Context Protocol server for Claude Desktop integration

## Planning & Project Status

**For current project status**, see `plan/STATUS.md`.

**Planning documentation** is organized in `plan/` with status-based structure:
- `01-active/` - Currently executing plans
- `02-completed/` - Successfully implemented
- `03-archived/` - Obsolete or superseded plans
- `04-reference/` - Guides and templates

## Development Commands

### CRITICAL: Pre-Commit Checklist (ALWAYS RUN BEFORE COMMITTING!)

```bash
# All three are MANDATORY before every commit:
make check            # Format + lint (Ruff)
make typecheck-fast   # Type check (mypy daemon)
make test             # Run tests (pytest)

# OR run all checks at once:
make all
```

- CI/CD will FAIL if any check fails. The codebase maintains 0 mypy errors, 0 lint errors, 0 format issues.
- **NEVER dismiss test failures** as "flaky", "pre-existing", or "environment-specific". Always investigate and fix.
- **For locale file changes** (`frontend/src/locales/`): also run `make frontend-i18n-check`

### Python CLI/Library
```bash
# Install
make install                    # Install package (uv sync)
make install-dev                # Install with all extras (uv sync --all-extras)

# Code quality
make check                      # Format + lint (REQUIRED BEFORE COMMIT!)
make format                     # Format only (Ruff)
make lint                       # Lint only (Ruff)
make lint-fix                   # Lint and auto-fix

# Type checking
make typecheck-fast             # mypy daemon (recommended, 10x faster)
make typecheck                  # mypy incremental
make typecheck-fresh            # mypy from scratch (clears cache)

# Testing
make test                       # Run unit/integration tests (excludes slow + e2e)
make test-cov                   # Run tests with coverage
pytest tests/unit/api/          # Run specific test directory
pytest tests/test_chunking.py   # Run specific test file
pytest -k "test_semantic"       # Run tests matching pattern
pytest -m "slow"                # Run only slow-marked tests

# Package management (uv)
make lock                       # Update uv.lock
make upgrade                    # Upgrade all dependencies
make add PACKAGE=name           # Add new dependency

# CLI
phentrieve --help
phentrieve query --interactive
phentrieve text process "clinical text here"
phentrieve data prepare         # Download and prepare HPO data
phentrieve index build          # Build vector index

# Cleaning
make clean                      # Remove build artifacts and caches
```

### Frontend (Vue.js)
```bash
make frontend-install           # Install dependencies
make frontend-lint              # Lint with ESLint 9
make frontend-format            # Format with Prettier
make frontend-dev               # Dev server (localhost:5734)
make frontend-build             # Production build
make frontend-test              # Run tests once (Vitest)
make frontend-test-cov          # Tests with coverage
make frontend-i18n-check        # Validate translation completeness (REQUIRED for locale changes!)
```

### Local Development (Recommended)

```bash
# Terminal 1: API with hot reload
make dev-api                    # http://localhost:8734, docs at /docs

# Terminal 2: Frontend with Vite HMR
make dev-frontend               # http://localhost:5734
```

**Port convention**: API uses 8734, frontend uses 5734 (HPOD on phone keypad).

**Critical path config**: The API runs from `api/` directory, so `PHENTRIEVE_DATA_ROOT_DIR=../data` in `api/local_api_config.env` points to project root `data/`. Using `./data` would incorrectly point to `api/data/`.

### MCP Server
```bash
make mcp-serve                  # Start MCP server (stdio transport for Claude Desktop)
make mcp-serve-http             # Start MCP server with HTTP transport (port 8734)
make mcp-info                   # Display MCP server configuration
```

### Docker
```bash
make docker-build               # Build images locally
make docker-up                  # Start containers (detached)
make docker-down                # Stop containers
make docker-dev                 # Development stack (local builds)
```

### E2E Testing (Docker)
```bash
make test-e2e                   # All E2E tests (security + health + API)
make test-e2e-security          # Security tests (non-root, read-only FS, capabilities)
make test-e2e-health            # Health check tests
make test-e2e-api               # API workflow tests
make test-e2e-fast              # Skip container rebuild (faster iteration)
make test-e2e-clean             # Cleanup test resources
```

### Benchmarking
```bash
phentrieve benchmark run                                    # Default tiny dataset
phentrieve benchmark run --test-file german/70cases_gemini_v1.json  # Specific dataset
phentrieve benchmark compare
phentrieve benchmark visualize
```

## Code Style & Conventions

### Python (3.10+)

- **Modern type syntax required**: Use `list[str]`, `dict[str, int]`, `str | None` (PEP 585/604). Do NOT use `List[str]`, `Optional[str]`.
- **Formatter/Linter**: Ruff with line length 88, rules: E/W/F/I/B/C4/UP/S
  - `S101` (assert) allowed in tests; `E501` (line length) ignored globally
- **Type checker**: mypy with Python 3.10 target, gradual typing (not strict)
- **Package manager**: `uv` (not pip)

### pytest Markers

Tests use markers configured in `pyproject.toml`:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - Docker end-to-end tests
- `@pytest.mark.slow` - Long-running tests

`make test` runs with `-m "not slow and not e2e"` by default.

### Frontend

- Vue 3 with Composition API
- Vuetify for UI components
- Pinia for state management (with persistence)
- Vue i18n for internationalization (EN, DE, ES, FR, NL)
- ESLint 9 + Prettier

### Test Directory Structure

**All tests MUST be placed in `tests/` only.** Never create a `tests_new/` folder.

- `tests/unit/` - Unit tests (organized by module: `api/`, `cli/`, `core/`, `retrieval/`, `text_processing/`)
- `tests/integration/` - Integration tests
- `tests/e2e/` - Docker E2E tests (security, health, API workflows)
- `tests/data/` - Test fixtures and benchmark datasets

## Architecture Overview

### Data Flow

1. Clinical text → Text processing pipeline → Semantic chunks with assertions
2. Chunks → Embedding → Vector search against HPO term database (ChromaDB)
3. Retrieval results → Optional re-ranking → Formatted output with text attribution

### Core Modules (`phentrieve/`)

- **`config.py`** - Central configuration constants and defaults
- **`embeddings.py`** - SBERT model loading, tensor operations, device management
- **`cli/`** - Typer-based CLI commands (query, text, benchmark, data, index, mcp)
- **`text_processing/`** - Chunking, ConText assertion detection (122 rules, 5 languages), HPO extraction
- **`indexing/`** - ChromaDB vector indexing
- **`retrieval/`** - Dense retrieval, cross-encoder re-ranking, aggregation
- **`evaluation/`** - Benchmarking, metrics calculation
- **`data_processing/`** - HPO parsing, SQLite database, data bundling

### API (`api/`)

- **Routers** (`api/routers/`): query, text_processing, similarity, config_info, health, system
- **Schemas** (`api/schemas/`): Pydantic request/response models
- **Dependencies** (`api/dependencies.py`): Dependency injection for model loading/caching
- **MCP** (`api/mcp/`): Model Context Protocol server (stdio + HTTP transports)
- OpenAPI docs at `/docs`

### Key Configuration Files

- **`phentrieve.yaml`** (from `phentrieve.yaml.template`) - Models, chunking, assertion detection
- **`api/local_api_config.env`** - API port (8734), logging, CORS, data root path
- **`frontend/vite.config.js`** - API proxy to localhost:8734, HMR config

### Caching Pattern

Thread-safe `@lru_cache` is used throughout for expensive operations (HPO graph loading, config, label mapping, ID normalization). Clear with `.cache_clear()` in tests.

### Key Architectural Notes

**Multilingual Support**: Embedding models are multilingual, HPO translations in `api/hpo_translations/`, text processing adapts via spaCy models, frontend has 5 locales.

**ChromaDB Indexing**: Vector indexes are per-embedding-model in `data/indexes/`. Index names: `{model_name_sanitized}`.

**Cross-Encoder Re-ranking**: Optional via `enable_reranker: true` in `phentrieve.yaml`. Default model: BAAI/bge-reranker-v2-m3. High-confidence dense matches (>=0.7) are preserved from demotion.

**ConText Assertion Detection**: medspaCy-based with direction-aware scope (FORWARD/BACKWARD/BIDIRECTIONAL), TERMINATE boundaries, PSEUDO false positives. 122 rules across EN/DE/ES/FR/NL. See `docs/advanced-topics/negation-detection.md`.

**HPO Term Details** (Issue #24): API `include_details` parameter enriches results with definitions/synonyms via batch-loading (`HPODatabase.get_terms_by_ids()`). Also available as `--include-details` CLI flag.

### Data Directory Structure

- `data/hpo_data.db` - SQLite database with HPO terms, ontology graph, metadata (~12 MB, via `phentrieve data prepare`)
- `data/hp.json` - Source HPO JSON file
- `data/indexes/` - ChromaDB vector indexes per embedding model
- `data/hf_cache/` - Hugging Face model cache
- `data/results/` - Benchmark results

### Benchmark Datasets (`tests/data/benchmarks/`)

Organized by language (`german/`, `en/`):
- `tiny_v1.json` (9 cases) - Quick testing, default
- `70cases_gemini_v1.json` (70 cases) - Medium evaluation
- `200cases_gemini_v1.json` (200 cases) - Comprehensive

### CI/CD

**Python CI** (3.10, 3.11, 3.12): Ruff format/lint, mypy (informational), pytest + Codecov
**Frontend CI** (Node 18): ESLint 9, Prettier, Vitest + Codecov, build verification
**Docker**: Build test on PRs, push to GHCR on main/tags
**Change detection**: Only runs jobs for modified code paths
**Workflows**: `.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml`

### Dependency Management

Dependabot runs weekly (Mondays 09:00 CET). PRs require manual review - no auto-merge.

**Pinned to major version** (breaking changes tracked): `sentence-transformers` (4.x), `chromadb` (1.x), `torch` (2.x), `typer` (0.16.x), Vue/Vuetify/Vite.
