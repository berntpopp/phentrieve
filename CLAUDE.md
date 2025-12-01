# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phentrieve is an AI-powered system for mapping clinical text to Human Phenotype Ontology (HPO) terms using a Retrieval-Augmented Generation (RAG) approach. It consists of three main components:

1. **Python CLI/Library** (`phentrieve/`) - Core functionality for text processing, HPO term retrieval, and benchmarking
2. **FastAPI Backend** (`api/`) - REST API exposing Phentrieve functionality
3. **Vue.js Frontend** (`frontend/`) - Web interface for interactive HPO term mapping

## Planning & Project Status

**For current project status**, see `plan/STATUS.md` - comprehensive snapshot of completed work, testing metrics, and next steps.

**Planning documentation** is organized in `plan/` with status-based structure:
- `01-active/` - Currently executing plans
- `02-completed/` - Successfully implemented (MASTER-PLAN.md, TESTING-MODERNIZATION-PLAN.md, LOCAL-DEV-ENVIRONMENT.md, HPO-PARSER-RESILIENCE-REFACTOR-V2-SIMPLIFIED.md)
- `03-archived/` - Obsolete or superseded plans
- `04-reference/` - Guides and templates
- `README.md` - Navigation guide and best practices for LLM-assisted development

**Current Status Highlights**:
- ✅ Tooling Modernization: 8/9 phases complete (Ruff, uv, mypy, ESLint 9, GHCR, Dependabot, CI/CD)
- ✅ Testing Infrastructure: 532 tests (490 unit/integration + 42 Docker E2E), 13% statement coverage
- ✅ Local Development: 100x faster than Docker with instant hot reload
- ✅ Code Quality: 0 linting errors, 0 type errors
- ✅ HPO Parser Resilience: Safe dictionary access (Issue #23)
- ✅ HPO Term Details: Definitions and synonyms in API responses with batch-loading optimization (Issue #24)

## Development Commands

### ⚠️ CRITICAL: Pre-Commit Checklist (ALWAYS RUN BEFORE COMMITTING!)

**MANDATORY checks before every commit:**

```bash
# 1. Format code (auto-fixes formatting issues)
make check

# 2. Type checking (must pass with 0 errors)
make typecheck-fast

# 3. Run tests (must pass with 0 failures)
make test

# OR run all checks at once:
make all
```

**Why this matters:**
- CI/CD will FAIL if any check fails
- Formatting issues cause build failures
- Type errors break the build
- Test failures indicate regressions
- **NEVER skip these checks before committing!**

**⚠️ CRITICAL: Never Ignore Test Failures!**
- **NEVER dismiss test failures as "flaky", "pre-existing", or "environment-specific"**
- **ALWAYS investigate and fix test failures** - they indicate real problems
- If a test threshold is too tight (e.g., performance tests), fix the test properly
- If a test is genuinely unreliable, fix the test design, don't ignore the failure
- Test failures in CI mean the code cannot be merged - period

**Common mistake:** Running tests locally but forgetting `make check` (formatting) → CI fails!

**For locale file changes (frontend/src/locales/):**
```bash
make frontend-i18n-check  # REQUIRED: Validates all locales have matching keys
```

### Python CLI/Library
```bash
# Install in development mode (using uv)
make install                                         # Install package (uv sync)
make install-dev                                     # Install with all extras (uv sync --all-extras)

# Development workflow (recommended)
make check                                           # Format + lint code (REQUIRED BEFORE COMMIT!)
make test                                            # Run tests (REQUIRED BEFORE COMMIT!)
make typecheck-fast                                  # Type check (REQUIRED BEFORE COMMIT!)
make all                                             # Clean + check + test (BEST PRACTICE!)

# IMPORTANT: Always run ALL checks before committing Python code changes
# The codebase maintains 0 mypy errors, 0 lint errors, 0 format issues - keep it that way!

# Code quality
make format                                          # Format with Ruff
make lint                                            # Lint with Ruff
make lint-fix                                        # Lint and auto-fix issues

# Type checking
make typecheck                                       # Type check with mypy (incremental + SQLite cache)
make typecheck-fast                                  # Fast type check using mypy daemon (recommended)
make typecheck-daemon-stop                           # Stop mypy daemon
make typecheck-fresh                                 # Type check from scratch (clears cache)

# Package management (uv)
make lock                                            # Update uv.lock file
make upgrade                                         # Upgrade all dependencies
make add PACKAGE=name                                # Add new dependency
make remove PACKAGE=name                             # Remove dependency

# Common CLI commands
phentrieve --help                                    # Show all commands
phentrieve query --interactive                       # Interactive HPO query mode
phentrieve text process "clinical text here"         # Extract HPO terms from text

# Data and index management
phentrieve data prepare                              # Download and prepare HPO data
phentrieve index build                               # Build vector index for HPO terms

# Benchmarking
phentrieve benchmark run                             # Run retrieval benchmarks
phentrieve benchmark compare                         # Compare benchmark results
phentrieve benchmark visualize                       # Generate result visualizations

# Testing (manual)
pytest tests/                                        # Run all tests
pytest tests/test_chunking.py                        # Run specific test file
pytest -k "test_semantic"                            # Run tests matching pattern
make test-cov                                        # Run tests with coverage

# Cleaning
make clean                                           # Remove build artifacts and caches
```

### Frontend (Vue.js)
```bash
# Using Makefile (recommended)
make frontend-install                                # Install dependencies
make frontend-lint                                   # Lint with ESLint 9
make frontend-format                                 # Format with Prettier
make frontend-dev                                    # Development server
make frontend-build                                  # Production build

# Or directly with npm
cd frontend/
npm install
npm run dev                                          # Development server
npm run build                                        # Production build
npm run preview                                      # Preview production build
npm run lint                                         # ESLint check
npm run lint:fix                                     # ESLint auto-fix
npm run format                                       # Prettier format
npm run format:check                                 # Prettier check
npm run test                                         # Vitest (watch mode)
npm run test:run                                     # Vitest (run once)
npm run test:ui                                      # Vitest with UI
npm run test:coverage                                # Vitest with coverage

# Using Makefile for testing
make frontend-test                                   # Run tests once
make frontend-test-ui                                # Run tests with UI
make frontend-test-cov                               # Run tests with coverage

# i18n translation validation (REQUIRED when changing locale files!)
make frontend-i18n-check                             # Validate translation completeness and congruence
make frontend-i18n-report                            # Generate detailed JSON report
```

### API (FastAPI)
```bash
cd api/
python run_api_local.py    # Run local development server
```

### Docker Development
```bash
# Using Makefile (recommended)
make docker-build                                    # Build Docker images locally
make docker-up                                       # Start containers (detached)
make docker-down                                     # Stop containers
make docker-logs                                     # View logs
make docker-dev                                      # Development stack

# Or directly with docker-compose
docker-compose -f docker-compose.dev.yml up          # Development (builds locally)
docker-compose up                                    # Production (uses GHCR images)
docker-compose down                                  # Stop containers

# Pull pre-built images from GitHub Container Registry
docker pull ghcr.io/berntpopp/phentrieve/api:latest
docker pull ghcr.io/berntpopp/phentrieve/frontend:latest

# Build and push to GHCR (requires authentication)
docker login ghcr.io -u USERNAME
docker-compose build
docker-compose push
```

**Docker Images**: Images are automatically built and pushed to GitHub Container Registry (GHCR) via GitHub Actions on:
- Push to `main` branch → `latest` tag
- Git tags `v*.*.*` → versioned tags (e.g., `v1.0.0`, `1.0`, `1`)
- Pull requests → test builds only (not pushed)

### Local Development (Fast - No Docker)

**⚡ Recommended for daily development** - 100x faster than Docker with instant hot reload!

For detailed documentation, see `plan/LOCAL-DEV-ENVIRONMENT.md`

#### First-Time Setup

```bash
# Run automated setup script
./scripts/setup-local-dev.sh

# This will:
# - Check prerequisites (Python, uv, Node.js, npm)
# - Install Python dependencies with uv (10-100x faster than pip)
# - Install frontend dependencies with npm
# - Verify data directory structure
# - Provide next steps

# If data preparation needed:
phentrieve data prepare    # Download HPO data
phentrieve index build     # Build vector indexes
```

#### Daily Development Workflow

**Option 1: Two Terminals (Recommended)**

```bash
# Terminal 1: Start API server with hot reload
make dev-api
# → API:      http://localhost:8734
# → API Docs: http://localhost:8734/docs
# → Hot reload: <1s on .py file changes

# Terminal 2: Start frontend with Vite HMR
make dev-frontend
# → Frontend: http://localhost:5734
# → HMR: <50ms on .vue/.ts file changes
```

**Option 2: View Instructions**

```bash
make dev-all    # Display setup instructions for both terminals
```

#### Performance Benefits

| Metric | Docker | Native Local | Improvement |
|--------|--------|--------------|-------------|
| **Cold Start** | 5-10 minutes | 2-3 seconds | 100-200x faster |
| **API Reload** | 3-5 seconds | <1 second | 5x faster |
| **Frontend HMR** | 2-4 seconds | <50ms | 40-80x faster |
| **Memory Usage** | ~2GB | ~500MB | 75% less |
| **CPU Idle** | 5-10% | <1% | 90% less |

#### Hot Reload Features

**API (FastAPI + Uvicorn)**:
- Automatically detects `.py` file changes
- Restarts server in <1 second
- Preserves ChromaDB connections
- Debug logging enabled by default
- Full stack traces in console

**Frontend (Vite HMR)**:
- Vue Fast Refresh preserves component state
- CSS hot injection (no page reload)
- Instant feedback (<50ms)
- Error overlay in browser
- Source maps for debugging

#### Configuration Files

**API Configuration**: `api/local_api_config.env`
```bash
API_PORT=8734                         # API server port (custom HPOD port)
LOG_LEVEL=DEBUG                       # Detailed logging
RELOAD=true                           # Enable hot reload
PHENTRIEVE_DATA_ROOT_DIR=../data      # IMPORTANT: Relative to api/ directory!
ALLOWED_ORIGINS=http://localhost:5734 # CORS for frontend
```

**IMPORTANT Path Configuration:**
- The API runs from `api/` directory via `cd api && python run_api_local.py`
- Therefore `PHENTRIEVE_DATA_ROOT_DIR=../data` points to project root `data/`
- Using `./data` would incorrectly point to `api/data` (doesn't exist)
- This must be set correctly or you'll get "503 Service Unavailable" errors

**Frontend Configuration**: `frontend/vite.config.js`
- API proxy: Forwards `/api/*` to `http://localhost:8734` (custom HPOD port)
- HMR overlay: Shows errors in browser
- Fast refresh: Vue component state preservation
- Source maps: Enabled for debugging

#### Troubleshooting

**API not starting?**
```bash
# Check if port 8734 is in use
lsof -i :8734
kill -9 <PID>

# Verify environment file exists
ls -la api/local_api_config.env

# Check Python dependencies
uv sync
```

**Frontend not connecting to API?**
```bash
# Check Vite proxy configuration
cat frontend/vite.config.js | grep -A 10 "proxy"

# Verify API is running
curl http://localhost:8734/health

# Check browser console for CORS errors
```

**Hot reload not working?**
```bash
# API: Verify watchfiles is installed
uv pip list | grep watchfiles

# Frontend: Check HMR connection in browser console
# Should see: [vite] connected.

# Restart with debug logging
cd api && fastapi dev run_api_local.py --log-level debug
```

**Port conflicts?**
```bash
# Find process using ports
lsof -i :8734  # API (custom HPOD port)
lsof -i :5734  # Frontend (custom HPOD port)

# Or use different ports
uvicorn api.run_api_local:app --reload --port 8001
```

**PyTorch CUDA/NVML warning at startup?**

As of recent updates, PyTorch CUDA/NVML warnings are automatically caught and logged nicely through the application's logger system:

```python
# In phentrieve/embeddings.py - warning is caught and logged at DEBUG level
logger.debug("PyTorch CUDA/NVML initialization: GPU monitoring unavailable. Using CPU device.")
```

**What this means**:
- NVML (NVIDIA Management Library) is for GPU monitoring
- PyTorch checks for CUDA/GPU availability at import time
- NVML initialization may fail in some environments (common behavior)
- The system correctly falls back to CPU execution
- **Zero functional impact** - the application works normally

**Implementation**:
- Warning is caught using Python's `warnings.catch_warnings()` at module import
- Logged at DEBUG level for visibility without clutter
- All code has proper conditional checks: `if torch.cuda.is_available():`
- No raw Python warnings printed to stderr

**No action needed** - the warning is handled automatically and logged cleanly.

#### When to Use Docker vs Local

**Use Local Development (Native) when**:
✅ Daily coding and testing
✅ Rapid iteration with hot reload
✅ Running on your local machine
✅ Need maximum performance
✅ Debugging with IDE integration

**Use Docker when**:
✅ Production deployment
✅ CI/CD testing pipelines
✅ Team onboarding ("works on my machine" prevention)
✅ Multi-service orchestration
✅ System-level dependency isolation

**Note**: Both environments use the same codebase - only configuration differs!

### E2E Testing (Docker)

**Docker-based End-to-End Tests** - Validates production configuration with 42 comprehensive tests.

```bash
# Run all E2E tests (requires Docker running)
make test-e2e                                    # All 42 E2E tests (security + health + API)

# Run specific E2E test categories
make test-e2e-security                           # 12 security tests (non-root, read-only FS, capabilities)
make test-e2e-health                             # 14 health check tests (endpoints, uptime, OOM protection)
make test-e2e-api                                # 17 API workflow tests (validation, formats, performance)

# Fast testing with existing containers
make test-e2e-fast                               # Skip container rebuild (faster for iteration)

# Cleanup E2E test resources
make test-e2e-clean                              # Stop containers and remove volumes
```

**E2E Test Coverage**:
- **Security Validation**: Non-root user (UID 10001), read-only filesystem, dropped capabilities, resource limits
- **Health Monitoring**: Container uptime, health endpoints, restart policies, memory protection
- **API Workflows**: Query validation, error handling, response formats, performance benchmarks

**Implementation Details**: See `plan/02-completed/TESTING-MODERNIZATION-PLAN.md` Phase 3 for test architecture and `tests_new/e2e/` for test code.

### Dependency Management

**Dependabot** automatically checks for dependency updates weekly (Mondays at 09:00 CET) for:
- **Python dependencies** (pyproject.toml) - Grouped minor/patch updates, pinned major versions for critical packages
- **npm dependencies** (frontend/package.json) - Grouped updates, manual review for framework majors
- **GitHub Actions** - Keeps workflow actions up to date
- **Docker base images** - Updates base images in Dockerfiles

**Important**: Dependabot PRs require manual review and merge. NO automatic merging is configured for security.

**Pinned Dependencies** (no major version auto-updates):
- `sentence-transformers` (4.x) - Breaking changes in 5.x
- `chromadb` (1.x) - Stable version
- `torch` (2.x) - PyTorch major versions
- `typer` (0.16.x) - Feature compatibility
- `vue`, `vuetify`, `vite` - Framework stability

Configuration: `.github/dependabot.yml`

### Continuous Integration (CI)

Automated CI pipeline runs on all pull requests and pushes to `main`/`develop` branches:

**Python CI** (Python 3.9, 3.10, 3.11):
- ✓ Ruff format check (`ruff format --check`)
- ✓ Ruff linting (`ruff check`)
- ✓ mypy type checking (informational, doesn't fail)
- ✓ pytest with coverage reporting
- ✓ Coverage upload to Codecov

**Frontend CI** (Node 18):
- ✓ ESLint 9 linting (`npm run lint`)
- ✓ Prettier format check (`npm run format:check`)
- ✓ Vitest tests with coverage (`npm run test:coverage`)
- ✓ Production build verification
- ✓ Coverage upload to Codecov

**Docker Build Test**:
- ✓ API Docker image build
- ✓ Frontend Docker image build
- ✓ Multi-platform support (linux/amd64)
- ✓ Layer caching for faster builds

**Optimizations**:
- Change detection: Only runs jobs for modified code paths
- Dependency caching: uv cache, npm cache
- Concurrency control: Cancels outdated workflow runs
- Matrix testing: Multiple Python versions in parallel

Workflows: `.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml`

## Architecture Overview

### Core Components

- **Text Processing Pipeline** (`phentrieve/text_processing/`): Handles clinical text chunking, assertion detection, and HPO extraction
- **Embedding & Indexing** (`phentrieve/indexing/`, `phentrieve/embeddings.py`): ChromaDB-based vector storage with multilingual embedding support
- **Retrieval System** (`phentrieve/retrieval/`): Dense retrieval with optional cross-encoder re-ranking
- **Evaluation Framework** (`phentrieve/evaluation/`): Comprehensive benchmarking and metrics calculation

### Data Flow

1. Clinical text → Text processing pipeline → Semantic chunks with assertions
2. Chunks → Embedding → Vector search against HPO term database
3. Retrieval results → Re-ranking → Formatted output with text attribution

### Key Configuration

- **Main config**: `phentrieve.yaml` (created from `phentrieve.yaml.template`)
- **API config**: `api/local_api_config.env`
- **HPO data**: `data/hpo_data.db` - SQLite database containing all HPO terms, ontology structure, and graph metadata (generated via `phentrieve data prepare`)
- **Embeddings cache**: `data/hf_cache/` for Hugging Face model cache

### Caching Patterns

**Thread-Safe Caching with `@lru_cache`:**
- HPO graph data loading (`load_hpo_graph_data()`) - Thread-safe, cached ontology structure
- Configuration loading (`load_user_config()`) - Cached YAML configuration
- HPO label mapping (`_get_hpo_label_map_api()`) - Cached term labels
- ID normalization (`normalize_id()`) - Cached ID format conversions

**Implementation Pattern:**
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def expensive_function(param: str) -> DataType:
    """Function with cached results."""
    # Expensive operation here
    return result

# Clear cache when needed (e.g., in tests):
expensive_function.cache_clear()
```

**Benefits:**
- Thread-safe by default (Python 3.9+)
- No mutable global state
- Easy to test with `.cache_clear()`
- Pythonic and idiomatic

### Testing Architecture

**⚠️ CRITICAL: Test Directory Structure**

**NEVER CREATE `tests_new/` FOLDER!** All tests MUST be placed in the `tests/` directory only.

- ✅ **Correct**: `tests/unit/`, `tests/e2e/`, `tests/integration/`
- ❌ **WRONG**: `tests_new/` (do not create this folder under ANY circumstances)

If you accidentally create `tests_new/`, immediately move all files to `tests/` and delete `tests_new/`.

**Test Structure** (157 total tests, 13% statement coverage):
- **Unit/Integration Tests** (`tests/`): 115 tests covering core functionality
  - Text processing: Chunking, assertion detection, HPO extraction
  - Retrieval: Dense retriever (100% coverage), embeddings (100% coverage), re-ranker (100% coverage)
  - Utilities: Output formatters (100% coverage), semantic metrics
- **Docker E2E Tests** (`tests/e2e/`): 42 tests validating production deployment
  - 12 security tests (non-root, read-only FS, capabilities, limits)
  - 14 health tests (endpoints, uptime, restart policies, OOM)
  - 17 API workflow tests (validation, formats, performance)

**Coverage Highlights**:
- 4 modules at 100% coverage (embeddings.py, dense_retriever.py, reranker.py, output_formatters.py)
- 13% overall statement coverage (622/4916 statements)
- All tests passing with 0 linting errors, 0 type errors

**Testing Commands**: See Development Commands section for `make test`, `make test-cov`, and `make test-e2e*` commands.

**Documentation**: Full testing plan in `plan/02-completed/TESTING-MODERNIZATION-PLAN.md`

### Data Directory Structure

When first run, Phentrieve creates data directories (configurable in `phentrieve.yaml`):
- `data/hpo_data.db` - SQLite database containing all HPO terms (labels, definitions, synonyms, comments), ontology graph structure (ancestors, depths), and metadata for 19,534 terms (~12 MB, generated via `phentrieve data prepare`)
- `data/hp.json` - Source HPO JSON file downloaded from the HPO ontology repository
- `data/indexes/` - ChromaDB vector indexes for different embedding models
- `data/hf_cache/` - Hugging Face model cache for faster loading
- `data/results/` - Benchmark results and evaluation metrics

### Benchmark Test Data

**Location**: `tests/data/benchmarks/`

Benchmark datasets for evaluating HPO retrieval performance organized by language:
- `german/` - German-language datasets (current)
- `en/` - English-language datasets (future)
- Other languages as needed

**Available Datasets**:
- `tiny_v1.json` (9 cases) - Quick testing, used by default
- `70cases_gemini_v1.json` (70 cases) - Medium evaluation
- `200cases_gemini_v1.json` (200 cases) - Comprehensive evaluation

**Usage**:
```bash
# Use default tiny dataset
phentrieve benchmark run

# Specify dataset by relative path
phentrieve benchmark run --test-file german/70cases_gemini_v1.json
```

**In Tests**:
```python
def test_something(benchmark_data_dir):
    from phentrieve.data_processing.test_data_loader import load_test_data
    dataset = load_test_data(str(benchmark_data_dir / "german/tiny_v1.json"))
    assert len(dataset) == 9
```

See `tests/data/benchmarks/README.md` for complete dataset documentation.

### Frontend Architecture

- Vue 3 with Composition API
- Vuetify for UI components
- Pinia for state management
- Vue i18n for internationalization (supports EN, DE, ES, FR, NL)
- Vite for build tooling

### API Architecture

- FastAPI with automatic OpenAPI documentation at `/docs`
- Modular router structure in `api/routers/`
- Pydantic schemas in `api/schemas/` for request/response validation
- Dependency injection pattern in `api/dependencies.py` for model loading and caching
- Health check and configuration info endpoints
- API runs on port 8000 by default (configurable in `local_api_config.env`)

**HPO Term Details Feature** (Issue #24):
- API endpoints support `include_details` parameter to enrich results with definitions and synonyms
- **Database Layer**: `HPODatabase.get_terms_by_ids()` - Batch query for efficient term lookup
- **Enrichment Layer**: `enrich_results_with_details()` - Pure function with thread-safe `@lru_cache`
- **Performance**: Batch-loading optimization prevents N-query problem (1 query instead of N queries)
- **CLI Support**: `--include-details` flag for `phentrieve query` and `phentrieve text process`
- **Frontend**: Expandable UI in ResultsDisplay.vue with i18n support (EN, DE, ES, FR, NL)

### Important Architectural Notes

**Multilingual Support**: The system supports multiple languages throughout the pipeline:
- Embedding models are multilingual or can be configured per-language
- HPO translations stored in `api/hpo_translations/`
- Text processing adapts to different languages via spaCy models

**ChromaDB Indexing**: Vector indexes are built per embedding model and stored separately:
- Index names follow pattern: `{model_name_sanitized}`
- Supports multiple concurrent indexes for benchmarking
- Build new indexes with `phentrieve index build`

**Cross-Encoder Re-ranking**: Optional re-ranking stage for improved precision:
- Configurable in `phentrieve.yaml` with `enable_reranker: true`
- Default model: BAAI/bge-reranker-v2-m3 (568M params, 100+ languages)
- Protected two-stage retrieval: High-confidence dense matches (≥0.7) preserved from demotion

**Text Processing Pipeline Components**:
- Semantic chunking strategies (sentence-based, token-based)
- **ConText Assertion Detection**: medspaCy-based algorithm with direction-aware scope detection
  - Negation detection with FORWARD/BACKWARD/BIDIRECTIONAL directions
  - TERMINATE scope boundaries prevent negation leakage across conjunctions
  - PSEUDO false positive prevention
  - 122 ConText rules across 5 languages (EN, DE, ES, FR, NL)
  - Resolves issue #79 (missing German negation terms)
  - See `docs/advanced-topics/negation-detection.md` for comprehensive documentation
- HPO term extraction from clinical text
- All configurable through CLI flags or config file