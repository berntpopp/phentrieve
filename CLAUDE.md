# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Phentrieve is an AI-powered system for mapping clinical text to Human Phenotype Ontology (HPO) terms using a Retrieval-Augmented Generation (RAG) approach. It consists of three main components:

1. **Python CLI/Library** (`phentrieve/`) - Core functionality for text processing, HPO term retrieval, and benchmarking
2. **FastAPI Backend** (`api/`) - REST API exposing Phentrieve functionality
3. **Vue.js Frontend** (`frontend/`) - Web interface for interactive HPO term mapping

## Development Commands

### Python CLI/Library
```bash
# Install in development mode (using uv)
make install                                         # Install package (uv sync)
make install-dev                                     # Install with all extras (uv sync --all-extras)

# Development workflow (recommended)
make check                                           # Format + lint code
make test                                            # Run tests
make all                                             # Clean + check + test

# Code quality
make format                                          # Format with Ruff
make lint                                            # Lint with Ruff
make lint-fix                                        # Lint and auto-fix issues
make typecheck                                       # Type check with mypy

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
- **HPO data**: `data/hpo_core_data/` contains processed HPO ontology files
- **Embeddings cache**: `data/hf_cache/` for Hugging Face model cache

### Testing

- Test files in `tests/` directory
- Focus on text processing components, chunking, and semantic metrics
- Use `pytest` for running tests
- Run specific tests with `pytest tests/test_file.py` or pattern matching with `-k`
- No current test coverage for API or frontend components

### Data Directory Structure

When first run, Phentrieve creates data directories (configurable in `phentrieve.yaml`):
- `data/hpo_core_data/` - Processed HPO ontology files, terms, and graph data
- `data/indexes/` - ChromaDB vector indexes for different embedding models
- `data/hf_cache/` - Hugging Face model cache for faster loading
- `data/results/` - Benchmark results and evaluation metrics

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
- Multiple re-ranker models supported (multilingual NLI, biomedical, domain-specific)
- Two modes: `cross-lingual` and `monolingual`

**Text Processing Pipeline Components**:
- Semantic chunking strategies (sentence-based, token-based)
- Assertion detection (negation, uncertainty, family history)
- HPO term extraction from clinical text
- All configurable through CLI flags or config file