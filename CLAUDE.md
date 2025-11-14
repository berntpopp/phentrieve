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
cd frontend/
npm install
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build
npm run lint         # ESLint with auto-fix
```

### API (FastAPI)
```bash
cd api/
python run_api_local.py    # Run local development server
```

### Docker Development
```bash
# Full stack development
docker-compose -f docker-compose.dev.yml up

# Production deployment
docker-compose up
```

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