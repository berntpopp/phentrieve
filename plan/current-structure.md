# Current System Structure

**Date:** 2025-11-14
**Branch:** main (feature/phase-0-preparation)
**Commit:** 3cffbae (before structure changes)
**Purpose:** Reference documentation for troubleshooting during modernization

---

## Project Overview

**Project Name:** phentrieve
**Version:** 0.2.0
**Python Requirement:** >=3.9
**Current Python Version:** 3.10.14
**License:** MIT

**Description:** A CLI tool for retrieving Human Phenotype Ontology (HPO) terms from clinical text using multilingual embeddings.

---

## Directory Structure

```
phentrieve/
├── .github/
│   └── workflows/
│       └── deploy-docs.yml          # GitHub Actions: Deploy documentation
├── api/                             # FastAPI backend
│   ├── routers/
│   │   ├── config_info_router.py
│   │   ├── health.py
│   │   ├── query_router.py
│   │   ├── similarity_router.py
│   │   └── text_processing_router.py
│   ├── schemas/
│   │   ├── config_info_schemas.py
│   │   ├── query_schemas.py
│   │   ├── similarity_schemas.py
│   │   └── text_processing_schemas.py
│   ├── dependencies.py
│   ├── main.py
│   └── run_api_local.py             # Local development server
├── data/                            # Data directory (not in version control)
│   ├── hpo_core_data/               # HPO ontology files
│   └── hf_cache/                    # Hugging Face model cache
├── frontend/                        # Vue.js frontend
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── locales/                 # i18n translations (EN, DE, ES, FR, NL)
│   │   ├── router/
│   │   ├── stores/                  # Pinia state management
│   │   ├── views/
│   │   ├── App.vue
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
├── phentrieve/                      # Core Python package
│   ├── cli/                         # CLI commands (Typer)
│   │   ├── benchmark_commands.py
│   │   ├── query_commands.py
│   │   ├── similarity_commands.py
│   │   ├── text_commands.py
│   │   └── utils.py
│   ├── data_processing/             # HPO data processing
│   │   ├── document_creator.py
│   │   └── hpo_parser.py
│   ├── evaluation/                  # Benchmarking and metrics
│   │   ├── benchmark_orchestrator.py
│   │   ├── comparison_orchestrator.py
│   │   ├── full_text_runner.py
│   │   ├── metrics.py
│   │   ├── result_analyzer.py
│   │   ├── runner.py
│   │   └── semantic_metrics.py
│   ├── indexing/                    # ChromaDB indexing
│   │   ├── chromadb_indexer.py
│   │   └── chromadb_orchestrator.py
│   ├── retrieval/                   # Dense retrieval and reranking
│   │   ├── api_helpers.py
│   │   ├── dense_retriever.py
│   │   ├── output_formatters.py
│   │   ├── output_formatters_new.py
│   │   ├── query_orchestrator.py
│   │   └── reranker.py
│   ├── text_processing/             # Clinical text processing
│   │   ├── assertion_detection.py
│   │   ├── chunkers.py
│   │   ├── hpo_extraction_orchestrator.py
│   │   └── pipeline.py
│   ├── visualization/               # Plotting utilities
│   │   └── plot_utils.py
│   ├── __init__.py
│   ├── __main__.py                  # Entry point for `python -m phentrieve`
│   ├── config.py                    # Configuration constants
│   ├── embeddings.py                # Embedding utilities
│   └── utils.py                     # Utility functions
├── plan/                            # Migration planning documents (Phase 0)
│   ├── baselines.md
│   ├── current-structure.md         # This file
│   ├── MASTER-PLAN.md
│   ├── phase-0-preparation.md
│   ├── phase-1-ruff-migration.md
│   ├── README.md
│   ├── requirements-baseline.txt
│   ├── ruff-pilot-report.md
│   ├── STATUS.md
│   ├── tooling-modernization-critique.md
│   └── uv-pilot-report.md
├── scripts/                         # Rollback and utility scripts (Phase 0)
│   ├── README.md
│   ├── rollback-phase-1.sh
│   └── rollback-phase-2.sh
├── tests/                           # Test suite (pytest)
│   ├── test_chunkers.py
│   ├── test_semantic_metrics.py
│   └── test_text_processing.py
├── .gitignore
├── CLAUDE.md                        # Claude Code instructions
├── docker-compose.dev.yml           # Docker development setup
├── docker-compose.yml               # Docker production setup
├── LICENSE
├── pyproject.toml                   # Project metadata and tool configs
├── README.md
└── phentrieve.yaml.template         # Configuration template

Total Python Files: 71 (as of Phase 0 baseline)
Total Lines of Code: ~15,000+ (estimate)
```

---

## Python Package Structure

### Core Components

#### 1. **CLI (`phentrieve/cli/`)**
- **Framework:** Typer with decorators
- **Entry Point:** `phentrieve` command (via `pyproject.toml`)
- **Commands:**
  - `phentrieve query` - Query HPO terms
  - `phentrieve text` - Text processing
  - `phentrieve similarity` - Term similarity
  - `phentrieve benchmark` - Run benchmarks

#### 2. **Text Processing (`phentrieve/text_processing/`)**
- **Purpose:** Clinical text chunking and assertion detection
- **Dependencies:** pysbd, spaCy (optional)
- **Key Classes:**
  - `SlidingWindowChunker` - Sentence-based chunking
  - `SemanticChunker` - Embedding-based chunking
  - `CombinedAssertionDetector` - Negation/uncertainty detection
  - `TextProcessingPipeline` - Orchestrates processing

#### 3. **Indexing (`phentrieve/indexing/`)**
- **Database:** ChromaDB (vector database)
- **Purpose:** Store and index HPO term embeddings
- **Key Functions:**
  - `index_chromadb()` - Create ChromaDB index
  - `generate_collection_name()` - Collection naming
  - Model-specific indexing

#### 4. **Retrieval (`phentrieve/retrieval/`)**
- **Purpose:** Dense retrieval with optional reranking
- **Components:**
  - `DenseRetriever` - Vector similarity search
  - `CrossEncoderReranker` - Reranking with cross-encoders
  - `QueryOrchestrator` - End-to-end query processing
- **Output Formats:** Text, JSON, JSONL

#### 5. **Evaluation (`phentrieve/evaluation/`)**
- **Purpose:** Benchmarking and metrics calculation
- **Metrics:**
  - MRR (Mean Reciprocal Rank)
  - Hit Rate @ K
  - Precision, Recall, F1
  - Ontological similarity
- **Key Files:**
  - `runner.py` - Main benchmark runner
  - `metrics.py` - Metric calculations
  - `result_analyzer.py` - Result analysis

#### 6. **Data Processing (`phentrieve/data_processing/`)**
- **Purpose:** Parse and process HPO ontology files
- **Key Functions:**
  - `parse_hpo_obo()` - Parse OBO format
  - `create_chromadb_documents()` - Create vector DB documents

---

## Configuration

### pyproject.toml

**Current Tool Configurations:**

#### Project Metadata
```toml
[project]
name = "phentrieve"
version = "0.2.0"
requires-python = ">=3.9"

[project.scripts]
phentrieve = "phentrieve.cli:app"
```

#### Black (Current Formatter)
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | env
  | venv
)/
'''
force-exclude = '''
/(
  __pycache__
)/
'''
```

#### Pytest
```toml
[tool.pytest.ini_options]
python_files = ["test_*.py", "*_test.py", "tests.py"]
testpaths = ["tests"]
norecursedirs = [
    ".*",       # Ignore hidden directories
    "build",
    "dist",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
]
```

#### Setuptools
```toml
[tool.setuptools]

[tool.setuptools.packages.find]
where = ["."]
include = ["phentrieve*", "api*"]
exclude = ["tests*", "build*", "dist*", ".venv*"]
```

**No Current Configurations For:**
- ❌ Ruff (will be added in Phase 1)
- ❌ mypy (will be added in Phase 3)
- ❌ uv (will be added in Phase 2)

---

## Dependencies

### Core Dependencies (from pyproject.toml)

**AI/ML:**
- `sentence-transformers>=2.2.2` - Embeddings (CURRENT: 4.1.0 baseline)
- `torch>=2.0.0` - PyTorch (CURRENT: 2.6.0 baseline)
- `numpy>=1.23.0` - Numerical computing

**Database:**
- `chromadb>=0.4.18` - Vector database (CURRENT: 1.0.6 baseline)

**Text Processing:**
- `pysbd>=0.3.4` - Sentence boundary detection
- `spacy>=3.6.0` (optional) - NLP processing

**CLI/UI:**
- `typer[all]>=0.9.0` - CLI framework (CURRENT: 0.16.0 baseline)
- `tqdm>=4.66.1` - Progress bars

**Data/Analysis:**
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization

**Configuration:**
- `PyYAML>=6.0` - YAML parsing
- `requests>=2.31.0` - HTTP requests

### Development Dependencies (Implicit)

**Currently Installed (from baseline):**
- `black==25.1.0` - Code formatter
- `ruff==0.14.1` - Linter (installed for testing)
- `pytest` - Testing framework

**Not Currently Installed:**
- ❌ `mypy` - Type checker
- ❌ `bandit` - Security linter
- ❌ `flake8` - Style guide enforcement
- ❌ `isort` - Import sorting

**Total Packages (from baseline):** 453

---

## Frontend Structure

### Vue.js Frontend

**Framework:** Vue 3 (Composition API)
**Build Tool:** Vite 4.4.11
**Version:** 0.1.0

#### Key Dependencies
```json
{
  "dependencies": {
    "vue": "^3.3.4",
    "vuetify": "^3.3.23",
    "pinia": "^3.0.2",
    "vue-i18n": "^9.13.0",
    "vue-router": "^4.2.5",
    "axios": "^1.6.2"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^4.4.0",
    "vite": "^4.4.11",
    "eslint": "^8.49.0",
    "eslint-plugin-vue": "^9.17.0"
  }
}
```

#### Current Linting Setup
- **ESLint:** 8.49.0 (OLD format, not flat config)
- **Config:** `.eslintrc` style (pre-ESLint 9)
- **Plugin:** `eslint-plugin-vue` 9.17.0

#### Scripts
```json
{
  "dev": "vite",
  "build": "vite build",
  "preview": "vite preview",
  "lint": "eslint . --ext .vue,.js,.jsx,.cjs,.mjs --fix --ignore-path .gitignore"
}
```

#### Internationalization (i18n)
- **Supported Languages:** EN, DE, ES, FR, NL
- **Location:** `frontend/src/locales/`

---

## API Structure

### FastAPI Backend

**Framework:** FastAPI
**Python Version:** 3.10.14
**Location:** `api/`

#### Routers
1. **Health (`health.py`)** - Health check endpoint
2. **Config Info (`config_info_router.py`)** - Configuration information
3. **Query (`query_router.py`)** - HPO term queries
4. **Similarity (`similarity_router.py`)** - Term similarity calculations
5. **Text Processing (`text_processing_router.py`)** - Text processing pipeline

#### Schemas (Pydantic)
- `QuerySchema` - Query request/response models
- `SimilaritySchema` - Similarity request/response models
- `TextProcessingSchema` - Text processing models
- `ConfigInfoSchema` - Configuration info models

#### Local Development
- **Script:** `api/run_api_local.py`
- **Config:** `api/local_api_config.env` (not in version control)

---

## CI/CD

### GitHub Actions

**Current Workflows:**
1. **`deploy-docs.yml`** - Deploy documentation
   - **Trigger:** Manual or on push
   - **Purpose:** Documentation deployment

**Missing CI/CD:**
- ❌ No automated testing workflow
- ❌ No linting workflow
- ❌ No type checking workflow
- ❌ No security scanning workflow
- ❌ No Docker build/push workflow

**Note:** CI/CD optimization is Phase 9 of the migration plan.

---

## Docker Setup

### Docker Compose

**Files:**
- `docker-compose.yml` - Production setup
- `docker-compose.dev.yml` - Development setup

**Services:**
- Frontend (Vue.js)
- Backend (FastAPI)
- Database (if applicable)

**Registry:**
- **Current:** DockerHub (will migrate to GHCR in Phase 7)

---

## Testing

### Pytest Configuration

**Test Location:** `tests/`
**Test Files:**
- `test_chunkers.py` - Chunking functionality
- `test_semantic_metrics.py` - Semantic metrics
- `test_text_processing.py` - Text processing pipeline

**Configuration:** See `pyproject.toml` `[tool.pytest.ini_options]`

**Coverage:** Unknown (not measured in baseline)

**Missing Tests:**
- ❌ API endpoint tests
- ❌ Frontend component tests (Vitest in Phase 5)
- ❌ Integration tests

---

## Build System

### Current Build Commands

**Python:**
```bash
# Install in development mode
pip install -e .

# Install with text processing dependencies
pip install -e ".[text_processing]"

# Format code
black phentrieve/ api/ tests/

# Run tests
pytest tests/
```

**Frontend:**
```bash
cd frontend/
npm install
npm run dev          # Development
npm run build        # Production build
npm run lint         # ESLint
```

**Docker:**
```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose up
```

**No Makefile:** ❌ Will be created in Phase 6

---

## Tooling Summary

### Currently Used Tools

| Tool | Purpose | Version | Status |
|------|---------|---------|--------|
| **Black** | Python formatter | 25.1.0 | ✅ Active |
| **Pytest** | Python testing | (installed) | ✅ Active |
| **ESLint** | JS/Vue linting | 8.49.0 | ✅ Active (old config) |
| **Vite** | Frontend build | 4.4.11 | ✅ Active |
| **Docker** | Containerization | (installed) | ✅ Active |
| **GitHub Actions** | CI/CD | N/A | ⚠️ Minimal |

### Tools to Add/Migrate

| Tool | Purpose | Phase | Status |
|------|---------|-------|--------|
| **Ruff** | Python linter/formatter | Phase 1 | ⏳ Pending |
| **uv** | Python package manager | Phase 2 | ⏳ Pending |
| **mypy** | Python type checker | Phase 3 | ⏳ Pending |
| **ESLint 9** | JS/Vue linting (flat config) | Phase 4 | ⏳ Pending |
| **Prettier** | Frontend formatting | Phase 4 | ⏳ Pending |
| **Vitest** | Frontend testing | Phase 5 | ⏳ Pending |
| **Makefile** | Unified build system | Phase 6 | ⏳ Pending |
| **GHCR** | Container registry | Phase 7 | ⏳ Pending |
| **Dependabot** | Dependency updates | Phase 8 | ⏳ Pending |

---

## Known Issues (Pre-Migration)

### From Ruff Pilot Report

1. **F823: Undefined local variable** (1 occurrence)
   - **File:** `phentrieve/retrieval/query_orchestrator.py:452`
   - **Severity:** HIGH (potential runtime bug)
   - **Status:** ⚠️ Must fix before or during Phase 1

2. **F841: Unused variables** (14 occurrences)
   - **Severity:** Medium (dead code)
   - **Status:** ⏳ Will fix in Phase 1

3. **F401: Unused imports** (57 occurrences)
   - **Severity:** Low (code cleanliness)
   - **Status:** ⏳ Auto-fix in Phase 1

4. **E402: Module imports not at top** (5 occurrences)
   - **File:** `phentrieve/cli/text_commands.py`
   - **Severity:** Low (style violation)
   - **Status:** ⏳ Will fix in Phase 1

5. **F811: Redefined while unused** (5 occurrences)
   - **Severity:** Medium (confusing code)
   - **Status:** ⏳ Auto-fix in Phase 1

### From uv Pilot Report

1. **sentence-transformers 4.1.0 → 5.1.2** (Major version bump)
   - **Risk:** Medium (API changes possible)
   - **Status:** ⏳ Will pin in Phase 2

2. **typer 0.16.0 → 0.9.4** (Downgrade!)
   - **Risk:** Medium (CLI behavior changes)
   - **Status:** ⏳ Will investigate in Phase 2

---

## Performance Baselines

### Python

**Black Format Time:** 5.978 seconds (71 files)
**Ruff Format Time:** 0.318 seconds (71 files) - **18.8x faster**

**Total Packages:** 453 (from `pip freeze`)

### Frontend

**Not measured in Phase 0** (Phase 4 scope)

---

## Configuration Files

### Key Files to Preserve

1. **`pyproject.toml`** - Project metadata and tool configs
2. **`phentrieve.yaml.template`** - Application config template
3. **`.gitignore`** - Git ignore patterns
4. **`docker-compose.yml`** - Docker production setup
5. **`docker-compose.dev.yml`** - Docker development setup
6. **`frontend/vite.config.js`** - Vite configuration
7. **`frontend/package.json`** - Frontend dependencies

### Files to Modify in Migration

**Phase 1 (Ruff):**
- ✏️ `pyproject.toml` - Add Ruff config, remove Black

**Phase 2 (uv):**
- ✏️ `pyproject.toml` - Update dependencies format
- ➕ `uv.lock` - Create lockfile

**Phase 3 (mypy):**
- ✏️ `pyproject.toml` - Add mypy config

**Phase 4 (Frontend):**
- ✏️ `frontend/package.json` - Update ESLint, add Prettier
- ➕ `frontend/eslint.config.js` - Flat config
- ➕ `frontend/.prettierrc` - Prettier config

**Phase 5 (Vitest):**
- ✏️ `frontend/package.json` - Add Vitest
- ✏️ `frontend/vite.config.js` - Add Vitest config

**Phase 6 (Makefile):**
- ➕ `Makefile` - Main Makefile
- ➕ `mk/python.mk` - Python targets
- ➕ `mk/frontend.mk` - Frontend targets
- ➕ `mk/docker.mk` - Docker targets

**Phase 7 (GHCR):**
- ✏️ `docker-compose.yml` - Update registry
- ✏️ `.github/workflows/` - Update workflows

**Phase 8 (Dependabot):**
- ➕ `.github/dependabot.yml` - Dependabot config

**Phase 9 (CI/CD):**
- ✏️ `.github/workflows/` - Optimize workflows

---

## Critical Paths for Rollback

### Files to Backup Before Migration

**Phase 1:**
- `pyproject.toml` (Black config)
- All Python source files (for formatting revert)

**Phase 2:**
- `plan/requirements-baseline.txt` ✅ (already created)
- `pyproject.toml` (dependencies format)
- `.venv/` (can recreate from baseline)

**Backup Strategy:**
- ✅ Git feature branches for each phase
- ✅ Baseline requirements saved in `plan/`
- ✅ Rollback scripts in `scripts/`
- ✅ Comprehensive documentation in `plan/`

---

## Version Control

**Current Branch:** feature/phase-0-preparation
**Main Branch:** main
**Remote:** GitHub

**Protected Files (Do Not Delete):**
- `plan/requirements-baseline.txt` - Critical for Phase 2 rollback
- `plan/baselines.md` - Performance baseline reference
- `plan/uv-pilot-report.md` - Version resolution reference
- `plan/ruff-pilot-report.md` - Code quality baseline

---

## Next Steps

**After Phase 0 completion:**
1. Review and merge Phase 0 PR
2. Begin Phase 1: Ruff Migration
3. Fix known issues (F823, F841, etc.)
4. Run comprehensive tests
5. Continue through Phases 2-9

**Reference Documents:**
- `plan/MASTER-PLAN.md` - Overall migration strategy
- `plan/phase-1-ruff-migration.md` - Next phase details
- `scripts/README.md` - Rollback procedures

---

**Document Status:** Complete ✅
**Last Updated:** 2025-11-14
**Maintained By:** Phase 0 preparation
**Purpose:** Troubleshooting reference for Phases 1-9
