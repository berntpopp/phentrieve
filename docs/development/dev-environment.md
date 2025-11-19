# Development Environment

This page explains how to set up a development environment for contributing to the Phentrieve project using modern Python tooling.

## Prerequisites

- **Python 3.10+** (Required for modern type hints)
- **Git** for version control
- **uv** (Recommended package manager - 10-100x faster than pip)
- **Node.js 18+** and npm (for frontend development)
- **Docker** (optional, for containerized development)
- **VSCode** or **PyCharm** (recommended IDEs)

## Toolchain

Phentrieve uses modern Python tooling for fast development and high code quality:

*   **uv**: Extremely fast package installer and resolver (written in Rust)
*   **Ruff**: Fast linting and formatting (100x faster than flake8 + black)
*   **mypy**: Static type checking for type safety
*   **pytest**: Testing framework with fixtures and parametrization
*   **Typer**: Modern CLI framework with rich help and validation

## Quick Start (Makefile)

The project includes a `Makefile` to automate common tasks.

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies including dev tools
make install-dev
```

This runs `uv sync --all-extras`, which:
- Creates a `.venv/` virtual environment automatically
- Installs core dependencies
- Installs API extras (FastAPI, ChromaDB)
- Installs text processing extras (spaCy models)
- Installs dev tools (Ruff, mypy, pytest)

### 2. Run Code Quality Checks

```bash
# Format code with Ruff (auto-fixes)
make format

# Lint code with Ruff (checks for issues)
make lint

# Auto-fix linting issues
make lint-fix

# Type check with mypy (fast daemon mode)
make typecheck-fast

# Or run all checks at once
make check
```

**IMPORTANT**: Always run `make check` before committing! CI will fail if code is not formatted/linted.

### 3. Run Tests

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov

# Run specific test file
pytest tests/test_chunking.py

# Run tests matching a pattern
pytest -k "test_semantic"
```

## Development Workflow

### Daily Workflow

```bash
# 1. Pull latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make code changes

# 4. Format and lint (auto-fix)
make format
make lint-fix

# 5. Type check (catches type errors early)
make typecheck-fast

# 6. Run tests
make test

# 7. Commit changes
git add .
git commit -m "feat: Add new feature"

# 8. Push and create PR
git push -u origin feature/your-feature-name
```

### Pre-Commit Checklist

**MANDATORY before every commit:**

```bash
# All-in-one check (recommended)
make all

# Or run individually:
make check          # Format + lint
make typecheck-fast # Type checking
make test           # Run tests
```

**Why this matters:**
- CI/CD will fail if checks don't pass
- Prevents broken code from being merged
- Maintains codebase quality standards

## Local Development Server (Fast - No Docker)

**⚡ 100x faster than Docker with instant hot reload!**

### Terminal 1: Start API Server

```bash
make dev-api
# API runs on http://localhost:8734
# API Docs at http://localhost:8734/docs
# Hot reload: <1s on .py file changes
```

### Terminal 2: Start Frontend

```bash
make dev-frontend
# Frontend runs on http://localhost:5734
# HMR: <50ms on .vue/.ts file changes
```

### Performance Benefits

| Metric | Docker | Native Local | Improvement |
|--------|--------|--------------|-------------|
| **Cold Start** | 5-10 min | 2-3 sec | 100-200x faster |
| **API Reload** | 3-5 sec | <1 sec | 5x faster |
| **Frontend HMR** | 2-4 sec | <50ms | 40-80x faster |

## Package Management with uv

### Common Operations

```bash
# Update lockfile after manual pyproject.toml edits
make lock

# Upgrade all dependencies to latest compatible versions
make upgrade

# Add a new dependency
make add PACKAGE=requests

# Add a dev dependency
uv add --dev pytest-cov

# Remove a dependency
make remove PACKAGE=requests

# Install optional extras individually
uv sync --extra api           # API dependencies only
uv sync --extra text          # Text processing only
uv sync --all-extras          # Everything (recommended for dev)
```

### Why uv?

- **10-100x faster** than pip for installing packages
- **Lockfile** (`uv.lock`) ensures reproducible builds
- **Better dependency resolution** catches conflicts early
- **Automatic venv management** no manual virtualenv creation
- **Drop-in replacement** for pip/pip-tools

## Frontend Development Setup

For working on the Vue 3 frontend:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Development server with HMR
npm run dev

# Or use Makefile from project root
make frontend-dev
```

### Frontend Commands

```bash
make frontend-lint          # ESLint 9 check
make frontend-format        # Prettier format
make frontend-test          # Run Vitest tests
make frontend-test-cov      # Tests with coverage
make frontend-build         # Production build
```

## IDE Configuration

### VSCode (Recommended)

**.vscode/settings.json:**
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "none",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll": true,
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.rulers": [88, 100]
  },
  "mypy-type-checker.importStrategy": "fromEnvironment",
  "ruff.nativeServer": true
}
```

**Recommended Extensions:**
- **Python** (ms-python.python)
- **Ruff** (charliermarsh.ruff) - Linting and formatting
- **mypy Type Checker** (ms-python.mypy-type-checker)
- **Pylance** (ms-python.vscode-pylance) - IntelliSense
- **Volar** (Vue.volar) - Vue 3 support
- **Docker** (ms-azuretools.vscode-docker)

### PyCharm

1. **Open project** in PyCharm
2. **Settings → Project → Python Interpreter**
3. **Add Interpreter → Existing**
4. **Select** `.venv/bin/python`
5. **Enable**:
   - Settings → Tools → Python Integrated Tools → Testing → pytest
   - Settings → Tools → Actions on Save → Reformat code
   - Settings → Editor → Inspections → mypy

**External Tools Configuration:**

Add Ruff as external tool:
- Program: `.venv/bin/ruff`
- Arguments: `check --fix $FilePath$`
- Working directory: `$ProjectFileDir$`

## Working with Git

### Branching Strategy

Phentrieve uses a feature branch workflow:

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Create bugfix branch
git checkout -b fix/issue-description

# Create documentation branch
git checkout -b docs/update-readme
```

### Commit Message Convention

Follow conventional commits:

```bash
# Feature
git commit -m "feat: Add HPO term caching"

# Bug fix
git commit -m "fix: Correct assertion detection for negation"

# Documentation
git commit -m "docs: Update installation guide for uv"

# Performance
git commit -m "perf: Optimize SQLite query with index"

# Refactor
git commit -m "refactor: Simplify chunker pipeline"

# Tests
git commit -m "test: Add E2E tests for Docker deployment"
```

### Creating Pull Requests

```bash
# Push branch
git push -u origin feature/your-feature-name

# Create PR via GitHub CLI
gh pr create --title "feat: Your feature" --body "Description..."

# Or create PR via web interface
# https://github.com/berntpopp/phentrieve/compare
```

## Environment Variables

Create a `.env` file in the project root for local development:

```bash
# Data directories (relative to project root)
PHENTRIEVE_DATA_ROOT_DIR=./data
PHENTRIEVE_DATA_DIR=./data
PHENTRIEVE_INDEX_DIR=./data/indexes
PHENTRIEVE_RESULTS_DIR=./data/results

# Logging
PHENTRIEVE_LOG_LEVEL=DEBUG

# Development mode (enables extra logging)
PHENTRIEVE_DEV_MODE=true

# GPU (optional)
CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

**Note**: The API uses `api/local_api_config.env` for its own configuration.

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=phentrieve --cov-report=html

# Specific test file
pytest tests/test_hpo_database.py

# Specific test function
pytest tests/test_hpo_database.py::test_load_all_terms

# Tests matching pattern
pytest -k "database"

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Types

- **Unit tests**: `tests/unit/` - Test individual functions/classes
- **Integration tests**: `tests/integration/` - Test component interactions
- **E2E tests**: `tests_new/e2e/` - Docker-based end-to-end tests

### Writing Tests

```python
import pytest
from phentrieve.data_processing.hpo_database import HPODatabase

def test_database_initialization():
    """Test that database initializes correctly."""
    db = HPODatabase(":memory:")
    db.initialize_schema()

    assert db.get_term_count() == 0
    assert db.get_schema_version() == 1
```

## Type Checking

### Running mypy

```bash
# Fast (daemon mode, recommended)
make typecheck-fast

# Full check (clears cache)
make typecheck-fresh

# Stop mypy daemon
make typecheck-daemon-stop
```

### Type Hints Best Practices

```python
from typing import Any
from pathlib import Path

def load_terms(db_path: Path | str) -> list[dict[str, Any]]:
    """
    Load terms from database.

    Args:
        db_path: Path to database file

    Returns:
        List of term dictionaries
    """
    ...
```

## GPU Development

### Checking GPU Availability

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check device
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Using Specific GPU

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Disable GPU (use CPU only)
export CUDA_VISIBLE_DEVICES=-1
```

## Docker Development

### Building and Running

```bash
# Build locally
make docker-build

# Start development stack
make docker-dev

# View logs
make docker-logs

# Stop containers
make docker-down
```

### Debugging Docker

```bash
# Shell into API container
docker exec -it phentrieve-api-1 /bin/bash

# Check API logs
docker logs phentrieve-api-1 --tail 100 -f

# Check resource usage
docker stats
```

## Troubleshooting

### uv Installation Issues

```bash
# Manual install via pip
pip install uv

# Or use pipx
pipx install uv

# Verify installation
uv --version
```

### Virtual Environment Not Activated

```bash
# Activate manually
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Port Conflicts

```bash
# Check what's using port 8734 (API)
lsof -i :8734

# Kill process
kill -9 <PID>
```

### mypy Daemon Issues

```bash
# Stop and restart daemon
make typecheck-daemon-stop
make typecheck-fast
```

## Getting Help

- **Documentation**: Browse `docs/` directory
- **GitHub Issues**: https://github.com/berntpopp/phentrieve/issues
- **Discussions**: https://github.com/berntpopp/phentrieve/discussions
- **CLI Help**: `phentrieve --help` or `phentrieve <command> --help`

!!! tip "Hot Reload"
    Use local development mode (`make dev-api` + `make dev-frontend`) for instant feedback. Docker is 100x slower for development iteration.

!!! warning "Pre-Commit Checks"
    ALWAYS run `make all` before committing. CI will fail if you don't run checks locally first.
