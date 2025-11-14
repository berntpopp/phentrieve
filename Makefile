.PHONY: help format lint typecheck check test clean all install lock upgrade add remove clean-venv frontend-install frontend-lint frontend-format frontend-dev frontend-build docker-build docker-up docker-down docker-logs

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@echo "Phentrieve Development Makefile"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Python Development

install: ## Install package with uv
	uv sync

install-dev: ## Install package with optional dependencies
	uv sync --all-extras

install-editable: ## Install in editable mode (for development)
	uv pip install -e .

format: ## Format Python code with Ruff
	ruff format phentrieve/ api/ tests/

lint: ## Lint Python code with Ruff
	ruff check phentrieve/ api/ tests/

lint-fix: ## Lint and auto-fix Python code
	ruff check phentrieve/ api/ tests/ --fix

typecheck: ## Type check with mypy (incremental with SQLite cache)
	mypy phentrieve/ api/

typecheck-fast: ## Fast type check using mypy daemon (first run starts daemon)
	@echo "Using mypy daemon for faster checking..."
	@dmypy run -- phentrieve/ api/ || (echo "Starting mypy daemon..." && dmypy start && dmypy run -- phentrieve/ api/)

typecheck-daemon-stop: ## Stop mypy daemon
	dmypy stop

typecheck-fresh: ## Type check from scratch (clear cache first)
	rm -rf .mypy_cache/
	mypy phentrieve/ api/

check: format lint ## Format and lint code

test: ## Run tests with pytest
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=phentrieve --cov=api --cov-report=html --cov-report=term

##@ Package Management

lock: ## Update uv.lock file
	uv lock

upgrade: ## Upgrade dependencies
	uv lock --upgrade

add: ## Add a new dependency (usage: make add PACKAGE=package-name)
	uv add $(PACKAGE)

remove: ## Remove a dependency (usage: make remove PACKAGE=package-name)
	uv remove $(PACKAGE)

##@ Frontend Development

frontend-install: ## Install frontend dependencies
	cd frontend && npm install

frontend-lint: ## Lint frontend code
	cd frontend && npm run lint

frontend-format: ## Format frontend code with Prettier
	cd frontend && npm run format

frontend-dev: ## Run frontend development server
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	cd frontend && npm run build

frontend-test: ## Run frontend tests with Vitest
	cd frontend && npm run test:run

frontend-test-ui: ## Run frontend tests with UI
	cd frontend && npm run test:ui

frontend-test-cov: ## Run frontend tests with coverage
	cd frontend && npm run test:coverage

##@ Docker

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-dev: ## Start development Docker stack
	docker-compose -f docker-compose.dev.yml up

##@ Cleaning

clean: ## Remove build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

clean-venv: ## Remove virtual environment
	rm -rf .venv

clean-data: ## Clean data caches (use with caution)
	@echo "⚠️  This will remove ChromaDB indices and model caches"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/chromadb_*
	rm -rf data/hf_cache/

##@ All-in-one

check-all: check frontend-lint frontend-test ## Check and test both Python and frontend code

all: clean check test ## Clean, check, and test Python code
