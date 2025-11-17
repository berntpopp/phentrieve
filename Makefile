.PHONY: help format lint typecheck check test clean all install install-text-processing lock upgrade add remove clean-venv frontend-install frontend-lint frontend-format frontend-dev frontend-build docker-build docker-up docker-down docker-logs dev-api dev-frontend dev-all test-api test-api-cov test-e2e test-e2e-security test-e2e-health test-e2e-api test-e2e-fast test-e2e-clean test-e2e-logs test-e2e-shell cov-package cov-api cov-frontend cov-all

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

install-dev: ## Install package with all optional dependencies (includes text_processing)
	uv sync --all-extras

install-text-processing: ## Install text processing dependencies (spaCy + model)
	uv sync --extra text_processing

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
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

##@ Local Development (Fast - No Docker)

dev-api: ## Start API with hot reload (custom port 8734)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  FastAPI Development Server (Hot Reload)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "API:       http://localhost:8734 (custom port - HPOD)"
	@echo "API Docs:  http://localhost:8734/docs"
	@echo ""
	@echo "Hot Reload: Enabled (<1s on .py file changes)"
	@echo "Press CTRL+C to stop"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@cd api && python run_api_local.py

dev-frontend: ## Start frontend with Vite HMR (custom port 5734)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Vite Development Server (Hot Module Replacement)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Frontend:  http://localhost:5734 (custom port - matches HPOD)"
	@echo ""
	@echo "HMR: Enabled (<50ms on file changes)"
	@echo "Press CTRL+C to stop"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@cd frontend && npm run dev

dev-all: ## Display instructions to start both API and frontend
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Fast Local Development Environment"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Open 2 terminals and run:"
	@echo ""
	@echo "  Terminal 1 (API):"
	@echo "    $$ make dev-api"
	@echo "    → http://localhost:8734 (custom HPOD port)"
	@echo "    → http://localhost:8734/docs (OpenAPI)"
	@echo ""
	@echo "  Terminal 2 (Frontend):"
	@echo "    $$ make dev-frontend"
	@echo "    → http://localhost:5734 (custom HPOD port)"
	@echo ""
	@echo "Custom Ports (avoid conflicts with other dev tools):"
	@echo "  • API: 8734 (HPOD - HPO Database on phone keypad)"
	@echo "  • Frontend: 5734 (matches API pattern)"
	@echo ""
	@echo "Performance:"
	@echo "  • API hot reload: <1s on .py changes"
	@echo "  • Frontend HMR: <50ms on .vue/.ts changes"
	@echo "  • 100x faster than Docker development"
	@echo ""
	@echo "First time setup:"
	@echo "  $$ ./scripts/setup-local-dev.sh"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

##@ CI/CD Pipeline (Local-First Best Practice)

ci: ci-python ci-frontend ## Run full CI pipeline locally (matches GitHub Actions 1:1)

ci-python: ## Run Python CI checks (matches GitHub Actions)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Python CI Pipeline"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Running Python CI checks (same as GitHub Actions)..."
	@echo ""
	@echo "[1/5] Install dependencies..."
	@uv sync --all-extras --dev
	@echo "✅ Dependencies installed"
	@echo ""
	@echo "[2/5] Ruff format check..."
	@ruff format --check phentrieve/ api/ tests/ || (echo "❌ Format check failed. Run: make format" && exit 1)
	@echo "✅ Format check passed"
	@echo ""
	@echo "[3/5] Ruff lint check..."
	@ruff check phentrieve/ api/ tests/ || (echo "❌ Lint check failed. Run: make lint-fix" && exit 1)
	@echo "✅ Lint check passed"
	@echo ""
	@echo "[4/5] mypy type check..."
	@uv run mypy phentrieve/ api/ || echo "⚠️  Type errors found (non-blocking)"
	@echo ""
	@echo "[5/5] pytest with coverage (unit + integration only)..."
	@uv run pytest tests/ -v -m "not e2e" --cov=phentrieve --cov=api --cov-report=xml --cov-report=term || (echo "❌ Tests failed" && exit 1)
	@echo "✅ Tests passed"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  ✅ Python CI Pipeline Complete"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ci-frontend: ## Run Frontend CI checks (matches GitHub Actions)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Frontend CI Pipeline"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Running Frontend CI checks (same as GitHub Actions)..."
	@echo ""
	@echo "[1/5] npm install (if needed)..."
	@cd frontend && npm ci
	@echo "✅ Dependencies installed"
	@echo ""
	@echo "[2/5] ESLint check..."
	@cd frontend && npm run lint || (echo "❌ ESLint failed. Run: make frontend-lint" && exit 1)
	@echo "✅ ESLint passed"
	@echo ""
	@echo "[3/5] Prettier format check..."
	@cd frontend && npm run format:check || (echo "❌ Format check failed. Run: make frontend-format" && exit 1)
	@echo "✅ Format check passed"
	@echo ""
	@echo "[4/5] Vitest tests with coverage..."
	@cd frontend && npm run test:coverage || (echo "❌ Tests failed" && exit 1)
	@echo "✅ Tests passed"
	@echo ""
	@echo "[5/5] Production build..."
	@cd frontend && VITE_API_URL=/api/v1 npm run build || (echo "❌ Build failed" && exit 1)
	@echo "✅ Build passed"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  ✅ Frontend CI Pipeline Complete"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ci-quick: ## Quick CI check (format + lint only, no tests)
	@echo "Running quick CI checks (format + lint)..."
	@ruff format --check phentrieve/ api/ tests/
	@ruff check phentrieve/ api/ tests/
	@cd frontend && npm run lint
	@cd frontend && npm run format:check
	@echo "✅ Quick CI checks passed"

##@ All-in-one

check-all: check frontend-lint frontend-test ## Check and test both Python and frontend code

all: clean check test ## Clean, check, and test Python code

# API Testing (requires PYTHONPATH for api module imports)
.PHONY: test-api
test-api:  ## Run API unit tests
	PYTHONPATH=$(PWD) python3 -m pytest tests/unit/api/ -v

.PHONY: test-api-cov
test-api-cov:  ## Run API tests with coverage
	PYTHONPATH=$(PWD) python3 -m pytest tests/unit/api/ --cov=api --cov-report=term-missing -v

##@ E2E Testing (Docker)

.PHONY: test-e2e
test-e2e:  ## Run all E2E tests (requires Docker)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Running E2E Docker Tests"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Prerequisites:"
	@echo "  • Docker Engine running"
	@echo "  • docker-compose V2 installed"
	@echo "  • HPO data prepared (data/hpo_core_data/)"
	@echo "  • ~6GB free disk space"
	@echo "  • ~4GB RAM available"
	@echo ""
	@echo "This will:"
	@echo "  1. Build Docker images (first run: ~5-10 min)"
	@echo "  2. Start containers with health checks"
	@echo "  3. Run security, health, and API tests"
	@echo "  4. Clean up containers and volumes"
	@echo ""
	@echo "Starting tests..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	pytest tests/e2e/ -v -m e2e

.PHONY: test-e2e-security
test-e2e-security:  ## Run E2E security tests only
	@echo "Running E2E Security Tests..."
	pytest tests/e2e/test_docker_security.py -v -m e2e

.PHONY: test-e2e-health
test-e2e-health:  ## Run E2E health check tests only
	@echo "Running E2E Health Tests..."
	pytest tests/e2e/test_docker_health.py -v -m e2e

.PHONY: test-e2e-api
test-e2e-api:  ## Run E2E API workflow tests only
	@echo "Running E2E API Workflow Tests..."
	pytest tests/e2e/test_api_e2e.py -v -m e2e

.PHONY: test-e2e-fast
test-e2e-fast:  ## Run E2E tests with existing containers (no rebuild)
	@echo "Running E2E tests with existing containers..."
	pytest tests/e2e/ -v -m e2e --reuse-containers

.PHONY: test-e2e-clean
test-e2e-clean:  ## Clean up E2E test Docker resources
	@echo "Cleaning up E2E test Docker resources..."
	docker-compose -f docker-compose.test.yml -p phentrieve_e2e_test down -v
	@echo "E2E test resources cleaned."

.PHONY: test-e2e-logs
test-e2e-logs:  ## View E2E test container logs
	docker-compose -f docker-compose.test.yml -p phentrieve_e2e_test logs -f

.PHONY: test-e2e-shell
test-e2e-shell:  ## Open shell in E2E test API container
	docker-compose -f docker-compose.test.yml -p phentrieve_e2e_test exec phentrieve_api_test sh

##@ Package Coverage Testing

.PHONY: cov-package
cov-package:  ## Coverage: Full phentrieve package (CLI + all core modules)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Coverage: Phentrieve Package (Full CLI Application)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@uv run pytest tests/unit/cli/ tests/unit/retrieval/ tests/unit/core/ \
		--cov=phentrieve --cov-report=html --cov-report=term-missing -v
	@echo ""
	@echo "📊 HTML Coverage Report: htmlcov/index.html"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

.PHONY: cov-api
cov-api:  ## Coverage: API package (FastAPI backend)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Coverage: API Package (FastAPI Backend)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@PYTHONPATH=$(PWD) python3 -m pytest tests/unit/api/ \
		--cov=api --cov-report=html --cov-report=term-missing -v
	@echo ""
	@echo "📊 HTML Coverage Report: htmlcov/index.html"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

.PHONY: cov-frontend
cov-frontend:  ## Coverage: Frontend package (Vue.js application)
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Coverage: Frontend Package (Vue.js Application)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@cd frontend && npm run test:coverage
	@echo ""
	@echo "📊 HTML Coverage Report: frontend/coverage/index.html"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

.PHONY: cov-all
cov-all: cov-package cov-api cov-frontend  ## Coverage: Run all package coverage reports (package + API + frontend)
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  ✅ All Coverage Reports Complete"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  • Phentrieve Package: htmlcov/index.html"
	@echo "  • API Package:        htmlcov/index.html"
	@echo "  • Frontend Package:   frontend/coverage/index.html"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
