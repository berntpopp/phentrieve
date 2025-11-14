.PHONY: help format lint check test clean all install

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@echo "Phentrieve Development Makefile"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Python Development

install: ## Install package in development mode
	pip install -e .

install-dev: ## Install package with dev dependencies
	pip install -e ".[text_processing]"

format: ## Format Python code with Ruff
	ruff format phentrieve/ api/ tests/

lint: ## Lint Python code with Ruff
	ruff check phentrieve/ api/ tests/

lint-fix: ## Lint and auto-fix Python code
	ruff check phentrieve/ api/ tests/ --fix

check: format lint ## Format and lint code

test: ## Run tests with pytest
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=phentrieve --cov=api --cov-report=html --cov-report=term

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

clean-data: ## Clean data caches (use with caution)
	@echo "⚠️  This will remove ChromaDB indices and model caches"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/chromadb_*
	rm -rf data/hf_cache/

##@ All-in-one

all: clean check test ## Clean, check, and test everything
