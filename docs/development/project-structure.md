# Project Structure

This page provides a detailed overview of the Phentrieve codebase organization.

## Repository Layout

The Phentrieve repository is organized into several main components:

```text
phentrieve/
├── api/                          # FastAPI Backend Application
│   ├── Dockerfile                # API container definition
│   ├── routers/                  # API endpoint groups
│   └── dependencies.py           # FastAPI dependency injection
│
├── frontend/                     # Vue/Vuetify Web Interface
│   ├── Dockerfile                # Frontend container definition
│   ├── nginx.conf                # Web server configuration
│   ├── src/                      # Vue application source
│   └── package.json              # Frontend dependencies
│
├── phentrieve/                   # Core Source Code Package
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface entry points
│   ├── config.py                 # Central config: paths, defaults, constants
│   ├── data_processing/          # Modules for loading/processing data
│   ├── embeddings.py             # Wrapper for loading embedding models
│   ├── indexing/                 # Modules for building indexes
│   ├── retrieval/                # Modules for querying indexes
│   ├── evaluation/               # Modules for benchmarking and metrics
│   ├── text_processing/          # Text chunking and assertion detection
│   └── utils.py                  # Shared utility functions
│
├── docker-compose.yml            # Production Docker deployment
├── docker-compose.dev.yml        # Local development overrides
├── setup_phentrieve.sh           # Automated deployment setup script
├── .env.docker.example           # Docker environment template
│
├── benchmark_results/            # Benchmark Outputs
│   ├── summaries/                # JSON summaries per run/model
│   ├── visualizations/           # Plot images
│   └── detailed/                 # Detailed CSV results per run
│
├── docs/                         # Documentation (this site)
└── tests/                        # Test suite
```

## Core Package Structure

The core `phentrieve` package contains the following key modules:

### CLI Module (`cli.py`)

The command-line interface entry points, implemented using the Click library. This module defines the main command groups and their subcommands.

### Configuration Module (`config.py`)

Central configuration for paths, defaults, and constants. This module handles:
- Environment variable processing
- Default path configurations
- Model registry and defaults

### Data Processing (`data_processing/`)

Modules for loading and processing HPO data:
- `hpo_processor.py`: HPO data parsing and extraction
- `document_creator.py`: Creating documents from HPO terms

### Embeddings (`embeddings.py`)

Wrapper for loading and using embedding models from HuggingFace.

### Indexing (`indexing/`)

Modules for building and managing vector indexes:
- `index_builder.py`: Creating ChromaDB indexes
- `index_manager.py`: Managing index lifecycle

### Retrieval (`retrieval/`)

Modules for querying indexes and processing results:
- `query_processor.py`: Core query processing

### Evaluation (`evaluation/`)

Modules for benchmarking and metrics:
- `benchmark_runner.py`: Benchmark execution
- `metrics.py`: Performance metrics calculation
- `visualizer.py`: Result visualization

### Text Processing (`text_processing/`)

Advanced text processing components:
- `chunkers.py`: Text chunking strategies
- `assertion.py`: Assertion detection
- `pipeline.py`: Processing pipeline

## API Structure

The API is implemented using FastAPI and organized into routers:

- `query_router.py`: Endpoints for querying HPO terms
- `text_router.py`: Endpoints for text processing
- `index_router.py`: Endpoints for index management

## Frontend Structure

The Vue.js frontend is organized as follows:

- `src/components/`: Vue components
  - `QueryInterface.vue`: Main query interface
  - `ResultsDisplay.vue`: HPO term results display
- `src/views/`: Vue views/pages
- `src/store/`: Vuex state management
- `src/router/`: Vue Router configuration

## Testing Structure

The test suite is organized by component:

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for component interactions
- `tests/e2e/`: End-to-end tests for full workflows
