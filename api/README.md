# Phentrieve API

This directory contains the FastAPI implementation for the Phentrieve project, providing RESTful API endpoints for HPO term querying, semantic similarity calculation, and health monitoring.

## Overview

The Phentrieve API provides a set of endpoints to interact with the Human Phenotype Ontology (HPO) data for:

- Querying clinical text to retrieve relevant HPO terms
- Calculating semantic similarity between HPO terms
- Health check endpoint for monitoring

## API Structure

```
api/
├── dependencies.py       # FastAPI dependency injection utilities
├── main.py              # FastAPI application entry point
├── Dockerfile           # Container definition for production deployment
├── local_api_config.env # Local environment configuration
├── run_api_local.py     # Local development server script
├── routers/
│   ├── health.py        # Health check endpoints
│   ├── query_router.py  # HPO term querying endpoints
│   └── similarity_router.py # Semantic similarity endpoints
└── schemas/
    ├── query_schemas.py    # Pydantic models for query API
    └── similarity_schemas.py # Pydantic models for similarity API
```

## Running Locally

### Prerequisites

- Python 3.8+
- Phentrieve core package installed
- HPO data initialized

### Installation

1. Install required dependencies:

```bash
pip install python-dotenv uvicorn fastapi
```

2. Make sure HPO data is initialized (if not already done):

```bash
python -m phentrieve.setup_hpo_index
```

### Running the API Server

From the `api` directory, run:

```bash
python run_api_local.py
```

This will:
1. Load environment variables from `local_api_config.env`
2. Ensure all required directories exist
3. Start the FastAPI server using uvicorn

### Configuration

Edit `local_api_config.env` to configure:

- Data directories
- Default models
- Hardware device (CPU/GPU)

Example configuration:

```
# Base directory for all Phentrieve data
PHENTRIEVE_DATA_ROOT_DIR=C:/development/rag-hpo-testing

# Derived paths - these will be created by the code if they don't exist
PHENTRIEVE_DATA_DIR=${PHENTRIEVE_DATA_ROOT_DIR}/hpo_core_data
PHENTRIEVE_INDEX_DIR=${PHENTRIEVE_DATA_ROOT_DIR}/indexes
PHENTRIEVE_RESULTS_DIR=${PHENTRIEVE_DATA_ROOT_DIR}/results
```

## Available Endpoints

### Health Check

- `GET /api/v1/health/`: Health status of the API

### HPO Term Query

- `GET /api/v1/query/`: Query for HPO terms with GET parameters
- `POST /api/v1/query/`: Query for HPO terms with JSON request body

### HPO Term Similarity

- `GET /api/v1/similarity/{term1_id}/{term2_id}`: Calculate semantic similarity between two HPO terms
  - Query parameter: `formula` (default: "hybrid")
  - Example: `/api/v1/similarity/HP:0001197/HP:0000750?formula=hybrid`

## API Documentation

When running locally, API documentation is available at:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Docker Deployment

For production deployment, use the provided Dockerfile:

```bash
docker build -t phentrieve-api .
docker run -p 8000:8000 -v /path/to/data:/phentrieve_data_mount phentrieve-api
```

## Troubleshooting

### Missing HPO Data

If you see errors about missing HPO data files:

```
ERROR: HPO graph data (ancestors or depths) is critically unavailable.
```

Run the setup script to initialize HPO data:

```bash
python -m phentrieve.setup_hpo_index
```

### Index Issues

If you encounter ChromaDB collection errors:

```
ERROR: DenseRetriever: Failed to connect to Chroma collection
```

Build the vector indexes:

```bash
python -m phentrieve.setup_hpo_index --build-index
```

### Path Configuration

Make sure all paths in `local_api_config.env` use forward slashes (`/`) for best compatibility across platforms.
