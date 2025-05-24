# User Guide Overview

This section provides comprehensive documentation on how to use Phentrieve's various features and interfaces.

## Available Interfaces

Phentrieve offers multiple ways to interact with the system:

1. **Command-Line Interface (CLI)**: The primary way to use Phentrieve's features, including text processing, querying, benchmarking, and more.

2. **API**: A FastAPI-based service that exposes Phentrieve's functionality through RESTful endpoints.

3. **Web Frontend**: A Vue.js-based web interface for user-friendly interaction with the system.

## Core Functionality

Phentrieve provides several key functionalities:

### HPO Term Retrieval

Map clinical text to relevant HPO terms using multilingual embedding models. This works across languages without requiring translation.

### Text Processing

Process clinical text with advanced features:
- Semantic chunking
- Assertion detection (negated, normal, uncertain)
- Evidence aggregation
- Confidence scoring

### Benchmarking

Evaluate and compare the performance of different embedding models for HPO term retrieval.

### Ontology Similarity

Calculate semantic similarity between HPO terms using the ontology graph structure.

## Section Contents

This User Guide section contains the following pages:

- [CLI Usage](cli-usage.md): Detailed guide on using the Phentrieve command-line interface
- [Text Processing Guide](text-processing-guide.md): How to process clinical text to extract HPO terms
- [Benchmarking Guide](benchmarking-guide.md): Running and interpreting benchmarks
- [Configuration Profiles](configuration-profiles.md): Configuring Phentrieve for different use cases
- [API Usage](api-usage.md): Using the Phentrieve API
- [Frontend Usage](frontend-usage.md): Using the Phentrieve web interface
