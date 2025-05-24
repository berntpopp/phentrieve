# Welcome to Phentrieve

Phentrieve is a comprehensive system for mapping clinical text in multiple languages to Human Phenotype Ontology (HPO) terms via a Retrieval-Augmented Generation (RAG) approach. The system supports multilingual text processing, benchmarking across various embedding models, and provides flexible interfaces through a Python package, API, and web frontend.

## Key Features

* **Multilingual HPO Term Mapping**: Map clinical text to HPO terms in multiple languages without translation
* **Advanced Text Processing**: Process clinical text with semantic chunking and assertion detection
* **Multiple Embedding Models**: Support for domain-specific, language-specific, and general multilingual models
* **Cross-Encoder Re-Ranking**: Improve retrieval precision with specialized re-ranking models
* **Comprehensive Benchmarking**: Evaluate and compare model performance with detailed metrics
* **Multiple Interfaces**: Command-line tools, FastAPI backend, and Vue.js frontend

## Core Concept

In clinical genomics and rare disease diagnosis, identifying phenotypic abnormalities in patient descriptions is a critical step. Traditional approaches often require translation when descriptions are in languages other than English, which can introduce inaccuracies.

Phentrieve implements a novel approach using **multilingual embedding models** that map semantically similar concepts from different languages to nearby points in the embedding space. This allows direct matching between non-English clinical descriptions and English-based HPO terminology.

## Dive Deeper

* [Getting Started](getting-started/installation.md): Install and set up Phentrieve
* [User Guide](user-guide/index.md): Learn how to use the CLI, API, and frontend
* [Core Concepts](core-concepts/index.md): Understand the underlying technology
* [Advanced Topics](advanced-topics/index.md): Explore text processing, benchmarking, and more
* [Deployment](deployment/index.md): Learn how to deploy Phentrieve in various environments
* [Development](development/index.md): Contribute to the Phentrieve project
