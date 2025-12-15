# Phentrieve

![Phentrieve Logo](docs/assets/phentrieve-logo.svg)

Phentrieve is an advanced AI-powered system for mapping clinical text to Human Phenotype Ontology (HPO) terms using a Retrieval-Augmented Generation (RAG) approach. It supports multiple languages and offers robust tools for benchmarking, text processing, and HPO term retrieval.

**For comprehensive documentation, please visit the [Phentrieve Documentation Site](https://berntpopp.github.io/phentrieve/).**

## Key Features

* Multilingual HPO term mapping using state-of-the-art embedding models
* Advanced text processing pipeline including semantic chunking and assertion detection
* Extensive benchmarking framework for model evaluation and comparison
* User-friendly interfaces: CLI, FastAPI backend, and Vue.js frontend
* Support for cross-encoder re-ranking to improve retrieval precision

## Benchmark Results

Performance on 570 German clinical terms (BioLORD-2023-M model):

| Retrieval Mode | MRR | Hit@1 | Hit@10 | Ont Sim@1 |
|----------------|-----|-------|--------|-----------|
| Single-vector | 0.695 | 55.8% | 94.0% | 79.9% |
| Multi-vector (all_max) | **0.892** | **84.0%** | **97.4%** | **91.9%** |

**+28% MRR improvement** with multi-vector retrieval using label, synonym, and definition embeddings.

## Quick Start

Install Phentrieve using pip:

```bash
pip install phentrieve
```

For detailed setup and usage instructions, including Docker deployment, please see our [Getting Started Guide](https://berntpopp.github.io/phentrieve/getting-started/installation/).

## Basic Usage

```bash
# Launch interactive query mode
phentrieve query --interactive

# Process clinical text to extract HPO terms
phentrieve text process "The patient exhibits microcephaly and frequent seizures."
```

Discover more commands and options in the [User Guide](https://berntpopp.github.io/phentrieve/user-guide/).

## Docker Deployment

Deploy Phentrieve using Docker Compose for production environments:

```bash
# Linux: Setup volume permissions (required)
sudo ./scripts/setup-docker-volumes.sh

# macOS/Windows: No setup needed, skip to next step

# Start services
docker-compose up -d

# Access the application
# - API: http://localhost:8000
# - Frontend: http://localhost:8080
```

For detailed deployment instructions, security best practices, and troubleshooting, see the [Docker Deployment Guide](docs/DOCKER-DEPLOYMENT.md).

---

[Full Documentation](https://berntpopp.github.io/phentrieve/) | [Contributing Guide](https://berntpopp.github.io/phentrieve/development/) | [License](LICENSE)