# Phentrieve

<p align="center">
  <img src="docs/assets/hpo-logo.svg" alt="Phentrieve Logo" width="150"/>
</p>

Phentrieve is an advanced AI-powered system for mapping clinical text to Human Phenotype Ontology (HPO) terms using a Retrieval-Augmented Generation (RAG) approach. It supports multiple languages and offers robust tools for benchmarking, text processing, and HPO term retrieval.

**For comprehensive documentation, please visit the [Phentrieve Documentation Site](https://berntpopp.github.io/phentrieve/).**

## Key Features

* Multilingual HPO term mapping using state-of-the-art embedding models
* Advanced text processing pipeline including semantic chunking and assertion detection
* Extensive benchmarking framework for model evaluation and comparison
* User-friendly interfaces: CLI, FastAPI backend, and Vue.js frontend
* Support for cross-encoder re-ranking to improve retrieval precision

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

---

[Full Documentation](https://berntpopp.github.io/phentrieve/) | [Contributing Guide](https://berntpopp.github.io/phentrieve/development/) | [License](LICENSE)