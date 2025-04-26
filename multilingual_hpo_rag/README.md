# Multilingual HPO RAG

A modular Python package for mapping clinical text in multiple languages to Human Phenotype Ontology (HPO) terms via a Retrieval-Augmented Generation (RAG) approach.

## Overview

This package provides functionality for:

1. **Building a vector index** of HPO terms using multilingual embedding models
2. **Retrieving relevant HPO terms** from clinical text descriptions
3. **Evaluating retrieval performance** using various metrics
4. **Visualizing and comparing results** across different embedding models

The system is designed to work with multiple languages thanks to multilingual embedding models, with a focus on mapping clinical text to standardized HPO terms.

## Project Structure

```
multilingual_hpo_rag/
├── data/                         # Data Sources & Test Cases
│   ├── hp.json                   # Original HPO download (generated)
│   ├── hpo_terms/                # Extracted terms (generated)
│   ├── hpo_ancestors.pkl         # Precomputed graph data (generated)
│   ├── hpo_term_depths.pkl       # Precomputed graph data (generated)
│   └── test_cases/               # Test cases for benchmarking
│
├── multilingual_hpo_rag/         # Core Source Code Package
│   ├── __init__.py
│   ├── config.py                 # Central config: paths, defaults, constants
│   ├── data_processing/          # Modules for loading/processing data
│   ├── embeddings.py             # Wrapper for loading embedding models
│   ├── indexing/                 # Modules for building indexes
│   ├── retrieval/                # Modules for querying indexes
│   ├── evaluation/               # Modules for benchmarking and metrics
│   └── utils.py                  # Shared utility functions
│
├── scripts/                      # Executable Workflow Scripts
│   ├── 01_prepare_hpo_data.py    # Downloads, parses, precomputes HPO graph data
│   ├── 02_build_index.py         # Builds ChromaDB index for a given model
│   ├── 03_run_benchmark.py       # Runs benchmark evaluation for models/configs
│   ├── 04_manage_results.py      # Compares/visualizes results from benchmark runs
│   └── run_interactive_query.py  # Runs the interactive query CLI
│
├── benchmark_results/            # Benchmark Outputs
│   ├── summaries/                # JSON summaries per run/model
│   ├── visualizations/           # Plot images
│   └── detailed/                 # Detailed CSV results per run
│
├── hpo_chroma_index/             # ChromaDB persistent storage (generated)
│
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/multilingual_hpo_rag.git
cd multilingual_hpo_rag
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare HPO Data

Download and prepare HPO ontology data:

```bash
python scripts/01_prepare_hpo_data.py
```

### 2. Build Index

Build a vector index using a specific embedding model:

```bash
python scripts/02_build_index.py --model-name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### 3. Run Interactive Queries

Start the interactive query tool:

```bash
python scripts/run_interactive_query.py --model-name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### 4. Run Benchmarks

Evaluate performance using benchmark test cases:

```bash
python scripts/03_run_benchmark.py --model-name "FremyCompany/BioLORD-2023-M"
```

### 5. Compare Results

Compare and visualize benchmark results:

```bash
# Compare results across models
python scripts/04_manage_results.py compare

# Visualize hit rate metrics
python scripts/04_manage_results.py visualize --metric hit_rate
```

## Model Support

This package has been tested with the following embedding models:

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `FremyCompany/BioLORD-2023-M`
- `sentence-transformers/distiluse-base-multilingual-cased-v2`
- `jinaai/jina-embeddings-v2-base-de`
- `sentence-transformers/LaBSE`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## GPU Acceleration

The package automatically uses CUDA GPU acceleration when available and gracefully falls back to CPU when unavailable.

## License

See LICENSE file for details.
