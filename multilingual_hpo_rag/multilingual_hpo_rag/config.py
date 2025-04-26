"""
Central configuration for the multilingual HPO RAG package.

This module contains constants, defaults, and configuration parameters used
throughout the multilingual_hpo_rag package.
"""

import os
from pathlib import Path


# Base directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
HPO_TERMS_DIR = os.path.join(DATA_DIR, "hpo_terms")
TEST_CASES_DIR = os.path.join(DATA_DIR, "test_cases")

# HPO data files
HPO_FILE_PATH = os.path.join(DATA_DIR, "hp.json")
HPO_ANCESTORS_FILE = os.path.join(DATA_DIR, "hpo_ancestors.pkl")
HPO_DEPTHS_FILE = os.path.join(DATA_DIR, "hpo_term_depths.pkl")

# ChromaDB settings
INDEX_DIR = os.path.join(ROOT_DIR, "hpo_chroma_index")

# Benchmark results directory
RESULTS_DIR = os.path.join(ROOT_DIR, "benchmark_results")
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")
DETAILED_DIR = os.path.join(RESULTS_DIR, "detailed")
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, "visualizations")

# Default models
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_BIOLORD_MODEL = "FremyCompany/BioLORD-2023-M"
JINA_MODEL_ID = "jinaai/jina-embeddings-v2-base-de"

# All models for benchmarking
BENCHMARK_MODELS = [
    "FremyCompany/BioLORD-2023-M",
    "jinaai/jina-embeddings-v2-base-de",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "BAAI/bge-m3",
    "Alibaba-NLP/gte-multilingual-base",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# Default parameters
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to display results
DEFAULT_TOP_K = 5  # Default number of results to return

# HPO root ID - used in HPO graph processing
HPO_ROOT_ID = "HP:0000001"
