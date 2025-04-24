# German HPO RAG Prototype

This project implements a multilingual Retrieval-Augmented Generation (RAG) system for mapping German clinical text to Human Phenotype Ontology (HPO) terms without requiring translation.

## Core Concept

In clinical genomics and rare disease diagnosis, identifying phenotypic abnormalities in patient descriptions is a critical step. When these descriptions are in languages other than English (like German), traditional approaches often require translation before matching against the English-based HPO terminology, which can introduce inaccuracies.

This project implements a novel approach that avoids translation by using a **multilingual embedding model**. The key insight is that a properly trained multilingual model can map semantically similar concepts from different languages to nearby points in the embedding space.

We use the `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model, which is specifically designed to map sentences from different languages into a shared semantic space, allowing direct semantic matching between German clinical text and English HPO terms.

## How It Works

The system operates in two phases:

### 1. Setup Phase (One-time)

- **HPO Data Acquisition**: Downloads the official HPO data in JSON format from JAX
- **HPO Term Extraction**: Processes the HPO data, focusing only on phenotypic abnormalities (under HP:0000118) while filtering out non-phenotype terms (like modes of inheritance)
- **Document Creation**: For each relevant HPO term, creates a comprehensive document containing:
  - HPO ID (e.g., HP:0000123)
  - Primary label/name in English
  - Definition
  - Synonyms
- **Embedding Generation**: Using the multilingual model, creates vector embeddings for each HPO term document
- **Index Building**: Stores these embeddings along with metadata in a local ChromaDB vector database for efficient similarity search

### 2. Query Phase

- **Input Processing**: Takes German clinical text as input, optionally splitting into sentences
- **Embedding Generation**: Maps the German text into the same embedding space using the identical model
- **Semantic Search**: Queries the ChromaDB index to find the closest HPO term embeddings to the German text embedding
- **Result Ranking**: Ranks results by similarity score and filters out low-confidence matches
- **Output Generation**: Returns the most relevant HPO terms with their IDs, names, definitions, and similarity scores

## Advantages

- **Direct semantic matching**: No error-prone intermediate translation step
- **Language-independent**: The model understands the meaning across languages
- **Offline operation**: All components run locally after initial setup
- **Robust to linguistic variations**: Can find relevant terms even when phrasing differs from known synonyms
- **Maintainable architecture**: Simpler than translation-based pipelines

## Current Implementation

Our current implementation successfully extracts and indexes over 18,000 HPO phenotypic abnormality terms. The system includes:

1. **Data processing pipeline**:
   - `download_hpo.py`: Downloads the HPO data from the official source
   - `extract_hpo_terms.py`: Parses the HPO hierarchy, extracts individual terms, and filters for phenotypic abnormalities
   - `setup_hpo_index.py`: Creates and populates the vector database

2. **Query interface**:
   - `german_hpo_rag.py`: CLI for entering German text and viewing matching HPO terms
   - Supports sentence-by-sentence processing for longer texts
   - Configurable similarity threshold and result count

### Technical Details

- **Embedding Model**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Vector Database**: ChromaDB (local, persistent)
- **HPO Version**: Latest available from JAX (currently 2025-03-03)
- **Batch Processing**: Terms are processed and indexed in batches to handle memory constraints

## Limitations & Future Work

- **No coordinate information**: The system identifies relevant HPO terms but not their precise positions in the input text
- **Performance variability**: Matching quality depends on how well the model handles clinical terminology in different languages
- **Semantic gap**: Clinical descriptions in German may use terminology patterns different from the English HPO terms
- **German compound words**: German's compound word structure presents challenges for semantic matching

### Planned Improvements

- Fine-tuning the embedding model on clinical-specific multilingual data
- Adding a secondary step for coordinate mapping
- Implementing a hybrid approach combining semantic search with other techniques
- Support for additional languages beyond German

## Setup and Usage

### Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Prepare the HPO Index (Run Once)

```bash
python download_hpo.py  # Downloads hp.json if needed
python extract_hpo_terms.py  # Extracts individual HPO terms
python setup_hpo_index.py  # Creates and populates the vector index
```

Note: The first run will download the model (~1.1 GB) and generate embeddings, which can be time-intensive.

### Run the CLI Tool

Basic usage:
```bash
python german_hpo_rag.py
```

With command-line arguments:

```bash
python german_hpo_rag.py --text "Der Patient zeigt eine Anomalie des Herzens" --similarity-threshold 0.2 --num-results 3
```

Options:

- `--text`: German text to process (if not provided, runs in interactive mode)
- `--similarity-threshold`: Minimum similarity score (0-1) to show results (default: 0.3)
- `--num-results`: Maximum number of results to display (default: 5)
- `--sentence-mode`: Process input text sentence by sentence

## File Structure

- `download_hpo.py`: Downloads HPO data from official source
- `extract_hpo_terms.py`: Extracts and filters HPO terms from the main data file
- `setup_hpo_index.py`: Builds the ChromaDB vector index
- `german_hpo_rag.py`: CLI tool for querying with German text
- `requirements.txt`: Project dependencies
- `data/`: Directory containing the HPO data and extracted terms
- `hpo_chroma_index/`: Directory containing the ChromaDB vector database

## Example Results

For the German query "Synophrys" (fused eyebrows):

```text
Query: 'Synophrys.'
1. HP:0000664 - Synophrys
   Similarity: 0.174
   Definition: Meeting of the medial eyebrows in the midline.
   Synonyms: Monobrow; Synophris; Unibrow
```

## References

- Human Phenotype Ontology: [https://hpo.jax.org/](https://hpo.jax.org/)
- Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)
- ChromaDB: [https://docs.trychroma.com/](https://docs.trychroma.com/)