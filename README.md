# Multilingual HPO RAG

A modular Python package for mapping clinical text in multiple languages to Human Phenotype Ontology (HPO) terms via a Retrieval-Augmented Generation (RAG) approach. Originally developed for German clinical text, the system now supports benchmarking across multiple multilingual embedding models.

## Project Structure

```text
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
```

## Core Concept

In clinical genomics and rare disease diagnosis, identifying phenotypic abnormalities in patient descriptions is a critical step. When these descriptions are in languages other than English (like German), traditional approaches often require translation before matching against the English-based HPO terminology, which can introduce inaccuracies.

This project implements a novel approach that avoids translation by using a **multilingual embedding model**. The key insight is that a properly trained multilingual model can map semantically similar concepts from different languages to nearby points in the embedding space.

We support multiple multilingual embedding models, with comprehensive benchmarking for performance comparison:

- Domain-specific models like `FremyCompany/BioLORD-2023-M`
- Language-specific models like `jinaai/jina-embeddings-v2-base-de`
- General multilingual models like `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `BAAI/bge-m3`, `Alibaba-NLP/gte-multilingual-base`, and more

This allows researchers to select the best model for their specific language and domain needs.

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

Our current implementation successfully extracts and indexes over 18,000 HPO phenotypic abnormality terms and provides comprehensive benchmarking with both exact-match and ontology-based semantic similarity metrics. The system includes:

1. **Data processing pipeline**:
   - `download_hpo.py`: Downloads the HPO data from the official source
   - `extract_hpo_terms.py`: Parses the HPO hierarchy, extracts individual terms, and filters for phenotypic abnormalities
   - `setup_hpo_index.py`: Creates and populates the vector database

2. **Query interface**:
   - `run_interactive_query.py`: CLI for entering multilingual text and viewing matching HPO terms
   - Supports sentence-by-sentence processing for longer texts
   - Configurable similarity threshold and result count

3. **Benchmarking system**:
   - `benchmark_rag.py`: Evaluates model performance using test cases with expected HPO terms
   - `manage_benchmarks.py`: Tool for running and comparing benchmarks across different models
   - `precompute_hpo_graph.py`: Precomputes HPO graph properties for ontology similarity metrics
   - Generates detailed performance metrics and visualizations

### Technical Details

- **Embedding Models**: Multiple models supported and benchmarked:
  - FremyCompany/BioLORD-2023-M (biomedical specialized)
  - jinaai/jina-embeddings-v2-base-de (German specialized)
  - T-Systems-onsite/cross-en-de-roberta-sentence-transformer (German-English cross-lingual)
  - sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (multilingual)
  - sentence-transformers/distiluse-base-multilingual-cased-v2 (multilingual lightweight)
  - BAAI/bge-m3 (retrieval-focused model)
  - sentence-transformers/LaBSE (translation alignment model)
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (lightweight model)
- **Vector Database**: ChromaDB (local, persistent)
- **HPO Version**: Latest available from JAX (currently 2025-03-03)
- **Batch Processing**: Terms are processed and indexed in batches to handle memory constraints
- **Ontology Metrics**: Semantic similarity calculations using HPO hierarchy depth and structure

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
- Expanding the ontology similarity metrics with additional measures (e.g., Lin, Wu-Palmer)

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
python multilingual_hpo_rag/scripts/01_prepare_hpo_data.py  # Downloads hp.json if needed and extracts terms
python multilingual_hpo_rag/scripts/02_build_index.py  # Creates and populates the vector index
```

Note: The first run will download the model (~1.1 GB) and generate embeddings, which can be time-intensive.

### Run the CLI Tool

Basic usage:

```bash
python multilingual_hpo_rag/scripts/run_interactive_query.py
```

With command-line arguments:

```bash
python multilingual_hpo_rag/scripts/run_interactive_query.py --text "Der Patient zeigt eine Anomalie des Herzens" --similarity-threshold 0.2 --num-results 3
```

Options:

- `--text`: German text to process (if not provided, runs in interactive mode)
- `--similarity-threshold`: Minimum similarity score (0-1) to show results (default: 0.3)
- `--num-results`: Maximum number of results to display (default: 5)
- `--sentence-mode`: Process input text sentence by sentence

## File Structure

- `multilingual_hpo_rag/scripts/01_prepare_hpo_data.py`: Downloads and prepares HPO data from official source
- `multilingual_hpo_rag/scripts/02_build_index.py`: Builds the ChromaDB vector index for a given model
- `multilingual_hpo_rag/scripts/run_interactive_query.py`: CLI tool for querying with multilingual text
- `multilingual_hpo_rag/scripts/03_run_benchmark.py`: Evaluates model performance with various metrics
- `multilingual_hpo_rag/scripts/04_manage_results.py`: Tool for running, comparing, and visualizing benchmark results

- `hpo_similarity.py`: Contains implementations of ontology-based similarity metrics
- `requirements.txt`: Project dependencies
- `data/`: Directory containing the HPO data, extracted terms, and graph data

- `hpo_chroma_index/`: Directory containing the ChromaDB vector database
- `benchmark_results/`: Directory containing benchmark output files and visualizations

## Example Results

For the German query "Synophrys" (fused eyebrows):

```text
Query: 'Synophrys.'
1. HP:0000664 - Synophrys
   Similarity: 0.174
   Definition: Meeting of the medial eyebrows in the midline.
   Synonyms: Monobrow; Synophris; Unibrow
```

## Benchmarking and Evaluation

The system includes a comprehensive benchmarking suite that evaluates model performance using two types of metrics:

### Exact Match Metrics

- **Mean Reciprocal Rank (MRR)**: The average of the reciprocal of the rank of the correct HPO term. Higher is better.
- **Hit Rate at K (Hit@K)**: The proportion of test cases where a correct HPO term appears in the top K results. Higher is better.

### Ontology Similarity Metrics

- **Ontology Similarity at K (OntSim@K)**: The average semantic similarity between the expected HPO terms and the top K retrieved terms, based on the HPO hierarchy. Higher is better.

These ontology-based metrics provide a more nuanced evaluation than exact matches alone because they account for the semantic relatedness of terms in the HPO hierarchy. For example, retrieving "Mild microcephaly" (HP:0040196) when the expected term is "Microcephaly" (HP:0000252) would get a high ontology similarity score due to their close relationship in the HPO hierarchy, despite not being an exact match.

Benchmark results are saved as:

- JSON summaries for each model
- CSV files with detailed metrics for all test cases
- Visualizations comparing model performance

### Understanding Ontology Similarity

#### How Ontology Similarity Works

The HPO is organized as a directed acyclic graph where terms have parent-child relationships defining increasingly specific phenotypes. Our ontology similarity implementation:

1. **Precomputes** a graph representation of the HPO including:
   - Each term's ancestors (all parent terms up to the root)
   - Each term's depth in the hierarchy (distance from root)

2. **Calculates similarity** between an expected HPO term and a retrieved HPO term using:
   - The depth of their Lowest Common Ancestor (LCA)
   - The depth of the terms themselves
   - A normalization factor to produce values between 0 and 1

3. **Aggregates** these similarities into the OntSim@K metric by:
   - For each expected term, finding its most similar term among the top-K retrieved results
   - Averaging these maximum similarities across all expected terms

#### Interpreting Similarity Values

Similarity values range from 0 to 1, where:

- **1.0**: Perfect match (same term)
- **~0.75-0.99**: Very close relationship (e.g., parent-child or siblings sharing a specific parent)
- **~0.50-0.74**: Moderate relationship (e.g., terms sharing a common ancestor a few levels up)
- **~0.25-0.49**: Distant relationship (e.g., terms sharing only general category ancestors)
- **~0.01-0.24**: Very distant relationship (e.g., terms under the same broad branches)
- **0.0**: No meaningful relationship (no common ancestor except the root)

#### Examples

1. **High Similarity (0.95)**:
   - Expected: "Microcephaly" (HP:0000252)
   - Retrieved: "Mild microcephaly" (HP:0040196)
   - Explanation: "Mild microcephaly" is a direct child of "Microcephaly" in the HPO hierarchy

2. **Medium Similarity (0.65)**:
   - Expected: "Seizure" (HP:0001250)
   - Retrieved: "Focal seizure" (HP:0007359)
   - Explanation: Both are types of seizures but in different subcategories of the nervous system abnormalities

3. **Low Similarity (0.30)**:
   - Expected: "Microcephaly" (HP:0000252)
   - Retrieved: "Intellectual disability" (HP:0001249)
   - Explanation: Both are neurological abnormalities but affect different aspects (brain size vs. cognitive function)

4. **Minimal Similarity (0.10)**:
   - Expected: "Microcephaly" (HP:0000252)
   - Retrieved: "Joint hypermobility" (HP:0001382)
   - Explanation: These terms come from entirely different branches of the HPO (neurological vs. skeletal)

#### Benefits in Model Evaluation

OntSim@K offers several advantages over exact match metrics:

- **Clinical relevance**: A model retrieving closely related terms is more useful than one retrieving unrelated terms
- **Partial credit**: Models are rewarded for retrieving terms semantically close to the expected ones
- **Hierarchy awareness**: The evaluation acknowledges the organized nature of medical knowledge

This allows for more nuanced comparison between models, especially in cases where exact matches are rare but semantically similar results are clinically valuable.

### Running Benchmarks

The system provides a command-line interface for running and comparing benchmarks. Here are the key commands:

#### Setting Up Models for Benchmarking

Before running benchmarks, you need to set up the embedding models and their corresponding ChromaDB collections:

```bash
# Set up a specific model
python multilingual_hpo_rag/scripts/04_manage_results.py setup --model-name "FremyCompany/BioLORD-2023-M"

# Or set up all supported models at once
python multilingual_hpo_rag/scripts/04_manage_results.py setup --all
```

#### Running Benchmark Tests

To evaluate model performance using the test cases:

```bash
# Benchmark a specific model
python multilingual_hpo_rag/scripts/04_manage_results.py run --model-name "FremyCompany/BioLORD-2023-M"

# Run benchmarks on all models
python multilingual_hpo_rag/scripts/04_manage_results.py run --all

# Run with detailed per-test-case results
python multilingual_hpo_rag/scripts/04_manage_results.py run --all --detailed

# Set a custom similarity threshold
python multilingual_hpo_rag/scripts/04_manage_results.py run --all --similarity-threshold 0.2
```

**Note:** The `run` command will benchmark models, generate result files, and also create a comparison table and visualization for the models just benchmarked. When using `--all`, this provides an immediate comparison of all models.

#### Comparing Previously Benchmarked Models

The `compare` command allows you to compare previously saved benchmark results without re-running the benchmarks:

```bash
# Compare all previously benchmarked models (loads saved results)
python multilingual_hpo_rag/scripts/04_manage_results.py compare

# Compare only specific models from previous benchmark runs
python multilingual_hpo_rag/scripts/04_manage_results.py compare --models "biolord_2023_m" "jina_embeddings_v2_base_de"
```

**When to use `compare` vs. `run --all`:**

- Use `run --all` when you need to execute new benchmarks and want results for all models at once

- Use `compare` when:
  - You've benchmarked models at different times and want to compare them later
  - You want to generate new visualizations without re-running time-consuming benchmarks
  - You want to create a focused comparison of just a few specific models
  - You've made changes to the visualization code and want to update visualizations for existing results

Both commands will display a table with all metrics (MRR, Hit@K, OntSim@K) and generate visualizations showing the relative performance of each model. Benchmark results and visualizations are saved to the `benchmark_results/` directory.

## References

- Human Phenotype Ontology: [https://hpo.jax.org/](https://hpo.jax.org/)
- Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)
- ChromaDB: [https://docs.trychroma.com/](https://docs.trychroma.com/)
- Semantic Similarity in Biomedical Ontologies: [https://doi.org/10.1371/journal.pcbi.1000443](https://doi.org/10.1371/journal.pcbi.1000443)