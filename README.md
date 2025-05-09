# Phentrieve

A modular Python package for mapping clinical text in multiple languages to Human Phenotype Ontology (HPO) terms via a Retrieval-Augmented Generation (RAG) approach. The system supports benchmarking across multiple multilingual embedding models to identify relevant HPO terms from clinical descriptions in various languages.

## Project Structure

```text
phentrieve/
├── data/                         # Data Sources & Test Cases
│   ├── hp.json                   # Original HPO download (generated)
│   ├── hpo_terms/                # Extracted terms (generated)
│   ├── hpo_ancestors.pkl         # Precomputed graph data (generated)
│   ├── hpo_term_depths.pkl       # Precomputed graph data (generated)
│   └── test_cases/               # Test cases for benchmarking
│
├── phentrieve/                   # Core Source Code Package
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

- **Input Processing**: Takes multilingual clinical text as input, optionally splitting into sentences
- **Embedding Generation**: Maps the text into the same embedding space using the identical model
- **Semantic Search**: Queries the ChromaDB index to find the closest HPO term embeddings to the input text embedding
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

## Advanced Features

### Cross-Encoder Re-ranking

The system supports re-ranking of retrieved candidate HPO terms using cross-encoder models, which can significantly improve the ranking precision. Two re-ranking modes are available:

1. **Cross-lingual Re-ranking** (default): Compares non-English queries directly with English HPO term labels
   - Uses a multilingual cross-encoder model (default: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
   - No translation files required
   - Suitable when no translations are available in your target language

2. **Monolingual Re-ranking**: Compares queries with translations of HPO terms in the same language
   - Uses a language-specific cross-encoder model
   - Requires translations of HPO terms in JSON format
   - Often produces more accurate rankings when translations are available

#### Translation File Format

For monolingual re-ranking, translation files must be provided in the following structure (example for German HPO translations):

```bash
[translation_dir]/
├── HP_0000123.json
├── HP_0000124.json
└── ...
```

Each JSON file should follow this format:

```json
{
  "lbl": "Translation of the main HPO term label",
  "meta": {
    "synonyms": [
      {"val": "Synonym 1 in target language"},
      {"val": "Synonym 2 in target language"}
    ]
  }
}
```

#### Example Usage

```bash
# Cross-lingual re-ranking (non-English query → English HPO)
phentrieve query --enable-reranker

# Monolingual re-ranking (using target language translations)
phentrieve query --enable-reranker --reranker-mode monolingual --translation-dir path/to/translations
```

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

Install Phentrieve:

```bash
pip install -e .
```

### Prepare HPO Data and Index (Run Once)

```bash
# Prepare HPO Data (Graph Properties & Extracted Terms)
phentrieve data prepare

# Build Vector Index for default model
phentrieve index build
```

Note: The first run will download the model (~1.1 GB) and generate embeddings, which can be time-intensive.

### Execution Methods

After installation, you can use the `phentrieve` command directly from your terminal:

```bash
# View available commands
phentrieve --help

# View help for a specific command
phentrieve query --help
```

### Run the CLI Tool

Basic usage:

```bash
phentrieve query --interactive
```

With command-line arguments:

```bash
phentrieve query --text "The patient shows an anomaly of the heart" --similarity-threshold 0.2 --num-results 3
```

Options:

- `--text`: Text to process (if not provided, runs in interactive mode)
- `--similarity-threshold`: Minimum similarity score (0-1) to show results (default: 0.3)
- `--num-results`: Maximum number of results to display (default: 5)
- `--sentence-mode`: Process input text sentence by sentence
- `--enable-reranker`: Enable cross-encoder re-ranking (default: False)
- `--reranker-mode`: Re-ranking mode, either 'cross-lingual' or 'monolingual' (default: cross-lingual)
- `--reranker-model`: Cross-encoder model to use for cross-lingual re-ranking
- `--monolingual-reranker-model`: Cross-encoder model to use for monolingual re-ranking
- `--translation-dir`: Directory containing HPO term translations for monolingual re-ranking
- `--rerank-count`: Number of candidates to re-rank (default: 50)

## File Structure

- `phentrieve/`: Main package directory
  - `data_processing/`: Modules for loading/processing data
  - `indexing/`: Modules for building indexes
  - `retrieval/`: Modules for querying indexes
  - `evaluation/`: Modules for benchmarking and metrics
  - `utils.py`: Shared utility functions
- `data/`: Directory containing the HPO data, extracted terms, and graph data
  - `hp.json`: Original HPO download (generated)
  - `hpo_terms/`: Extracted terms (generated)
  - `hpo_ancestors.pkl`: Precomputed graph data (generated)
  - `hpo_term_depths.pkl`: Precomputed graph data (generated)
  - `results/`: Benchmark results and visualizations
- `hpo_chroma_index/`: Directory containing the ChromaDB vector database

## Example Results

Example query for "Synophrys" (fused eyebrows):

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

- **Ontology Similarity at K (MaxOntSim@K)**: For each expected term in a test case, this metric finds the highest semantic similarity score against any of the top K retrieved terms. These maximum scores are then averaged across all expected terms in the test case. Finally, these per-test-case average maximum similarities are averaged over all test cases. A score of 1.0 indicates perfect or very close semantic matches for all expected terms, while lower scores indicate less relevance. Higher is better.

These ontology-based metrics provide a more nuanced evaluation than exact matches alone because they account for the semantic relatedness of terms in the HPO hierarchy. For example, retrieving "Mild microcephaly" (HP:0040196) when the expected term is "Microcephaly" (HP:0000252) would get a high ontology similarity score due to their close relationship in the HPO hierarchy, despite not being an exact match.

Benchmark results are saved as:

- JSON summaries for each model
- CSV files with detailed metrics for all test cases
- Visualizations comparing model performance

### Understanding Ontology Similarity

The HPO is organized as a directed acyclic graph (DAG) where terms have parent-child relationships defining increasingly specific phenotypes. Our ontology similarity implementation leverages this structure.

### Core Calculation Steps

#### Precomputation (01_prepare_hpo_data.py)

- A graph representation of the HPO is built.
- For every HPO term, its ancestors (all parent terms up to the true ontology root, HP:0000001) are determined and stored in data/hpo_ancestors.pkl.
- The depth of each term (its shortest distance from HP:0000001) is calculated and stored in data/hpo_term_depths.pkl.

#### Lowest Common Ancestor (LCA)

- For any two HPO terms (e.g., an expected term t1 and a retrieved term t2), their LCA is found. The LCA is their deepest shared ancestor in the HPO graph.

#### Similarity Calculation

- Once the LCA and the depths of t1, t2, and LCA(t1, t2) are known, a similarity score is computed. This system supports multiple similarity formulas, selectable during benchmarking.

### Available Similarity Formulas

The choice of formula can be specified using the `--similarity-formula` option when running benchmarks (e.g., `phentrieve benchmark run --similarity-formula simple_resnik_like`).

#### hybrid (Default Formula)

This formula combines aspects of Resnik and Lin similarity:

```python
Sim(t1, t2) = (0.7 * depth_factor) + (0.3 * distance_factor)
```

Where:

- `depth_factor = D(LCA(t1, t2)) / D_max_ontology`
  - D(LCA(t1, t2)) is the depth of the Lowest Common Ancestor.
  - D_max_ontology is the maximum depth of any term in the entire HPO.
  - This component reflects the shared specificity of the terms, normalized by the overall depth of the ontology.
- `distance_factor = 1 - (total_path_length_to_LCA / (D(t1) + D(t2)))`
  - total_path_length_to_LCA is the sum of path lengths from t1 to LCA and t2 to LCA.
  - D(t1) and D(t2) are the depths of the terms being compared.
  - This component reflects the structural closeness of the terms to their LCA.

**Characteristics**: This formula aims for a nuanced score by considering both shared information (via LCA depth) and structural proximity. It tends to give slightly higher scores to direct parent-child relationships than to sibling relationships if the parent's depth is the same as the siblings' common parent.

#### simple_resnik_like

This formula is a simpler, Resnik-like measure using depth as a proxy for Information Content (IC):

```python
Sim(t1, t2) = D(LCA(t1, t2)) / max(D(t1), D(t2))
```

(If max(D(t1), D(t2)) is 0, the score is 0, unless t1 and t2 are identical and are the root, then it's 1).

**Characteristics**: This formula is more straightforward. It normalizes the LCA's depth by the depth of the deeper of the two terms being compared. For parent-child pairs (P, C), it resolves to D(P) / D(C). Sibling pairs sharing a common parent P will have the same similarity score as a P-C pair where C is a child of P. Scores approach 1 for closely related terms deep in the ontology.

### Interpreting Similarity Values

Regardless of the formula, similarity values generally range from 0 to 1:

- **1.0**: Perfect match (the terms are identical).
- **~0.75-0.99**: Very close relationship (e.g., parent-child or siblings sharing a very specific parent). The exact range depends on the formula and term depths.
- **~0.50-0.74**: Moderate relationship (e.g., terms sharing a common ancestor a few levels up).
- **~0.25-0.49**: Distant relationship (e.g., terms sharing only general category ancestors).
- **~0.01-0.24**: Very distant relationship.
- **0.0**: No meaningful semantic relationship found based on the ontology structure (e.g., no common ancestor other than potentially the ultimate root, or one of the terms is not found in the precomputed data).

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
python -m phentrieve.scripts.04_manage_results setup --model-name "FremyCompany/BioLORD-2023-M"

# Or set up all supported models at once
python -m phentrieve.scripts.04_manage_results setup --all
```

#### Running Benchmark Tests

To evaluate model performance using the test cases:

```bash
# Benchmark a specific model
phentrieve benchmark run --model-name "FremyCompany/BioLORD-2023-M"

# Run benchmarks on all models
phentrieve benchmark run --all-models

# Run with a specific similarity formula
phentrieve benchmark run --similarity-formula simple_resnik_like

# Run with detailed per-test-case results
phentrieve benchmark run --detailed
```

**Note:** The `run` command will benchmark models, generate result files, and also create a comparison table and visualization for the models just benchmarked. When using `--all`, this provides an immediate comparison of all models.

#### Comparing Previously Benchmarked Models

The system allows you to compare previously saved benchmark results without re-running the benchmarks:

```bash
# Compare all previously benchmarked models (loads saved results)
phentrieve benchmark compare

# Generate visualizations from benchmark results
phentrieve benchmark visualize

# Generate visualizations with specific metrics
phentrieve benchmark visualize --metrics mrr,hit_rate
```

**When to use the different benchmark commands:**

- Use `benchmark run` when you need to execute benchmarks for specific or all models
- Use `benchmark compare` when you want to compare previously benchmarked models without rerunning them
- Use `benchmark visualize` when you want to generate or update visualizations for existing benchmark results

Benchmark results and visualizations are saved to your configured results directory (default: `data/results/`). The visualizations include comparative plots for MRR, Hit@K, MaxOntSim@K and heatmaps showing the performance of all models across multiple metrics.

## References

- Human Phenotype Ontology: [https://hpo.jax.org/](https://hpo.jax.org/)
- Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)
- ChromaDB: [https://docs.trychroma.com/](https://docs.trychroma.com/)
- Semantic Similarity in Biomedical Ontologies: [https://doi.org/10.1371/journal.pcbi.1000443](https://doi.org/10.1371/journal.pcbi.1000443)
