# Phentrieve

A comprehensive system for mapping clinical text in multiple languages to Human Phenotype Ontology (HPO) terms via a Retrieval-Augmented Generation (RAG) approach. Phentrieve includes a robust Python package, FastAPI backend, and Vue/Vuetify frontend. The system supports benchmarking across multiple multilingual embedding models to identify relevant HPO terms from clinical descriptions in various languages.

## Project Structure

```text
phentrieve/
├── api/                          # FastAPI Backend Application
│   ├── Dockerfile                # API container definition
│   ├── routers/                  # API endpoint groups
│   └── dependencies.py           # FastAPI dependency injection
│
├── frontend/                     # Vue/Vuetify Web Interface
│   ├── Dockerfile                # Frontend container definition
│   ├── nginx.conf                # Web server configuration
│   ├── src/                      # Vue application source
│   └── package.json              # Frontend dependencies
│
├── phentrieve/                   # Core Source Code Package
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface entry points
│   ├── config.py                 # Central config: paths, defaults, constants
│   ├── data_processing/          # Modules for loading/processing data
│   ├── embeddings.py             # Wrapper for loading embedding models
│   ├── indexing/                 # Modules for building indexes
│   ├── retrieval/                # Modules for querying indexes
│   ├── evaluation/               # Modules for benchmarking and metrics
│   ├── text_processing/          # Text chunking and assertion detection 
│   └── utils.py                  # Shared utility functions
│
├── docker-compose.yml            # Production Docker deployment
├── docker-compose.dev.yml        # Local development overrides
├── setup_phentrieve.sh           # Automated deployment setup script
├── .env.docker.example           # Docker environment template
│
├── benchmark_results/            # Benchmark Outputs
│   ├── summaries/                # JSON summaries per run/model
│   ├── visualizations/           # Plot images
│   └── detailed/                 # Detailed CSV results per run
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

The system operates in three main layers:

1. **Core Package**: The underlying Python library that handles data processing, embedding generation, and retrieval
2. **API Layer**: A FastAPI-based service that exposes the core functionality through RESTful endpoints
3. **Frontend Layer**: A Vue.js-based web interface for user-friendly interaction

The workflow follows two primary phases:

### 1. Setup Phase (One-time)

- **HPO Data Acquisition**: Downloads the official HPO data in JSON format from JAX
- **HPO Term Extraction**: Processes the HPO data, extracting ALL HPO terms with a focus on phenotypic abnormalities
- **Document Creation**: For each HPO term, creates a comprehensive document containing:
  - HPO ID (e.g., HP:0000123)
  - Primary label/name in English
  - Definition
  - Synonyms
- **Embedding Generation**: Using a selected multilingual model, creates vector embeddings for each HPO term document
- **Index Building**: Stores these embeddings along with metadata in a local ChromaDB vector database for efficient similarity search

For Docker deployments, the `setup_phentrieve.sh` script automates this process, creating necessary directories, establishing network connections with Nginx Proxy Manager, and preparing the HPO data and index.

### 2. Query Phase

- **Input Processing**: Takes multilingual clinical text as input, with optional chunking and assertion detection
- **Embedding Generation**: Creates vector embeddings of the input text using the same model
- **Similarity Search**: Queries the ChromaDB index to find the most similar HPO term documents
- **Optional Re-ranking**: Can apply cross-encoder models to improve ranking precision
- **Result Filtering**: Ranks results by similarity score and filters out low-confidence matches
- **Output Generation**: Returns the most relevant HPO terms with their IDs, names, definitions, similarity scores, and assertion status (if applicable)

## Advantages

- **Direct semantic matching**: No error-prone intermediate translation step
- **Language-independent**: The model understands the meaning across languages

## Current Implementation

The system consists of several integrated components:

- **Core Python Package**:
  - **Data Preparation**: Extracts phenotype terms from HPO JSON, builds semantic documents
  - **Embedding Models**: Uses SentenceTransformers multilingual models with GPU acceleration when available
  - **Vector Database**: ChromaDB for efficient similarity search
  - **Text Processing**: Chunking strategies and assertion detection for detailed text analysis
  - **Evaluation Framework**: Comprehensive benchmarking system with multiple metrics
    - Basic retrieval metrics: MRR, Hit@K, recall, precision
    - HPO graph precomputation: Precomputes HPO graph properties for ontology similarity metrics
    - Generates detailed performance metrics and visualizations

- **API Layer**:
  - FastAPI-based REST endpoints for querying HPO terms
  - Dependency injection for efficient model and retriever management
  - Health checks and error handling
  - Interactive API documentation (OpenAPI)

- **Frontend**:
  - Vue.js framework with Vuetify UI components
  - Interactive query interface
  - Results display with HPO term details
  - User-friendly configuration options
  
- **Deployment**:
  - Docker containers for both API and frontend
  - Nginx for serving the frontend
  - Nginx Proxy Manager integration for reverse proxying and SSL
  - Setup automation script for easy deployment

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

## Text Processing Features

Phentrieve now includes robust text processing capabilities for extracting HPO terms from clinical text, with support for:

### Flexible Text Chunking

The system provides multiple text chunking strategies that can be combined in a pipeline:

- **Paragraph Chunking**: Splits text based on blank lines
- **Sentence Chunking**: Uses language-specific rules to separate sentences
- **Semantic Chunking**: Groups sentences by semantic similarity
- **Fine-grained Punctuation Chunking**: Further splits text at punctuation marks like periods, commas, and semicolons

Three predefined strategies are available via the `--strategy` option:

- **simple**: Paragraph chunking + Sentence chunking
- **semantic** (default): Paragraph chunking + Semantic chunking
- **detailed**: Paragraph chunking + Semantic chunking + Fine-grained punctuation chunking for more granular analysis

Chunking configuration can be specified via command-line parameters or through YAML/JSON configuration files.

### Assertion Detection

The system can detect the assertion status of medical concepts in text:

- **Affirmed**: The phenotype is positively asserted (default)
- **Negated**: The phenotype is explicitly negated (e.g., "no microcephaly", "denies seizures")
- **Normal**: The finding is described as normal or within normal limits
- **Uncertain**: The phenotype is mentioned with uncertainty

Assertion detection uses both keyword-based and dependency-based approaches that can be configured based on user preference. The system implements a priority-based logic for determining assertion status:

1. Dependency-based negation has highest priority
2. Dependency-based normality has second priority
3. Keyword-based negation has third priority
4. Keyword-based normality has fourth priority

This prioritization ensures the most accurate detection of assertion status, particularly for complex clinical text where the context and grammatical structure are important for proper interpretation.

### Multilingual Assertion Detection

Phentrieve supports assertion detection in multiple languages, including English and German, by utilizing language-specific SpaCy models and predefined negation/normality cues. For German clinical text, the system identifies terms like "kein", "keine", "nicht", and "ohne" (negation) as well as "normal", "unauffällig", and "o.B." (normality).

All text processing, including chunking and assertion detection, uses the language parameter to ensure appropriate language models and cues are applied throughout the pipeline.

### HPO Term Extraction

The system processes chunked text and extracts relevant HPO terms while maintaining assertion status:

- **Evidence Aggregation**: Combines evidence from multiple chunks for the same HPO term
- **Confidence Scoring**: Calculates confidence scores based on similarity and evidence count
- **Result Filtering**: Filters results based on confidence thresholds and allows taking only the top term per chunk
- **Multi-format Output**: Supports JSON, CSV, and other output formats for easy integration

### Using Text Processing

```bash
# Process text directly
phentrieve text process "Patient presents with hearing loss and developmental delay"

# Only include high-confidence terms (confidence >= 0.7)
phentrieve text process "Patient presents with hearing loss" --min-confidence 0.7

# Only include the highest-scored term for each text chunk
phentrieve text process "Patient presents with hearing loss" --top-term-per-chunk

# Process a file with semantic chunking
phentrieve text process --input-file clinical_note.txt --strategy semantic

# Use a specific model and output format
phentrieve text process --input-file notes.txt --model "FremyCompany/BioLORD-2023-M" --output-format csv_hpo_list

# Just perform chunking without HPO term extraction
phentrieve text chunk "Patient presents with progressive hearing loss that began in childhood."

# Process text with confidence threshold and only getting top term per chunk
phentrieve text process "Patient has microcephaly but no seizures" --min-confidence 0.4 --top-term-per-chunk
```

### Filtering Options for Text Processing

- `--min-confidence`: Set a threshold for minimum similarity score (0.0-1.0) to include an HPO term
- `--top-term-per-chunk`: Return only the highest-scoring HPO term for each text chunk
- `--strategy`: Choose text chunking strategy (simple, semantic, detailed)
- `--language`: Specify text language for accurate chunking and assertion detection (e.g., 'en', 'de')

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

# Install required SpaCy language models for dependency parsing
python -m spacy download en_core_web_sm  # For English text
python -m spacy download de_core_news_sm  # For German text
```

### Data Setup

1. **Prepare HPO Data (Graph Properties & Extracted Terms):**
   This step downloads `hp.json`, extracts all HPO terms into `data/hpo_terms/`, and precomputes `data/hpo_ancestors.pkl` and `data/hpo_term_depths.pkl`.

   ```bash
   phentrieve data prepare
   ```

   Use `--force` to re-download and re-process if needed.

2. **Build Vector Index:**
   This step creates the ChromaDB vector index for a specified model (or all benchmark models).

   ```bash
   # For the default model (e.g., BioLORD)
   phentrieve index build
   
   # Or specify a model:
   phentrieve index build --model-name "BAAI/bge-m3"
   
   # To build for all benchmark models:
   phentrieve index build --all-models
   ```

### Interactive Querying

```bash
# For interactive querying with text input
phentrieve query --interactive
```

### HPO Term Similarity Calculation

Calculate semantic similarity between two specific HPO terms using the ontology graph structure:

```bash
# Calculate similarity between two HPO terms using the default 'hybrid' formula
phentrieve similarity calculate HP:0000252 HP:0001250

# Calculate using the simple Resnik-like formula
phentrieve similarity calculate HP:0000252 HP:0001250 --formula simple_resnik_like

# Enable debug logging
phentrieve similarity calculate HP:0000252 HP:0001250 --debug
```

The output includes:
- Information about both HPO terms (ID and label)
- The formula used for calculation
- The similarity score (0.0 to 1.0, where 1.0 means identical terms)
- The Lowest Common Ancestor (LCA) in the ontology, including its depth

This is useful for researchers who want to quantify the semantic relationship between phenotypic abnormalities based on their position in the HPO ontology tree.

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

#### Precomputation (phentrieve data prepare)

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
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Or set up all supported models at once
phentrieve index build --all-models
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

## Web Application Interface

Phentrieve now includes a modern web interface for easier interaction with the HPO query functionality, consisting of:

1. A FastAPI backend that exposes the core Phentrieve query functionality via REST API
2. A Vue.js 3 frontend with Vuetify 3 that provides an intuitive user interface
3. Docker configuration for easy deployment

### Web API

The API exposes the HPO term mapping functionality via a RESTful endpoint:

- **Endpoint**: `/api/v1/query/`
- **Method**: POST
- **Request Body**: JSON matching the `QueryRequest` schema

  ```json
  {
    "text": "Patient has microcephaly and seizures",
    "model_name": "FremyCompany/BioLORD-2023-M",
    "similarity_threshold": 0.3,
    "enable_reranker": true
  }
  ```

- **Response**: JSON with query results and matching HPO terms

Key API features:

- Support for all embedding models in the core Phentrieve package
- Optional cross-encoder reranking for improved result quality
- Automatic language detection for multilingual text
- Model caching for improved performance

### Web Frontend

The Vue/Vuetify frontend provides:

- A clean, modern interface using Vuetify 3 components
- Text input area for multilingual clinical text
- Controls for model selection and similarity threshold

Phentrieve can be deployed in multiple ways depending on your needs. The recommended approach for production is using Docker with Nginx Proxy Manager, but local development options are also available.

### Docker Deployment with Nginx Proxy Manager (Recommended for Servers)

This deployment method provides a production-ready setup with proper SSL termination and domain routing.

#### Prerequisites

- Docker and Docker Compose V2 installed on your server
- Nginx Proxy Manager (NPM) already set up and running
- Domain name with DNS records pointing to your server
- Linux server (Ubuntu recommended)

#### Deployment Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/phentrieve.git
   cd phentrieve
   ```

2. **Configure environment variables**

   ```bash
   # Copy example files
   cp .env.docker.example .env.docker
   
   # Edit the .env.docker file and set at minimum:
   # - PHENTRIEVE_HOST_DATA_DIR (absolute path where data will be stored)
   # - NPM_SHARED_NETWORK_NAME (your NPM network, typically 'npm_default')
   # - VITE_API_URL_PUBLIC and VITE_FRONTEND_URL_PUBLIC (your domain names)
   nano .env.docker
   ```

3. **Run the setup script**

   The setup script automates several tasks:
   - Creates necessary host data directories
   - Creates shared Docker network with NPM if needed
   - Prepares HPO core data (if not already present)
   - Builds default model index (if not already present)

   ```bash
   chmod +x setup_phentrieve.sh
   ./setup_phentrieve.sh
   ```

4. **Configure DNS records**

   Ensure your domain registrar has A/AAAA records pointing to your server for:
   - Your main frontend domain (e.g., phentrieve.example.com)
   - Your API domain (e.g., phentrieve-api.example.com)

5. **Configure Nginx Proxy Manager**

   In the NPM web interface, add two Proxy Hosts:

   **Frontend:**
   - Domain: phentrieve.example.com
   - Scheme: http
   - Forward Hostname/IP: phentrieve_frontend (Docker service name)
   - Forward Port: 80
   - SSL: Request Let's Encrypt certificate, Force SSL

   **API:**
   - Domain: phentrieve-api.example.com
   - Scheme: http
   - Forward Hostname/IP: phentrieve_api (Docker service name)
   - Forward Port: 8000
   - SSL: Request Let's Encrypt certificate, Force SSL

6. **Start the Phentrieve services**

   ```bash
   docker compose -f docker-compose.yml --env-file .env.docker up -d --build
   ```

7. **Access your application**

   Open your browser and navigate to your frontend domain (https://phentrieve.example.com)

### Local Development with Docker Compose (Direct Port Access)

For local development and testing, you can run Phentrieve with direct port access:

```bash
# Create a local environment file
cp .env.docker.example .env.local

# Edit .env.local with local settings
# Set PHENTRIEVE_HOST_DATA_DIR to an absolute path on your machine
# You can leave the URL variables as they are

# Start both API and frontend containers with development overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.local up --build

# Access the frontend at http://localhost:8080
# API is available at http://localhost:8001/api/v1
# API docs at http://localhost:8001/docs
```

The `docker-compose.dev.yml` override file maps the container ports to your host and configures the frontend to communicate with the local API instance.

### Local Python Environment Installation

For CLI usage or advanced development, you can install Phentrieve directly in a Python environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Phentrieve in development mode
pip install -e .

# Install required spaCy models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm  # If working with German

# Verify installation
phentrieve --help
```

### Running Tests

Phentrieve includes a comprehensive test suite that ensures all components work correctly. Tests can be run using pytest (recommended) or Python's built-in unittest framework:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_sliding_window_chunker.py

# Run tests with debug logging
pytest --log-cli-level=DEBUG

# Run a specific test case
pytest phentrieve/text_processing/tests/test_sliding_window_chunker.py::TestSlidingWindowSemanticSplitter::test_semantic_splitting
```

Alternatively, you can use Python's unittest module:

```bash
# Run all tests in a specific file
python -m unittest tests/test_sliding_window_chunker.py
```

#### Test Organization

Tests are organized in the following directories:

- `tests/`: High-level component and integration tests
- `phentrieve/text_processing/tests/`: Unit tests for text processing modules

### Initial Setup

If using Docker, the `setup_phentrieve.sh` script handles this automatically. For local Python installations:

```bash
# Download and process HPO data
phentrieve data prepare

# Build the index for a specific model
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Or build multiple indexes for benchmarking
phentrieve index build --all-models
```

### Data Management

All Phentrieve data is stored in configurable directories:

- **For Docker deployments**: In the directory specified by `PHENTRIEVE_HOST_DATA_DIR` in your `.env.docker` file
- **For local installations**: In the default locations specified in `phentrieve/config.py` (configurable via environment variables)

The data structure includes:

```text
/your/data/dir/
├── hpo_core_data/    # HPO source files (hp.json, etc.)
├── indexes/          # ChromaDB persistent storage
├── results/          # Benchmark results
└── hpo_translations/ # Translation files (if used)
```

### Web Application Components

Phentrieve provides a complete web application for easy interaction with the system:

#### API

The FastAPI backend provides RESTful endpoints:

- **Main Query Endpoint**: `/api/v1/query/`
- **Method**: POST
- **Example Request**:
  ```json
  {
    "text": "Der Patient zeigt Mikrozephalie und Krampfanfälle",
    "model_name": "FremyCompany/BioLORD-2023-M",
    "num_results": 5,
    "similarity_threshold": 0.3,
    "enable_reranker": true
  }
  ```

Complete API documentation is available at the `/docs` endpoint when running the API.

#### Frontend

The Vue/Vuetify frontend provides:

- User-friendly query interface
- Model selection dropdown
- Result display with HPO details and similarity scores
- Option toggles for reranking and other features

## Benchmarking Results

Extensive benchmarking has been performed to evaluate different embedding models for HPO term retrieval. Results show significant performance variations across models:

### GPU-Accelerated Results

Benchmarking with GPU acceleration (CUDA) shows the following metrics:

- **BioLORD-2023-M** (domain-specific biomedical model):
  - MRR: 0.5361
  - HR@1: 0.3333
  - HR@3: 0.6667
  - HR@5: 0.7778
  - HR@10: 1.0
  - Recall: 1.0

- **Jina-v2-base-de** (German language-specific model):
  - MRR: 0.3708
  - HR@1: 0.2222
  - HR@3: 0.4444
  - HR@5: 0.5556
  - HR@10: 0.7778
  - Recall: 0.7778

### Cross-Encoder Re-Ranking Model Comparison

Three different cross-encoder models have been tested for re-ranking HPO terms:

1. **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** (original implementation)
   - General multilingual retrieval model
   - Returns negative scores (higher/less negative = better)
   - Small and efficient, but not domain-specific

2. **MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7** (current default)
   - Multilingual natural language inference model
   - Returns probability distributions for entailment/neutral/contradiction
   - Strong multilingual capabilities with good performance

3. **ncbi/MedCPT-Cross-Encoder**
   - Biomedical domain-specific cross-encoder
   - Developed by NCBI specifically for medical text matching
   - Provides better understanding of medical relationships
   - Complements BioLORD bi-encoder for medical terminology

### Key Findings

- Domain-specific models (BioLORD) consistently outperform language-specific models for medical terminology retrieval
- GPU acceleration significantly improves processing speed while maintaining quality
- Cross-encoder re-ranking improves precision at the cost of additional processing time
- The choice between multilingual and domain-specific cross-encoders depends on the specific use case

## References

- Human Phenotype Ontology: [HPO website](https://hpo.jax.org/)
- Sentence Transformers: [SBERT website](https://www.sbert.net/)
- ChromaDB: [Documentation](https://docs.trychroma.com/)
- Semantic Similarity in Biomedical Ontologies: [Publication](https://doi.org/10.1371/journal.pcbi.1000443)
