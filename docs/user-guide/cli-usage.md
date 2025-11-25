# CLI Usage Guide

Phentrieve provides a comprehensive command-line interface (CLI) built with Typer for accessing all its functionality. This guide covers the main commands and their options.

## Command Structure

Phentrieve commands follow this general structure:

```bash
phentrieve <command> <subcommand> [options]
```

## Global Options

Available for all commands:

*   `--debug`: Enable verbose logging for debugging purposes
*   `--version`: Show version information and exit
*   `--help`: Show help message for any command

## Available Commands

### Data Management

```bash
# Download and process HPO data
phentrieve data prepare

# Clean all data directories (use with caution)
phentrieve data clean
```

### Index Management

```bash
# Build index for a specific model
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Build indexes for all supported models
phentrieve index build --all-models

# Clean indexes (removes all vector stores)
phentrieve index clean
```

### Interactive Querying

The query command allows you to find HPO terms that match a given text:

```bash
# Launch interactive query mode
phentrieve query --interactive

# Query with specific text
phentrieve query --text "The patient shows microcephaly and seizures"
```

#### Query Options

- `--text`: Text to process (if not provided, runs in interactive mode)
- `--similarity-threshold`: Minimum similarity score (0-1) to show results (default: 0.3)
- `--num-results`: Maximum number of results to display (default: 5)
- `--model-name`: Embedding model to use (default: "FremyCompany/BioLORD-2023-M")
- `--enable-reranker`: Enable cross-encoder reranking for improved precision
- `--reranker-model`: Cross-encoder model (default: "BAAI/bge-reranker-v2-m3")
- `--rerank-count`: Number of candidates to pass to reranker (default: 50)

### Text Processing

Process clinical text to extract HPO terms with advanced pipeline:

```bash
# Basic processing with default strategy
phentrieve text process "Patient has arachnodactyly but no scoliosis"

# Process with specific chunking strategy
phentrieve text process "..." --strategy sliding_window_punct_conj_cleaned

# Output as JSON Lines for machine parsing (useful for pipelines)
phentrieve text process "..." --output-format json_lines

# Process from file and save to file
phentrieve text process --input-file notes.txt --output-file results.jsonl \
  --output-format json_lines

# Override sliding window parameters for fine-tuning
phentrieve text process "..." \
  --strategy sliding_window_punct_conj_cleaned \
  --window-size 10 \
  --step-size 2 \
  --threshold 0.4 \
  --min-segment 5
```

#### Available Chunking Strategies

*   **`simple`**: Paragraph → Sentence splitting (fastest, least granular)
*   **`sliding_window`**: Semantic sliding window only
*   **`sliding_window_punct_conj_cleaned`** (Default): Full pipeline with:
    - Paragraph splitting
    - Sentence splitting
    - Fine-grained punctuation splitting
    - Conjunction splitting
    - Semantic sliding window
    - Final chunk cleaning

#### Output Formats

*   **`json_lines`** (Default): JSON Lines format - one JSON object per line (machine-readable)
*   **`rich_json_summary`**: Rich JSON with complete metadata (human + machine readable)
*   **`csv_hpo_list`**: CSV format with HPO IDs and labels (spreadsheet-friendly)

Example JSON Lines output:
```json
{"chunk_index":0,"chunk_text":"Patient has arachnodactyly","assertion":"affirmed","matches":[{"hpo_id":"HP:0001166","label":"Arachnodactyly","score":0.89}]}
{"chunk_index":1,"chunk_text":"no scoliosis","assertion":"negated","matches":[{"hpo_id":"HP:0002650","label":"Scoliosis","score":0.92}]}
```

#### Text Processing Options

- `--min-confidence`: Minimum similarity score threshold (0.0-1.0, default: 0.3)
- `--top-term-per-chunk`: Return only the highest-scoring HPO term per chunk (boolean)
- `--strategy`: Chunking strategy (see above)
- `--language`: Text language for accurate processing (en, de, es, fr, nl)
- `--output-format`: Output format (json_lines, rich_json_summary, csv_hpo_list)

**Sliding Window Parameters** (override config for all strategies using sliding window):
- `--window-size`: Window size in tokens (default: 7)
- `--step-size`: Step size in tokens (default: 1)
- `--threshold`: Semantic similarity threshold for splitting (default: 0.5)
- `--min-segment`: Minimum segment length in words (default: 3)

**Advanced Options**:
- `--input-file`, `-i`: Read text from file instead of argument
- `--output-file`, `-f`: Save results to file instead of stdout
- `--cross-language-hpo-retrieval`: Enable retrieval of HPO terms in a different language

### HPO Term Similarity

Calculate semantic similarity between two specific HPO terms:

```bash
# Calculate similarity between two HPO terms
phentrieve similarity calculate HP:0001250 HP:0001251 --formula hybrid
```

#### Similarity Options

- `--formula`: Similarity formula to use (hybrid, resnik, lin, jc, ic)

### Benchmarking

Run benchmarks to evaluate model performance:

```bash
# Run a benchmark with default settings
phentrieve benchmark run

# Run a benchmark with a specific model
phentrieve benchmark run --model-name "FremyCompany/BioLORD-2023-M"
```

#### Benchmarking Options

- `--model-name`: Model to benchmark
- `--test-file`: Path to test cases file
- `--output-dir`: Directory to save benchmark results
- `--enable-reranker`: Include reranking in the benchmark
- `--gpu`: Use GPU acceleration if available

## Getting Help

For any command, you can add `--help` to see available options:

```bash
phentrieve --help
phentrieve query --help
phentrieve text process --help
```

## Environment Variables

Phentrieve's behavior can be configured through environment variables:

- `PHENTRIEVE_DATA_DIR`: Base directory for all Phentrieve data
- `PHENTRIEVE_HPO_DATA_DIR`: Directory for HPO data files
- `PHENTRIEVE_INDEX_DIR`: Directory for vector indexes
- `PHENTRIEVE_RESULTS_DIR`: Directory for benchmark results
- `PHENTRIEVE_TRANSLATIONS_DIR`: Directory for translation files (if used)
