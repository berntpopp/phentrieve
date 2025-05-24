# CLI Usage Guide

Phentrieve provides a comprehensive command-line interface (CLI) for accessing all its functionality. This guide covers the main commands and their options.

## Command Structure

Phentrieve commands follow this general structure:

```bash
phentrieve <command> <subcommand> [options]
```

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
- `--reranker-mode`: Reranking mode, either "crosslingual" (default) or "monolingual"
- `--reranker-model`: Cross-encoder model to use for reranking

### Text Processing

Process clinical text to extract HPO terms:

```bash
# Process a text file
phentrieve text process --input-file clinical_notes.txt --output-file results.json

# Process text directly
phentrieve text process "The patient exhibits microcephaly and frequent seizures."
```

#### Text Processing Options

- `--min-confidence`: Set a threshold for minimum similarity score (0.0-1.0)
- `--top-term-per-chunk`: Return only the highest-scoring HPO term for each text chunk
- `--strategy`: Choose text chunking strategy (simple, semantic, detailed, sliding_window)
- `--language`: Specify text language for accurate chunking and assertion detection
- `--window-size`: Size of the sliding window (for sliding_window strategy)
- `--step-size`: Step size for the sliding window (for sliding_window strategy)
- `--threshold`: Semantic similarity threshold for chunking
- `--min-segment`: Minimum segment length for semantic chunking

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
