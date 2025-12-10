# Benchmarking Guide

This page provides a guide to using Phentrieve's benchmarking capabilities for evaluating model performance.

## Introduction

Benchmarking is essential for evaluating the performance of different embedding models and configurations. Phentrieve includes a comprehensive benchmarking framework that allows you to compare model performance using standardized metrics.

## Running Benchmarks

```bash
# Run a benchmark with default settings
phentrieve benchmark run

# Run a benchmark with a specific model
phentrieve benchmark run --model-name "FremyCompany/BioLORD-2023-M"

# Run a benchmark with re-ranking enabled
phentrieve benchmark run --enable-reranker
```

## Benchmark Metrics

The benchmarking framework calculates several information retrieval metrics:

- **Mean Reciprocal Rank (MRR)**: Average position of the first relevant result
- **Hit Rate at K (HR@K)**: Proportion of queries with a relevant result in the top K positions
- **Recall**: Proportion of relevant items that are retrieved

## Interpreting Results

Benchmark results are stored in the `results/` directory:

- `summaries/`: JSON summaries for each model
- `visualizations/`: Charts and plots comparing model performance
- `detailed/`: Detailed CSV results

## Extraction Benchmarking

Evaluate document-level HPO extraction against gold-standard annotations:

```bash
# Run on PhenoBERT test data (306 documents)
phentrieve benchmark extraction run tests/data/en/phenobert/

# Run on specific dataset (GeneReviews: 10 docs, good for quick tests)
phentrieve benchmark extraction run tests/data/en/phenobert/ --dataset GeneReviews

# High precision mode (fewer false positives)
phentrieve benchmark extraction run tests/data/en/phenobert/ --top-term-only

# Custom thresholds
phentrieve benchmark extraction run tests/data/en/phenobert/ \
    --chunk-threshold 0.6 --min-confidence 0.6 --num-results 2
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | all | PhenoBERT subset: `all`, `GSC_plus`, `ID_68`, `GeneReviews` |
| `--num-results` | 3 | HPO candidates per chunk |
| `--chunk-threshold` | 0.5 | Minimum similarity for chunk matching |
| `--min-confidence` | 0.5 | Minimum confidence for final results |
| `--top-term-only` | false | Keep only best match per chunk |

### Extraction Metrics

- **Precision**: Proportion of predicted terms that are correct
- **Recall**: Proportion of gold terms that were found
- **F1 Score**: Harmonic mean of precision and recall
- **Bootstrap CI**: 95% confidence intervals via bootstrap sampling

### Comparing Results

```bash
# Compare two benchmark runs
phentrieve benchmark extraction compare results/run1/extraction_results.json \
    results/run2/extraction_results.json

# Generate report from multiple runs
phentrieve benchmark extraction report results/
```

## Multi-Vector vs Single-Vector Comparison

Compare the performance of single-vector embeddings against multi-vector embeddings with different aggregation strategies:

```bash
# Compare with default strategies
phentrieve benchmark compare-vectors

# Compare specific strategies on a dataset
phentrieve benchmark compare-vectors \
    --test-file german/200cases_gemini_v1.json \
    --strategies "label_synonyms_max,all_max,label_only"

# Skip single-vector (only compare multi-vector strategies)
phentrieve benchmark compare-vectors --no-single \
    --strategies "label_synonyms_max,all_max,all_weighted"
```

### Aggregation Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `label_synonyms_max` | Best match between label and synonyms | **Recommended default** |
| `label_only` | Match only against label vectors | High precision |
| `all_max` | Best match across all components | Balanced |
| `all_weighted` | Weighted combination of all components | Custom tuning |

### Example Results

Results from 200-case German benchmark dataset:

| Mode | Strategy | MRR | Hit@1 | Hit@10 |
|------|----------|-----|-------|--------|
| single-vector | - | 0.824 | 74.0% | 95.0% |
| multi-vector | label_synonyms_max | **0.937** | **91.0%** | **98.0%** |
| multi-vector | label_only | 0.943 | 92.0% | 97.5% |
| multi-vector | all_max | 0.934 | 90.5% | 98.5% |

Multi-vector embeddings consistently outperform single-vector by **+13-21% MRR**.

## Further Reading

For more advanced benchmarking information, see the [Benchmarking Framework](../advanced-topics/benchmarking-framework.md) page in the Advanced Topics section.
