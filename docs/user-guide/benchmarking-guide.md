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

## Further Reading

For more advanced benchmarking information, see the [Benchmarking Framework](../advanced-topics/benchmarking-framework.md) page in the Advanced Topics section.
