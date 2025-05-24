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

## Further Reading

For more advanced benchmarking information, see the [Benchmarking Framework](../advanced-topics/benchmarking-framework.md) page in the Advanced Topics section.
