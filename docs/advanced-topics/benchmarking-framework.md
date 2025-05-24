# Benchmarking Framework

Phentrieve includes a comprehensive benchmarking framework for evaluating and comparing different embedding models and configurations. This page explains how to use the framework and interpret the results.

## Overview

The benchmarking framework tests how well different models map clinical text to the correct HPO terms. It uses a set of test cases with ground truth HPO terms and measures various metrics to evaluate performance.

## Running Benchmarks

### Basic Usage

```bash
# Run a benchmark with default settings
phentrieve benchmark run

# Run a benchmark with a specific model
phentrieve benchmark run --model-name "FremyCompany/BioLORD-2023-M"

# Run a benchmark with re-ranking enabled
phentrieve benchmark run --enable-reranker
```

### Advanced Options

```bash
# Specify a custom test file
phentrieve benchmark run --test-file path/to/test_cases.json

# Set a custom output directory for results
phentrieve benchmark run --output-dir path/to/results

# Run benchmark with GPU acceleration
phentrieve benchmark run --gpu
```

## Metrics

The benchmarking framework calculates several standard information retrieval metrics:

### Mean Reciprocal Rank (MRR)

Measures the average position of the first relevant result in the ranked list. Higher is better.

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

Where $rank_i$ is the position of the first relevant result for query $i$.

### Hit Rate at K (HR@K)

The proportion of queries where a relevant result appears in the top K positions. Higher is better.

$$HR@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{1}(rank_i \leq K)$$

Where $\mathbb{1}(rank_i \leq K)$ is 1 if the rank is less than or equal to K, and 0 otherwise.

### Recall

The proportion of relevant items that are retrieved.

$$Recall = \frac{\text{Number of relevant items retrieved}}{\text{Total number of relevant items}}$$

## Benchmark Results

According to our project memories, GPU-accelerated benchmarking has shown these results:

### BioLORD-2023-M (Domain-specific Model)

- MRR: 0.5361
- HR@1: 0.3333
- HR@3: 0.6667
- HR@5: 0.7778
- HR@10: 1.0
- Recall: 1.0

### Jina-v2-base-de (German-specific Model)

- MRR: 0.3708
- HR@1: 0.2222
- HR@3: 0.4444
- HR@5: 0.5556
- HR@10: 0.7778
- Recall: 0.7778

These results demonstrate that domain-specific models (BioLORD) consistently outperform language-specific models for medical terminology retrieval.

## Visualizing Results

The benchmarking framework generates visualizations to help interpret the results:

- Bar charts comparing metrics across models
- Precision-recall curves
- Hit rate curves

These visualizations are saved in the `results/visualizations/` directory.

## Customizing Benchmarks

### Custom Test Cases

You can create custom test cases for benchmarking. The test file should be in JSON format:

```json
[
  {
    "text": "The patient exhibits microcephaly and seizures.",
    "hpo_ids": ["HP:0000252", "HP:0001250"],
    "language": "en"
  },
  {
    "text": "Der Patient zeigt Mikrozephalie und Krampfanfälle.",
    "hpo_ids": ["HP:0000252", "HP:0001250"],
    "language": "de"
  }
]
```

### Custom Metrics

You can implement custom metrics by extending the benchmarking framework:

```python
from phentrieve.evaluation.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self):
        super().__init__(name="custom_metric")
    
    def calculate(self, results):
        # Implement your custom metric calculation
        return value
```
