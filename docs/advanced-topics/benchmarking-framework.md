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

### Normalized Discounted Cumulative Gain (NDCG@K)

A ranking-aware metric that rewards relevant results at higher positions. Higher is better.

$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$
$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

Where $rel_i$ is the relevance score of the item at position $i$, and $IDCG@K$ is the ideal DCG.

### Recall@K

The proportion of relevant items that appear in the top K results.

$$Recall@K = \frac{|Retrieved@K \cap Relevant|}{|Relevant|}$$

### Precision@K

The proportion of top K results that are relevant.

$$Precision@K = \frac{|Retrieved@K \cap Relevant|}{K}$$

### Mean Average Precision (MAP@K)

The mean of average precision scores for each query, considering only the first K results.

$$AP@K = \frac{1}{\min(K, |Relevant|)} \sum_{k=1}^{K} Precision@k \times rel_k$$
$$MAP@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} AP@K_i$$

### Maximum Ontology Similarity (MaxOntSim@K)

A domain-specific metric that measures the maximum semantic similarity between any expected HPO term and any retrieved term in the top K results, using ontology-based similarity measures.

## Benchmark Results

The framework now includes comprehensive metrics aligned with industry standards (MTEB/BEIR). Example results from recent benchmarking:

### BioLORD-2023-M (Domain-specific Model)

- **MRR**: 0.5361
- **HR@1**: 0.3333, **HR@3**: 0.6667, **HR@5**: 0.7778, **HR@10**: 1.0
- **NDCG@1**: 0.3333, **NDCG@3**: 0.6111, **NDCG@5**: 0.7222, **NDCG@10**: 0.8611
- **Recall@1**: 0.1667, **Recall@3**: 0.5000, **Recall@5**: 0.6667, **Recall@10**: 1.0
- **Precision@1**: 1.0000, **Precision@3**: 1.0000, **Precision@5**: 1.0000, **Precision@10**: 1.0000
- **MAP@1**: 0.3333, **MAP@3**: 0.6111, **MAP@5**: 0.7222, **MAP@10**: 0.8611
- **MaxOntSim@1**: 0.8, **MaxOntSim@3**: 0.9, **MaxOntSim@5**: 0.95, **MaxOntSim@10**: 1.0

### Jina-v2-base-de (German-specific Model)

- **MRR**: 0.3708
- **HR@1**: 0.2222, **HR@3**: 0.4444, **HR@5**: 0.5556, **HR@10**: 0.7778
- **NDCG@1**: 0.2222, **NDCG@3**: 0.4074, **NDCG@5**: 0.4815, **NDCG@10**: 0.6296
- **Recall@1**: 0.1111, **Recall@3**: 0.3333, **Recall@5**: 0.4444, **Recall@10**: 0.7778
- **Precision@1**: 1.0000, **Precision@3**: 1.0000, **Precision@5**: 1.0000, **Precision@10**: 0.8889
- **MAP@1**: 0.2222, **MAP@3**: 0.4074, **MAP@5**: 0.4815, **MAP@10**: 0.6296
- **MaxOntSim@1**: 0.7, **MaxOntSim@3**: 0.8, **MaxOntSim@5**: 0.85, **MaxOntSim@10**: 0.95

These results demonstrate that domain-specific models (BioLORD) consistently outperform language-specific models for medical terminology retrieval. The new metrics provide more nuanced evaluation of ranking quality and retrieval effectiveness.

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
