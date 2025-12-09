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

$$AP@K = \frac{1}{\min(K, |Relevant|)} \sum_{k=1}^{K} Precision@k \times \mathbb{1}(\text{item at rank } k \text{ is relevant})$$
$$MAP@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} AP@K_i$$

Where $\mathbb{1}(\text{item at rank } k \text{ is relevant})$ is 1 if the item at position $k$ is relevant, and 0 otherwise.
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

## Extraction Benchmarking

The extraction benchmark evaluates document-level HPO extraction from clinical text against gold-standard annotations.

### Overview

Unlike retrieval benchmarks (which test single-term lookups), extraction benchmarks evaluate the full pipeline:

1. **Text chunking**: Split documents into semantic units
2. **HPO retrieval**: Find candidate terms for each chunk
3. **Aggregation**: Deduplicate and rank results across chunks
4. **Assertion detection**: Identify present vs. absent phenotypes

### Test Data Format

Extraction benchmarks use the PhenoBERT directory format:

```
tests/data/en/phenobert/
├── GSC_plus/annotations/     # 228 documents
├── ID_68/annotations/        # 68 documents
├── GeneReviews/annotations/  # 10 documents
└── conversion_report.json
```

Each annotation file:

```json
{
  "doc_id": "GSC+_1003450",
  "full_text": "Clinical description...",
  "annotations": [
    {
      "hpo_id": "HP:0001250",
      "assertion_status": "affirmed",
      "evidence_spans": [{"start_char": 14, "end_char": 27}]
    }
  ]
}
```

### Extraction Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Precision | TP / (TP + FP) | Fraction of predictions that are correct |
| Recall | TP / (TP + FN) | Fraction of gold terms that were found |
| F1 | 2 × P × R / (P + R) | Harmonic mean of precision and recall |

**Averaging strategies:**

- **Micro**: Aggregate TP/FP/FN across all documents, then calculate
- **Macro**: Calculate per-document, then average
- **Weighted**: Weight by document size (gold term count)

### Tuning Parameters

The extraction benchmark exposes key pipeline parameters:

| Parameter | Effect on Precision | Effect on Recall |
|-----------|--------------------:|----------------:|
| `--num-results` ↓ | ↑ Higher | ↓ Lower |
| `--chunk-threshold` ↑ | ↑ Higher | ↓ Lower |
| `--min-confidence` ↑ | ↑ Higher | ↓ Lower |
| `--top-term-only` | ↑ Much higher | ↓ Much lower |

### Example Results

GeneReviews dataset (10 documents, 237 annotations):

| Configuration | Precision | Recall | F1 |
|--------------|----------:|-------:|---:|
| Default | 0.088 | 0.422 | 0.145 |
| `--top-term-only` | 0.143 | 0.262 | 0.185 |

### Running Extraction Benchmarks

```bash
# Quick test (10 documents)
phentrieve benchmark extraction run tests/data/en/phenobert/ \
    --dataset GeneReviews --no-bootstrap-ci

# Full evaluation (306 documents)
phentrieve benchmark extraction run tests/data/en/phenobert/ \
    --model "FremyCompany/BioLORD-2023-M"

# Compare configurations
phentrieve benchmark extraction compare \
    results/default/extraction_results.json \
    results/top_term/extraction_results.json
```
