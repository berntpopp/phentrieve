# HPO Extraction Benchmark Implementation Plan

**Status:** Active Implementation
**Created:** 2025-12-08
**Timeline:** 4 weeks
**Framework Document:** [HPO-EXTRACTION-BENCHMARKING-FRAMEWORK.md](./HPO-EXTRACTION-BENCHMARKING-FRAMEWORK.md)
**Primary Issue:** [#17 - Full Clinical Text HPO Extraction Benchmark](https://github.com/berntpopp/phentrieve/issues/17)

## Executive Summary

This document provides the concrete implementation plan for the HPO Extraction Benchmarking Framework. It translates the research and design work into actionable development tasks with clear deliverables, timelines, and integration points.

**Key Distinction:** This focuses on **document-level extraction evaluation** (full clinical texts with multiple HPO terms), not single-term retrieval benchmarking.

## Implementation Phases

### Phase 1: Core Metrics & Infrastructure (Week 1)

#### 1.1 Base Evaluation Module

**File:** `phentrieve/evaluation/extraction_metrics.py`

```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats

@dataclass
class ExtractionResult:
    """Single document extraction result."""
    doc_id: str
    predicted: List[Tuple[str, str]]  # (hpo_id, assertion)
    gold: List[Tuple[str, str]]       # (hpo_id, assertion)

@dataclass
class CorpusMetrics:
    """Corpus-level evaluation metrics."""
    micro: Dict[str, float]  # precision, recall, f1
    macro: Dict[str, float]  # precision, recall, f1
    weighted: Dict[str, float]  # weighted by doc size
    confidence_intervals: Dict[str, Tuple[float, float]]

class CorpusExtractionMetrics:
    """Document-level HPO extraction evaluation."""

    def __init__(self, averaging: str = "micro"):
        self.averaging = averaging

    def calculate_metrics(
        self,
        results: List[ExtractionResult]
    ) -> CorpusMetrics:
        """Calculate corpus-level metrics with different averaging strategies."""
        pass

    def bootstrap_confidence_intervals(
        self,
        results: List[ExtractionResult],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for metrics."""
        pass
```

#### 1.2 Document Aggregation Strategy

**File:** `phentrieve/evaluation/document_aggregation.py`

```python
from typing import List, Set, Dict
from collections import defaultdict

class DocumentAggregator:
    """Aggregate chunk-level predictions to document-level."""

    def __init__(self, strategy: str = "union"):
        """
        Args:
            strategy: "union", "intersection", "weighted", "threshold"
        """
        self.strategy = strategy

    def aggregate_chunks(
        self,
        chunk_predictions: List[Dict[str, float]]
    ) -> Set[str]:
        """Aggregate HPO predictions from multiple chunks."""
        if self.strategy == "union":
            return self._union_aggregation(chunk_predictions)
        elif self.strategy == "intersection":
            return self._intersection_aggregation(chunk_predictions)
        elif self.strategy == "weighted":
            return self._weighted_aggregation(chunk_predictions)
        elif self.strategy == "threshold":
            return self._threshold_aggregation(chunk_predictions)

    def _union_aggregation(self, chunks):
        """Take all unique HPO terms from all chunks."""
        pass

    def _weighted_aggregation(self, chunks):
        """Weight by confidence scores and chunk importance."""
        pass
```

#### 1.3 Evaluation Pipeline

**File:** `phentrieve/benchmark/extraction_benchmark.py`

```python
import json
from pathlib import Path
from typing import Dict, List, Optional
import typer

class ExtractionBenchmark:
    """Main benchmark runner for document-level extraction."""

    def __init__(
        self,
        model_name: str,
        config_path: Optional[Path] = None
    ):
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.extractor = self._init_extractor()

    def run_benchmark(
        self,
        test_file: Path,
        output_dir: Path
    ) -> Dict:
        """Run extraction benchmark on test dataset."""
        # Load test data
        test_data = self._load_test_data(test_file)

        # Process each document
        results = []
        for doc in test_data["documents"]:
            extracted = self.extractor.extract(doc["text"])
            results.append({
                "doc_id": doc["id"],
                "predicted": extracted,
                "gold": doc["gold_hpo_terms"]
            })

        # Calculate metrics
        evaluator = CorpusExtractionMetrics()
        metrics = evaluator.calculate_metrics(results)

        # Save results
        self._save_results(results, metrics, output_dir)

        return metrics
```

### Phase 2: Assertion & Joint Metrics (Week 2)

#### 2.1 Joint F1 Implementation

**File:** `phentrieve/evaluation/assertion_metrics.py`

```python
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import confusion_matrix

class AssertionMetrics:
    """Evaluate HPO term + assertion jointly."""

    ASSERTION_TYPES = ["PRESENT", "ABSENT", "UNCERTAIN"]

    def calculate_joint_f1(
        self,
        predicted: List[Tuple[str, str]],
        gold: List[Tuple[str, str]]
    ) -> float:
        """
        Calculate F1 where match requires both:
        1. Correct HPO term
        2. Correct assertion
        """
        # Convert to sets for comparison
        pred_set = set(predicted)
        gold_set = set(gold)

        # Calculate joint matches
        true_positives = len(pred_set & gold_set)
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if pred_set else 0
        recall = true_positives / (true_positives + false_negatives) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def assertion_confusion_matrix(
        self,
        predicted_assertions: List[str],
        gold_assertions: List[str]
    ) -> np.ndarray:
        """Build confusion matrix for assertion types."""
        return confusion_matrix(
            gold_assertions,
            predicted_assertions,
            labels=self.ASSERTION_TYPES
        )

    def stratified_metrics(
        self,
        results: List[ExtractionResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics stratified by assertion type."""
        metrics_by_assertion = {}

        for assertion_type in self.ASSERTION_TYPES:
            # Filter results by assertion type
            filtered_results = self._filter_by_assertion(
                results, assertion_type
            )

            # Calculate standard metrics
            evaluator = CorpusExtractionMetrics()
            metrics = evaluator.calculate_metrics(filtered_results)
            metrics_by_assertion[assertion_type] = metrics.micro

        return metrics_by_assertion
```

#### 2.2 Enhanced Assertion Detection

```python
class AssertionDetector:
    """Enhanced assertion detection with joint evaluation."""

    def __init__(self, enable_context: bool = True):
        self.enable_context = enable_context
        self.context_analyzer = self._init_context_analyzer()

    def detect_assertion(
        self,
        text: str,
        hpo_term: str,
        span: Tuple[int, int]
    ) -> str:
        """
        Detect assertion type for HPO term in text.
        Returns: PRESENT, ABSENT, or UNCERTAIN
        """
        if self.enable_context:
            # Use ConText algorithm
            return self.context_analyzer.analyze(text, span)
        else:
            # Simple keyword-based detection
            return self._simple_detection(text, span)
```

### Phase 3: Relaxed Matching & Hierarchical Evaluation (Week 3)

#### 3.1 Hierarchical Matching System

**File:** `phentrieve/evaluation/hierarchical_matching.py`

```python
from enum import Enum
from typing import Optional, Dict, Tuple
import networkx as nx

class MatchType(Enum):
    """Types of hierarchical matches."""
    EXACT = "exact"
    ANCESTOR = "ancestor"      # Predicted is ancestor of gold
    DESCENDANT = "descendant"  # Predicted is descendant of gold
    SIBLING = "sibling"        # Share immediate parent
    COUSIN = "cousin"          # Share grandparent
    UNRELATED = "unrelated"    # No close relationship

class HierarchicalMatcher:
    """Match predictions with hierarchical relationships."""

    def __init__(self, hpo_graph: nx.DiGraph):
        self.hpo_graph = hpo_graph
        self.ic_scores = self._calculate_information_content()

    def classify_match(
        self,
        predicted_id: str,
        gold_id: str
    ) -> MatchType:
        """Classify the relationship between predicted and gold HPO terms."""
        if predicted_id == gold_id:
            return MatchType.EXACT

        # Check ancestor/descendant relationships
        if self._is_ancestor(predicted_id, gold_id):
            return MatchType.ANCESTOR
        elif self._is_ancestor(gold_id, predicted_id):
            return MatchType.DESCENDANT

        # Check sibling/cousin relationships
        pred_parents = self._get_parents(predicted_id)
        gold_parents = self._get_parents(gold_id)

        if pred_parents & gold_parents:
            return MatchType.SIBLING

        pred_grandparents = self._get_grandparents(predicted_id)
        gold_grandparents = self._get_grandparents(gold_id)

        if pred_grandparents & gold_grandparents:
            return MatchType.COUSIN

        return MatchType.UNRELATED

    def calculate_partial_credit(
        self,
        match_type: MatchType,
        predicted_id: str,
        gold_id: str
    ) -> float:
        """
        Calculate partial credit based on match type and distance.
        Returns value between 0 and 1.
        """
        credit_map = {
            MatchType.EXACT: 1.0,
            MatchType.ANCESTOR: self._ancestor_credit(predicted_id, gold_id),
            MatchType.DESCENDANT: self._descendant_credit(predicted_id, gold_id),
            MatchType.SIBLING: 0.7,
            MatchType.COUSIN: 0.5,
            MatchType.UNRELATED: 0.0
        }
        return credit_map[match_type]

    def _ancestor_credit(self, ancestor_id: str, descendant_id: str) -> float:
        """Credit based on specificity loss."""
        distance = nx.shortest_path_length(
            self.hpo_graph, ancestor_id, descendant_id
        )
        return max(0.5, 1.0 - (distance * 0.1))

    def _descendant_credit(self, descendant_id: str, ancestor_id: str) -> float:
        """Credit based on over-specificity."""
        distance = nx.shortest_path_length(
            self.hpo_graph, ancestor_id, descendant_id
        )
        return max(0.7, 1.0 - (distance * 0.05))
```

#### 3.2 Data-Driven Threshold Selection

**File:** `phentrieve/evaluation/threshold_optimization.py`

```python
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import precision_recall_curve

class ThresholdOptimizer:
    """Learn optimal similarity thresholds from validation data."""

    def __init__(self, optimization_metric: str = "f1"):
        self.optimization_metric = optimization_metric
        self.learned_thresholds = {}

    def optimize_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        model_name: str
    ) -> float:
        """
        Find optimal threshold for a specific model.

        Args:
            scores: Similarity scores from model
            labels: Binary labels (1 = correct, 0 = incorrect)
            model_name: Name of the model for storing threshold

        Returns:
            Optimal threshold value
        """
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)

        if self.optimization_metric == "f1":
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        elif self.optimization_metric == "precision_at_recall":
            # Find highest precision at minimum recall (e.g., 0.8)
            min_recall = 0.8
            valid_idx = np.where(recalls >= min_recall)[0]
            if len(valid_idx) > 0:
                optimal_idx = valid_idx[np.argmax(precisions[valid_idx])]
            else:
                optimal_idx = 0

        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Store for this model
        self.learned_thresholds[model_name] = optimal_threshold

        return optimal_threshold

    def apply_threshold(
        self,
        scores: Dict[str, float],
        model_name: str
    ) -> List[str]:
        """Apply learned threshold to filter predictions."""
        threshold = self.learned_thresholds.get(model_name, 0.7)  # Default to 0.7
        return [
            hpo_id for hpo_id, score in scores.items()
            if score >= threshold
        ]
```

### Phase 4: CLI Integration & Reporting (Week 4)

#### 4.1 CLI Commands

**File:** `phentrieve/benchmark/extraction_cli.py`

```python
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="HPO Extraction Benchmarking")
console = Console()

@app.command()
def run(
    test_file: Path = typer.Argument(
        ...,
        help="Path to test dataset JSON file"
    ),
    model: str = typer.Option(
        "BAAI/bge-m3",
        help="Embedding model to use"
    ),
    output_dir: Path = typer.Option(
        Path("results/extraction"),
        help="Output directory for results"
    ),
    averaging: str = typer.Option(
        "micro",
        help="Averaging strategy: micro, macro, or weighted"
    ),
    include_assertions: bool = typer.Option(
        True,
        help="Include assertion detection in evaluation"
    ),
    relaxed_matching: bool = typer.Option(
        False,
        help="Enable hierarchical relaxed matching"
    ),
    bootstrap_ci: bool = typer.Option(
        True,
        help="Calculate bootstrap confidence intervals"
    ),
):
    """Run extraction benchmark on test dataset."""

    console.print(f"[bold cyan]Running extraction benchmark[/bold cyan]")
    console.print(f"Test file: {test_file}")
    console.print(f"Model: {model}")

    # Initialize benchmark
    benchmark = ExtractionBenchmark(model)

    # Configure evaluation options
    config = {
        "averaging": averaging,
        "include_assertions": include_assertions,
        "relaxed_matching": relaxed_matching,
        "bootstrap_ci": bootstrap_ci
    }

    # Run benchmark
    with console.status("Processing documents..."):
        results = benchmark.run_benchmark(test_file, output_dir, config)

    # Display results
    _display_results(results)

@app.command()
def compare(
    result1: Path = typer.Argument(..., help="First result file"),
    result2: Path = typer.Argument(..., help="Second result file"),
    output_file: Optional[Path] = typer.Option(None, help="Save comparison")
):
    """Compare two extraction benchmark results."""

    console.print("[bold cyan]Comparing benchmark results[/bold cyan]")

    # Load results
    r1 = _load_results(result1)
    r2 = _load_results(result2)

    # Create comparison table
    table = Table(title="Extraction Benchmark Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Result 1", style="magenta")
    table.add_column(f"Result 2", style="magenta")
    table.add_column("Difference", style="yellow")

    # Add metrics
    for metric in ["precision", "recall", "f1", "joint_f1"]:
        val1 = r1["corpus_metrics"]["micro"].get(metric, 0)
        val2 = r2["corpus_metrics"]["micro"].get(metric, 0)
        diff = val2 - val1

        table.add_row(
            metric.capitalize(),
            f"{val1:.3f}",
            f"{val2:.3f}",
            f"{diff:+.3f}" if diff != 0 else "="
        )

    console.print(table)

    # Statistical significance test
    if "confidence_intervals" in r1 and "confidence_intervals" in r2:
        _test_significance(r1, r2)

    # Save if requested
    if output_file:
        _save_comparison(r1, r2, output_file)

@app.command()
def report(
    results_dir: Path = typer.Argument(
        ...,
        help="Directory containing benchmark results"
    ),
    output_format: str = typer.Option(
        "markdown",
        help="Output format: markdown, html, or latex"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        help="Save report to file"
    )
):
    """Generate comprehensive benchmark report."""

    console.print("[bold cyan]Generating extraction benchmark report[/bold cyan]")

    # Load all results in directory
    results = _load_all_results(results_dir)

    # Generate report
    reporter = ExtractionReporter(output_format)
    report = reporter.generate_report(results)

    # Display or save
    if output_file:
        output_file.write_text(report)
        console.print(f"[green]Report saved to {output_file}[/green]")
    else:
        console.print(report)
```

#### 4.2 Visualization & Reporting

**File:** `phentrieve/benchmark/extraction_reporter.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd

class ExtractionReporter:
    """Generate reports and visualizations for extraction benchmarks."""

    def __init__(self, output_format: str = "markdown"):
        self.output_format = output_format
        sns.set_theme(style="whitegrid")

    def generate_report(self, results: List[Dict]) -> str:
        """Generate comprehensive benchmark report."""
        if self.output_format == "markdown":
            return self._markdown_report(results)
        elif self.output_format == "html":
            return self._html_report(results)
        elif self.output_format == "latex":
            return self._latex_report(results)

    def plot_pr_curve(self, results: Dict, save_path: Path):
        """Plot precision-recall curve."""
        plt.figure(figsize=(8, 6))

        # Extract PR points
        precisions = results.get("pr_curve", {}).get("precisions", [])
        recalls = results.get("pr_curve", {}).get("recalls", [])

        plt.plot(recalls, precisions, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for HPO Extraction')
        plt.grid(True, alpha=0.3)

        # Add F1 iso-curves
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            x = np.linspace(0.01, 1)
            y = f1 * x / (2 * x - f1)
            plt.plot(x[y >= 0], y[y >= 0], 'k--', alpha=0.2)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_assertion_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Path
    ):
        """Plot confusion matrix for assertion detection."""
        plt.figure(figsize=(8, 6))

        labels = ["PRESENT", "ABSENT", "UNCERTAIN"]
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )

        plt.title('Assertion Detection Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_hierarchical_matching_distribution(
        self,
        match_types: Dict[str, int],
        save_path: Path
    ):
        """Plot distribution of hierarchical match types."""
        plt.figure(figsize=(10, 6))

        # Create bar plot
        types = list(match_types.keys())
        counts = list(match_types.values())

        bars = plt.bar(types, counts, color='steelblue')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                str(count),
                ha='center',
                va='bottom'
            )

        plt.xlabel('Match Type')
        plt.ylabel('Count')
        plt.title('Distribution of Hierarchical Match Types')
        plt.xticks(rotation=45)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
```

### Test Implementation

**File:** `tests/test_extraction_metrics.py`

```python
import pytest
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult
)

class TestCorpusExtractionMetrics:

    def test_micro_averaging(self):
        """Test micro-averaged metrics calculation."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT"), ("HP:0002360", "ABSENT")],
                gold=[("HP:0001250", "PRESENT"), ("HP:0001251", "PRESENT")]
            ),
            ExtractionResult(
                doc_id="doc2",
                predicted=[("HP:0001252", "PRESENT")],
                gold=[("HP:0001252", "PRESENT"), ("HP:0001253", "ABSENT")]
            )
        ]

        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_metrics(results)

        # 2 true positives, 2 false positives, 2 false negatives
        assert metrics.micro["precision"] == 0.5  # 2/4
        assert metrics.micro["recall"] == 0.5     # 2/4
        assert metrics.micro["f1"] == 0.5

    def test_macro_averaging(self):
        """Test macro-averaged metrics calculation."""
        # Test implementation
        pass

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap CI calculation."""
        # Test implementation
        pass
```

## Integration Points

### 1. With Existing Codebase

- **Retrieval Pipeline**: Reuse `phentrieve.retrieval.dense_retriever`
- **Text Processing**: Use `phentrieve.text_processing.chunker`
- **HPO Database**: Access via `phentrieve.database.hpo_database`
- **Embeddings**: Use existing `phentrieve.embeddings`

### 2. CLI Integration

Add to main CLI in `phentrieve/cli.py`:

```python
# Add new command group
app.add_typer(
    extraction_cli.app,
    name="extraction",
    help="Document-level extraction benchmarking"
)
```

### 3. Configuration

Add to `phentrieve.yaml.template`:

```yaml
extraction_benchmark:
  averaging_strategy: micro  # micro, macro, weighted
  enable_assertions: true
  enable_relaxed_matching: false
  bootstrap_samples: 1000
  confidence_level: 0.95
```

## Test Data Specification

### Required Format

```json
{
  "metadata": {
    "dataset_name": "clinical_extraction_v1",
    "language": "en",
    "annotation_guidelines": "URL or description",
    "inter_annotator_agreement": 0.85,
    "is_exhaustive": false
  },
  "documents": [
    {
      "id": "doc_001",
      "text": "The patient presents with seizures and developmental delay...",
      "gold_hpo_terms": [
        {
          "id": "HP:0001250",
          "assertion": "PRESENT",
          "span": [27, 35]  # Optional: character offsets
        },
        {
          "id": "HP:0001263",
          "assertion": "PRESENT",
          "span": [40, 58]
        }
      ],
      "metadata": {
        "source": "clinical_notes",
        "specialty": "neurology"
      }
    }
  ]
}
```

### Validation Requirements

- All HPO IDs must exist in current ontology
- Assertions must be from: PRESENT, ABSENT, UNCERTAIN
- Document text must be non-empty
- Unique document IDs required

## Success Metrics

### Week 1 Deliverables
- [ ] Core metrics module with micro/macro/weighted averaging
- [ ] Document aggregation strategies implemented
- [ ] Basic evaluation pipeline functional
- [ ] 10+ unit tests passing

### Week 2 Deliverables
- [ ] Joint F1 metric implemented
- [ ] Assertion confusion matrix generation
- [ ] Stratified metrics by assertion type
- [ ] 20+ unit tests passing

### Week 3 Deliverables
- [ ] Hierarchical matching with 6 match types
- [ ] Partial credit calculation
- [ ] Data-driven threshold optimization
- [ ] 30+ unit tests passing

### Week 4 Deliverables
- [ ] CLI commands integrated
- [ ] Report generation (markdown, HTML)
- [ ] Visualization plots (PR curve, confusion matrix)
- [ ] Complete documentation
- [ ] 40+ unit tests with >80% coverage

## Dependencies

### Required Python Packages
- numpy>=1.24.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- pandas>=2.0.0
- networkx>=3.0 (for hierarchical matching)

### Existing Phentrieve Modules
- phentrieve.retrieval
- phentrieve.text_processing
- phentrieve.database
- phentrieve.embeddings

## Risk Mitigation

### Risk 1: Test Data Quality
**Mitigation**: Start with synthetic test data if real annotations unavailable

### Risk 2: Performance on Large Documents
**Mitigation**: Implement batch processing and optional sampling

### Risk 3: Complex Hierarchical Relationships
**Mitigation**: Start with simple exact/ancestor/descendant, add complex relationships later

## Next Steps

1. **Immediate**: Create directory structure and base files
2. **Day 1-2**: Implement CorpusExtractionMetrics class
3. **Day 3-4**: Add document aggregation strategies
4. **Day 5**: Create basic CLI integration
5. **Week 2**: Begin assertion metrics implementation

## Related Issues

- **Primary**: [#17](https://github.com/berntpopp/phentrieve/issues/17) - Full Clinical Text HPO Extraction Benchmark
- **Deferred**: [#130](https://github.com/berntpopp/phentrieve/issues/130) - Dataset quality (can validate after implementation)
- **Deferred**: [#25](https://github.com/berntpopp/phentrieve/issues/25) - Chunking strategies (evaluate using this framework)
- **Future**: [#126](https://github.com/berntpopp/phentrieve/issues/126) - Negation particles (diagnose with assertion metrics)

## References

- [HPO Extraction Framework Document](./HPO-EXTRACTION-BENCHMARKING-FRAMEWORK.md)
- [Single-Term Retrieval Benchmark](../02-completed/SINGLE-TERM-RETRIEVAL-BENCHMARK.md)
- [SemEval Clinical Concept Extraction](https://www.aclweb.org/anthology/S13-2056/)
- [Information Content in Ontologies](https://doi.org/10.1186/1471-2105-7-302)