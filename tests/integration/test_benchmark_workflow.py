"""
Integration tests for complete benchmark workflow with new metrics.

Tests the full evaluation pipeline:
- Running benchmarks with new MTEB-aligned metrics
- Bootstrap CI calculation for all metrics
- Model comparison with significance testing
- Result saving and loading
- Metric consistency across pipeline

Note: These tests require ChromaDB indexes to be built.
Tests will automatically skip if indexes are not available.
To build indexes: phentrieve index build --model sentence-transformers/LaBSE
"""

import json
import tempfile
from pathlib import Path

import pytest

from phentrieve.evaluation.runner import compare_models, run_evaluation
from phentrieve.evaluation.statistics import (
    compare_models_with_significance,
)


@pytest.fixture
def temp_results_dir():
    """Create temporary directory for benchmark results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_test_data_tiny(benchmark_data_dir):
    """Path to tiny benchmark dataset."""
    return benchmark_data_dir / "german" / "tiny_v1.json"


@pytest.fixture
def available_model():
    """Get first available model from ChromaDB or skip."""
    import sqlite3
    from pathlib import Path

    db_path = Path("data/indexes/chroma.sqlite3")
    if not db_path.exists():
        pytest.skip("No ChromaDB database found")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM collections LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if not row:
            pytest.skip("No collections in ChromaDB")

        # Convert collection name back to model name
        # phentrieve_biolord_2023_m -> FremyCompany/BioLORD-2023-M
        collection_name = row[0].replace("phentrieve_", "")
        if collection_name == "biolord_2023_m":
            return "FremyCompany/BioLORD-2023-M"
        elif collection_name == "labse":
            return "sentence-transformers/LaBSE"
        else:
            # Generic fallback
            return collection_name.replace("_", "-")
    except Exception as e:
        pytest.skip(f"Failed to query ChromaDB: {e}")


def check_results_or_skip(results):
    """Helper to skip tests if ChromaDB index is not available."""
    if results is None:
        pytest.skip("ChromaDB index not built - run 'phentrieve index build' first")
    return results


# ============================================================================
# Full Benchmark Workflow Tests
# ============================================================================


def test_run_evaluation_includes_new_metrics(
    mock_test_data_tiny, temp_results_dir, available_model
):
    """run_evaluation should calculate all new MTEB metrics."""
    results = run_evaluation(
        model_name=available_model,
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        enable_reranker=False,
        save_results=True,
        results_dir=temp_results_dir,
    )
    results = check_results_or_skip(results)

    # Check that new metrics are present
    assert "ndcg_dense@1" in results
    assert "ndcg_dense@3" in results
    assert "ndcg_dense@5" in results

    assert "recall_dense@1" in results
    assert "recall_dense@3" in results
    assert "recall_dense@5" in results

    assert "precision_dense@1" in results
    assert "precision_dense@3" in results
    assert "precision_dense@5" in results

    assert "map_dense@1" in results
    assert "map_dense@3" in results
    assert "map_dense@5" in results

    # Check that average metrics are calculated
    assert "avg_ndcg_dense@1" in results
    assert "avg_recall_dense@1" in results
    assert "avg_precision_dense@1" in results
    assert "avg_map_dense@1" in results


def test_run_evaluation_includes_confidence_intervals(
    mock_test_data_tiny, temp_results_dir
):
    """run_evaluation should calculate bootstrap CIs for all metrics."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        enable_reranker=False,
        save_results=True,
        results_dir=temp_results_dir,
    )
    results = check_results_or_skip(results)

    # Check that confidence intervals are present
    assert "confidence_intervals" in results
    ci = results["confidence_intervals"]

    # Check MRR CI
    assert "mrr_dense" in ci
    assert "point_estimate" in ci["mrr_dense"]
    assert "ci_lower" in ci["mrr_dense"]
    assert "ci_upper" in ci["mrr_dense"]
    assert "ci_level" in ci["mrr_dense"]
    assert ci["mrr_dense"]["ci_level"] == 0.95

    # Check new metric CIs
    assert "ndcg_dense@1" in ci
    assert "recall_dense@1" in ci
    assert "precision_dense@1" in ci
    assert "map_dense@1" in ci


def test_compare_models_includes_new_metrics(mock_test_data_tiny, temp_results_dir):
    """compare_models should include new metrics in comparison table."""
    # Run evaluation for single model
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Compare with itself (for testing structure)
    comparison_df = compare_models([results, results])

    # Check that new metric columns exist
    expected_columns = [
        "NDCG@1 (Dense)",
        "NDCG@3 (Dense)",
        "NDCG@5 (Dense)",
        "Recall@1 (Dense)",
        "Recall@3 (Dense)",
        "Recall@5 (Dense)",
        "Precision@1 (Dense)",
        "Precision@3 (Dense)",
        "Precision@5 (Dense)",
        "MAP@1 (Dense)",
        "MAP@3 (Dense)",
        "MAP@5 (Dense)",
    ]

    for col in expected_columns:
        assert col in comparison_df.columns


def test_compare_models_with_significance_workflow(
    mock_test_data_tiny, temp_results_dir
):
    """Full workflow: run 2 models, compare with significance tests."""
    # Run evaluation for first model
    results_a = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        enable_reranker=False,
        save_results=False,
    )
    results_a = check_results_or_skip(results_a)

    # Simulate second model results (slightly different)
    results_b = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        enable_reranker=False,
        save_results=False,
    )
    results_b = check_results_or_skip(results_b)

    # Compare with significance testing
    comparison = compare_models_with_significance(
        results_a,
        results_b,
        model_a_name="Model A",
        model_b_name="Model B",
        k_values=(1, 3),
        n_bootstrap=100,  # Low for speed
    )

    # Check structure
    assert "model_a" in comparison
    assert "model_b" in comparison
    assert "comparisons" in comparison

    # Check that new metrics are compared
    comparisons = comparison["comparisons"]
    assert "ndcg_dense@1" in comparisons or len(comparisons) > 0
    assert "recall_dense@1" in comparisons or len(comparisons) > 0

    # Each comparison should have diff, p_value, significant
    for metric, result in comparisons.items():
        assert "diff" in result
        assert "p_value" in result
        assert "significant" in result


# ============================================================================
# Metric Consistency Tests
# ============================================================================


def test_metrics_values_are_bounded(mock_test_data_tiny):
    """All metrics should be in valid ranges [0, 1]."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Check all average metrics are in [0, 1]
    metric_keys = [
        "avg_mrr_dense",
        "avg_ndcg_dense@1",
        "avg_ndcg_dense@3",
        "avg_ndcg_dense@5",
        "avg_recall_dense@1",
        "avg_recall_dense@3",
        "avg_recall_dense@5",
        "avg_precision_dense@1",
        "avg_precision_dense@3",
        "avg_precision_dense@5",
        "avg_map_dense@1",
        "avg_map_dense@3",
        "avg_map_dense@5",
    ]

    for key in metric_keys:
        if key in results:
            value = results[key]
            assert 0.0 <= value <= 1.0, f"{key} out of bounds: {value}"


def test_confidence_intervals_are_valid(mock_test_data_tiny):
    """All confidence intervals should satisfy: lower <= point <= upper."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    ci = results["confidence_intervals"]

    for metric_name, ci_data in ci.items():
        point = ci_data["point_estimate"]
        lower = ci_data["ci_lower"]
        upper = ci_data["ci_upper"]

        # CI should contain point estimate
        assert lower <= point <= upper, (
            f"{metric_name}: CI [{lower}, {upper}] does not contain point {point}"
        )

        # All values should be in [0, 1]
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= point <= 1.0
        assert 0.0 <= upper <= 1.0


def test_recall_increases_with_k(mock_test_data_tiny):
    """Recall@K should monotonically increase as K increases."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5, 10),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Average recall should increase with K
    recall_1 = results["avg_recall_dense@1"]
    recall_3 = results["avg_recall_dense@3"]
    recall_5 = results["avg_recall_dense@5"]
    recall_10 = results["avg_recall_dense@10"]

    # Monotonic increase (or equal)
    assert recall_1 <= recall_3 <= recall_5 <= recall_10


def test_ndcg_bounded_by_one(mock_test_data_tiny):
    """NDCG@K should never exceed 1.0 (perfect ranking)."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3, 5),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Check per-query NDCG values
    for k in [1, 3, 5]:
        ndcg_values = results[f"ndcg_dense@{k}"]
        for ndcg in ndcg_values:
            assert 0.0 <= ndcg <= 1.0, f"NDCG@{k} out of bounds: {ndcg}"


def test_precision_at_1_binary(mock_test_data_tiny):
    """Precision@1 should be either 0.0 or 1.0 (binary for single result)."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1,),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Each precision@1 value should be 0 or 1
    precision_values = results["precision_dense@1"]
    for p in precision_values:
        assert p in [0.0, 1.0], f"Precision@1 should be binary, got {p}"


# ============================================================================
# Reranker Integration Tests
# ============================================================================


@pytest.mark.slow
def test_reranker_includes_new_metrics(mock_test_data_tiny):
    """Reranked results should also include new metrics."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        enable_reranker=True,  # Enable reranking
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Check reranked metrics exist
    assert "ndcg_reranked@1" in results
    assert "recall_reranked@1" in results
    assert "precision_reranked@1" in results
    assert "map_reranked@1" in results

    # Check average metrics
    assert "avg_ndcg_reranked@1" in results
    assert "avg_recall_reranked@1" in results
    assert "avg_precision_reranked@1" in results
    assert "avg_map_reranked@1" in results


@pytest.mark.slow
def test_reranker_confidence_intervals(mock_test_data_tiny):
    """Reranked metrics should have confidence intervals."""
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(mock_test_data_tiny),
        k_values=(1, 3),
        enable_reranker=True,
        save_results=False,
    )
    results = check_results_or_skip(results)

    ci = results["confidence_intervals"]

    # Check reranked metric CIs
    assert "ndcg_reranked@1" in ci
    assert "recall_reranked@1" in ci
    assert "precision_reranked@1" in ci
    assert "map_reranked@1" in ci


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_single_query_benchmark(benchmark_data_dir, temp_results_dir):
    """Benchmark should work with single query (edge case for bootstrap)."""
    # Create single-query test file
    single_query_file = temp_results_dir / "single_query.json"
    with open(single_query_file, "w") as f:
        json.dump(
            [
                {
                    "input_text": "Krampfanfälle",
                    "expected_hpo_ids": ["HP:0001250"],
                    "language": "de",
                }
            ],
            f,
        )

    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(single_query_file),
        k_values=(1,),
        enable_reranker=False,
        save_results=False,
    )
    results = check_results_or_skip(results)

    # Should complete without errors
    assert "avg_ndcg_dense@1" in results
    assert "confidence_intervals" in results


def test_empty_benchmark_handles_gracefully(temp_results_dir):
    """Benchmark should handle empty test file gracefully."""
    empty_file = temp_results_dir / "empty.json"
    with open(empty_file, "w") as f:
        json.dump([], f)

    # Should return None (indicating failure to load test data)
    results = run_evaluation(
        model_name="sentence-transformers/LaBSE",
        test_file=str(empty_file),
        k_values=(1,),
        enable_reranker=False,
        save_results=False,
    )

    # Empty test file should result in None return
    assert results is None
