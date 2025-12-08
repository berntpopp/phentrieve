"""
Unit tests for statistical analysis utilities (bootstrap CI, significance tests).

Tests cover:
- Bootstrap confidence interval calculation
- Paired bootstrap significance testing
- CI calculation for all benchmark metrics
- Model comparison with significance tests
- Edge cases (empty data, single value, identical distributions)
"""

import numpy as np
import pytest

from phentrieve.evaluation.statistics import (
    bootstrap_confidence_interval,
    calculate_bootstrap_ci_for_metrics,
    compare_models_with_significance,
    paired_bootstrap_test,
)

# ============================================================================
# Bootstrap Confidence Interval Tests
# ============================================================================


def test_bootstrap_ci_basic():
    """Bootstrap CI should produce interval around mean for normal data."""
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    point, ci_lower, ci_upper = bootstrap_confidence_interval(
        values, n_bootstrap=1000, confidence_level=0.95
    )

    # Point estimate should be close to mean
    assert point == pytest.approx(np.mean(values), abs=1e-6)

    # CI should contain the mean
    assert ci_lower <= point <= ci_upper

    # CI width should be reasonable (not too wide or narrow)
    ci_width = ci_upper - ci_lower
    assert 0 < ci_width < 1.0


def test_bootstrap_ci_empty_values():
    """Bootstrap CI should return zeros for empty input."""
    point, ci_lower, ci_upper = bootstrap_confidence_interval([])
    assert point == 0.0
    assert ci_lower == 0.0
    assert ci_upper == 0.0


def test_bootstrap_ci_single_value():
    """Bootstrap CI should collapse to single value when only one data point."""
    values = [0.75]
    point, ci_lower, ci_upper = bootstrap_confidence_interval(values, n_bootstrap=100)

    # All should be the same value
    assert point == pytest.approx(0.75, abs=1e-6)
    assert ci_lower == pytest.approx(0.75, abs=1e-6)
    assert ci_upper == pytest.approx(0.75, abs=1e-6)


def test_bootstrap_ci_confidence_levels():
    """Wider confidence level should produce wider interval."""
    values = [0.1, 0.3, 0.5, 0.7, 0.9]

    _, ci_lower_95, ci_upper_95 = bootstrap_confidence_interval(
        values, n_bootstrap=1000, confidence_level=0.95
    )
    _, ci_lower_80, ci_upper_80 = bootstrap_confidence_interval(
        values, n_bootstrap=1000, confidence_level=0.80
    )

    # 95% CI should be wider than 80% CI
    width_95 = ci_upper_95 - ci_lower_95
    width_80 = ci_upper_80 - ci_lower_80
    assert width_95 >= width_80


def test_bootstrap_ci_custom_metric():
    """Bootstrap CI should support custom aggregation functions."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Test with median instead of mean
    point, ci_lower, ci_upper = bootstrap_confidence_interval(
        values, n_bootstrap=1000, metric_fn=np.median
    )

    # Point estimate should be median
    assert point == pytest.approx(np.median(values), abs=1e-6)
    assert ci_lower <= point <= ci_upper


def test_bootstrap_ci_reproducibility():
    """Bootstrap CI should be reproducible with same random seed."""
    values = [0.5, 0.6, 0.7, 0.8, 0.9]

    np.random.seed(42)
    point1, lower1, upper1 = bootstrap_confidence_interval(values, n_bootstrap=100)

    np.random.seed(42)
    point2, lower2, upper2 = bootstrap_confidence_interval(values, n_bootstrap=100)

    assert point1 == pytest.approx(point2, abs=1e-6)
    assert lower1 == pytest.approx(lower2, abs=1e-6)
    assert upper1 == pytest.approx(upper2, abs=1e-6)


def test_bootstrap_ci_high_variance_data():
    """Bootstrap CI should produce wider intervals for high variance data."""
    low_variance = [0.50, 0.51, 0.49, 0.50, 0.51]
    high_variance = [0.10, 0.90, 0.20, 0.80, 0.50]

    _, lower_lv, upper_lv = bootstrap_confidence_interval(
        low_variance, n_bootstrap=1000
    )
    _, lower_hv, upper_hv = bootstrap_confidence_interval(
        high_variance, n_bootstrap=1000
    )

    width_lv = upper_lv - lower_lv
    width_hv = upper_hv - lower_hv

    # High variance should have wider CI
    assert width_hv > width_lv


# ============================================================================
# Paired Bootstrap Test
# ============================================================================


def test_paired_bootstrap_identical_distributions():
    """Paired test should find no significant difference for identical data."""
    values_a = [0.5, 0.6, 0.7, 0.8, 0.9]
    values_b = [0.5, 0.6, 0.7, 0.8, 0.9]

    p_value, significant = paired_bootstrap_test(values_a, values_b, n_bootstrap=1000)

    # p-value should be high (close to 1.0)
    assert p_value > 0.05
    assert not significant


def test_paired_bootstrap_clearly_different():
    """Paired test should detect significant difference for clearly different data."""
    values_a = [0.9, 0.95, 0.92, 0.93, 0.91]  # High performance
    values_b = [0.1, 0.15, 0.12, 0.13, 0.11]  # Low performance

    p_value, significant = paired_bootstrap_test(values_a, values_b, n_bootstrap=1000)

    # p-value should be very low (close to 0.0)
    assert p_value < 0.05
    assert significant


def test_paired_bootstrap_mismatched_lengths():
    """Paired test should raise error for mismatched lengths."""
    values_a = [0.5, 0.6, 0.7]
    values_b = [0.5, 0.6]  # Different length

    with pytest.raises(ValueError, match="Must have same number of queries"):
        paired_bootstrap_test(values_a, values_b)


def test_paired_bootstrap_empty_values():
    """Paired test should handle empty inputs gracefully."""
    p_value, significant = paired_bootstrap_test([], [], n_bootstrap=100)

    assert p_value == 1.0
    assert not significant


def test_paired_bootstrap_single_pair():
    """Paired test should work with single paired observation."""
    values_a = [0.9]
    values_b = [0.5]

    p_value, significant = paired_bootstrap_test(values_a, values_b, n_bootstrap=100)

    # Should detect difference even with single pair
    assert 0.0 <= p_value <= 1.0
    assert isinstance(significant, bool)


def test_paired_bootstrap_null_hypothesis():
    """Paired test should correctly implement null hypothesis testing."""
    # Small but consistent difference
    values_a = [0.60, 0.61, 0.62, 0.63, 0.64]
    values_b = [0.50, 0.51, 0.52, 0.53, 0.54]

    p_value, significant = paired_bootstrap_test(values_a, values_b, n_bootstrap=5000)

    # Consistent 0.1 difference should be significant
    assert p_value < 0.05
    assert significant


def test_paired_bootstrap_reproducibility():
    """Paired test should be reproducible with same random seed."""
    values_a = [0.5, 0.6, 0.7, 0.8, 0.9]
    values_b = [0.4, 0.5, 0.6, 0.7, 0.8]

    np.random.seed(42)
    p1, sig1 = paired_bootstrap_test(values_a, values_b, n_bootstrap=100)

    np.random.seed(42)
    p2, sig2 = paired_bootstrap_test(values_a, values_b, n_bootstrap=100)

    assert p1 == pytest.approx(p2, abs=1e-6)
    assert sig1 == sig2


# ============================================================================
# Calculate Bootstrap CI for All Metrics
# ============================================================================


@pytest.fixture
def mock_benchmark_results():
    """Mock benchmark results with all metric types."""
    return {
        "mrr_dense": [0.5, 0.6, 0.7],
        "mrr_reranked": [0.6, 0.7, 0.8],
        "hit_rate_dense@1": [0.3, 0.4, 0.5],
        "hit_rate_dense@3": [0.5, 0.6, 0.7],
        "hit_rate_dense@5": [0.6, 0.7, 0.8],
        "hit_rate_dense@10": [0.7, 0.8, 0.9],
        "ndcg_dense@1": [0.3, 0.4, 0.5],
        "ndcg_dense@3": [0.5, 0.6, 0.7],
        "ndcg_dense@5": [0.6, 0.7, 0.8],
        "ndcg_dense@10": [0.7, 0.8, 0.9],
        "recall_dense@1": [0.2, 0.3, 0.4],
        "recall_dense@3": [0.4, 0.5, 0.6],
        "recall_dense@5": [0.5, 0.6, 0.7],
        "recall_dense@10": [0.6, 0.7, 0.8],
        "precision_dense@1": [0.9, 1.0, 1.0],
        "precision_dense@3": [0.8, 0.9, 1.0],
        "precision_dense@5": [0.7, 0.8, 0.9],
        "precision_dense@10": [0.6, 0.7, 0.8],
        "map_dense@1": [0.3, 0.4, 0.5],
        "map_dense@3": [0.5, 0.6, 0.7],
        "map_dense@5": [0.6, 0.7, 0.8],
        "map_dense@10": [0.7, 0.8, 0.9],
        "max_ont_similarity_dense@1": [0.7, 0.8, 0.9],
        "max_ont_similarity_dense@3": [0.8, 0.9, 1.0],
        "max_ont_similarity_dense@5": [0.9, 1.0, 1.0],
        "max_ont_similarity_dense@10": [1.0, 1.0, 1.0],
    }


def test_calculate_bootstrap_ci_all_metrics(mock_benchmark_results):
    """CI calculation should process all metric types."""
    ci_results = calculate_bootstrap_ci_for_metrics(
        mock_benchmark_results, k_values=(1, 3, 5, 10), n_bootstrap=100
    )

    # Check MRR metrics
    assert "mrr_dense" in ci_results
    assert "mrr_reranked" in ci_results

    # Check structure of CI results
    mrr_ci = ci_results["mrr_dense"]
    assert "point_estimate" in mrr_ci
    assert "ci_lower" in mrr_ci
    assert "ci_upper" in mrr_ci
    assert "ci_level" in mrr_ci
    assert mrr_ci["ci_level"] == 0.95


def test_calculate_bootstrap_ci_k_values(mock_benchmark_results):
    """CI calculation should respect k_values parameter."""
    ci_results = calculate_bootstrap_ci_for_metrics(
        mock_benchmark_results, k_values=(1, 5), n_bootstrap=100
    )

    # Should have CIs for k=1 and k=5
    assert "ndcg_dense@1" in ci_results
    assert "ndcg_dense@5" in ci_results

    # Should NOT have CIs for k=3 and k=10
    assert "ndcg_dense@3" not in ci_results
    assert "ndcg_dense@10" not in ci_results


def test_calculate_bootstrap_ci_missing_metrics():
    """CI calculation should handle missing metrics gracefully."""
    minimal_results = {
        "mrr_dense": [0.5, 0.6, 0.7],
        # Missing all other metrics
    }

    ci_results = calculate_bootstrap_ci_for_metrics(
        minimal_results, k_values=(1, 3), n_bootstrap=100
    )

    # Should only have MRR CI
    assert "mrr_dense" in ci_results
    assert len(ci_results) == 1


def test_calculate_bootstrap_ci_empty_lists():
    """CI calculation should skip metrics with empty value lists."""
    results_with_empty = {
        "mrr_dense": [0.5, 0.6, 0.7],
        "ndcg_dense@1": [],  # Empty list
    }

    ci_results = calculate_bootstrap_ci_for_metrics(
        results_with_empty, k_values=(1,), n_bootstrap=100
    )

    # Should have MRR but not NDCG
    assert "mrr_dense" in ci_results
    assert "ndcg_dense@1" not in ci_results


def test_calculate_bootstrap_ci_confidence_level(mock_benchmark_results):
    """CI calculation should respect custom confidence levels."""
    ci_results_95 = calculate_bootstrap_ci_for_metrics(
        mock_benchmark_results, k_values=(1,), n_bootstrap=100, confidence_level=0.95
    )
    ci_results_80 = calculate_bootstrap_ci_for_metrics(
        mock_benchmark_results, k_values=(1,), n_bootstrap=100, confidence_level=0.80
    )

    mrr_95 = ci_results_95["mrr_dense"]
    mrr_80 = ci_results_80["mrr_dense"]

    # 95% CI should be wider than 80% CI
    width_95 = mrr_95["ci_upper"] - mrr_95["ci_lower"]
    width_80 = mrr_80["ci_upper"] - mrr_80["ci_lower"]
    assert width_95 >= width_80


def test_calculate_bootstrap_ci_all_metric_types(mock_benchmark_results):
    """CI calculation should cover all 6 metric types."""
    ci_results = calculate_bootstrap_ci_for_metrics(
        mock_benchmark_results, k_values=(1,), n_bootstrap=100
    )

    # All metric types should be present for k=1
    expected_metrics = [
        "hit_rate_dense@1",
        "max_ont_similarity_dense@1",
        "ndcg_dense@1",
        "recall_dense@1",
        "precision_dense@1",
        "map_dense@1",
    ]

    for metric in expected_metrics:
        assert metric in ci_results


# ============================================================================
# Model Comparison with Significance
# ============================================================================


@pytest.fixture
def mock_results_model_a():
    """Mock results for Model A (better performance)."""
    return {
        "mrr_dense": [0.7, 0.8, 0.9],
        "ndcg_dense@1": [0.6, 0.7, 0.8],
        "recall_dense@1": [0.5, 0.6, 0.7],
    }


@pytest.fixture
def mock_results_model_b():
    """Mock results for Model B (worse performance)."""
    return {
        "mrr_dense": [0.4, 0.5, 0.6],
        "ndcg_dense@1": [0.3, 0.4, 0.5],
        "recall_dense@1": [0.2, 0.3, 0.4],
    }


def test_compare_models_basic_structure(mock_results_model_a, mock_results_model_b):
    """Model comparison should return proper structure."""
    comparison = compare_models_with_significance(
        mock_results_model_a,
        mock_results_model_b,
        model_a_name="BioLORD",
        model_b_name="Jina",
        k_values=(1,),
        n_bootstrap=100,
    )

    # Check structure
    assert "model_a" in comparison
    assert "model_b" in comparison
    assert "comparisons" in comparison

    assert comparison["model_a"] == "BioLORD"
    assert comparison["model_b"] == "Jina"


def test_compare_models_significance_detection(
    mock_results_model_a, mock_results_model_b
):
    """Model comparison should detect significant differences."""
    comparison = compare_models_with_significance(
        mock_results_model_a,
        mock_results_model_b,
        k_values=(1,),
        n_bootstrap=1000,
    )

    # MRR difference should be significant (0.8 vs 0.5)
    mrr_comp = comparison["comparisons"]["mrr_dense"]
    assert mrr_comp["diff"] > 0  # Model A is better
    assert mrr_comp["significant"]  # Difference is significant
    assert mrr_comp["p_value"] < 0.05


def test_compare_models_identical_results():
    """Model comparison should find no significance for identical results."""
    identical_results = {
        "mrr_dense": [0.6, 0.7, 0.8],
        "ndcg_dense@1": [0.5, 0.6, 0.7],
    }

    comparison = compare_models_with_significance(
        identical_results,
        identical_results,
        k_values=(1,),
        n_bootstrap=100,
    )

    # MRR should have p-value > 0.05 (not significant)
    mrr_comp = comparison["comparisons"]["mrr_dense"]
    assert abs(mrr_comp["diff"]) < 1e-6  # Difference should be ~0
    assert not mrr_comp["significant"]
    assert mrr_comp["p_value"] > 0.05


def test_compare_models_missing_metrics():
    """Model comparison should skip metrics not present in both models."""
    results_a = {
        "mrr_dense": [0.7, 0.8, 0.9],
        "ndcg_dense@1": [0.6, 0.7, 0.8],
    }
    results_b = {
        "mrr_dense": [0.4, 0.5, 0.6],
        # Missing ndcg_dense@1
    }

    comparison = compare_models_with_significance(
        results_a, results_b, k_values=(1,), n_bootstrap=100
    )

    # Should only compare MRR
    assert "mrr_dense" in comparison["comparisons"]
    assert "ndcg_dense@1" not in comparison["comparisons"]


def test_compare_models_empty_results():
    """Model comparison should handle empty metric lists gracefully."""
    results_a = {"mrr_dense": []}
    results_b = {"mrr_dense": []}

    comparison = compare_models_with_significance(
        results_a, results_b, k_values=(1,), n_bootstrap=100
    )

    # Should not include comparison for empty metrics
    assert "mrr_dense" not in comparison["comparisons"]


def test_compare_models_all_k_values(mock_results_model_a, mock_results_model_b):
    """Model comparison should respect k_values parameter."""
    # Add metrics for multiple k values
    for k in [1, 3, 5, 10]:
        mock_results_model_a[f"ndcg_dense@{k}"] = [0.6, 0.7, 0.8]
        mock_results_model_b[f"ndcg_dense@{k}"] = [0.3, 0.4, 0.5]

    comparison = compare_models_with_significance(
        mock_results_model_a,
        mock_results_model_b,
        k_values=(1, 3, 5, 10),
        n_bootstrap=100,
    )

    # Should have comparisons for all k values
    for k in [1, 3, 5, 10]:
        assert f"ndcg_dense@{k}" in comparison["comparisons"]
