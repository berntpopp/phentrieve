"""
Statistical analysis utilities for benchmark evaluation.

This module provides functions for calculating confidence intervals and
significance tests for model comparison in information retrieval benchmarks.
"""

from typing import Any

import numpy as np


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        values: Per-query metric values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    if not values:
        return 0.0, 0.0, 0.0

    values_array = np.array(values)
    n = len(values_array)

    # Vectorized bootstrap resampling for performance
    rng = np.random.default_rng()
    indices = rng.choice(n, size=(n_bootstrap, n), replace=True)
    bootstrap_means = values_array[indices].mean(axis=1)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    point_estimate = float(np.mean(values_array))

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10000,
) -> tuple[float, bool]:
    """
    Paired bootstrap significance test for model comparison.

    Args:
        values_a: Per-query metrics for model A
        values_b: Per-query metrics for model B
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (p_value, is_significant_at_0.05)
    """
    if len(values_a) != len(values_b):
        raise ValueError("Must have same number of queries for paired test")

    if not values_a:
        return 1.0, False

    differences = np.array(values_a) - np.array(values_b)
    observed_diff = np.mean(differences)

    # Null hypothesis: no difference (center at 0)
    centered_diffs = differences - observed_diff

    # Bootstrap under null
    count_extreme = 0
    for _ in range(n_bootstrap):
        sample = np.random.choice(
            centered_diffs, size=len(centered_diffs), replace=True
        )
        if abs(np.mean(sample)) >= abs(observed_diff):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap
    is_significant = p_value < 0.05

    return p_value, is_significant


def calculate_bootstrap_ci_for_metrics(
    results: dict[str, Any],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for all metrics in benchmark results.

    Args:
        results: Benchmark results dictionary from run_evaluation
        k_values: K values to calculate CIs for
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CIs

    Returns:
        Dictionary with CI information for each metric
    """
    ci_results: dict[str, dict[str, float]] = {}

    # MRR metric
    if "mrr_dense" in results and results["mrr_dense"]:
        point, lower, upper = bootstrap_confidence_interval(
            results["mrr_dense"], n_bootstrap, confidence_level
        )
        ci_results["mrr_dense"] = {
            "point_estimate": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_level": confidence_level,
        }

    # K-dependent metrics (dense only - reranking removed)
    metric_types = [
        "hit_rate",
        "max_ont_similarity",
        "ndcg",
        "recall",
        "precision",
        "map",
    ]

    for metric_type in metric_types:
        for k in k_values:
            key = f"{metric_type}_dense@{k}"
            if key in results and results[key]:
                point, lower, upper = bootstrap_confidence_interval(
                    results[key], n_bootstrap, confidence_level
                )
                ci_results[key] = {
                    "point_estimate": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_level": confidence_level,
                }

    return ci_results


def compare_models_with_significance(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """
    Compare two models using paired bootstrap significance tests.

    Args:
        results_a: Benchmark results for model A
        results_b: Benchmark results for model B
        model_a_name: Display name for model A
        model_b_name: Display name for model B
        k_values: K values to compare
        n_bootstrap: Number of bootstrap samples for significance tests

    Returns:
        Dictionary with comparison results and significance tests
    """
    comparisons: dict[str, dict[str, Any]] = {}

    # MRR comparison
    if (
        "mrr_dense" in results_a
        and results_a["mrr_dense"]
        and "mrr_dense" in results_b
        and results_b["mrr_dense"]
    ):
        p_value, significant = paired_bootstrap_test(
            results_a["mrr_dense"], results_b["mrr_dense"], n_bootstrap
        )
        diff = float(np.mean(results_a["mrr_dense"]) - np.mean(results_b["mrr_dense"]))
        comparisons["mrr_dense"] = {
            "diff": diff,
            "p_value": p_value,
            "significant": significant,
        }

    # K-dependent metrics (dense only - reranking removed)
    metric_types = [
        "hit_rate",
        "max_ont_similarity",
        "ndcg",
        "recall",
        "precision",
        "map",
    ]

    for metric_type in metric_types:
        for k in k_values:
            key = f"{metric_type}_dense@{k}"
            if (
                key in results_a
                and results_a[key]
                and key in results_b
                and results_b[key]
            ):
                p_value, significant = paired_bootstrap_test(
                    results_a[key], results_b[key], n_bootstrap
                )
                diff = float(np.mean(results_a[key]) - np.mean(results_b[key]))
                comparisons[key] = {
                    "diff": diff,
                    "p_value": p_value,
                    "significant": significant,
                }

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "comparisons": comparisons,
    }
