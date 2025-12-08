"""
Statistical analysis utilities for benchmark evaluation.

This module provides functions for calculating confidence intervals and
significance tests for model comparison in information retrieval benchmarks.
"""

from collections.abc import Sequence
from typing import Callable

import numpy as np
import numpy.typing as npt


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    metric_fn: Callable[[npt.ArrayLike], float] = np.mean,  # type: ignore[assignment]
) -> tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        values: Per-query metric values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        metric_fn: Aggregation function (default: mean)

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    if not values:
        return 0.0, 0.0, 0.0

    values_array = np.array(values)
    n = len(values_array)

    # Bootstrap resampling
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample = values_array[sample_indices]
        bootstrap_estimates.append(metric_fn(sample))

    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    point_estimate = float(metric_fn(values_array))

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
    results: dict,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
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
    ci_results = {}

    # MRR metrics
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

    if "mrr_reranked" in results and results["mrr_reranked"]:
        point, lower, upper = bootstrap_confidence_interval(
            results["mrr_reranked"], n_bootstrap, confidence_level
        )
        ci_results["mrr_reranked"] = {
            "point_estimate": point,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_level": confidence_level,
        }

    # K-dependent metrics
    metric_types = [
        "hit_rate",
        "max_ont_similarity",
        "ndcg",
        "recall",
        "precision",
        "map",
    ]

    for metric_type in metric_types:
        for method in ["dense", "reranked"]:
            for k in k_values:
                key = f"{metric_type}_{method}@{k}"
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
    results_a: dict,
    results_b: dict,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    n_bootstrap: int = 10000,
) -> dict:
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
    comparison = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "comparisons": {},
    }

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
        comparisons: dict[str, dict[str, float | bool]] = {}
        comparisons["mrr_dense"] = {
            "diff": diff,
            "p_value": p_value,
            "significant": significant,
        }
        comparison["comparisons"] = comparisons

    # K-dependent metrics
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
                # Ensure comparisons dict exists
                if "comparisons" not in comparison:
                    comparison["comparisons"] = {}
                comp_dict = comparison["comparisons"]
                if isinstance(comp_dict, dict):
                    comp_dict[key] = {
                        "diff": diff,
                        "p_value": p_value,
                        "significant": significant,
                    }

    return comparison
