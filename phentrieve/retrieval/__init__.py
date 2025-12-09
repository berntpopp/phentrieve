"""
Retrieval modules for the Phentrieve package.

This package contains modules for retrieving HPO terms from vector indices.
"""

from phentrieve.retrieval.aggregation import (
    AggregationStrategy,
    aggregate_multi_vector_results,
    aggregate_scores,
    group_results_by_hpo_id,
)

__all__ = [
    "AggregationStrategy",
    "aggregate_multi_vector_results",
    "aggregate_scores",
    "group_results_by_hpo_id",
]
