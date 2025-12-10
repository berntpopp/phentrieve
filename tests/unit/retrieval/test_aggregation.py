"""Unit tests for multi-vector aggregation module.

Tests the aggregation strategies for combining component scores
from multi-vector HPO term embeddings.
"""

import pytest

from phentrieve.retrieval.aggregation import (
    AggregationStrategy,
    aggregate_multi_vector_results,
    aggregate_scores,
    group_results_by_hpo_id,
)


class TestAggregateScores:
    """Test individual score aggregation."""

    def test_label_only_strategy(self):
        """Test label_only strategy returns label score."""
        result = aggregate_scores(
            label_score=0.9,
            synonym_scores=[0.8, 0.7],
            definition_score=0.6,
            strategy=AggregationStrategy.LABEL_ONLY,
        )
        assert result == 0.9

    def test_label_only_with_no_label(self):
        """Test label_only returns 0 when label is None."""
        result = aggregate_scores(
            label_score=None,
            synonym_scores=[0.8, 0.7],
            definition_score=0.6,
            strategy=AggregationStrategy.LABEL_ONLY,
        )
        assert result == 0.0

    def test_label_synonyms_max_takes_best(self):
        """Test label_synonyms_max returns max of label and synonyms."""
        result = aggregate_scores(
            label_score=0.7,
            synonym_scores=[0.9, 0.6],
            definition_score=0.5,
            strategy=AggregationStrategy.LABEL_SYNONYMS_MAX,
        )
        assert result == 0.9  # Best synonym wins

    def test_label_synonyms_max_label_wins(self):
        """Test label_synonyms_max when label is best."""
        result = aggregate_scores(
            label_score=0.95,
            synonym_scores=[0.8, 0.7],
            definition_score=0.6,
            strategy=AggregationStrategy.LABEL_SYNONYMS_MAX,
        )
        assert result == 0.95

    def test_all_max_strategy(self):
        """Test all_max returns maximum across all components."""
        result = aggregate_scores(
            label_score=0.7,
            synonym_scores=[0.8],
            definition_score=0.95,
            strategy=AggregationStrategy.ALL_MAX,
        )
        assert result == 0.95

    def test_all_min_strategy(self):
        """Test all_min returns minimum across all components."""
        result = aggregate_scores(
            label_score=0.7,
            synonym_scores=[0.8, 0.9],
            definition_score=0.6,
            strategy=AggregationStrategy.ALL_MIN,
        )
        assert result == 0.6

    def test_all_weighted_strategy(self):
        """Test all_weighted returns weighted average."""
        result = aggregate_scores(
            label_score=0.8,
            synonym_scores=[0.9, 0.7],  # max = 0.9
            definition_score=0.6,
            strategy=AggregationStrategy.ALL_WEIGHTED,
            weights={"label": 0.5, "synonyms": 0.3, "definition": 0.2},
        )
        # Expected: (0.5 * 0.8 + 0.3 * 0.9 + 0.2 * 0.6) / 1.0 = 0.4 + 0.27 + 0.12 = 0.79
        assert abs(result - 0.79) < 0.01

    def test_all_weighted_with_missing_components(self):
        """Test all_weighted normalizes when components missing."""
        result = aggregate_scores(
            label_score=0.8,
            synonym_scores=[],  # No synonyms
            definition_score=None,  # No definition
            strategy=AggregationStrategy.ALL_WEIGHTED,
            weights={"label": 0.5, "synonyms": 0.3, "definition": 0.2},
        )
        # Only label available, so result should be label score
        assert result == 0.8

    def test_custom_formula_simple(self):
        """Test custom formula evaluation."""
        result = aggregate_scores(
            label_score=0.8,
            synonym_scores=[0.9, 0.7],
            definition_score=0.6,
            strategy=AggregationStrategy.CUSTOM,
            custom_formula="max(label, max(synonyms))",
        )
        assert result == 0.9

    def test_custom_formula_weighted(self):
        """Test custom formula with weights."""
        result = aggregate_scores(
            label_score=0.8,
            synonym_scores=[0.9],
            definition_score=0.6,
            strategy=AggregationStrategy.CUSTOM,
            custom_formula="0.5 * label + 0.5 * max(synonyms)",
        )
        # Expected: 0.5 * 0.8 + 0.5 * 0.9 = 0.4 + 0.45 = 0.85
        assert abs(result - 0.85) < 0.01

    def test_empty_scores_returns_zero(self):
        """Test empty scores returns 0."""
        result = aggregate_scores(
            label_score=None,
            synonym_scores=[],
            definition_score=None,
            strategy=AggregationStrategy.ALL_MAX,
        )
        assert result == 0.0

    def test_invalid_strategy_raises_error(self):
        """Test invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            aggregate_scores(
                label_score=0.8,
                synonym_scores=[0.9],
                definition_score=0.6,
                strategy="invalid_strategy",
            )

    def test_custom_without_formula_raises_error(self):
        """Test custom strategy without formula raises error."""
        with pytest.raises(ValueError, match="custom_formula required"):
            aggregate_scores(
                label_score=0.8,
                synonym_scores=[0.9],
                definition_score=0.6,
                strategy=AggregationStrategy.CUSTOM,
                custom_formula=None,
            )


class TestGroupResultsByHpoId:
    """Test grouping ChromaDB results by HPO ID."""

    def test_groups_by_hpo_id(self):
        """Test results are grouped by HPO ID."""
        results = {
            "ids": [["HP:0001__label__0", "HP:0001__synonym__0", "HP:0002__label__0"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001", "component": "label", "label": "Term 1"},
                    {"hpo_id": "HP:0001", "component": "synonym", "label": "Term 1"},
                    {"hpo_id": "HP:0002", "component": "label", "label": "Term 2"},
                ]
            ],
            "similarities": [[0.9, 0.8, 0.7]],
        }

        grouped = group_results_by_hpo_id(results)

        assert "HP:0001" in grouped
        assert "HP:0002" in grouped
        assert grouped["HP:0001"]["label"] == 0.9
        assert grouped["HP:0001"]["synonyms"] == [0.8]
        assert grouped["HP:0002"]["label"] == 0.7

    def test_handles_distances_instead_of_similarities(self):
        """Test converts distances to similarities."""
        results = {
            "ids": [["HP:0001__label__0"]],
            "metadatas": [
                [{"hpo_id": "HP:0001", "component": "label", "label": "Term"}]
            ],
            "distances": [[0.2]],  # distance, not similarity
        }

        grouped = group_results_by_hpo_id(results)

        # similarity = 1.0 - distance = 0.8
        assert grouped["HP:0001"]["label"] == 0.8

    def test_empty_results_returns_empty_dict(self):
        """Test empty results returns empty dict."""
        results = {"ids": [[]], "metadatas": [[]], "similarities": [[]]}
        grouped = group_results_by_hpo_id(results)
        assert grouped == {}


class TestAggregateMultiVectorResults:
    """Test full multi-vector result aggregation."""

    def test_aggregates_and_sorts_results(self):
        """Test results are aggregated and sorted by score."""
        results = {
            "ids": [
                [
                    "HP:0001__label__0",
                    "HP:0001__synonym__0",
                    "HP:0002__label__0",
                    "HP:0002__synonym__0",
                ]
            ],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001", "component": "label", "label": "Term 1"},
                    {"hpo_id": "HP:0001", "component": "synonym", "label": "Term 1"},
                    {"hpo_id": "HP:0002", "component": "label", "label": "Term 2"},
                    {"hpo_id": "HP:0002", "component": "synonym", "label": "Term 2"},
                ]
            ],
            "similarities": [[0.7, 0.8, 0.9, 0.6]],
        }

        aggregated = aggregate_multi_vector_results(
            results, strategy=AggregationStrategy.LABEL_SYNONYMS_MAX
        )

        # HP:0002 should be first (label=0.9 > HP:0001's max=0.8)
        assert len(aggregated) == 2
        assert aggregated[0]["hpo_id"] == "HP:0002"
        assert aggregated[0]["similarity"] == 0.9
        assert aggregated[1]["hpo_id"] == "HP:0001"
        assert aggregated[1]["similarity"] == 0.8

    def test_includes_component_scores(self):
        """Test component scores are included in results."""
        results = {
            "ids": [["HP:0001__label__0", "HP:0001__synonym__0"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001", "component": "label", "label": "Term 1"},
                    {"hpo_id": "HP:0001", "component": "synonym", "label": "Term 1"},
                ]
            ],
            "similarities": [[0.9, 0.8]],
        }

        aggregated = aggregate_multi_vector_results(results)

        assert aggregated[0]["component_scores"]["label"] == 0.9
        assert aggregated[0]["component_scores"]["synonyms"] == [0.8]

    def test_filters_by_min_similarity(self):
        """Test results below threshold are filtered."""
        results = {
            "ids": [["HP:0001__label__0", "HP:0002__label__0"]],
            "metadatas": [
                [
                    {"hpo_id": "HP:0001", "component": "label", "label": "Term 1"},
                    {"hpo_id": "HP:0002", "component": "label", "label": "Term 2"},
                ]
            ],
            "similarities": [[0.9, 0.3]],
        }

        aggregated = aggregate_multi_vector_results(results, min_similarity=0.5)

        assert len(aggregated) == 1
        assert aggregated[0]["hpo_id"] == "HP:0001"
