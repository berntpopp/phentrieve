"""Unit tests for semantic metrics module (pytest style)."""

import pytest
from unittest.mock import patch

from phentrieve.evaluation.semantic_metrics import (
    calculate_assertion_accuracy,
    calculate_semantically_aware_set_based_prf1,
)


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


class TestAssertionAccuracy:
    """Test cases for assertion accuracy calculation."""

    def test_perfect_matching(self):
        """Test assertion accuracy with perfect matching."""
        matched_pairs = [
            (
                {"id": "HP:0001250", "status": "affirmed", "name": "Seizure"},
                {
                    "hpo_id": "HP:0001250",
                    "assertion_status": "affirmed",
                    "label": "Seizure",
                },
            ),
            (
                {
                    "id": "HP:0012638",
                    "status": "negated",
                    "name": "Abnormality of nervous system physiology",
                },
                {
                    "hpo_id": "HP:0012638",
                    "assertion_status": "negated",
                    "label": "Abnormality of nervous system physiology",
                },
            ),
        ]

        accuracy, correct, total = calculate_assertion_accuracy(matched_pairs)

        assert accuracy == 100.0
        assert correct == 2
        assert total == 2

    def test_partial_matching(self):
        """Test assertion accuracy with partial matching."""
        matched_pairs = [
            (
                {"id": "HP:0001250", "status": "affirmed", "name": "Seizure"},
                {
                    "hpo_id": "HP:0001250",
                    "assertion_status": "affirmed",
                    "label": "Seizure",
                },
            ),
            (
                {
                    "id": "HP:0012638",
                    "status": "negated",
                    "name": "Abnormality of nervous system physiology",
                },
                {
                    "hpo_id": "HP:0012638",
                    "assertion_status": "affirmed",
                    "label": "Abnormality of nervous system physiology",
                },
            ),
        ]

        accuracy, correct, total = calculate_assertion_accuracy(matched_pairs)

        assert accuracy == 50.0
        assert correct == 1
        assert total == 2

    def test_empty_matched_pairs(self):
        """Test assertion accuracy with empty matched pairs."""
        matched_pairs = []

        accuracy, correct, total = calculate_assertion_accuracy(matched_pairs)

        assert accuracy == 0.0
        assert correct == 0
        assert total == 0


class TestSemanticPRF1:
    """Test cases for semantically aware P/R/F1 calculation."""

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_exact_matches_only(self, mock_sim):
        """Test PRF1 with exact matches only."""
        # Mock won't be called for exact matches
        mock_sim.return_value = 0.5

        extracted = [
            {"id": "HP:0001250", "name": "Seizure", "status": "affirmed"},
            {
                "id": "HP:0012638",
                "name": "Abnormality of nervous system physiology",
                "status": "affirmed",
            },
        ]

        ground_truth = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "assertion_status": "affirmed",
            },
            {
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "assertion_status": "affirmed",
            },
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted, ground_truth, target_assertion_status="affirmed"
        )

        # Perfect match expected
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1_score"] == 1.0
        assert results["tp_count"] == 2
        assert results["fp_count"] == 0
        assert results["fn_count"] == 0
        assert len(results["tp_matched_pairs_list"]) == 2

        # Mock should not have been called for exact matches
        mock_sim.assert_not_called()

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_semantic_matches(self, mock_sim):
        """Test PRF1 with semantic matches."""

        def sim_side_effect(extracted_id, ground_truth_id, formula):
            if extracted_id == "HP:0001251" and ground_truth_id == "HP:0001250":
                return 0.8  # High similarity
            return 0.1  # Low similarity for other pairs

        mock_sim.side_effect = sim_side_effect

        extracted = [
            {"id": "HP:0001251", "name": "Atonic seizure", "status": "affirmed"},
            {"id": "HP:0000001", "name": "All", "status": "affirmed"},
        ]

        ground_truth = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "assertion_status": "affirmed",
            },
            {
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "assertion_status": "affirmed",
            },
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted,
            ground_truth,
            target_assertion_status="affirmed",
            semantic_similarity_threshold=0.7,
        )

        # One semantic match expected (HP:0001251 -> HP:0001250)
        assert results["precision"] == 0.5
        assert results["recall"] == 0.5
        assert results["f1_score"] == 0.5
        assert results["tp_count"] == 1
        assert results["fp_count"] == 1  # HP:0000001 unmatched
        assert results["fn_count"] == 1  # HP:0012638 unmatched

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_no_matches(self, mock_sim):
        """Test PRF1 with no matches."""
        # Always return low similarity
        mock_sim.return_value = 0.1

        extracted = [{"id": "HP:0001251", "name": "Atonic seizure", "status": "affirmed"}]

        ground_truth = [
            {
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "assertion_status": "affirmed",
            }
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted,
            ground_truth,
            target_assertion_status="affirmed",
            semantic_similarity_threshold=0.7,
        )

        # No matches expected
        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
        assert results["f1_score"] == 0.0
        assert results["tp_count"] == 0
        assert results["fp_count"] == 1  # extracted term unmatched
        assert results["fn_count"] == 1  # ground truth term unmatched

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_mixed_matches(self, mock_sim):
        """Test PRF1 with mix of exact and semantic matches."""

        def sim_side_effect(extracted_id, ground_truth_id, formula):
            if extracted_id == "HP:0002187" and ground_truth_id == "HP:0012638":
                return 0.75  # Semantic match
            return 0.1  # Low similarity for other pairs

        mock_sim.side_effect = sim_side_effect

        extracted = [
            {"id": "HP:0001250", "name": "Seizure", "status": "affirmed"},
            {
                "id": "HP:0002187",
                "name": "Intellectual disability",
                "status": "affirmed",
            },
        ]

        ground_truth = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "assertion_status": "affirmed",
            },
            {
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "assertion_status": "affirmed",
            },
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted,
            ground_truth,
            target_assertion_status="affirmed",
            semantic_similarity_threshold=0.7,
        )

        # One exact + one semantic match
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1_score"] == 1.0
        assert results["tp_count"] == 2

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_multilingual_compatibility(self, mock_sim):
        """Test compatibility with multilingual terms."""
        # Mock similarity for cross-lingual matches
        mock_sim.return_value = 0.85

        extracted = [
            {"id": "HP:0001250", "name": "Anfall", "status": "affirmed"},  # German
            {"id": "HP:0000739", "name": "Angst", "status": "affirmed"},  # German
        ]

        ground_truth = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "assertion_status": "affirmed",
            },
            {
                "hpo_id": "HP:0000739",
                "label": "Anxiety",
                "assertion_status": "affirmed",
            },
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted, ground_truth, target_assertion_status="affirmed"
        )

        # Exact ID matches should work regardless of language
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1_score"] == 1.0
