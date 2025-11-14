"""
Unit tests for the semantic metrics module.

Tests the semantically-aware P/R/F1 calculations and assertion accuracy metrics.
"""

import unittest
from unittest.mock import patch

from phentrieve.evaluation.semantic_metrics import (
    calculate_assertion_accuracy,
    calculate_semantically_aware_set_based_prf1,
)


class TestSemanticMetrics(unittest.TestCase):
    """Test cases for semantic metrics module."""

    def test_calculate_assertion_accuracy_perfect(self):
        """Test assertion accuracy with perfect matching."""
        # Create some matched pairs with matching assertion statuses
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

        self.assertEqual(accuracy, 100.0)
        self.assertEqual(correct, 2)
        self.assertEqual(total, 2)

    def test_calculate_assertion_accuracy_partial(self):
        """Test assertion accuracy with partial matching."""
        # Create some matched pairs with some mismatched assertion statuses
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

        self.assertEqual(accuracy, 50.0)
        self.assertEqual(correct, 1)
        self.assertEqual(total, 2)

    def test_calculate_assertion_accuracy_empty(self):
        """Test assertion accuracy with empty matched pairs."""
        matched_pairs = []

        accuracy, correct, total = calculate_assertion_accuracy(matched_pairs)

        self.assertEqual(accuracy, 0.0)
        self.assertEqual(correct, 0)
        self.assertEqual(total, 0)

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_calculate_semantically_aware_set_based_prf1_exact_matches(self, mock_sim):
        """Test PRF1 with exact matches only."""
        # Mock won't be called for exact matches, but set up anyway
        mock_sim.return_value = 0.5

        # Create extracted and ground truth annotations with exact matches
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
        self.assertEqual(results["precision"], 1.0)
        self.assertEqual(results["recall"], 1.0)
        self.assertEqual(results["f1_score"], 1.0)
        self.assertEqual(results["tp_count"], 2)
        self.assertEqual(results["fp_count"], 0)
        self.assertEqual(results["fn_count"], 0)
        self.assertEqual(len(results["tp_matched_pairs_list"]), 2)

        # Mock should not have been called for exact matches
        mock_sim.assert_not_called()

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_calculate_semantically_aware_set_based_prf1_semantic_matches(
        self, mock_sim
    ):
        """Test PRF1 with semantic matches."""

        # Set up mock to return high similarity for specific pairs
        def sim_side_effect(extracted_id, ground_truth_id, formula):
            if extracted_id == "HP:0001251" and ground_truth_id == "HP:0001250":
                return 0.8  # High similarity
            return 0.1  # Low similarity for other pairs

        mock_sim.side_effect = sim_side_effect

        # Create extracted and ground truth annotations with semantic matches
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

        # One semantic match expected
        self.assertEqual(results["precision"], 0.5)
        self.assertEqual(results["recall"], 0.5)
        self.assertEqual(results["f1_score"], 0.5)
        self.assertEqual(results["tp_count"], 1)
        self.assertEqual(results["fp_count"], 1)
        self.assertEqual(results["fn_count"], 1)
        self.assertEqual(len(results["tp_matched_pairs_list"]), 1)

        # Mock should have been called for semantic matching
        self.assertTrue(mock_sim.called)

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_calculate_semantically_aware_set_based_prf1_mixed_matches(self, mock_sim):
        """Test PRF1 with a mix of exact and semantic matches."""

        # Set up mock to return high similarity for specific pairs
        def sim_side_effect(extracted_id, ground_truth_id, formula):
            if extracted_id == "HP:0001251" and ground_truth_id == "HP:0001253":
                return 0.8  # High similarity
            return 0.1  # Low similarity for other pairs

        mock_sim.side_effect = sim_side_effect

        # Create extracted and ground truth annotations with mixed matches
        extracted = [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "status": "affirmed",
            },  # Exact match
            {
                "id": "HP:0001251",
                "name": "Atonic seizure",
                "status": "affirmed",
            },  # Semantic match
            {"id": "HP:0000001", "name": "All", "status": "affirmed"},  # No match
        ]

        ground_truth = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "assertion_status": "affirmed",
            },  # Exact match
            {
                "hpo_id": "HP:0001253",
                "label": "Abnormal electroencephalogram",
                "assertion_status": "affirmed",
            },  # Semantic match
            {
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "assertion_status": "affirmed",
            },  # No match
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted,
            ground_truth,
            target_assertion_status="affirmed",
            semantic_similarity_threshold=0.7,
        )

        # Two matches expected (1 exact, 1 semantic)
        self.assertAlmostEqual(results["precision"], 2 / 3)
        self.assertAlmostEqual(results["recall"], 2 / 3)
        self.assertAlmostEqual(results["f1_score"], 2 / 3)
        self.assertEqual(results["tp_count"], 2)
        self.assertEqual(results["fp_count"], 1)
        self.assertEqual(results["fn_count"], 1)
        self.assertEqual(len(results["tp_matched_pairs_list"]), 2)

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_calculate_semantically_aware_set_based_prf1_no_matches(self, mock_sim):
        """Test PRF1 with no matches."""
        # Mock won't be called if there are no potential matches
        mock_sim.return_value = 0.1  # Low similarity

        # Create extracted and ground truth annotations with no matches
        extracted = [
            {"id": "HP:0001251", "name": "Atonic seizure", "status": "affirmed"},
            {"id": "HP:0000001", "name": "All", "status": "affirmed"},
        ]

        ground_truth = [
            {
                "hpo_id": "HP:0012638",
                "label": "Abnormality of nervous system physiology",
                "assertion_status": "affirmed",
            },
            {
                "hpo_id": "HP:0002315",
                "label": "Headache",
                "assertion_status": "affirmed",
            },
        ]

        results = calculate_semantically_aware_set_based_prf1(
            extracted,
            ground_truth,
            target_assertion_status="affirmed",
            semantic_similarity_threshold=0.7,
        )

        # No matches expected
        self.assertEqual(results["precision"], 0.0)
        self.assertEqual(results["recall"], 0.0)
        self.assertEqual(results["f1_score"], 0.0)
        self.assertEqual(results["tp_count"], 0)
        self.assertEqual(results["fp_count"], 2)
        self.assertEqual(results["fn_count"], 2)
        self.assertEqual(len(results["tp_matched_pairs_list"]), 0)

    @patch("phentrieve.evaluation.semantic_metrics.calculate_semantic_similarity")
    def test_multilingual_compatibility(self, mock_sim):
        """Test compatibility with multilingual terms (based on memory of model testing)."""
        # Mock similarity for cross-lingual matches
        mock_sim.return_value = 0.85

        # Create extracted and ground truth annotations with German and English terms
        extracted = [
            {
                "id": "HP:0001250",
                "name": "Anfall",
                "status": "affirmed",
            },  # German for "Seizure"
            {
                "id": "HP:0000739",
                "name": "Angst",
                "status": "affirmed",
            },  # German for "Anxiety"
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
            extracted,
            ground_truth,
            target_assertion_status="affirmed",
            semantic_similarity_threshold=0.7,
        )

        # Perfect match expected based on HPO IDs despite different languages
        self.assertEqual(results["precision"], 1.0)
        self.assertEqual(results["recall"], 1.0)
        self.assertEqual(results["f1_score"], 1.0)
        self.assertEqual(results["tp_count"], 2)
        self.assertEqual(results["fp_count"], 0)
        self.assertEqual(results["fn_count"], 0)
        self.assertEqual(len(results["tp_matched_pairs_list"]), 2)


if __name__ == "__main__":
    unittest.main()
