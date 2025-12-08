"""Tests for extraction metrics module."""

import pytest

from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult,
    _calculate_prf,
    _doc_metrics,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_calculate_prf_basic(self):
        """Test basic precision/recall/f1 calculation."""
        precision, recall, f1 = _calculate_prf(tp=8, fp=2, fn=2)
        assert precision == pytest.approx(0.8)
        assert recall == pytest.approx(0.8)
        assert f1 == pytest.approx(0.8)

    def test_calculate_prf_perfect(self):
        """Test perfect precision/recall/f1."""
        precision, recall, f1 = _calculate_prf(tp=10, fp=0, fn=0)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_calculate_prf_zero_predictions(self):
        """Test with no predictions."""
        precision, recall, f1 = _calculate_prf(tp=0, fp=0, fn=5)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_calculate_prf_all_wrong(self):
        """Test with all wrong predictions."""
        precision, recall, f1 = _calculate_prf(tp=0, fp=5, fn=5)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_doc_metrics(self):
        """Test single document metrics calculation."""
        result = ExtractionResult(
            doc_id="doc1",
            predicted=[("HP:0001", "PRESENT"), ("HP:0002", "PRESENT")],
            gold=[("HP:0001", "PRESENT"), ("HP:0003", "PRESENT")],
        )
        precision, recall, f1, gold_count = _doc_metrics(result)
        # 1 TP, 1 FP, 1 FN
        assert precision == pytest.approx(0.5)
        assert recall == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)
        assert gold_count == 2


class TestCorpusExtractionMetrics:
    """Test corpus-level metrics."""

    def test_micro_averaging(self):
        """Test micro-averaged metrics calculation."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001250", "PRESENT"), ("HP:0002360", "ABSENT")],
                gold=[("HP:0001250", "PRESENT"), ("HP:0001251", "PRESENT")],
            ),
            ExtractionResult(
                doc_id="doc2",
                predicted=[("HP:0001252", "PRESENT")],
                gold=[("HP:0001252", "PRESENT"), ("HP:0001253", "ABSENT")],
            ),
        ]

        evaluator = CorpusExtractionMetrics(averaging="micro")
        metrics = evaluator.calculate_metrics(results)

        # Doc1: 1 TP (HP:0001250), 1 FP (HP:0002360), 1 FN (HP:0001251)
        # Doc2: 1 TP (HP:0001252), 0 FP, 1 FN (HP:0001253)
        # Total: 2 TP, 1 FP, 2 FN
        assert metrics.micro["precision"] == pytest.approx(2 / 3)  # 2/(2+1)
        assert metrics.micro["recall"] == pytest.approx(0.5)  # 2/(2+2)
        # F1 = 2 * (2/3) * 0.5 / ((2/3) + 0.5) = 2 * (1/3) / (7/6) = (2/3) / (7/6) = 4/7
        assert metrics.micro["f1"] == pytest.approx(4 / 7, rel=0.01)

    def test_macro_averaging(self):
        """Test macro-averaged metrics calculation."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "PRESENT")],
            ),
            ExtractionResult(
                doc_id="doc2",
                predicted=[("HP:0002", "PRESENT"), ("HP:0003", "PRESENT")],
                gold=[("HP:0002", "PRESENT")],
            ),
        ]

        evaluator = CorpusExtractionMetrics(averaging="macro")
        metrics = evaluator.calculate_metrics(results)

        # Doc1: P=1, R=1, F1=1
        # Doc2: P=0.5, R=1, F1=0.667
        # Macro: P=0.75, R=1, F1=0.833
        assert metrics.macro["precision"] == pytest.approx(0.75)
        assert metrics.macro["recall"] == pytest.approx(1.0)
        assert metrics.macro["f1"] == pytest.approx(5 / 6, rel=0.01)

    def test_weighted_averaging(self):
        """Test weighted-averaged metrics calculation."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "PRESENT")],  # weight=1
            ),
            ExtractionResult(
                doc_id="doc2",
                predicted=[("HP:0002", "PRESENT"), ("HP:0003", "PRESENT")],
                gold=[
                    ("HP:0002", "PRESENT"),
                    ("HP:0004", "PRESENT"),
                    ("HP:0005", "PRESENT"),
                ],  # weight=3
            ),
        ]

        evaluator = CorpusExtractionMetrics(averaging="weighted")
        metrics = evaluator.calculate_metrics(results)

        # Doc1: P=1, R=1, F1=1, weight=1
        # Doc2: P=0.5 (1TP, 1FP), R=0.333 (1TP, 2FN), F1=0.4, weight=3
        # Weighted P = (1*1 + 0.5*3) / 4 = 2.5/4 = 0.625
        # Weighted R = (1*1 + 0.333*3) / 4 = 2/4 = 0.5
        assert metrics.weighted["precision"] == pytest.approx(0.625)
        assert metrics.weighted["recall"] == pytest.approx(0.5)

    def test_calculate_all_metrics(self):
        """Test calculating all averaging strategies at once."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "PRESENT")],
            ),
        ]

        evaluator = CorpusExtractionMetrics()
        metrics = evaluator.calculate_all_metrics(results)

        # All should have values
        assert metrics.micro["f1"] == 1.0
        assert metrics.macro["f1"] == 1.0
        assert metrics.weighted["f1"] == 1.0

    def test_empty_results(self):
        """Test with empty results."""
        evaluator = CorpusExtractionMetrics()
        metrics = evaluator.calculate_all_metrics([])

        assert metrics.micro["precision"] == 0.0
        assert metrics.micro["recall"] == 0.0
        assert metrics.micro["f1"] == 0.0

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap CI calculation."""
        # Create some results
        results = [
            ExtractionResult(
                doc_id=f"doc{i}",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "PRESENT")],
            )
            for i in range(10)
        ]

        evaluator = CorpusExtractionMetrics()
        ci = evaluator.bootstrap_confidence_intervals(results, n_bootstrap=100)

        # With perfect results, CI should be tight around 1.0
        assert "precision" in ci
        assert "recall" in ci
        assert "f1" in ci
        assert ci["precision"][0] <= ci["precision"][1]
        assert ci["f1"][0] >= 0.9  # Should be close to 1.0

    def test_bootstrap_empty_results(self):
        """Test bootstrap CI with empty results."""
        evaluator = CorpusExtractionMetrics()
        ci = evaluator.bootstrap_confidence_intervals([])

        assert ci["precision"] == (0.0, 0.0)
        assert ci["recall"] == (0.0, 0.0)
        assert ci["f1"] == (0.0, 0.0)

    def test_unknown_averaging_strategy(self):
        """Test that unknown averaging strategy raises error."""
        evaluator = CorpusExtractionMetrics(averaging="unknown")
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "PRESENT")],
            )
        ]

        with pytest.raises(ValueError, match="Unknown averaging strategy"):
            evaluator.calculate_metrics(results)


class TestAssertionAwareMatching:
    """Test that assertion status affects matching."""

    def test_same_hpo_different_assertion_no_match(self):
        """Same HPO ID with different assertion should NOT match."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "ABSENT")],
            )
        ]

        evaluator = CorpusExtractionMetrics()
        metrics = evaluator.calculate_metrics(results)

        # Should be 0 TP, 1 FP, 1 FN
        assert metrics.micro["precision"] == 0.0
        assert metrics.micro["recall"] == 0.0

    def test_same_hpo_same_assertion_matches(self):
        """Same HPO ID with same assertion should match."""
        results = [
            ExtractionResult(
                doc_id="doc1",
                predicted=[("HP:0001", "PRESENT")],
                gold=[("HP:0001", "PRESENT")],
            )
        ]

        evaluator = CorpusExtractionMetrics()
        metrics = evaluator.calculate_metrics(results)

        assert metrics.micro["precision"] == 1.0
        assert metrics.micro["recall"] == 1.0
        assert metrics.micro["f1"] == 1.0
