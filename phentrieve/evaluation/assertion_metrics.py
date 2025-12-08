
import numpy as np
from sklearn.metrics import confusion_matrix

from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    ExtractionResult,
)


class AssertionMetrics:
    """Evaluate HPO term + assertion jointly."""

    ASSERTION_TYPES = ["PRESENT", "ABSENT", "UNCERTAIN"]

    def calculate_joint_f1(
        self,
        predicted: list[tuple[str, str]],
        gold: list[tuple[str, str]]
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
        predicted_assertions: list[str],
        gold_assertions: list[str]
    ) -> np.ndarray:
        """Build confusion matrix for assertion types."""
        return confusion_matrix(
            gold_assertions,
            predicted_assertions,
            labels=self.ASSERTION_TYPES
        )

    def stratified_metrics(
        self,
        results: list[ExtractionResult]
    ) -> dict[str, dict[str, float]]:
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

    def _filter_by_assertion(
        self,
        results: list[ExtractionResult],
        assertion_type: str
    ) -> list[ExtractionResult]:
        """Filter results to only include specific assertion type."""
        filtered_results = []

        for result in results:
            # Filter gold terms by assertion
            gold_filtered = [
                (hpo_id, assertion) for hpo_id, assertion in result.gold
                if assertion == assertion_type
            ]

            # Filter predicted terms by assertion
            pred_filtered = [
                (hpo_id, assertion) for hpo_id, assertion in result.predicted
                if assertion == assertion_type
            ]

            if gold_filtered or pred_filtered:  # Only include if there are terms
                filtered_results.append(ExtractionResult(
                    doc_id=result.doc_id,
                    predicted=pred_filtered,
                    gold=gold_filtered
                ))

        return filtered_results

class AssertionDetector:
    """Enhanced assertion detection with joint evaluation."""

    def __init__(self, enable_context: bool = True):
        self.enable_context = enable_context
        self.context_analyzer = self._init_context_analyzer()

    def _init_context_analyzer(self):
        """Initialize context analyzer for assertion detection."""
        # Placeholder: will implement ConText algorithm later
        return None

    def detect_assertion(
        self,
        text: str,
        hpo_term: str,
        span: tuple[int, int]
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

    def _simple_detection(self, text: str, span: tuple[int, int]) -> str:
        """Simple keyword-based assertion detection."""
        # Extract context around the span
        start, end = span
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()

        # Check for negation keywords
        negation_keywords = [
            "no", "not", "without", "denies", "denied", "negative",
            "absent", "lack of", "free of", "rule out", "ruled out"
        ]

        if any(keyword in context for keyword in negation_keywords):
            return "ABSENT"

        # Check for uncertainty keywords
        uncertainty_keywords = [
            "possible", "potential", "maybe", "uncertain", "questionable",
            "suspected", "likely", "probable", "differential"
        ]

        if any(keyword in context for keyword in uncertainty_keywords):
            return "UNCERTAIN"

        # Default to present
        return "PRESENT"
