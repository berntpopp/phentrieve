from collections import defaultdict


class DocumentAggregator:
    """Aggregate chunk-level predictions to document-level."""

    # Default thresholds for aggregation strategies
    DEFAULT_WEIGHTED_THRESHOLD: float = 0.5
    DEFAULT_CHUNK_THRESHOLD: float = 0.7

    def __init__(
        self,
        strategy: str = "union",
        weighted_threshold: float | None = None,
        chunk_threshold: float | None = None,
    ):
        """
        Args:
            strategy: "union", "intersection", "weighted", "threshold"
            weighted_threshold: Threshold for weighted aggregation (default: 0.5)
            chunk_threshold: Threshold for threshold aggregation (default: 0.7)
        """
        self.strategy = strategy
        self.weighted_threshold = weighted_threshold or self.DEFAULT_WEIGHTED_THRESHOLD
        self.chunk_threshold = chunk_threshold or self.DEFAULT_CHUNK_THRESHOLD

    def aggregate_chunks(self, chunk_predictions: list[dict[str, float]]) -> set[str]:
        """Aggregate HPO predictions from multiple chunks."""
        if self.strategy == "union":
            return self._union_aggregation(chunk_predictions)
        elif self.strategy == "intersection":
            return self._intersection_aggregation(chunk_predictions)
        elif self.strategy == "weighted":
            return self._weighted_aggregation(chunk_predictions)
        elif self.strategy == "threshold":
            return self._threshold_aggregation(chunk_predictions)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

    def _union_aggregation(self, chunk_predictions: list[dict[str, float]]) -> set[str]:
        """Take all unique HPO terms from all chunks."""
        all_terms: set[str] = set()
        for chunk_pred in chunk_predictions:
            all_terms.update(chunk_pred.keys())
        return all_terms

    def _intersection_aggregation(
        self, chunk_predictions: list[dict[str, float]]
    ) -> set[str]:
        """Take only HPO terms that appear in all chunks."""
        if not chunk_predictions:
            return set()

        # Start with terms from first chunk
        common_terms = set(chunk_predictions[0].keys())

        # Intersect with terms from other chunks
        for chunk_pred in chunk_predictions[1:]:
            common_terms &= set(chunk_pred.keys())

        return common_terms

    def _weighted_aggregation(
        self, chunk_predictions: list[dict[str, float]]
    ) -> set[str]:
        """Weight by confidence scores and chunk importance."""
        # Placeholder: simple average of scores
        term_scores: defaultdict[str, float] = defaultdict(float)
        term_counts: defaultdict[str, int] = defaultdict(int)

        for chunk_pred in chunk_predictions:
            for term, score in chunk_pred.items():
                term_scores[term] += score
                term_counts[term] += 1

        # Average scores
        avg_scores = {
            term: term_scores[term] / term_counts[term] for term in term_scores
        }

        # Return terms above threshold
        return {
            term
            for term, score in avg_scores.items()
            if score >= self.weighted_threshold
        }

    def _threshold_aggregation(
        self, chunk_predictions: list[dict[str, float]]
    ) -> set[str]:
        """Apply threshold to individual chunk predictions and union."""
        thresholded_terms: set[str] = set()
        for chunk_pred in chunk_predictions:
            thresholded_terms.update(
                term
                for term, score in chunk_pred.items()
                if score >= self.chunk_threshold
            )
        return thresholded_terms
