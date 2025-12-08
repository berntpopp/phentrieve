import numpy as np
from sklearn.metrics import precision_recall_curve


class ThresholdOptimizer:
    """Learn optimal similarity thresholds from validation data."""

    def __init__(self, optimization_metric: str = "f1"):
        self.optimization_metric = optimization_metric
        self.learned_thresholds: dict[str, float] = {}

    def optimize_threshold(
        self, scores: np.ndarray, labels: np.ndarray, model_name: str
    ) -> float:
        """
        Find optimal threshold for a specific model.

        Args:
            scores: Similarity scores from model
            labels: Binary labels (1 = correct, 0 = incorrect)
            model_name: Name of the model for storing threshold

        Returns:
            Optimal threshold value
        """
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)

        optimal_idx: int
        if self.optimization_metric == "f1":
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
            optimal_idx = int(np.argmax(f1_scores))
        elif self.optimization_metric == "precision_at_recall":
            # Find highest precision at minimum recall (e.g., 0.8)
            min_recall = 0.8
            valid_idx = np.where(recalls >= min_recall)[0]
            if len(valid_idx) > 0:
                optimal_idx = int(valid_idx[np.argmax(precisions[valid_idx])])
            else:
                optimal_idx = 0

        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        )

        # Store for this model
        self.learned_thresholds[model_name] = optimal_threshold

        return optimal_threshold

    def apply_threshold(self, scores: dict[str, float], model_name: str) -> list[str]:
        """Apply learned threshold to filter predictions."""
        threshold = self.learned_thresholds.get(model_name, 0.7)  # Default to 0.7
        return [hpo_id for hpo_id, score in scores.items() if score >= threshold]
