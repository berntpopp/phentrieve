"""Interactive query session state management.

Encapsulates state needed across interactive query sessions in the CLI.
"""

from typing import Any

from phentrieve.retrieval.dense_retriever import DenseRetriever


class InteractiveState:
    """Container for interactive mode state across query sessions."""

    model: Any | None = None
    retriever: DenseRetriever | None = None
    cross_encoder: Any | None = None  # CrossEncoder type from sentence_transformers
    query_assertion_detector: Any | None = None  # CombinedAssertionDetector
    # Multi-vector settings
    multi_vector: bool = False
    aggregation_strategy: str = "label_synonyms_max"
    component_weights: dict[str, float] | None = None
    custom_formula: str | None = None


# Singleton instance for interactive mode
interactive_state = InteractiveState()
