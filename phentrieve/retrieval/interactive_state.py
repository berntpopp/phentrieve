"""Interactive query session state management.

Encapsulates state needed across interactive query sessions in the CLI.
"""

from typing import Any

from phentrieve.retrieval.dense_retriever import DenseRetriever


class InteractiveState:
    """Container for interactive mode state across query sessions.

    Attributes are set in ``__init__`` rather than as class attributes so
    that each instance has its own state (class-level attributes would be
    shared across all instances, which is surprising in tests and any
    future non-singleton usage).
    """

    def __init__(self) -> None:
        self.model: Any | None = None
        self.retriever: DenseRetriever | None = None
        self.query_assertion_detector: Any | None = None  # CombinedAssertionDetector
        # Multi-vector settings
        self.multi_vector: bool = False
        self.aggregation_strategy: str = "label_synonyms_max"
        self.component_weights: dict[str, float] | None = None
        self.custom_formula: str | None = None


# Singleton instance for interactive mode
interactive_state = InteractiveState()
