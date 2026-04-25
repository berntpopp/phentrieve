"""Return type for orchestrate_hpo_extraction.

Implements __iter__ and __getitem__ to preserve legacy 2-tuple unpacking
(aggregated_results, chunk_results) while exposing raw_query_results as a
new field for callers that need access to unfiltered retrieval scores
(specifically the adaptive rechunker).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """Return value of orchestrate_hpo_extraction.

    Backward compatibility: iteration and indexing yield the legacy 2-tuple
    ``(aggregated_results, chunk_results)``. Attribute access exposes
    ``raw_query_results`` for new callers that need the unfiltered top-K
    similarity output from ``DenseRetriever.query_batch`` (e.g. adaptive
    re-chunking, which needs scores below ``chunk_retrieval_threshold``).
    """

    aggregated_results: list[dict[str, Any]]
    chunk_results: list[dict[str, Any]]
    raw_query_results: list[dict[str, Any]] = field(default_factory=list)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """Yield (aggregated_results, chunk_results) for legacy 2-tuple unpack."""
        yield self.aggregated_results
        yield self.chunk_results

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        """Index 0 -> aggregated_results, 1 -> chunk_results. For legacy callers."""
        if idx == 0:
            return self.aggregated_results
        if idx == 1:
            return self.chunk_results
        raise IndexError(f"OrchestrationResult index {idx} out of range (0..1)")

    def __len__(self) -> int:
        return 2  # iteration yields 2 elements
