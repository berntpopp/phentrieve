"""HARD CONTRACT: at most ``max_depth`` additional ``retriever.query_batch``
calls during adaptive rechunking, even when every chunk flags as poor at
every level.

This is the cost-model invariant from Spec B. The
``precomputed_query_results`` parameter on ``orchestrate_hpo_extraction``
is what makes the bound tight; a regression that re-queries parents during
the final aggregation pass would fail this test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    run_adaptive_rechunking,
)

pytestmark = pytest.mark.integration


def _make_raw(similarities: list[float]) -> dict[str, Any]:
    """Build a query_batch-shaped raw result dict for a single chunk."""
    return {
        "ids": [[f"HP:{i:07d}" for i in range(len(similarities))]],
        "metadatas": [
            [{"id": f"HP:{i:07d}", "label": "x"} for i in range(len(similarities))]
        ],
        "similarities": [similarities],
        "distances": [[1 - s for s in similarities]],
        "documents": [[""] * len(similarities)],
    }


def test_at_most_max_depth_query_batch_calls() -> None:
    """Worst case fan-out: every chunk flags at every reachable level.

    The hard cap is ``config.max_depth`` calls to ``retriever.query_batch``.
    A regression that re-queries parents during the final aggregation pass
    (e.g., dropping ``precomputed_query_results`` somewhere) would fail
    this assertion.
    """
    retriever = MagicMock()
    # Simulate every level still flagging poor. Side effects supply one
    # query_batch return value per recursion level. ``max_depth=2`` so we
    # provide at most two levels of fan-out.
    retriever.query_batch.side_effect = [
        # depth 1: 3 children (one sentence each), all still poor
        [_make_raw([0.5, 0.49]) for _ in range(3)],
        # depth 2: would-be grandchildren; sentence-level parents cannot
        # subdivide further, so this is unused but kept defensively.
        [_make_raw([0.55, 0.54]) for _ in range(3)],
    ]

    processed = [
        {
            "text": "First sentence. Second sentence. Third sentence.",
            "status": "AFFIRMED",
            "start_char": 0,
            "end_char": 49,
            "source_indices": {"processing_stages": ["sentence"]},
        }
    ]
    chunk_results = [
        {"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}
    ]
    raw = [_make_raw([0.4, 0.39])]

    config = AdaptiveRechunkingConfig(
        enabled=True,
        max_depth=2,
        quality_threshold=0.6,
        margin_threshold=0.03,
        score_improvement_gate=0.05,
        max_sentences_per_subchunk=1,
        overlap_sentences=0,
        min_chunk_chars=5,
    )

    # ``run_adaptive_rechunking`` may re-aggregate via
    # ``orchestrate_hpo_extraction`` which is patched away by injecting
    # precomputed raws; we still want to be sure no other retrieval calls
    # leak through.
    run_adaptive_rechunking(
        processed_chunks=processed,
        chunk_results=chunk_results,
        raw_query_results=raw,
        retriever=retriever,
        language="en",
        config=config,
        num_results_per_chunk=10,
        chunk_retrieval_threshold=0.7,
        min_confidence_for_aggregated=0.0,
        include_details=False,
    )

    assert retriever.query_batch.call_count <= config.max_depth, (
        f"query_batch was called {retriever.query_batch.call_count} times; "
        f"max_depth={config.max_depth} should be the cap. "
        f"This fails if the precomputed_query_results contract regresses "
        f"and parents (or accepted children) get re-queried."
    )
