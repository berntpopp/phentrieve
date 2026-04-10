"""Characterization tests for orchestrate_hpo_extraction.

These tests lock the current behavior of the orchestrator before its
decomposition. They must continue to pass unchanged through the refactor.
"""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

pytestmark = pytest.mark.unit


def _make_mock_retriever(batch_results):
    """Build a mock DenseRetriever whose query_batch returns ``batch_results``."""
    retriever = MagicMock()
    retriever.query_batch.return_value = batch_results
    return retriever


def _chroma_batch_entry(items):
    """Build one ChromaDB-style batch entry from a list of
    (hpo_id, label, similarity) tuples."""
    metadatas = [[{"hpo_id": hpo_id, "label": label} for hpo_id, label, _ in items]]
    similarities = [[sim for _, _, sim in items]]
    return {"metadatas": metadatas, "similarities": similarities}


class TestEmptyAndSingleChunk:
    def test_empty_chunks_returns_empty_lists(self):
        retriever = _make_mock_retriever([])
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=[],
            retriever=retriever,
        )
        assert aggregated == []
        assert chunk_results == []

    def test_single_chunk_single_match(self):
        retriever = _make_mock_retriever(
            [_chroma_batch_entry([("HP:0001250", "Seizure", 0.9)])]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["Patient had a seizure."],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert len(chunk_results) == 1
        assert chunk_results[0]["chunk_idx"] == 0
        assert chunk_results[0]["chunk_text"] == "Patient had a seizure."
        assert len(chunk_results[0]["matches"]) == 1
        assert chunk_results[0]["matches"][0]["id"] == "HP:0001250"
        assert len(aggregated) == 1
        assert aggregated[0]["id"] == "HP:0001250"
        assert aggregated[0]["rank"] == 1
        assert aggregated[0]["count"] == 1


class TestThresholdFiltering:
    def test_matches_below_threshold_are_dropped(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry(
                    [
                        ("HP:0001250", "Seizure", 0.9),  # keep
                        ("HP:0000001", "All", 0.2),  # drop, below 0.5
                    ]
                )
            ]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert len(chunk_results[0]["matches"]) == 1
        assert chunk_results[0]["matches"][0]["id"] == "HP:0001250"

    def test_min_confidence_for_aggregated_filters_aggregated_only(self):
        retriever = _make_mock_retriever(
            [_chroma_batch_entry([("HP:0001250", "Seizure", 0.6)])]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
            min_confidence_for_aggregated=0.8,  # Above 0.6
        )
        # Chunk match survives chunk filter (0.6 >= 0.5)
        assert len(chunk_results[0]["matches"]) == 1
        # But aggregated is empty because avg_score 0.6 < 0.8
        assert aggregated == []


class TestTopTermPerChunk:
    def test_top_term_per_chunk_keeps_only_first(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry(
                    [
                        ("HP:0001250", "Seizure", 0.9),
                        ("HP:0001251", "Ataxia", 0.8),
                    ]
                )
            ]
        )
        _, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
            top_term_per_chunk=True,
        )
        assert len(chunk_results[0]["matches"]) == 1
        assert chunk_results[0]["matches"][0]["id"] == "HP:0001250"


class TestMultiChunkAggregation:
    def test_same_hpo_across_chunks_is_aggregated(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)]),
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.7)]),
            ]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["chunk a", "chunk b"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert len(chunk_results) == 2
        assert len(aggregated) == 1
        term = aggregated[0]
        assert term["id"] == "HP:0001250"
        assert term["count"] == 2
        assert term["score"] == pytest.approx(0.9)  # max
        assert term["avg_score"] == pytest.approx(0.8)  # (0.9 + 0.7) / 2
        assert sorted(term["chunks"]) == [0, 1]
        # Top evidence chunk should be the one with max_score (0.9)
        assert term["top_evidence_chunk_idx"] == 0


class TestAssertionStatuses:
    def test_assertion_status_propagated_to_matches_and_aggregated(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)]),
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.8)]),
            ]
        )
        aggregated, chunk_results = orchestrate_hpo_extraction(
            text_chunks=["a", "b"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
            assertion_statuses=["affirmed", "negated"],
        )
        assert chunk_results[0]["matches"][0]["assertion_status"] == "affirmed"
        assert chunk_results[1]["matches"][0]["assertion_status"] == "negated"
        # Aggregated: most-common; here tied 1-1, so Counter returns the
        # first one encountered ("affirmed").
        assert aggregated[0]["assertion_status"] == "affirmed"
        assert aggregated[0]["status"] == "affirmed"  # alias


class TestRankingAndOrdering:
    def test_results_sorted_by_avg_score_then_count_desc(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry(
                    [
                        ("HP:0001250", "Seizure", 0.9),
                        ("HP:0001251", "Ataxia", 0.7),
                    ]
                )
            ]
        )
        aggregated, _ = orchestrate_hpo_extraction(
            text_chunks=["x"],
            retriever=retriever,
            chunk_retrieval_threshold=0.5,
        )
        assert [t["id"] for t in aggregated] == ["HP:0001250", "HP:0001251"]
        assert [t["rank"] for t in aggregated] == [1, 2]


class TestRetrieverInteraction:
    def test_retriever_query_batch_called_with_all_chunks(self):
        retriever = _make_mock_retriever(
            [
                _chroma_batch_entry([("HP:0001250", "Seizure", 0.9)]),
                _chroma_batch_entry([]),
            ]
        )
        orchestrate_hpo_extraction(
            text_chunks=["a", "b"],
            retriever=retriever,
            num_results_per_chunk=7,
            chunk_retrieval_threshold=0.5,
        )
        retriever.query_batch.assert_called_once()
        kwargs = retriever.query_batch.call_args.kwargs
        assert kwargs["texts"] == ["a", "b"]
        assert kwargs["n_results"] == 7
        assert kwargs["include_similarities"] is True
