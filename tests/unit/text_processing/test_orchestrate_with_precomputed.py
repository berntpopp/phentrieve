"""Tests for the new precomputed_query_results parameter and
OrchestrationResult return type on orchestrate_hpo_extraction."""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.orchestration_result import OrchestrationResult


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    # Standard query_batch return shape.
    r.query_batch.return_value = [
        {
            "ids": [["HP:0001250"]],
            "metadatas": [[{"id": "HP:0001250", "label": "Seizure"}]],
            "similarities": [[0.85]],
            "distances": [[0.15]],
            "documents": [[""]],
        }
    ]
    return r


def test_returns_orchestration_result(mock_retriever):
    result = orchestrate_hpo_extraction(
        text_chunks=["seizures"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
    )
    assert isinstance(result, OrchestrationResult)
    assert result.aggregated_results
    assert result.chunk_results
    assert result.raw_query_results  # populated from query_batch


def test_legacy_unpack_still_works(mock_retriever):
    aggregated, chunks = orchestrate_hpo_extraction(
        text_chunks=["seizures"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
    )
    assert isinstance(aggregated, list)
    assert isinstance(chunks, list)


def test_precomputed_skips_retrieval(mock_retriever):
    raw = [
        {
            "ids": [["HP:0001"]],
            "metadatas": [[{"id": "HP:0001", "label": "Foo"}]],
            "similarities": [[0.9]],
            "distances": [[0.1]],
            "documents": [[""]],
        }
    ]
    result = orchestrate_hpo_extraction(
        text_chunks=["any text"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
        precomputed_query_results=raw,
    )
    # query_batch should NOT have been called.
    assert mock_retriever.query_batch.call_count == 0
    assert result.raw_query_results == raw
    assert any(t["id"] == "HP:0001" for t in result.aggregated_results)


def test_precomputed_value_drives_aggregation(mock_retriever):
    raw = [
        {
            "ids": [["HP:0002"]],
            "metadatas": [[{"id": "HP:0002", "label": "Bar"}]],
            "similarities": [[0.95]],
            "distances": [[0.05]],
            "documents": [[""]],
        }
    ]
    # mock_retriever would have returned HP:0001250 (from fixture) if called,
    # but precomputed should drive aggregation entirely.
    result = orchestrate_hpo_extraction(
        text_chunks=["unused"],
        retriever=mock_retriever,
        num_results_per_chunk=1,
        precomputed_query_results=raw,
    )
    ids = [t["id"] for t in result.aggregated_results]
    assert "HP:0002" in ids
    assert "HP:0001250" not in ids  # precomputed wins
