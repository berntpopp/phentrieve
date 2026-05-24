"""Tests for the top-level run_adaptive_rechunking orchestration."""

from unittest.mock import MagicMock

import pytest

from phentrieve.retrieval.adaptive_rechunker import (
    AdaptiveRechunkingConfig,
    AdaptiveRechunkingResult,
    run_adaptive_rechunking,
)


def make_raw(similarities: list[float]) -> dict:
    return {
        "ids": [[f"HP:{i:07d}" for i in range(len(similarities))]],
        "metadatas": [
            [{"id": f"HP:{i:07d}", "label": "x"} for i in range(len(similarities))]
        ],
        "similarities": [similarities],
        "distances": [[1 - s for s in similarities]],
        "documents": [[""] * len(similarities)],
    }


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.detect_index_type.return_value = "single_vector"
    return retriever


@pytest.fixture
def basic_inputs():
    """One good chunk + one poor chunk for happy-path tests."""
    processed_chunks = [
        {
            "text": "Good chunk.",
            "status": "AFFIRMED",
            "start_char": 0,
            "end_char": 11,
            "source_indices": {"processing_stages": ["sentence"]},
        },
        {
            "text": (
                "Poor multi-finding sentence one. "
                "Poor multi-finding sentence two. "
                "Poor multi-finding sentence three."
            ),
            "status": "AFFIRMED",
            "start_char": 12,
            "end_char": 113,
            "source_indices": {"processing_stages": ["sentence"]},
        },
    ]
    chunk_results = [
        {
            "chunk_idx": 0,
            "chunk_text": processed_chunks[0]["text"],
            "matches": [{"id": "HP:0001", "name": "Good", "score": 0.9}],
        },
        {
            "chunk_idx": 1,
            "chunk_text": processed_chunks[1]["text"],
            "matches": [],
        },  # below threshold
    ]
    raw_query_results = [
        make_raw([0.9, 0.8]),  # good chunk
        make_raw([0.4, 0.39]),  # poor chunk: low score AND low margin
    ]
    return processed_chunks, chunk_results, raw_query_results


class TestRunAdaptiveRechunking:
    def test_multi_vector_retriever_queries_children_with_query_batch_multi_vector(
        self,
    ):
        processed = [
            {
                "text": "Sentence one. Sentence two. Sentence three.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 44,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]
        retriever = MagicMock()
        retriever.detect_index_type.return_value = "multi_vector"
        retriever.query_batch_multi_vector.return_value = [make_raw([0.9, 0.5])] * 5

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.55,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=1,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=5,
        )

        run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=retriever,
            language="en",
            config=config,
            num_results_per_chunk=6,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        retriever.query_batch_multi_vector.assert_called_once()
        call_kwargs = retriever.query_batch_multi_vector.call_args.kwargs
        assert call_kwargs["n_results"] == 6
        assert call_kwargs["texts"]
        retriever.query_batch.assert_not_called()

    def test_single_vector_retriever_queries_children_with_query_batch(
        self, mock_retriever
    ):
        processed = [
            {
                "text": "Sentence one. Sentence two. Sentence three.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 44,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]
        mock_retriever.query_batch.return_value = [make_raw([0.9, 0.5])] * 5

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.55,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=1,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=5,
        )

        run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=6,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        mock_retriever.query_batch.assert_called_once()
        assert (
            mock_retriever.query_batch.call_args.kwargs["include_similarities"] is True
        )
        mock_retriever.query_batch_multi_vector.assert_not_called()

    def test_disabled_returns_inputs_unchanged(self, mock_retriever, basic_inputs):
        chunks, results, raw = basic_inputs
        config = AdaptiveRechunkingConfig(enabled=False)
        out = run_adaptive_rechunking(
            processed_chunks=chunks,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )
        assert isinstance(out, AdaptiveRechunkingResult)
        assert out.processed_chunks == chunks
        assert out.chunk_results == results
        assert out.meta["enabled"] is False
        # No retrieval calls were made.
        child_retrieval_calls = (
            mock_retriever.query_batch.call_count
            + mock_retriever.query_batch_multi_vector.call_count
        )
        assert child_retrieval_calls == 0

    def test_no_poor_chunks_no_op(self, mock_retriever):
        """All chunks are fine - no subdivision, no extra retrieval calls."""
        processed = [
            {
                "text": "Good chunk one.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 15,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [
            {
                "chunk_idx": 0,
                "chunk_text": processed[0]["text"],
                "matches": [{"id": "HP:0001", "name": "x", "score": 0.9}],
            }
        ]
        raw = [make_raw([0.95, 0.85])]
        config = AdaptiveRechunkingConfig(enabled=True)

        out = run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )
        child_retrieval_calls = (
            mock_retriever.query_batch.call_count
            + mock_retriever.query_batch_multi_vector.call_count
        )
        assert child_retrieval_calls == 0
        assert out.meta["trigger_count"] == 0
        assert out.meta["subdivided_count"] == 0

    def test_one_poor_chunk_subdivided_with_improving_children(self, mock_retriever):
        """One poor chunk subdivided into children that improve via the gate."""
        processed = [
            {
                "text": "First sentence here. Second sentence here. Third sentence here.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 64,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]  # poor: low score, low margin

        # Mock retriever returns improved scores for children. We expect 1
        # retrieval call (the children) at depth 1. With max_sentences=3 and
        # overlap=0 the parent (3 sentences) yields a single child window; with
        # overlap=1 it yields multiple windows. We don't pin child count here
        # except via the side_effect length below.
        # Children improve enough to pass the gate. Oversize the side_effect
        # so we don't depend on the exact number of windows the chunker
        # emits for our 3-sentence parent.
        mock_retriever.query_batch.return_value = [
            make_raw([0.9, 0.5]),  # top_1 = 0.9 > 0.4 + 0.05 -> improves!
        ] * 5

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.55,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=1,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=10,
        )

        out = run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        # Exactly one retrieval call for the children at this single
        # recursion level.
        child_retrieval_calls = (
            mock_retriever.query_batch.call_count
            + mock_retriever.query_batch_multi_vector.call_count
        )
        assert child_retrieval_calls == 1
        assert out.meta["enabled"] is True
        assert out.meta["trigger_count"] == 1
        assert out.meta["subdivided_count"] == 1
        assert out.meta["reverted_count"] == 0
        # The parent chunk was replaced by one or more children in the
        # final flat list; children carry the depth-1 stage marker.
        assert all(
            "adaptive_rechunker_depth_1"
            in c.get("source_indices", {}).get("processing_stages", [])
            for c in out.processed_chunks
        )

    def test_call_count_invariant_at_max_depth_2(self, mock_retriever):
        """The hard cost-model contract: 1 call per recursion level."""
        processed = [
            {
                "text": "Sentence one. Sentence two. Sentence three.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 44,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]

        # Children always also "poor" so recursion continues.
        # Children improve enough to pass the gate, then still flag at depth 2.
        # Worst case: each level produces 2 children, so we return enough raw
        # results for any number of children we might emit.
        depth_1_child = make_raw([0.6, 0.59])  # improves over 0.4 -> keeps
        depth_2_child = make_raw([0.9, 0.5])  # improves further over 0.6

        mock_retriever.query_batch.side_effect = [
            [depth_1_child] * 5,  # depth 1 query (oversize is fine)
            [depth_2_child] * 5,  # depth 2 query
        ]

        config = AdaptiveRechunkingConfig(
            enabled=True,
            quality_threshold=0.7,  # higher to keep flagging
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_depth=2,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=5,
        )

        run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        # AT MOST 2 retrieval calls (depth 1 + depth 2). Hard invariant.
        child_retrieval_calls = (
            mock_retriever.query_batch.call_count
            + mock_retriever.query_batch_multi_vector.call_count
        )
        assert child_retrieval_calls <= 2

    def test_recursion_respects_max_depth(self, mock_retriever):
        """max_depth=1 means: subdivide once, do not recurse further."""
        processed = [
            {
                "text": "Sentence one. Sentence two. Sentence three.",
                "status": "AFFIRMED",
                "start_char": 0,
                "end_char": 44,
                "source_indices": {"processing_stages": ["sentence"]},
            },
        ]
        results = [{"chunk_idx": 0, "chunk_text": processed[0]["text"], "matches": []}]
        raw = [make_raw([0.4, 0.39])]
        # Children also flag as poor - at max_depth=1 we should not recurse.
        mock_retriever.query_batch.return_value = [make_raw([0.6, 0.59])] * 5

        config = AdaptiveRechunkingConfig(
            enabled=True,
            max_depth=1,
            quality_threshold=0.7,
            margin_threshold=0.03,
            score_improvement_gate=0.05,
            max_sentences_per_subchunk=2,
            overlap_sentences=0,
            min_chunk_chars=5,
        )

        out = run_adaptive_rechunking(
            processed_chunks=processed,
            chunk_results=results,
            raw_query_results=raw,
            retriever=mock_retriever,
            language="en",
            config=config,
            num_results_per_chunk=10,
            chunk_retrieval_threshold=0.7,
            min_confidence_for_aggregated=0.0,
            include_details=False,
        )

        child_retrieval_calls = (
            mock_retriever.query_batch.call_count
            + mock_retriever.query_batch_multi_vector.call_count
        )
        assert child_retrieval_calls == 1
        assert out.meta["max_depth_reached"] == 1
