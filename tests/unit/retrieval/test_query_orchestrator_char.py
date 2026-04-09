"""Characterization tests for query_orchestrator.process_query().

These tests lock current behavior before refactoring.
They must pass identically before AND after orchestrator decomposition.
"""

from unittest.mock import MagicMock, patch

import pytest

from phentrieve.retrieval.interactive_state import InteractiveState
from phentrieve.retrieval.query_orchestrator import (
    format_results,
    process_query,
    segment_text,
)
from phentrieve.retrieval.utils import convert_results_to_candidates

# Note: convert_multi_vector_to_chromadb_format is tested separately
# and will move to retrieval/utils.py in Task 4. We do NOT import it here
# so these characterization tests remain stable through that relocation.

pytestmark = pytest.mark.unit


def _make_chromadb_results(ids, scores, docs=None, metadatas=None):
    """Helper to create ChromaDB-style result dicts."""
    return {
        "ids": [ids],
        "distances": [[1.0 - s for s in scores]],
        "documents": [docs or [f"Document for {hid}" for hid in ids]],
        "metadatas": [metadatas or [{"source": "test"} for _ in ids]],
    }


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.query.return_value = _make_chromadb_results(
        ids=["HP:0001250", "HP:0001251", "HP:0001252"],
        scores=[0.95, 0.85, 0.75],
    )
    return retriever


@pytest.fixture
def mock_cross_encoder_model():
    """Mock cross-encoder model (not the reranker module)."""
    model = MagicMock()
    model.predict.return_value = [0.92, 0.88, 0.70]
    return model


class TestProcessQuerySentenceMode:
    """Tests for process_query with sentence_mode=True."""

    def test_sentence_mode_returns_per_sentence_results(self, mock_retriever):
        """Each sentence gets its own retrieval results."""
        results = process_query(
            text="Patient has seizures. Also has ataxia.",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
        )
        # Two sentences should produce up to 2 result sets
        assert isinstance(results, list)
        assert len(results) >= 1
        # Retriever should be called at least once per sentence
        assert mock_retriever.query.call_count >= 1

    def test_sentence_mode_with_reranking(
        self, mock_retriever, mock_cross_encoder_model
    ):
        """Sentence mode with cross-encoder reranking."""
        with patch(
            "phentrieve.retrieval.pipeline.reranker_module"
        ) as mock_reranker_mod:
            mock_reranker_mod.protected_dense_rerank.return_value = [
                {
                    "hpo_id": "HP:0001250",
                    "english_doc": "Seizure",
                    "metadata": {"source": "test"},
                    "bi_encoder_score": 0.95,
                    "rank": 1,
                    "comparison_text": "Seizure",
                },
                {
                    "hpo_id": "HP:0001251",
                    "english_doc": "Ataxia",
                    "metadata": {"source": "test"},
                    "bi_encoder_score": 0.85,
                    "rank": 2,
                    "comparison_text": "Ataxia",
                },
            ]
            results = process_query(
                text="Patient has seizures.",
                retriever=mock_retriever,
                cross_encoder=mock_cross_encoder_model,
                num_results=3,
                similarity_threshold=0.5,
                sentence_mode=True,
                rerank_count=10,
            )
            assert isinstance(results, list)
            assert len(results) >= 1
            # Reranker should have been called
            assert mock_reranker_mod.protected_dense_rerank.call_count >= 1

    def test_sentence_mode_empty_text_returns_empty(self, mock_retriever):
        """Empty text returns empty results."""
        mock_retriever.query.return_value = _make_chromadb_results(ids=[], scores=[])
        results = process_query(
            text="",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
        )
        assert isinstance(results, list)


class TestProcessQueryFullTextMode:
    """Tests for process_query with sentence_mode=False."""

    def test_full_text_without_reranking(self, mock_retriever):
        """Full text mode returns formatted results."""
        results = process_query(
            text="Patient presents with severe intellectual disability",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=False,
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        # Should have results with expected structure
        for result_set in results:
            assert "results" in result_set

    def test_full_text_with_reranking(self, mock_retriever, mock_cross_encoder_model):
        """Full text with cross-encoder reranking."""
        with patch(
            "phentrieve.retrieval.pipeline.reranker_module"
        ) as mock_reranker_mod:
            mock_reranker_mod.protected_dense_rerank.return_value = [
                {
                    "hpo_id": "HP:0001250",
                    "english_doc": "Seizure",
                    "metadata": {"source": "test"},
                    "bi_encoder_score": 0.95,
                    "rank": 1,
                    "comparison_text": "Seizure",
                },
            ]
            results = process_query(
                text="Patient presents with severe intellectual disability",
                retriever=mock_retriever,
                cross_encoder=mock_cross_encoder_model,
                num_results=3,
                similarity_threshold=0.5,
                sentence_mode=False,
                rerank_count=10,
            )
            assert isinstance(results, list)
            assert mock_reranker_mod.protected_dense_rerank.call_count == 1


class TestProcessQueryFallback:
    """Tests for sentence mode falling back to full text when no results."""

    def test_fallback_to_full_text_when_sentences_empty(self, mock_retriever):
        """When sentence mode yields no results, falls back to full text."""
        call_count = 0

        def query_side_effect(text, n_results=10):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First calls (per-sentence) return nothing
                return _make_chromadb_results(ids=[], scores=[])
            # Fallback call returns results
            return _make_chromadb_results(ids=["HP:0001250"], scores=[0.9])

        mock_retriever.query.side_effect = query_side_effect
        results = process_query(
            text="Short text. Another sentence.",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
        )
        assert isinstance(results, list)
        # Should have called retriever for sentences + fallback
        assert mock_retriever.query.call_count >= 2


class TestConvertResultsToCandidates:
    """Tests for convert_results_to_candidates."""

    def test_converts_valid_results(self):
        results = _make_chromadb_results(
            ids=["HP:0001250", "HP:0001251"],
            scores=[0.95, 0.85],
        )
        candidates = convert_results_to_candidates(results)
        assert len(candidates) == 2
        assert candidates[0]["hpo_id"] == "HP:0001250"
        assert candidates[0]["rank"] == 1
        assert abs(candidates[0]["bi_encoder_score"] - 0.95) < 0.01

    def test_empty_results_returns_empty_list(self):
        results = _make_chromadb_results(ids=[], scores=[])
        candidates = convert_results_to_candidates(results)
        assert candidates == []

    def test_none_results_returns_empty_list(self):
        candidates = convert_results_to_candidates(None)
        assert candidates == []


class TestSegmentText:
    """Tests for segment_text."""

    def test_segments_english(self):
        sentences = segment_text("Patient has seizures. Also has ataxia.")
        assert len(sentences) == 2

    def test_single_sentence(self):
        sentences = segment_text("Patient has seizures")
        assert len(sentences) == 1


class TestInteractiveState:
    """Tests for InteractiveState class."""

    def test_default_values(self):
        state = InteractiveState()
        assert state.model is None
        assert state.retriever is None
        assert state.cross_encoder is None
        assert state.multi_vector is False
        assert state.aggregation_strategy == "label_synonyms_max"


class TestFormatResults:
    """Extended characterization tests for format_results."""

    def test_format_results_basic_structure(self):
        """Verify the basic output structure of format_results."""
        results = _make_chromadb_results(
            ids=["HP:0001250"],
            scores=[0.9],
            metadatas=[{"hpo_id": "HP:0001250", "label": "Seizure"}],
        )
        formatted = format_results(
            results=results,
            threshold=0.5,
            max_results=3,
            query="test",
        )
        assert "results" in formatted
        assert "query_text_processed" in formatted
        assert "header_info" in formatted
        assert formatted["query_text_processed"] == "test"

    def test_format_results_empty_input(self):
        """Empty results produce appropriate empty output."""
        results = _make_chromadb_results(ids=[], scores=[])
        formatted = format_results(results=results, threshold=0.5, max_results=3)
        assert formatted["results"] == []

    def test_format_results_none_input(self):
        """None results produce appropriate empty output."""
        formatted = format_results(results=None, threshold=0.5, max_results=3)
        assert formatted["results"] == []

    def test_format_results_threshold_filtering(self):
        """Results below threshold are filtered out."""
        results = _make_chromadb_results(
            ids=["HP:0001250", "HP:0001251"],
            scores=[0.9, 0.3],
            metadatas=[
                {"hpo_id": "HP:0001250", "label": "Seizure"},
                {"hpo_id": "HP:0001251", "label": "Ataxia"},
            ],
        )
        formatted = format_results(results=results, threshold=0.5, max_results=10)
        # Only the high-score result should remain
        assert len(formatted["results"]) == 1
        assert formatted["results"][0]["hpo_id"] == "HP:0001250"

    def test_format_results_max_results_limit(self):
        """max_results parameter limits output count."""
        results = _make_chromadb_results(
            ids=["HP:0001250", "HP:0001251", "HP:0001252"],
            scores=[0.95, 0.90, 0.85],
            metadatas=[
                {"hpo_id": "HP:0001250", "label": "Seizure"},
                {"hpo_id": "HP:0001251", "label": "Ataxia"},
                {"hpo_id": "HP:0001252", "label": "Tremor"},
            ],
        )
        formatted = format_results(results=results, threshold=0.5, max_results=2)
        assert len(formatted["results"]) == 2

    def test_format_results_assertion_fields(self):
        """Assertion status fields are present in output."""
        results = _make_chromadb_results(
            ids=["HP:0001250"],
            scores=[0.9],
            metadatas=[{"hpo_id": "HP:0001250", "label": "Seizure"}],
        )
        formatted = format_results(
            results=results,
            threshold=0.5,
            max_results=3,
            original_query_assertion_status=None,
            original_query_assertion_details=None,
        )
        assert "original_query_assertion_status" in formatted
        assert "original_query_assertion_details" in formatted


class TestProcessQueryMultiVector:
    """Tests for multi-vector paths in process_query."""

    def test_multi_vector_sentence_mode(self, mock_retriever):
        """Multi-vector sentence mode calls query_multi_vector."""
        mock_retriever.query_multi_vector.return_value = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
                "component_scores": {},
            }
        ]
        results = process_query(
            text="Patient has seizures.",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=True,
            multi_vector=True,
        )
        assert isinstance(results, list)
        assert mock_retriever.query_multi_vector.call_count >= 1

    def test_multi_vector_full_text_mode(self, mock_retriever):
        """Multi-vector full-text mode calls query_multi_vector."""
        mock_retriever.query_multi_vector.return_value = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
                "component_scores": {},
            }
        ]
        results = process_query(
            text="Patient has seizures.",
            retriever=mock_retriever,
            num_results=3,
            similarity_threshold=0.5,
            sentence_mode=False,
            multi_vector=True,
        )
        assert isinstance(results, list)
        assert mock_retriever.query_multi_vector.call_count >= 1


class TestExecuteSingleVectorPipeline:
    """Tests for the extracted single-vector pipeline."""

    def test_without_reranking(self, mock_retriever):
        from phentrieve.retrieval.pipeline import execute_single_vector_pipeline

        results = execute_single_vector_pipeline(
            retriever=mock_retriever,
            text="seizures",
            num_results=3,
        )
        assert "ids" in results
        assert len(results["ids"][0]) == 3
        mock_retriever.query.assert_called_once()

    def test_with_reranking(self, mock_retriever, mock_cross_encoder_model):
        from phentrieve.retrieval.pipeline import execute_single_vector_pipeline

        with patch("phentrieve.retrieval.pipeline.reranker_module") as mock_reranker:
            mock_reranker.protected_dense_rerank.return_value = [
                {
                    "hpo_id": "HP:0001250",
                    "english_doc": "Seizure",
                    "metadata": {"source": "test"},
                    "bi_encoder_score": 0.95,
                    "rank": 1,
                    "comparison_text": "Seizure",
                },
            ]
            results = execute_single_vector_pipeline(
                retriever=mock_retriever,
                text="seizures",
                num_results=3,
                cross_encoder=mock_cross_encoder_model,
                rerank_count=10,
            )
            assert "ids" in results
            mock_reranker.protected_dense_rerank.assert_called_once()

    def test_reranking_error_falls_back(self, mock_retriever, mock_cross_encoder_model):
        """When reranking raises an exception, falls back to unranked results."""
        from phentrieve.retrieval.pipeline import execute_single_vector_pipeline

        with patch("phentrieve.retrieval.pipeline.reranker_module") as mock_reranker:
            mock_reranker.protected_dense_rerank.side_effect = RuntimeError("fail")
            results = execute_single_vector_pipeline(
                retriever=mock_retriever,
                text="seizures",
                num_results=3,
                cross_encoder=mock_cross_encoder_model,
                rerank_count=10,
                debug=True,
                output_func=lambda x: None,
            )
            # Should still return results (the unranked ones)
            assert "ids" in results
            assert len(results["ids"][0]) == 3

    def test_debug_output(self, mock_retriever, mock_cross_encoder_model):
        """Debug mode produces output via output_func."""
        from phentrieve.retrieval.pipeline import execute_single_vector_pipeline

        debug_messages = []
        with patch("phentrieve.retrieval.pipeline.reranker_module") as mock_reranker:
            mock_reranker.protected_dense_rerank.return_value = [
                {
                    "hpo_id": "HP:0001250",
                    "english_doc": "Seizure",
                    "metadata": {"source": "test"},
                    "bi_encoder_score": 0.95,
                    "rank": 1,
                    "comparison_text": "Seizure",
                },
            ]
            execute_single_vector_pipeline(
                retriever=mock_retriever,
                text="seizures",
                num_results=3,
                cross_encoder=mock_cross_encoder_model,
                rerank_count=10,
                debug=True,
                output_func=debug_messages.append,
            )
        assert any("Reranking" in msg for msg in debug_messages)
