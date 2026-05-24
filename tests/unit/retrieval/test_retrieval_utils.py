"""Tests for phentrieve/retrieval/utils.py shared utilities."""

import pytest

from phentrieve.retrieval.utils import (
    convert_multi_vector_to_chromadb_format,
    query_chunk_candidates,
)

pytestmark = pytest.mark.unit


class TestConvertMultiVectorToChromadbFormat:
    """Tests for convert_multi_vector_to_chromadb_format."""

    def test_empty_list_returns_empty_structure(self):
        result = convert_multi_vector_to_chromadb_format([])
        assert result == {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }

    def test_default_does_not_include_similarities(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.91,
            }
        ]

        converted = convert_multi_vector_to_chromadb_format(results)

        assert "similarities" not in converted

    def test_include_similarities_true_adds_nested_scores(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.91,
                "matched_component": "synonym",
                "matched_text": "Convulsions",
            },
            {
                "hpo_id": "HP:0001251",
                "label": "Ataxia",
                "similarity": 0.75,
            },
        ]

        converted = convert_multi_vector_to_chromadb_format(
            results,
            include_similarities=True,
        )

        assert converted["ids"] == [["HP:0001250", "HP:0001251"]]
        assert converted["similarities"] == [[0.91, 0.75]]
        assert converted["distances"] == [[pytest.approx(0.09), pytest.approx(0.25)]]
        assert converted["metadatas"][0][0]["hpo_id"] == "HP:0001250"
        assert converted["metadatas"][0][0]["matched_component"] == "synonym"
        assert converted["metadatas"][0][0]["matched_text"] == "Convulsions"

    def test_empty_result_includes_empty_similarities_when_requested(self):
        converted = convert_multi_vector_to_chromadb_format(
            [],
            include_similarities=True,
        )

        assert converted == {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "similarities": [[]],
        }

    def test_single_result(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
                "component_scores": {"label": 0.9, "synonym": 0.85},
            }
        ]
        converted = convert_multi_vector_to_chromadb_format(results)
        assert converted["ids"] == [["HP:0001250"]]
        assert converted["documents"] == [["Seizure"]]
        assert len(converted["distances"][0]) == 1
        assert abs(converted["distances"][0][0] - 0.1) < 0.01
        assert converted["metadatas"][0][0]["component_scores"] == {
            "label": 0.9,
            "synonym": 0.85,
        }

    def test_multiple_results(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
            },
            {
                "hpo_id": "HP:0001251",
                "label": "Ataxia",
                "similarity": 0.8,
            },
        ]
        converted = convert_multi_vector_to_chromadb_format(results)
        assert len(converted["ids"][0]) == 2
        assert converted["ids"][0][0] == "HP:0001250"
        assert converted["ids"][0][1] == "HP:0001251"

    def test_missing_label_uses_default(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "similarity": 0.9,
            }
        ]
        converted = convert_multi_vector_to_chromadb_format(results)
        assert converted["documents"][0][0] == ""
        assert converted["metadatas"][0][0]["label"] == ""

    def test_missing_similarity_uses_zero(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
            }
        ]
        converted = convert_multi_vector_to_chromadb_format(results)
        assert converted["distances"][0][0] == 1.0  # 1.0 - 0.0

    def test_no_component_scores_omits_key(self):
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
            }
        ]
        converted = convert_multi_vector_to_chromadb_format(results)
        assert "component_scores" not in converted["metadatas"][0][0]

    def test_chromadb_format_structure(self):
        """Verify the nested list structure matches ChromaDB format."""
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.9,
            }
        ]
        converted = convert_multi_vector_to_chromadb_format(results)
        # ChromaDB format wraps each field in an outer list
        assert isinstance(converted["ids"], list)
        assert isinstance(converted["ids"][0], list)
        assert isinstance(converted["distances"], list)
        assert isinstance(converted["distances"][0], list)
        assert isinstance(converted["documents"], list)
        assert isinstance(converted["documents"][0], list)
        assert isinstance(converted["metadatas"], list)
        assert isinstance(converted["metadatas"][0], list)


class _RecordingRetriever:
    def __init__(
        self,
        index_type: str = "single_vector",
        detection_error: Exception | None = None,
    ):
        self.index_type = index_type
        self.detection_error = detection_error
        self.batch_calls: list[dict[str, object]] = []
        self.batch_multi_vector_calls: list[dict[str, object]] = []

    def detect_index_type(self) -> str:
        if self.detection_error is not None:
            raise self.detection_error
        return self.index_type

    def query_batch(
        self,
        *,
        texts: list[str],
        n_results: int,
        include_similarities: bool,
    ) -> list[dict[str, object]]:
        self.batch_calls.append(
            {
                "texts": texts,
                "n_results": n_results,
                "include_similarities": include_similarities,
            }
        )
        return [{"ids": [["single"]], "metadatas": [[]], "similarities": [[]]}]

    def query_batch_multi_vector(
        self,
        *,
        texts: list[str],
        n_results: int,
    ) -> list[dict[str, object]]:
        self.batch_multi_vector_calls.append(
            {
                "texts": texts,
                "n_results": n_results,
            }
        )
        return [{"ids": [["multi"]], "metadatas": [[]], "similarities": [[]]}]


class TestQueryChunkCandidates:
    def test_uses_multi_vector_batch_when_index_type_matches(self):
        retriever = _RecordingRetriever(index_type="multi_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a", "chunk b"],
            n_results=7,
        )

        assert results[0]["ids"] == [["multi"]]
        assert retriever.batch_multi_vector_calls == [
            {"texts": ["chunk a", "chunk b"], "n_results": 7}
        ]
        assert retriever.batch_calls == []

    def test_uses_query_batch_for_single_vector_index(self):
        retriever = _RecordingRetriever(index_type="single_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=3,
        )

        assert results[0]["ids"] == [["single"]]
        assert retriever.batch_calls == [
            {
                "texts": ["chunk a"],
                "n_results": 3,
                "include_similarities": True,
            }
        ]
        assert retriever.batch_multi_vector_calls == []

    def test_detection_error_falls_back_to_query_batch(self):
        retriever = _RecordingRetriever(detection_error=RuntimeError("metadata failed"))

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=4,
        )

        assert results[0]["ids"] == [["single"]]
        assert len(retriever.batch_calls) == 1
        assert retriever.batch_multi_vector_calls == []

    def test_unknown_index_type_falls_back_to_query_batch(self):
        retriever = _RecordingRetriever(index_type="legacy_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=6,
        )

        assert results[0]["ids"] == [["single"]]
        assert retriever.batch_calls == [
            {
                "texts": ["chunk a"],
                "n_results": 6,
                "include_similarities": True,
            }
        ]
        assert retriever.batch_multi_vector_calls == []

    def test_include_similarities_false_is_forwarded_to_query_batch(self):
        retriever = _RecordingRetriever(index_type="single_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=3,
            include_similarities=False,
        )

        assert results[0]["ids"] == [["single"]]
        assert retriever.batch_calls == [
            {
                "texts": ["chunk a"],
                "n_results": 3,
                "include_similarities": False,
            }
        ]
        assert retriever.batch_multi_vector_calls == []

    def test_missing_detect_index_type_falls_back_to_query_batch(self):
        class NoDetectIndexType:
            def __init__(self):
                self.batch_calls: list[dict[str, object]] = []

            def query_batch(
                self,
                *,
                texts: list[str],
                n_results: int,
                include_similarities: bool,
            ) -> list[dict[str, object]]:
                self.batch_calls.append(
                    {
                        "texts": texts,
                        "n_results": n_results,
                        "include_similarities": include_similarities,
                    }
                )
                return [{"ids": [["single"]], "metadatas": [[]], "similarities": [[]]}]

        retriever = NoDetectIndexType()

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=4,
        )

        assert results[0]["ids"] == [["single"]]
        assert retriever.batch_calls == [
            {
                "texts": ["chunk a"],
                "n_results": 4,
                "include_similarities": True,
            }
        ]

    def test_multi_vector_index_without_method_falls_back_to_query_batch(self):
        class NoMultiVectorMethod(_RecordingRetriever):
            query_batch_multi_vector = None

        retriever = NoMultiVectorMethod(index_type="multi_vector")

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=5,
        )

        assert results[0]["ids"] == [["single"]]
        assert len(retriever.batch_calls) == 1

    def test_dynamic_mock_attributes_do_not_force_multi_vector_route(self):
        from unittest.mock import MagicMock

        retriever = MagicMock()
        retriever.detect_index_type.return_value = MagicMock()
        retriever.query_batch.return_value = [
            {"ids": [["single"]], "metadatas": [[]], "similarities": [[]]}
        ]

        results = query_chunk_candidates(
            retriever=retriever,
            text_chunks=["chunk a"],
            n_results=2,
        )

        assert results[0]["ids"] == [["single"]]
        retriever.query_batch.assert_called_once_with(
            texts=["chunk a"],
            n_results=2,
            include_similarities=True,
        )
        retriever.query_batch_multi_vector.assert_not_called()
