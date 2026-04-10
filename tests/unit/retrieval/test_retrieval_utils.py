"""Tests for phentrieve/retrieval/utils.py shared utilities."""

import pytest

from phentrieve.retrieval.utils import convert_multi_vector_to_chromadb_format

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
