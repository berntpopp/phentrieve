"""Real unit tests for output_formatters module (actual code execution)."""

import json

import pytest

from phentrieve.retrieval.output_formatters import (
    format_results_as_json,
    format_results_as_jsonl,
    format_results_as_text,
)

pytestmark = pytest.mark.unit


class TestFormatResultsAsText:
    """Test format_results_as_text function with real logic execution."""

    def test_single_result_basic(self):
        """Test formatting a single basic result."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 2 matching HPO terms:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Test Term 1",
                        "similarity": 0.95,
                    },
                    {
                        "rank": 2,
                        "hpo_id": "HP:0000002",
                        "label": "Test Term 2",
                        "similarity": 0.85,
                    },
                ],
            }
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=False)

        # Assert
        assert "Found 2 matching HPO terms:" in output
        assert "HP:0000001" in output
        assert "Test Term 1" in output
        assert "0.95" in output
        assert "HP:0000002" in output
        assert "Test Term 2" in output
        assert "0.85" in output

    def test_sentence_mode_multiple_segments(self):
        """Test formatting with sentence mode and multiple segments."""
        # Arrange
        results = [
            {
                "query_text_processed": "segment 1",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term 1",
                        "similarity": 0.9,
                    }
                ],
            },
            {
                "query_text_processed": "segment 2",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000002",
                        "label": "Term 2",
                        "similarity": 0.8,
                    }
                ],
            },
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=True)

        # Assert
        assert "==== Results for: segment 1 ====" in output
        assert "==== Results for: segment 2 ====" in output
        assert "HP:0000001" in output
        assert "HP:0000002" in output

    def test_with_assertion_status(self):
        """Test formatting with assertion status."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "original_query_assertion_status": "negated",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=False)

        # Assert
        assert "Detected Assertion for Original Input Query: NEGATED" in output

    def test_with_assertion_status_value_fallback(self):
        """Test formatting with assertion status value fallback."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "original_query_assertion_status_value": "uncertain",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=False)

        # Assert
        assert "Detected Assertion for Original Input Query: UNCERTAIN" in output

    def test_with_reranking_info(self):
        """Test formatting with cross-encoder re-ranking information."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                        "cross_encoder_score": 0.95,
                        "original_rank": 3,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=False)

        # Assert
        assert "re-ranked from #3" in output
        assert "cross-encoder: 0.95" in output

    def test_empty_results(self):
        """Test formatting with empty results."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "No matching HPO terms found.",
                "results": [],
            }
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=False)

        # Assert
        assert "No matching HPO terms found." in output
        # Should not contain any HPO IDs
        assert "HP:" not in output

    def test_no_results_field(self):
        """Test formatting when results field is missing."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "No results.",
            }
        ]

        # Act
        output = format_results_as_text(results, sentence_mode=False)

        # Assert
        assert "No results." in output


class TestFormatResultsAsJson:
    """Test format_results_as_json function with real logic execution."""

    def test_single_result_not_sentence_mode(self):
        """Test JSON formatting for single result (not sentence mode)."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Test Term",
                        "similarity": 0.95,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_json(results, sentence_mode=False)
        parsed = json.loads(output)

        # Assert
        assert isinstance(parsed, dict)  # Single object, not array
        assert parsed["query_text_processed"] == "test query"
        assert len(parsed["hpo_terms"]) == 1
        assert parsed["hpo_terms"][0]["hpo_id"] == "HP:0000001"
        assert parsed["hpo_terms"][0]["name"] == "Test Term"
        assert parsed["hpo_terms"][0]["confidence"] == 0.95

    def test_sentence_mode_multiple_results(self):
        """Test JSON formatting for sentence mode with multiple results."""
        # Arrange
        results = [
            {
                "query_text_processed": "query 1",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term 1",
                        "similarity": 0.9,
                    }
                ],
            },
            {
                "query_text_processed": "query 2",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000002",
                        "label": "Term 2",
                        "similarity": 0.8,
                    }
                ],
            },
        ]

        # Act
        output = format_results_as_json(results, sentence_mode=True)
        parsed = json.loads(output)

        # Assert
        assert isinstance(parsed, list)  # Array for multiple results
        assert len(parsed) == 2
        assert parsed[0]["query_text_processed"] == "query 1"
        assert parsed[1]["query_text_processed"] == "query 2"

    def test_with_cross_encoder_scores(self):
        """Test JSON formatting with cross-encoder scores."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                        "cross_encoder_score": 0.95,
                        "original_rank": 2,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_json(results, sentence_mode=False)
        parsed = json.loads(output)

        # Assert
        assert parsed["hpo_terms"][0]["cross_encoder_score"] == 0.95
        assert parsed["hpo_terms"][0]["original_rank"] == 2

    def test_assertion_status_fallback(self):
        """Test JSON formatting with assertion status fallback."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "original_query_assertion_status_value": "negated",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_json(results, sentence_mode=False)
        parsed = json.loads(output)

        # Assert
        assert parsed["original_query_assertion_status"] == "negated"

    def test_empty_results(self):
        """Test JSON formatting with empty results."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "No terms found.",
                "results": [],
            }
        ]

        # Act
        output = format_results_as_json(results, sentence_mode=False)
        parsed = json.loads(output)

        # Assert
        assert parsed["hpo_terms"] == []

    def test_retrieval_info_structure(self):
        """Test that retrieval_info is correctly structured."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Custom header info",
                "results": [],
            }
        ]

        # Act
        output = format_results_as_json(results, sentence_mode=False)
        parsed = json.loads(output)

        # Assert
        assert "retrieval_info" in parsed
        assert parsed["retrieval_info"]["header"] == "Custom header info"


class TestFormatResultsAsJsonl:
    """Test format_results_as_jsonl function with real logic execution."""

    def test_single_result_jsonl(self):
        """Test JSONL formatting for a single result."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Test Term",
                        "similarity": 0.95,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_jsonl(results)
        lines = output.strip().split("\n")

        # Assert
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["query_text_processed"] == "test query"
        assert len(parsed["hpo_terms"]) == 1

    def test_multiple_results_jsonl(self):
        """Test JSONL formatting for multiple results."""
        # Arrange
        results = [
            {
                "query_text_processed": "query 1",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term 1",
                        "similarity": 0.9,
                    }
                ],
            },
            {
                "query_text_processed": "query 2",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000002",
                        "label": "Term 2",
                        "similarity": 0.8,
                    }
                ],
            },
        ]

        # Act
        output = format_results_as_jsonl(results)
        lines = output.strip().split("\n")

        # Assert
        assert len(lines) == 2
        parsed1 = json.loads(lines[0])
        parsed2 = json.loads(lines[1])
        assert parsed1["query_text_processed"] == "query 1"
        assert parsed2["query_text_processed"] == "query 2"

    def test_jsonl_with_cross_encoder(self):
        """Test JSONL formatting with cross-encoder scores."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                        "cross_encoder_score": 0.95,
                        "original_rank": 3,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_jsonl(results)
        parsed = json.loads(output)

        # Assert
        assert parsed["hpo_terms"][0]["cross_encoder_score"] == 0.95
        assert parsed["hpo_terms"][0]["original_rank"] == 3

    def test_jsonl_assertion_status(self):
        """Test JSONL formatting with assertion status."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "Found 1 term:",
                "original_query_assertion_status": "uncertain",
                "results": [
                    {
                        "rank": 1,
                        "hpo_id": "HP:0000001",
                        "label": "Term",
                        "similarity": 0.9,
                    }
                ],
            }
        ]

        # Act
        output = format_results_as_jsonl(results)
        parsed = json.loads(output)

        # Assert
        assert parsed["original_query_assertion_status"] == "uncertain"

    def test_jsonl_empty_results(self):
        """Test JSONL formatting with empty results."""
        # Arrange
        results = [
            {
                "query_text_processed": "test query",
                "header_info": "No terms found.",
                "results": [],
            }
        ]

        # Act
        output = format_results_as_jsonl(results)
        parsed = json.loads(output)

        # Assert
        assert parsed["hpo_terms"] == []
