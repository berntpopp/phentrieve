"""
Unit tests for CLI query enrichment with HPO term details.

Tests cover:
- Helper function for enriching structured query results
- Output formatters with definition and synonyms
- Integration with CLI command (mocked)
"""

import pytest

from phentrieve.cli.query_commands import enrich_query_results_with_details
from phentrieve.retrieval.output_formatters import (
    format_results_as_json,
    format_results_as_jsonl,
    format_results_as_text,
)

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture
def sample_structured_results():
    """Sample structured query results (without details)."""
    return [
        {
            "query_text_processed": "Patient has seizures",
            "header_info": "Found 2 matching HPO terms:",
            "original_query_assertion_status": "present",
            "results": [
                {
                    "rank": 1,
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "similarity": 0.95,
                },
                {
                    "rank": 2,
                    "hpo_id": "HP:0002119",
                    "label": "Ventriculomegaly",
                    "similarity": 0.87,
                },
            ],
        }
    ]


@pytest.fixture
def sample_enriched_results():
    """Sample structured query results (with details)."""
    return [
        {
            "query_text_processed": "Patient has seizures",
            "header_info": "Found 2 matching HPO terms:",
            "original_query_assertion_status": "present",
            "results": [
                {
                    "rank": 1,
                    "hpo_id": "HP:0001250",
                    "label": "Seizure",
                    "similarity": 0.95,
                    "definition": "A seizure is an intermittent abnormality of nervous system physiology",
                    "synonyms": ["Seizures", "Epileptic seizure"],
                },
                {
                    "rank": 2,
                    "hpo_id": "HP:0002119",
                    "label": "Ventriculomegaly",
                    "similarity": 0.87,
                    "definition": "An increase in size of the ventricular system",
                    "synonyms": ["Ventricular dilatation"],
                },
            ],
        }
    ]


class TestEnrichQueryResults:
    """Test the enrich_query_results_with_details helper function."""

    def test_enrich_empty_results(self):
        """Test enrichment with empty results."""
        results = []
        enriched = enrich_query_results_with_details(results)
        assert enriched == []

    def test_enrich_results_structure_preserved(self, sample_structured_results):
        """Test that structure is preserved after enrichment."""
        enriched = enrich_query_results_with_details(sample_structured_results)

        # Structure preserved
        assert len(enriched) == 1
        assert "query_text_processed" in enriched[0]
        assert "header_info" in enriched[0]
        assert "results" in enriched[0]
        assert len(enriched[0]["results"]) == 2

    def test_enrich_with_missing_database(self, sample_structured_results, tmp_path):
        """Test graceful handling when database doesn't exist."""
        enriched = enrich_query_results_with_details(
            sample_structured_results, data_dir_override=str(tmp_path)
        )

        # Should return results with None details
        assert len(enriched) == 1
        assert enriched[0]["results"][0]["definition"] is None
        assert enriched[0]["results"][0]["synonyms"] is None

    def test_enrich_preserves_metadata(
        self, sample_structured_results, tmp_path, monkeypatch
    ):
        """Test that metadata (query_text, header_info, assertion) is preserved."""

        # Mock enrichment to add dummy details
        def mock_enrich(results, data_dir_override=None):
            return [
                {**r, "definition": "Test def", "synonyms": ["Test syn"]}
                for r in results
            ]

        monkeypatch.setattr(
            "phentrieve.retrieval.details_enrichment.enrich_results_with_details",
            mock_enrich,
        )

        enriched = enrich_query_results_with_details(sample_structured_results)

        # Metadata preserved
        assert enriched[0]["query_text_processed"] == "Patient has seizures"
        assert enriched[0]["header_info"] == "Found 2 matching HPO terms:"
        assert enriched[0]["original_query_assertion_status"] == "present"


class TestTextFormatterWithDetails:
    """Test text formatter with definition and synonyms."""

    def test_format_without_details(self, sample_structured_results):
        """Test text format without details (baseline)."""
        output = format_results_as_text(sample_structured_results, sentence_mode=False)

        assert "HP:0001250" in output
        assert "Seizure" in output
        assert "0.95" in output
        # No details
        assert "Definition:" not in output
        assert "Synonyms:" not in output

    def test_format_with_details(self, sample_enriched_results):
        """Test text format with definition and synonyms."""
        output = format_results_as_text(sample_enriched_results, sentence_mode=False)

        # Basic info
        assert "HP:0001250" in output
        assert "Seizure" in output

        # Details added
        assert "Definition:" in output
        assert "nervous system physiology" in output
        assert "Synonyms:" in output
        assert "Epileptic seizure" in output

    def test_format_with_partial_details(self, sample_structured_results):
        """Test text format when only some fields have details."""
        # Add details to only first result
        sample_structured_results[0]["results"][0]["definition"] = (
            "A seizure is an abnormality"
        )
        sample_structured_results[0]["results"][0]["synonyms"] = None

        output = format_results_as_text(sample_structured_results, sentence_mode=False)

        # Definition shown
        assert "Definition:" in output
        assert "abnormality" in output

        # Synonyms not shown for None
        definition_lines = [
            line for line in output.split("\n") if "Definition:" in line
        ]
        synonym_lines = [line for line in output.split("\n") if "Synonyms:" in line]
        assert len(definition_lines) == 1
        assert len(synonym_lines) == 0  # Not shown because it's None


class TestJSONFormatterWithDetails:
    """Test JSON formatter with definition and synonyms."""

    def test_format_without_details(self, sample_structured_results):
        """Test JSON format without details (baseline)."""
        import json

        output = format_results_as_json(sample_structured_results, sentence_mode=False)
        data = json.loads(output)

        # Basic structure
        assert "hpo_terms" in data
        assert len(data["hpo_terms"]) == 2

        # No details
        first_term = data["hpo_terms"][0]
        assert "definition" not in first_term
        assert "synonyms" not in first_term

    def test_format_with_details(self, sample_enriched_results):
        """Test JSON format with definition and synonyms."""
        import json

        output = format_results_as_json(sample_enriched_results, sentence_mode=False)
        data = json.loads(output)

        # Details included
        first_term = data["hpo_terms"][0]
        assert "definition" in first_term
        assert "nervous system" in first_term["definition"]
        assert "synonyms" in first_term
        assert isinstance(first_term["synonyms"], list)
        assert "Epileptic seizure" in first_term["synonyms"]

    def test_format_preserves_other_fields(self, sample_enriched_results):
        """Test that other fields (rank, hpo_id, etc.) are preserved."""
        import json

        # Add cross-encoder score
        sample_enriched_results[0]["results"][0]["cross_encoder_score"] = 0.92
        sample_enriched_results[0]["results"][0]["original_rank"] = 3

        output = format_results_as_json(sample_enriched_results, sentence_mode=False)
        data = json.loads(output)

        first_term = data["hpo_terms"][0]
        assert first_term["rank"] == 1
        assert first_term["hpo_id"] == "HP:0001250"
        assert first_term["name"] == "Seizure"
        assert first_term["confidence"] == 0.95
        assert first_term["cross_encoder_score"] == 0.92
        assert first_term["original_rank"] == 3
        assert "definition" in first_term
        assert "synonyms" in first_term


class TestJSONLFormatterWithDetails:
    """Test JSON Lines formatter with definition and synonyms."""

    def test_format_without_details(self, sample_structured_results):
        """Test JSONL format without details (baseline)."""
        import json

        output = format_results_as_jsonl(sample_structured_results)
        lines = output.strip().split("\n")

        assert len(lines) == 1
        data = json.loads(lines[0])

        # No details
        first_term = data["hpo_terms"][0]
        assert "definition" not in first_term
        assert "synonyms" not in first_term

    def test_format_with_details(self, sample_enriched_results):
        """Test JSONL format with definition and synonyms."""
        import json

        output = format_results_as_jsonl(sample_enriched_results)
        lines = output.strip().split("\n")

        assert len(lines) == 1
        data = json.loads(lines[0])

        # Details included
        first_term = data["hpo_terms"][0]
        assert "definition" in first_term
        assert "nervous system" in first_term["definition"]
        assert "synonyms" in first_term
        assert "Epileptic seizure" in first_term["synonyms"]

    def test_format_multiple_result_sets(self, sample_enriched_results):
        """Test JSONL with multiple result sets (sentence mode)."""
        import json

        # Duplicate result set
        results = sample_enriched_results + sample_enriched_results

        output = format_results_as_jsonl(results)
        lines = output.strip().split("\n")

        # One line per result set
        assert len(lines) == 2

        # Each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "hpo_terms" in data
            assert data["hpo_terms"][0]["definition"] is not None


class TestOutputFormatterEdgeCases:
    """Test edge cases in output formatters."""

    def test_empty_definition_not_shown_in_text(self, sample_structured_results):
        """Test that empty string definition is not shown in text output."""
        sample_structured_results[0]["results"][0]["definition"] = ""
        sample_structured_results[0]["results"][0]["synonyms"] = []

        output = format_results_as_text(sample_structured_results, sentence_mode=False)

        # Empty values should not create output lines
        assert "Definition:" not in output
        assert "Synonyms:" not in output

    def test_none_details_not_shown_in_text(self, sample_structured_results):
        """Test that None definition/synonyms are not shown in text output."""
        sample_structured_results[0]["results"][0]["definition"] = None
        sample_structured_results[0]["results"][0]["synonyms"] = None

        output = format_results_as_text(sample_structured_results, sentence_mode=False)

        assert "Definition:" not in output
        assert "Synonyms:" not in output

    def test_long_synonym_list_formatted(self, sample_structured_results):
        """Test that long synonym lists are properly formatted."""
        sample_structured_results[0]["results"][0]["synonyms"] = [
            "Synonym 1",
            "Synonym 2",
            "Synonym 3",
            "Synonym 4",
            "Synonym 5",
        ]

        output = format_results_as_text(sample_structured_results, sentence_mode=False)

        assert "Synonyms:" in output
        assert "Synonym 1" in output
        assert "Synonym 5" in output
        # Check comma-separated format
        assert "Synonym 1, Synonym 2" in output
