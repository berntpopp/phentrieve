"""Unit tests for hallucination filtering in tool-guided annotation."""

import logging
from unittest.mock import MagicMock

import pytest

from phentrieve.llm.types import (
    AnnotationMode,
    AssertionStatus,
    HPOAnnotation,
    ToolCall,
)


class TestHallucinationFiltering:
    """Tests for _filter_against_candidates method."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.model = "github/gpt-4o"
        provider.temperature = 0.0
        provider.supports_tools.return_value = True
        return provider

    @pytest.fixture
    def strategy(self, mock_provider):
        """Create a ToolGuidedStrategy with mock provider."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        return ToolGuidedStrategy(mock_provider, mode=AnnotationMode.TOOL_TEXT)

    def _create_annotation(
        self, hpo_id: str, term_name: str = "Test Term"
    ) -> HPOAnnotation:
        """Helper to create an HPOAnnotation."""
        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=term_name,
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.9,
            source_mode=AnnotationMode.TOOL_TEXT,
        )

    def _create_tool_call_with_results(self, hpo_ids: list[str]) -> ToolCall:
        """Helper to create a tool call with HPO results (actual list format)."""
        # This matches what _process_clinical_text() actually returns
        result = [
            {
                "hpo_id": hpo_id,
                "term_name": f"Term {hpo_id}",
                "assertion": "affirmed",
                "score": 0.85,
                "evidence_text": "test chunk",
            }
            for hpo_id in hpo_ids
        ]
        return ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result=result,
        )

    def _create_tool_call_with_dict_results(self, hpo_ids: list[str]) -> ToolCall:
        """Helper to create a tool call with HPO results (legacy dict format)."""
        matches = [
            {"hpo_id": hpo_id, "term_name": f"Term {hpo_id}"} for hpo_id in hpo_ids
        ]
        return ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result={
                "results": [
                    {
                        "chunk_text": "test chunk",
                        "matches": matches,
                    }
                ]
            },
        )

    def test_filter_removes_hallucinated_ids(self, strategy):
        """Test that HPO IDs not in tool results are removed."""
        # Tool returned these candidates
        tool_calls = [
            self._create_tool_call_with_results(
                ["HP:0001249", "HP:0012758", "HP:0001252"]
            )
        ]

        # LLM returned these annotations (HP:0001324 is hallucinated)
        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0001324", "Muscle weakness"),  # HALLUCINATED
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 1
        assert filtered[0].hpo_id == "HP:0001249"
        # HP:0001324 should be removed

    def test_filter_keeps_all_when_all_valid(self, strategy):
        """Test that all annotations are kept when all are in tool results."""
        # Tool returned these candidates
        tool_calls = [
            self._create_tool_call_with_results(
                ["HP:0001249", "HP:0001252", "HP:0012758"]
            )
        ]

        # LLM returned these annotations (all valid)
        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0001252", "Motor delay"),
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 2
        assert {a.hpo_id for a in filtered} == {"HP:0001249", "HP:0001252"}

    def test_filter_handles_empty_tool_results(self, strategy):
        """Test behavior when tool returns no HPO matches."""
        # Tool returned no matches
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result={"results": []},
        )

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
        ]

        # Should return all annotations (no filtering when no candidates)
        filtered = strategy._filter_against_candidates(annotations, [tool_call])
        assert len(filtered) == 1

    def test_filter_handles_no_tool_calls(self, strategy):
        """Test behavior when no tool calls were made."""
        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
        ]

        # No tool calls - should return all annotations
        filtered = strategy._filter_against_candidates(annotations, [])
        assert len(filtered) == 1

    def test_filter_handles_non_process_tool_calls(self, strategy):
        """Test that non-process_clinical_text tool calls are ignored."""
        # Tool call with different name
        tool_call = ToolCall(
            name="search_hpo_terms",  # Different tool
            arguments={"query": "test"},
            result={"matches": [{"hpo_id": "HP:0001249"}]},
        )

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0001252", "Motor delay"),
        ]

        # Should return all annotations (no filtering for other tools)
        filtered = strategy._filter_against_candidates(annotations, [tool_call])
        assert len(filtered) == 2

    def test_filter_logs_hallucination_warning(self, strategy, caplog):
        """Test that hallucination detection is logged as warning."""
        tool_calls = [self._create_tool_call_with_results(["HP:0001249"])]

        annotations = [
            self._create_annotation("HP:0001249", "Valid term"),
            self._create_annotation("HP:0001324", "Hallucinated term"),
        ]

        with caplog.at_level(logging.WARNING, logger="phentrieve.llm.annotation"):
            strategy._filter_against_candidates(annotations, tool_calls)

        # Check that hallucination warning was logged
        assert any(
            "HALLUCINATION DETECTED" in record.message for record in caplog.records
        )
        assert any("HP:0001324" in record.message for record in caplog.records)

    def test_filter_logs_info_summary(self, strategy, caplog):
        """Test that filtering summary is logged at INFO level."""
        tool_calls = [self._create_tool_call_with_results(["HP:0001249"])]

        annotations = [
            self._create_annotation("HP:0001249", "Valid term"),
            self._create_annotation("HP:0001324", "Hallucinated term"),
        ]

        with caplog.at_level(logging.INFO, logger="phentrieve.llm.annotation"):
            strategy._filter_against_candidates(annotations, tool_calls)

        # Check that summary was logged
        assert any(
            "1/2 annotations kept" in record.message for record in caplog.records
        )

    def test_filter_handles_multiple_results_list_format(self, strategy):
        """Test filtering with multiple HPO results in list format (actual format)."""
        # Tool result as list - what _process_clinical_text() actually returns
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result=[
                {"hpo_id": "HP:0001249", "term_name": "Intellectual disability"},
                {"hpo_id": "HP:0001252", "term_name": "Motor delay"},
                {"hpo_id": "HP:0012758", "term_name": "Another term"},
            ],
        )

        annotations = [
            self._create_annotation("HP:0001249"),  # Valid
            self._create_annotation("HP:0012758"),  # Valid
            self._create_annotation("HP:0099999"),  # Hallucinated
        ]

        filtered = strategy._filter_against_candidates(annotations, [tool_call])

        assert len(filtered) == 2
        assert {a.hpo_id for a in filtered} == {"HP:0001249", "HP:0012758"}

    def test_filter_handles_legacy_dict_format(self, strategy):
        """Test filtering with legacy dict format (backwards compatibility)."""
        # Tool result with nested dict structure
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result={
                "results": [
                    {
                        "chunk_text": "first chunk",
                        "matches": [{"hpo_id": "HP:0001249"}],
                    },
                    {
                        "chunk_text": "second chunk",
                        "matches": [{"hpo_id": "HP:0001252"}, {"hpo_id": "HP:0012758"}],
                    },
                ]
            },
        )

        annotations = [
            self._create_annotation("HP:0001249"),  # From chunk 1
            self._create_annotation("HP:0012758"),  # From chunk 2
            self._create_annotation("HP:0099999"),  # Hallucinated
        ]

        filtered = strategy._filter_against_candidates(annotations, [tool_call])

        assert len(filtered) == 2
        assert {a.hpo_id for a in filtered} == {"HP:0001249", "HP:0012758"}

    def test_filter_handles_malformed_tool_result(self, strategy):
        """Test graceful handling of malformed tool results."""
        # Tool result with unexpected structure
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result={"unexpected": "structure"},
        )

        annotations = [
            self._create_annotation("HP:0001249"),
        ]

        # Should return all annotations (no valid candidates to filter against)
        filtered = strategy._filter_against_candidates(annotations, [tool_call])
        assert len(filtered) == 1

    def test_filter_handles_none_result(self, strategy):
        """Test handling of None tool result."""
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result=None,  # type: ignore[arg-type]
        )

        annotations = [
            self._create_annotation("HP:0001249"),
        ]

        filtered = strategy._filter_against_candidates(annotations, [tool_call])
        assert len(filtered) == 1
