"""Unit tests for hallucination filtering in tool-guided annotation."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    AssertionStatus,
    HPOAnnotation,
    TokenUsage,
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

    @pytest.fixture
    def term_strategy(self, mock_provider):
        """Create a ToolGuidedStrategy with TOOL_TERM mode."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        return ToolGuidedStrategy(mock_provider, mode=AnnotationMode.TOOL_TERM)

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
        """Helper to create a process_clinical_text tool call with results."""
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

    def _create_query_tool_call(self, hpo_ids: list[str]) -> ToolCall:
        """Helper to create a query_hpo_terms tool call with results."""
        result = [
            {
                "hpo_id": hpo_id,
                "term_name": f"Term {hpo_id}",
                "score": 0.85,
            }
            for hpo_id in hpo_ids
        ]
        return ToolCall(
            name="query_hpo_terms",
            arguments={"query": "test phrase"},
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
        tool_calls = [
            self._create_tool_call_with_results(
                ["HP:0001249", "HP:0012758", "HP:0001252"]
            )
        ]

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0001324", "Muscle weakness"),  # HALLUCINATED
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 1
        assert filtered[0].hpo_id == "HP:0001249"

    def test_filter_keeps_all_when_all_valid(self, strategy):
        """Test that all annotations are kept when all are in tool results."""
        tool_calls = [
            self._create_tool_call_with_results(
                ["HP:0001249", "HP:0001252", "HP:0012758"]
            )
        ]

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0001252", "Motor delay"),
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 2
        assert {a.hpo_id for a in filtered} == {"HP:0001249", "HP:0001252"}

    def test_filter_handles_empty_tool_results(self, strategy):
        """Test behavior when tool returns no HPO matches."""
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result={"results": []},
        )

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
        ]

        filtered = strategy._filter_against_candidates(annotations, [tool_call])
        assert len(filtered) == 1

    def test_filter_handles_no_tool_calls(self, strategy):
        """Test behavior when no tool calls were made."""
        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
        ]

        filtered = strategy._filter_against_candidates(annotations, [])
        assert len(filtered) == 1

    def test_filter_handles_unrecognized_tool_calls(self, strategy):
        """Test that unrecognized tool names are ignored."""
        tool_call = ToolCall(
            name="some_other_tool",
            arguments={"query": "test"},
            result={"matches": [{"hpo_id": "HP:0001249"}]},
        )

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0001252", "Motor delay"),
        ]

        filtered = strategy._filter_against_candidates(annotations, [tool_call])
        assert len(filtered) == 2

    def test_filter_logs_hallucination_warning(self, strategy, caplog):
        """Test that hallucination detection is logged as warning."""
        tool_calls = [self._create_tool_call_with_results(["HP:0001249"])]

        annotations = [
            self._create_annotation("HP:0001249", "Valid term"),
            self._create_annotation("HP:0001324", "Hallucinated term"),
        ]

        with caplog.at_level(
            logging.WARNING, logger="phentrieve.llm.annotation.tool_guided"
        ):
            strategy._filter_against_candidates(annotations, tool_calls)

        assert any("[FILTER] Removing" in record.message for record in caplog.records)
        assert any("HP:0001324" in record.message for record in caplog.records)

    def test_filter_logs_info_summary(self, strategy, caplog):
        """Test that filtering summary is logged at INFO level."""
        tool_calls = [self._create_tool_call_with_results(["HP:0001249"])]

        annotations = [
            self._create_annotation("HP:0001249", "Valid term"),
            self._create_annotation("HP:0001324", "Hallucinated term"),
        ]

        with caplog.at_level(
            logging.INFO, logger="phentrieve.llm.annotation.tool_guided"
        ):
            strategy._filter_against_candidates(annotations, tool_calls)

        assert any(
            "kept 1 of 2 annotations" in record.message for record in caplog.records
        )

    def test_filter_handles_multiple_results_list_format(self, strategy):
        """Test filtering with multiple HPO results in list format."""
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
        tool_call = ToolCall(
            name="process_clinical_text",
            arguments={"text": "test", "language": "en"},
            result={"unexpected": "structure"},
        )

        annotations = [
            self._create_annotation("HP:0001249"),
        ]

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


class TestToolTermHallucinationFiltering:
    """Tests for hallucination filtering with query_hpo_terms tool calls (Mode 2)."""

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
        """Create a ToolGuidedStrategy with TOOL_TERM mode."""
        from phentrieve.llm.annotation.tool_guided import ToolGuidedStrategy

        return ToolGuidedStrategy(mock_provider, mode=AnnotationMode.TOOL_TERM)

    def _create_annotation(
        self, hpo_id: str, term_name: str = "Test Term"
    ) -> HPOAnnotation:
        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=term_name,
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.9,
            source_mode=AnnotationMode.TOOL_TERM,
        )

    def _create_query_tool_call(self, hpo_ids: list[str]) -> ToolCall:
        result = [
            {"hpo_id": hpo_id, "term_name": f"Term {hpo_id}", "score": 0.85}
            for hpo_id in hpo_ids
        ]
        return ToolCall(
            name="query_hpo_terms",
            arguments={"query": "test phrase"},
            result=result,
        )

    def test_filter_removes_ids_not_in_query_results(self, strategy):
        """Test that HPO IDs not returned by query_hpo_terms are removed."""
        tool_calls = [
            self._create_query_tool_call(["HP:0001249", "HP:0001252"]),
            self._create_query_tool_call(["HP:0012758"]),
        ]

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0099999", "Hallucinated term"),
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 1
        assert filtered[0].hpo_id == "HP:0001249"

    def test_filter_merges_candidates_from_multiple_queries(self, strategy):
        """Test that candidates from all query_hpo_terms calls are merged."""
        tool_calls = [
            self._create_query_tool_call(["HP:0001249"]),
            self._create_query_tool_call(["HP:0001252"]),
            self._create_query_tool_call(["HP:0012758"]),
        ]

        annotations = [
            self._create_annotation("HP:0001249"),
            self._create_annotation("HP:0001252"),
            self._create_annotation("HP:0012758"),
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 3

    def test_filter_keeps_all_when_all_in_query_results(self, strategy):
        """Test that all annotations are kept when all match query results."""
        tool_calls = [
            self._create_query_tool_call(["HP:0001249", "HP:0001252", "HP:0012758"]),
        ]

        annotations = [
            self._create_annotation("HP:0001249"),
            self._create_annotation("HP:0001252"),
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 2

    def test_filter_handles_mixed_tool_calls(self, strategy):
        """Test filtering with both query_hpo_terms and process_clinical_text."""
        tool_calls = [
            self._create_query_tool_call(["HP:0001249"]),
            ToolCall(
                name="process_clinical_text",
                arguments={"text": "test", "language": "en"},
                result=[{"hpo_id": "HP:0001252", "term_name": "Motor delay"}],
            ),
        ]

        annotations = [
            self._create_annotation("HP:0001249"),
            self._create_annotation("HP:0001252"),
            self._create_annotation("HP:0099999"),  # Hallucinated
        ]

        filtered = strategy._filter_against_candidates(annotations, tool_calls)

        assert len(filtered) == 2
        assert {a.hpo_id for a in filtered} == {"HP:0001249", "HP:0001252"}


class TestDirectModeHallucinationFiltering:
    """Tests for pipeline-level hallucination filtering in DIRECT mode."""

    def _create_annotation(
        self, hpo_id: str, term_name: str = "Test Term"
    ) -> HPOAnnotation:
        return HPOAnnotation(
            hpo_id=hpo_id,
            term_name=term_name,
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.9,
            source_mode=AnnotationMode.DIRECT,
        )

    def test_filter_removes_hallucinated_ids(self):
        """Test that pipeline filters DIRECT mode output against retrieval."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(validate_hpo_ids=False)

        annotations = [
            self._create_annotation("HP:0001249", "Intellectual disability"),
            self._create_annotation("HP:0099999", "Hallucinated term"),
        ]

        # Mock the ToolExecutor to return known candidates
        mock_executor = MagicMock()
        mock_executor.execute.return_value = [
            {"hpo_id": "HP:0001249", "term_name": "Intellectual disability"},
            {"hpo_id": "HP:0001252", "term_name": "Motor delay"},
        ]
        pipeline._tool_executor = mock_executor

        filtered = pipeline._filter_direct_against_retrieval(
            annotations, "test text", "en"
        )

        assert len(filtered) == 1
        assert filtered[0].hpo_id == "HP:0001249"
        mock_executor.execute.assert_called_once_with(
            "process_clinical_text",
            {"text": "test text", "language": "en"},
        )

    def test_filter_keeps_all_when_all_valid(self):
        """Test that all annotations are kept when all match retrieval."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(validate_hpo_ids=False)

        annotations = [
            self._create_annotation("HP:0001249"),
            self._create_annotation("HP:0001252"),
        ]

        mock_executor = MagicMock()
        mock_executor.execute.return_value = [
            {"hpo_id": "HP:0001249"},
            {"hpo_id": "HP:0001252"},
            {"hpo_id": "HP:0012758"},
        ]
        pipeline._tool_executor = mock_executor

        filtered = pipeline._filter_direct_against_retrieval(
            annotations, "test text", "en"
        )

        assert len(filtered) == 2

    def test_filter_passes_through_on_empty_candidates(self):
        """Test that annotations pass through when retrieval returns nothing."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(validate_hpo_ids=False)

        annotations = [self._create_annotation("HP:0001249")]

        mock_executor = MagicMock()
        mock_executor.execute.return_value = []
        pipeline._tool_executor = mock_executor

        filtered = pipeline._filter_direct_against_retrieval(
            annotations, "test text", "en"
        )

        assert len(filtered) == 1

    def test_filter_passes_through_on_executor_error(self):
        """Test that annotations pass through when executor raises."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(validate_hpo_ids=False)

        annotations = [self._create_annotation("HP:0001249")]

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = RuntimeError("Pipeline not available")
        pipeline._tool_executor = mock_executor

        filtered = pipeline._filter_direct_against_retrieval(
            annotations, "test text", "en"
        )

        # Should return unfiltered on error
        assert len(filtered) == 1

    def test_pipeline_run_applies_direct_filter(self):
        """Test that pipeline.run() applies hallucination filtering for DIRECT mode."""
        from phentrieve.llm.pipeline import LLMAnnotationPipeline

        pipeline = LLMAnnotationPipeline(validate_hpo_ids=False)

        # Mock the strategy to return annotations with one hallucinated ID
        mock_strategy = MagicMock()
        mock_strategy.annotate.return_value = AnnotationResult(
            annotations=[
                self._create_annotation("HP:0001249"),
                self._create_annotation("HP:0099999"),  # Hallucinated
            ],
            input_text="test text",
            language="en",
            mode=AnnotationMode.DIRECT,
            model="github/gpt-4o",
            token_usage=TokenUsage(),
        )
        pipeline._strategies[AnnotationMode.DIRECT] = mock_strategy

        # Mock retrieval to only know HP:0001249
        mock_executor = MagicMock()
        mock_executor.execute.return_value = [
            {"hpo_id": "HP:0001249", "term_name": "Intellectual disability"},
        ]
        pipeline._tool_executor = mock_executor

        with patch.object(pipeline, "_detect_language", return_value="en"):
            result = pipeline.run(
                text="test text",
                mode=AnnotationMode.DIRECT,
                language="en",
            )

        assert len(result.annotations) == 1
        assert result.annotations[0].hpo_id == "HP:0001249"
