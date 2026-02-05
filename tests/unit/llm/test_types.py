"""Unit tests for LLM annotation types."""

from datetime import datetime

import pytest

from phentrieve.llm.types import (
    AnnotationMode,
    AnnotationResult,
    AssertionStatus,
    HPOAnnotation,
    LLMResponse,
    PostProcessingStep,
    ToolCall,
)


class TestAnnotationMode:
    """Tests for AnnotationMode enum."""

    def test_mode_values(self):
        """Test that all modes have correct string values."""
        assert AnnotationMode.DIRECT.value == "direct"
        assert AnnotationMode.TOOL_TERM.value == "tool_term"
        assert AnnotationMode.TOOL_TEXT.value == "tool_text"

    def test_mode_from_string(self):
        """Test creating mode from string value."""
        assert AnnotationMode("direct") == AnnotationMode.DIRECT
        assert AnnotationMode("tool_term") == AnnotationMode.TOOL_TERM
        assert AnnotationMode("tool_text") == AnnotationMode.TOOL_TEXT

    def test_invalid_mode_raises(self):
        """Test that invalid mode string raises ValueError."""
        with pytest.raises(ValueError):
            AnnotationMode("invalid_mode")


class TestAssertionStatus:
    """Tests for AssertionStatus enum."""

    def test_status_values(self):
        """Test that all statuses have correct string values."""
        assert AssertionStatus.AFFIRMED.value == "affirmed"
        assert AssertionStatus.NEGATED.value == "negated"
        assert AssertionStatus.UNCERTAIN.value == "uncertain"


class TestPostProcessingStep:
    """Tests for PostProcessingStep enum."""

    def test_step_values(self):
        """Test that all steps have correct string values."""
        assert PostProcessingStep.VALIDATION.value == "validation"
        assert PostProcessingStep.REFINEMENT.value == "refinement"
        assert PostProcessingStep.ASSERTION_REVIEW.value == "assertion_review"
        assert PostProcessingStep.CONSISTENCY.value == "consistency"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(
            name="query_hpo_terms",
            arguments={"query": "seizures", "num_results": 5},
        )
        assert tc.name == "query_hpo_terms"
        assert tc.arguments == {"query": "seizures", "num_results": 5}
        assert tc.result is None
        assert isinstance(tc.timestamp, datetime)

    def test_tool_call_with_result(self):
        """Test creating a tool call with result."""
        result = [{"hpo_id": "HP:0001250", "term_name": "Seizure"}]
        tc = ToolCall(
            name="query_hpo_terms",
            arguments={"query": "seizures"},
            result=result,
        )
        assert tc.result == result

    def test_tool_call_immutable(self):
        """Test that ToolCall is frozen (immutable)."""
        tc = ToolCall(name="test", arguments={})
        with pytest.raises(AttributeError):
            tc.name = "changed"


class TestHPOAnnotation:
    """Tests for HPOAnnotation dataclass."""

    def test_basic_annotation(self):
        """Test creating a basic annotation."""
        ann = HPOAnnotation(
            hpo_id="HP:0001250",
            term_name="Seizure",
        )
        assert ann.hpo_id == "HP:0001250"
        assert ann.term_name == "Seizure"
        assert ann.assertion == AssertionStatus.AFFIRMED
        assert ann.confidence == 1.0

    def test_annotation_with_all_fields(self):
        """Test creating annotation with all fields."""
        ann = HPOAnnotation(
            hpo_id="HP:0001250",
            term_name="Seizure",
            assertion=AssertionStatus.NEGATED,
            confidence=0.85,
            evidence_text="no seizures",
            evidence_start=10,
            evidence_end=21,
            definition="A seizure is...",
            synonyms=["Convulsion", "Fit"],
            source_mode=AnnotationMode.TOOL_TEXT,
            raw_score=0.92,
        )
        assert ann.assertion == AssertionStatus.NEGATED
        assert ann.confidence == 0.85
        assert ann.evidence_text == "no seizures"
        assert ann.definition == "A seizure is..."
        assert ann.synonyms == ["Convulsion", "Fit"]
        assert ann.source_mode == AnnotationMode.TOOL_TEXT

    def test_annotation_to_dict(self):
        """Test converting annotation to dictionary."""
        ann = HPOAnnotation(
            hpo_id="HP:0001250",
            term_name="Seizure",
            assertion=AssertionStatus.AFFIRMED,
            confidence=0.95,
        )
        d = ann.to_dict()
        assert d["hpo_id"] == "HP:0001250"
        assert d["term_name"] == "Seizure"
        assert d["assertion"] == "affirmed"
        assert d["confidence"] == 0.95


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_basic_response(self):
        """Test creating a basic LLM response."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4o",
            provider="openai",
        )
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.tool_calls == []

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tc = ToolCall(name="test", arguments={})
        response = LLMResponse(
            content=None,
            model="gpt-4o",
            provider="openai",
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        assert response.content is None
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1


class TestAnnotationResult:
    """Tests for AnnotationResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic result."""
        result = AnnotationResult(
            annotations=[],
            input_text="Patient has seizures.",
            language="en",
            mode=AnnotationMode.DIRECT,
            model="github/gpt-4o",
        )
        assert result.input_text == "Patient has seizures."
        assert result.language == "en"
        assert result.mode == AnnotationMode.DIRECT
        assert result.model == "github/gpt-4o"
        assert result.prompt_version == "v1.0.0"

    def test_result_with_annotations(self):
        """Test result with annotations."""
        annotations = [
            HPOAnnotation(
                hpo_id="HP:0001250",
                term_name="Seizure",
                assertion=AssertionStatus.AFFIRMED,
            ),
            HPOAnnotation(
                hpo_id="HP:0001627",
                term_name="Cardiac abnormality",
                assertion=AssertionStatus.NEGATED,
            ),
        ]
        result = AnnotationResult(
            annotations=annotations,
            input_text="Seizures present, no cardiac issues.",
            language="en",
            mode=AnnotationMode.TOOL_TEXT,
            model="github/gpt-4o",
        )
        assert len(result.annotations) == 2
        assert len(result.affirmed_annotations) == 1
        assert len(result.negated_annotations) == 1
        assert result.hpo_ids == ["HP:0001250"]

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = AnnotationResult(
            annotations=[
                HPOAnnotation(hpo_id="HP:0001250", term_name="Seizure"),
            ],
            input_text="Test text",
            language="en",
            mode=AnnotationMode.DIRECT,
            model="github/gpt-4o",
        )
        d = result.to_dict()
        assert d["input_text"] == "Test text"
        assert d["language"] == "en"
        assert d["mode"] == "direct"
        assert d["model"] == "github/gpt-4o"
        assert len(d["annotations"]) == 1
        assert d["annotations"][0]["hpo_id"] == "HP:0001250"
