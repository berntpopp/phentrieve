"""
Type definitions for the LLM annotation system.

This module defines the core data types used throughout the LLM annotation
system, including annotation results, tool calls, and configuration enums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AnnotationMode(str, Enum):
    """
    Annotation mode determining how the LLM produces HPO annotations.

    Attributes:
        DIRECT: LLM outputs HPO IDs directly from its training knowledge.
            Single API call, fastest, but may hallucinate non-existent terms.
        TOOL_TERM: LLM extracts clinical phrases, queries Phentrieve's
            query_hpo_terms() tool, then selects best matches.
        TOOL_TEXT: LLM sends text sections to Phentrieve's process_clinical_text()
            tool which runs full pipeline (chunking, assertion detection, retrieval),
            then LLM validates and selects from candidates.
    """

    DIRECT = "direct"
    TOOL_TERM = "tool_term"
    TOOL_TEXT = "tool_text"


class AssertionStatus(str, Enum):
    """
    Assertion status for an HPO annotation.

    Indicates whether the phenotype is present (affirmed), absent (negated),
    or has uncertain status.
    """

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


class PostProcessingStep(str, Enum):
    """
    Optional post-processing steps to validate and refine annotations.

    Attributes:
        VALIDATION: Re-check annotations against original text to remove
            false positives and adjust confidence scores.
        REFINEMENT: Check if more specific HPO terms should be used
            (e.g., upgrade "Seizure" to "Recurrent seizure" if supported).
        ASSERTION_REVIEW: Validate negation detection accuracy.
        CONSISTENCY: Cross-reference annotations for contradictions.
    """

    VALIDATION = "validation"
    REFINEMENT = "refinement"
    ASSERTION_REVIEW = "assertion_review"
    CONSISTENCY = "consistency"
    COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class TimingEvent:
    """A single timed operation within the annotation pipeline.

    Attributes:
        label: Short description of the operation (e.g., "LLM call #1", "tool: process_clinical_text").
        duration_seconds: Wall-clock time in seconds.
        category: "llm", "tool", or "postprocess".
    """

    label: str
    duration_seconds: float
    category: str  # "llm" | "tool" | "postprocess"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "label": self.label,
            "duration_seconds": round(self.duration_seconds, 3),
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class PostProcessingStats:
    """Statistics from a single post-processing step.

    Attributes:
        step: Which post-processing step produced these stats.
        annotations_in: Number of annotations before this step.
        annotations_out: Number of annotations after this step.
        removed: Number of annotations removed.
        added: Number of annotations added (e.g., corrected terms).
        assertions_changed: Number of assertion status corrections.
        terms_refined: Number of terms upgraded to more specific HPO IDs.
        details: Optional human-readable detail strings.
    """

    step: str
    annotations_in: int = 0
    annotations_out: int = 0
    removed: int = 0
    added: int = 0
    assertions_changed: int = 0
    terms_refined: int = 0
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "annotations_in": self.annotations_in,
            "annotations_out": self.annotations_out,
            "removed": self.removed,
            "added": self.added,
            "assertions_changed": self.assertions_changed,
            "terms_refined": self.terms_refined,
            "details": list(self.details),
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    Represents a tool call made by the LLM during annotation.

    Attributes:
        name: The name of the tool called (e.g., "query_hpo_terms").
        arguments: The arguments passed to the tool as a dictionary.
        result: The result returned by the tool (if available).
        timestamp: When the tool call was made.
    """

    name: str
    arguments: dict[str, Any]
    result: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class TokenUsage:
    """
    Token usage statistics for LLM API calls.

    Tracks cumulative token usage across multiple API calls within a single
    annotation operation (including tool call iterations and post-processing).

    Attributes:
        prompt_tokens: Total input/prompt tokens used.
        completion_tokens: Total output/completion tokens generated.
        total_tokens: Total tokens (prompt + completion).
        api_calls: Number of API calls made.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    llm_time_seconds: float = 0.0
    tool_time_seconds: float = 0.0
    timing_events: list[TimingEvent] = field(default_factory=list)

    def add(self, usage: dict[str, int]) -> None:
        """
        Add token usage from an LLM response.

        Args:
            usage: Token usage dict from LLMResponse.usage.
        """
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.api_calls += 1

    def merge(self, other: TokenUsage) -> None:
        """
        Merge token usage from another TokenUsage instance.

        Args:
            other: Another TokenUsage instance to merge.
        """
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.api_calls += other.api_calls
        self.llm_time_seconds += other.llm_time_seconds
        self.tool_time_seconds += other.tool_time_seconds
        self.timing_events.extend(other.timing_events)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "llm_time_seconds": round(self.llm_time_seconds, 3),
            "tool_time_seconds": round(self.tool_time_seconds, 3),
            "timing_events": [e.to_dict() for e in self.timing_events],
        }

    @classmethod
    def from_response(cls, usage: dict[str, int], llm_time: float = 0.0) -> TokenUsage:
        """
        Create TokenUsage from an LLM response usage dict.

        Args:
            usage: Token usage dict from LLMResponse.usage.
            llm_time: Time in seconds spent on the LLM API call.

        Returns:
            New TokenUsage instance.
        """
        return cls(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            api_calls=1,
            llm_time_seconds=llm_time,
        )


@dataclass(slots=True)
class HPOAnnotation:
    """
    A single HPO term annotation extracted from clinical text.

    Attributes:
        hpo_id: The HPO identifier (e.g., "HP:0001250").
        term_name: The HPO term name (e.g., "Seizure").
        assertion: Whether the phenotype is affirmed, negated, or uncertain.
        confidence: Confidence score from 0.0 to 1.0.
        evidence_text: The text span supporting this annotation.
        evidence_start: Character offset where evidence starts in original text.
        evidence_end: Character offset where evidence ends in original text.
        definition: HPO term definition (if include_details enabled).
        synonyms: Alternative names for this term (if include_details enabled).
        source_mode: Which annotation mode produced this annotation.
        raw_score: Original retrieval/similarity score before normalization.
    """

    hpo_id: str
    term_name: str
    assertion: AssertionStatus = AssertionStatus.AFFIRMED
    confidence: float = 1.0
    evidence_text: str | None = None
    evidence_start: int | None = None
    evidence_end: int | None = None
    definition: str | None = None
    synonyms: list[str] = field(default_factory=list)
    source_mode: AnnotationMode | None = None
    raw_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hpo_id": self.hpo_id,
            "term_name": self.term_name,
            "assertion": self.assertion.value,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "evidence_start": self.evidence_start,
            "evidence_end": self.evidence_end,
            "definition": self.definition,
            "synonyms": self.synonyms,
            "source_mode": self.source_mode.value if self.source_mode else None,
            "raw_score": self.raw_score,
        }


@dataclass(slots=True)
class LLMResponse:
    """
    Raw response from an LLM provider.

    Attributes:
        content: The text content of the response.
        model: The model that generated this response.
        provider: The provider used (e.g., "github", "gemini").
        finish_reason: Why the response ended (e.g., "stop", "tool_calls").
        tool_calls: List of tool calls requested by the model.
        usage: Token usage statistics.
        raw_response: The complete raw response from the provider.
    """

    content: str | None
    model: str
    provider: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnnotationResult:
    """
    Complete result of an LLM annotation operation.

    This is the primary output type from the annotation pipeline, containing
    all extracted HPO annotations along with full provenance information for
    reproducibility.

    Attributes:
        annotations: List of HPO annotations extracted from the text.
        input_text: The original clinical text that was annotated.
        language: Detected or specified language of the input.
        mode: The annotation mode used.
        model: The LLM model used (e.g., "github/gpt-4o").
        prompt_version: Version of the prompt template used.
        temperature: Temperature setting used for generation.
        tool_calls: Complete log of all tool calls made.
        post_processing_steps: Which post-processing steps were applied.
        raw_llm_response: The raw response from the LLM for debugging.
        processing_time_seconds: Total time taken for annotation.
        token_usage: Cumulative token usage across all API calls.
        timestamp: When the annotation was performed.
        error: Error message if annotation failed.
    """

    annotations: list[HPOAnnotation]
    input_text: str
    language: str
    mode: AnnotationMode
    model: str
    prompt_version: str = "v1.0.0"
    temperature: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    post_processing_steps: list[PostProcessingStep] = field(default_factory=list)
    post_processing_stats: list[PostProcessingStats] = field(default_factory=list)
    raw_llm_response: str | None = None
    processing_time_seconds: float | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None

    @property
    def affirmed_annotations(self) -> list[HPOAnnotation]:
        """Get only affirmed (present) phenotype annotations."""
        return [a for a in self.annotations if a.assertion == AssertionStatus.AFFIRMED]

    @property
    def negated_annotations(self) -> list[HPOAnnotation]:
        """Get only negated (absent) phenotype annotations."""
        return [a for a in self.annotations if a.assertion == AssertionStatus.NEGATED]

    @property
    def hpo_ids(self) -> list[str]:
        """Get list of all HPO IDs (affirmed only by default)."""
        return [a.hpo_id for a in self.affirmed_annotations]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "annotations": [a.to_dict() for a in self.annotations],
            "input_text": self.input_text,
            "language": self.language,
            "mode": self.mode.value,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "timestamp": tc.timestamp.isoformat(),
                }
                for tc in self.tool_calls
            ],
            "post_processing_steps": [s.value for s in self.post_processing_steps],
            "post_processing_stats": [s.to_dict() for s in self.post_processing_stats],
            "raw_llm_response": self.raw_llm_response,
            "processing_time_seconds": self.processing_time_seconds,
            "token_usage": self.token_usage.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }
