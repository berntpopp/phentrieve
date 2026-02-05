"""
Pydantic schemas for LLM annotation API endpoints.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AnnotationModeEnum(str, Enum):
    """Annotation mode for LLM-based annotation."""

    DIRECT = "direct"
    TOOL_TERM = "tool_term"
    TOOL_TEXT = "tool_text"


class PostProcessingStepEnum(str, Enum):
    """Post-processing steps that can be applied after annotation."""

    VALIDATION = "validation"
    REFINEMENT = "refinement"
    ASSERTION_REVIEW = "assertion_review"


class AssertionStatusEnum(str, Enum):
    """Assertion status for an HPO annotation."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


class LLMAnnotationRequest(BaseModel):
    """Request schema for LLM annotation endpoint."""

    text_content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Clinical text to annotate.",
        json_schema_extra={
            "example": "Patient has recurrent seizures and mild intellectual disability."
        },
    )
    model: str = Field(
        default="github/gpt-4o",
        description="LLM model to use (e.g., 'github/gpt-4o', 'gemini/gemini-1.5-pro').",
        json_schema_extra={"example": "github/gpt-4o"},
    )
    mode: AnnotationModeEnum = Field(
        default=AnnotationModeEnum.TOOL_TEXT,
        description="Annotation mode: direct (LLM outputs HPO directly), tool_term (LLM queries for terms), tool_text (full pipeline).",
    )
    language: str = Field(
        default="auto",
        description="Language code (en, de, es, fr, nl) or 'auto' for detection.",
        json_schema_extra={"example": "en"},
    )
    postprocess: Optional[list[PostProcessingStepEnum]] = Field(
        default=None,
        description="Optional post-processing steps to apply.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic).",
    )
    validate_hpo_ids: bool = Field(
        default=True,
        description="Whether to validate HPO IDs against the database.",
    )
    include_details: bool = Field(
        default=False,
        description="Include HPO term definitions and synonyms.",
    )
    output_format: str = Field(
        default="json",
        description="Output format: 'json' or 'phenopacket'.",
        json_schema_extra={"example": "json"},
    )


class HPOAnnotationItem(BaseModel):
    """A single HPO annotation in the response."""

    hpo_id: str = Field(
        ...,
        description="HPO identifier (e.g., 'HP:0001250').",
        json_schema_extra={"example": "HP:0001250"},
    )
    term_name: str = Field(
        ...,
        description="HPO term name.",
        json_schema_extra={"example": "Seizure"},
    )
    assertion: AssertionStatusEnum = Field(
        default=AssertionStatusEnum.AFFIRMED,
        description="Whether the phenotype is present (affirmed), absent (negated), or uncertain.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0.",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description="Text span supporting this annotation.",
    )
    definition: Optional[str] = Field(
        default=None,
        description="HPO term definition (if include_details=true).",
    )
    synonyms: Optional[list[str]] = Field(
        default=None,
        description="Alternative names for this term (if include_details=true).",
    )


class LLMAnnotationResponse(BaseModel):
    """Response schema for LLM annotation endpoint."""

    annotations: list[HPOAnnotationItem] = Field(
        ...,
        description="List of HPO annotations extracted from the text.",
    )
    input_text: str = Field(
        ...,
        description="The original input text.",
    )
    language: str = Field(
        ...,
        description="Detected or specified language.",
    )
    model: str = Field(
        ...,
        description="LLM model used.",
    )
    mode: AnnotationModeEnum = Field(
        ...,
        description="Annotation mode used.",
    )
    prompt_version: str = Field(
        ...,
        description="Version of the prompt template used.",
    )
    post_processing_steps: list[PostProcessingStepEnum] = Field(
        default_factory=list,
        description="Post-processing steps that were applied.",
    )
    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Total processing time in seconds.",
    )
    phenopacket: Optional[dict] = Field(
        default=None,
        description="Phenopacket representation (if output_format='phenopacket').",
    )


class AvailableModelsResponse(BaseModel):
    """Response schema for available models endpoint."""

    models: dict[str, list[str]] = Field(
        ...,
        description="Available models organized by provider.",
    )
