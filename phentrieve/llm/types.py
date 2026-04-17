from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from phentrieve.llm.config import (
    DEFAULT_LLM_LANGUAGE,
    DEFAULT_LLM_MODE,
    DEFAULT_PROVIDER_TEMPERATURE,
)


class AnnotationMode(StrEnum):
    DIRECT = "direct"
    TOOL_TERM = "tool_term"
    TOOL_TEXT = "tool_text"
    TWO_PHASE = "two_phase"


class AssertionStatus(StrEnum):
    PRESENT = "present"
    AFFIRMED = PRESENT
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


class PostProcessingStep(StrEnum):
    VALIDATION = "validation"
    REFINEMENT = "refinement"
    ASSERTION_REVIEW = "assertion_review"
    CONSISTENCY = "consistency"
    COMBINED = "combined"


class LLMPhenotype(BaseModel):
    term_id: str
    label: str
    evidence: str | None = None
    assertion: str = AssertionStatus.PRESENT.value
    category: str | None = None
    evidence_records: list[LLMPhenotypeEvidence] = Field(default_factory=list)


class LLMExtractedPhenotype(BaseModel):
    phrase: str = Field(
        description=(
            "A concise phenotype phrase copied from the source text. "
            "Use short noun phrases rather than full sentences."
        ),
    )
    category: Literal["Abnormal", "Normal", "Suspected", "Family_History", "Other"] = (
        Field(
            description=(
                "Classification of the extracted phrase as patient-present, negated, "
                "uncertain, family history, or non-phenotype metadata."
            )
        )
    )


class LLMExtractedPhenotypes(BaseModel):
    phenotypes: list[LLMExtractedPhenotype] = Field(
        default_factory=list,
        description="Distinct phenotype phrases extracted from the clinical text.",
    )


class LLMGroundedExtractedPhenotype(BaseModel):
    phrase: str = Field(...)
    category: Literal["Abnormal", "Normal", "Suspected", "Family_History", "Other"]
    chunk_ids: list[int] = Field(min_length=1)
    evidence_text: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class LLMGroundedExtractedPhenotypes(BaseModel):
    phenotypes: list[LLMGroundedExtractedPhenotype] = Field(default_factory=list)


class LLMPhenotypeEvidence(BaseModel):
    phrase: str
    evidence_text: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)
    start_char: int | None = None
    end_char: int | None = None
    match_method: str = "unknown"


class LLMMappingSelection(BaseModel):
    phrase: str
    hpo_id: str | None = None


class LLMBatchMappingSelection(BaseModel):
    item_id: str
    hpo_id: str | None = None


class LLMBatchMappingSelections(BaseModel):
    mappings: list[LLMBatchMappingSelection] = Field(default_factory=list)


class LLMMeta(BaseModel):
    llm_provider: str = "gemini"
    llm_model: str
    llm_mode: str
    prompt_version: str = "v1"
    token_input: int | None = None
    token_output: int | None = None
    token_count_source: Literal["exact", "estimated"] | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    request_count: int = 0
    phase_timings: dict[str, float] = Field(default_factory=dict)
    phase_counts: dict[str, int] = Field(default_factory=dict)
    phase_request_counts: dict[str, int] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)


class LLMPipelineConfig(BaseModel):
    provider: str = "gemini"
    model: str
    base_url: str | None = None
    mode: str = DEFAULT_LLM_MODE
    language: str | None = DEFAULT_LLM_LANGUAGE
    seed: int | None = None


class LLMExtractionResult(BaseModel):
    terms: list[LLMPhenotype] = Field(default_factory=list)
    meta: LLMMeta


class LLMTermsResult(BaseModel):
    terms: list[LLMPhenotype] = Field(default_factory=list)


class LLMToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None


class LLMResponse(BaseModel):
    content: str | None = None
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    temperature: float = DEFAULT_PROVIDER_TEMPERATURE


@dataclass(frozen=True)
class GroundedChunk:
    chunk_id: int
    text: str
    start_char: int | None
    end_char: int | None
    status: str


@dataclass(frozen=True)
class ExtractionGroup:
    group_id: int
    chunk_ids: list[int] = field(default_factory=list)
    text: str = ""
    estimated_prompt_tokens: int = 0
