from __future__ import annotations

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


class LLMExtractedPhenotype(BaseModel):
    phrase: str = Field(
        description=(
            "A concise phenotype phrase copied from the source text. "
            "Use short noun phrases rather than full sentences."
        ),
        max_length=160,
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
        max_length=128,
    )


class LLMMeta(BaseModel):
    llm_model: str
    llm_mode: str
    prompt_version: str = "v1"
    token_input: int | None = None
    token_output: int | None = None


class LLMPipelineConfig(BaseModel):
    model: str
    mode: str = DEFAULT_LLM_MODE
    language: str | None = DEFAULT_LLM_LANGUAGE


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
