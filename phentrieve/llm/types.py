from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from phentrieve.llm.config import (
    DEFAULT_LLM_LANGUAGE,
    DEFAULT_LLM_MODE,
    DEFAULT_PROVIDER_NAME,
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


Phase1Mode = Literal["ungrouped", "grouped_large", "grouped_small"]
Phase1FailureClass = (
    Literal[
        "structured_refusal",
        "provider_timeout",
        "structured_json_invalid",
        "structured_schema_validation_failed",
        "provider_transport_error",
        "provider_auth_error",
        "provider_config_error",
        "provider_execution_error",
    ]
    | None
)


class LLMPhenotype(BaseModel):
    term_id: str
    label: str
    evidence: str | None = None
    assertion: str = AssertionStatus.PRESENT.value
    # Who the phenotype belongs to, orthogonal to the present/absent assertion
    # (LLM-1). Kept distinct from ``assertion`` so a proband finding and a
    # relative's mention of the same HPO id never collapse into one term.
    experiencer: str = "proband"
    # The negated portion of an "X without Y" phrase (LLM-2): X is present and
    # only the qualifier Y is absent. None when the phrase is not partially
    # negated.
    negated_qualifier: str | None = None
    category: str | None = None
    confidence: float | None = None
    score: float | None = None
    evidence_records: list[LLMPhenotypeEvidence] = Field(default_factory=list)


# Orthogonal classification axes added to the Phase-1 extraction schemas (LLM-2),
# declared before the legacy ``category`` label so Gemini -- which emits keys in
# schema order -- reasons about experiencer and assertion first. ``category`` stays
# authoritative downstream; these axes scaffold the model's reasoning and
# ``negated_qualifier`` captures the negated portion of an "X without Y" phrase.
_EXPERIENCER_DESCRIPTION = (
    "Decide FIRST, independently of presence: who the phenotype belongs to. "
    "'family_history' for a relative, 'other' for non-clinical metadata, "
    "otherwise 'proband'."
)
_ASSERTION_DESCRIPTION = (
    "Whether the phenotype is present in the experiencer. A negation cue negates "
    "only the noun phrase it directly modifies: in 'X without Y', X is present and "
    "only Y is absent."
)
_NEGATED_QUALIFIER_DESCRIPTION = (
    "For an 'X without Y' phrase where X is present and only the qualifier Y is "
    "absent, the negated qualifier Y; otherwise null."
)


class LLMExtractedPhenotype(BaseModel):
    phrase: str = Field(
        description=(
            "A concise phenotype phrase copied from the source text. "
            "Use short noun phrases rather than full sentences."
        ),
    )
    experiencer: Literal["proband", "family_history", "other"] = Field(
        default="proband", description=_EXPERIENCER_DESCRIPTION
    )
    assertion: Literal["present", "absent", "uncertain"] = Field(
        default="present", description=_ASSERTION_DESCRIPTION
    )
    negated_qualifier: str | None = Field(
        default=None, description=_NEGATED_QUALIFIER_DESCRIPTION
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
    evidence_text: str | None = None
    experiencer: Literal["proband", "family_history", "other"] = Field(
        default="proband", description=_EXPERIENCER_DESCRIPTION
    )
    assertion: Literal["present", "absent", "uncertain"] = Field(
        default="present", description=_ASSERTION_DESCRIPTION
    )
    negated_qualifier: str | None = Field(
        default=None, description=_NEGATED_QUALIFIER_DESCRIPTION
    )
    category: Literal["Abnormal", "Normal", "Suspected", "Family_History", "Other"]
    chunk_ids: list[int] = Field(min_length=1)
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
    llm_provider: str = DEFAULT_PROVIDER_NAME
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
    provider: str = DEFAULT_PROVIDER_NAME
    model: str
    base_url: str | None = None
    mode: str = DEFAULT_LLM_MODE
    language: str | None = DEFAULT_LLM_LANGUAGE
    seed: int | None = None
    capture_phase1_debug: bool = False


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


# ``from __future__ import annotations`` makes every annotation a string, so
# LLMPhenotype.evidence_records (list[LLMPhenotypeEvidence]) is an unresolved
# forward reference at class-definition time -- LLMPhenotypeEvidence is declared
# later in this module. Pydantic would otherwise resolve it lazily on the first
# validation, and two threads validating concurrently (the grouped phase-1 path
# runs provider calls in parallel) can race that rebuild, raising a spurious
# "Input should be a valid instance of LLMPhenotypeEvidence". Resolve it eagerly
# at import so model validation is thread-safe.
LLMPhenotype.model_rebuild()
