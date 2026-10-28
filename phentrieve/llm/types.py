from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
    # When the finding applies: current | historical | unknown (``unknown``
    # when the phase-1 model did not state it).
    temporality: str = "unknown"
    # The negated portion of an "X without Y" phrase (LLM-2): X is present and
    # only the qualifier Y is absent. None when the phrase is not partially
    # negated.
    negated_qualifier: str | None = None
    # For a GENERATED excluded finding derived from a source finding's
    # ``negated_qualifier`` (B3): the surface text Y that was retrieved and
    # mapped. None on ordinary (non-derived) findings.
    qualifier_surface_text: str | None = None
    # Provenance of how the term was resolved. Set to
    # ``"negated_qualifier_derived"`` on a B3-generated excluded finding so
    # downstream consumers can distinguish it from a directly-extracted term.
    match_method: str | None = None
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
    "'family_history' for a relative of the patient, 'other' for another person "
    "who is not a relative (e.g. a neighbour or donor), otherwise 'proband'."
)
_ASSERTION_DESCRIPTION = (
    "Whether the phenotype is present in the experiencer. A negation cue applies "
    "to every finding of the coordinated list it introduces, up to a contrast "
    "word, a new clause or the end of the sentence; in 'X without Y', X is "
    "present and only Y is absent."
)
_TEMPORALITY_DESCRIPTION = (
    "When the phenotype applies: 'current' (now or in the reported episode), "
    "'historical' (only in the past and no longer present), or 'unknown'."
)
_TEMPORALITY_VALUES = frozenset({"current", "historical", "unknown"})


def _coerce_temporality(value: Any) -> str:
    """Map a value outside the temporality contract to ``unknown``.

    One stray value (e.g. "past") from a provider without schema enforcement
    must not fail validation of the whole phase-1 response.
    """
    normalized = str(value).strip().lower() if isinstance(value, str) else ""
    return normalized if normalized in _TEMPORALITY_VALUES else "unknown"


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
    temporality: Literal["current", "historical", "unknown"] = Field(
        default="unknown", description=_TEMPORALITY_DESCRIPTION
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

    _normalize_temporality = field_validator("temporality", mode="before")(
        _coerce_temporality
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
    temporality: Literal["current", "historical", "unknown"] = Field(
        default="unknown", description=_TEMPORALITY_DESCRIPTION
    )
    negated_qualifier: str | None = Field(
        default=None, description=_NEGATED_QUALIFIER_DESCRIPTION
    )
    category: Literal["Abnormal", "Normal", "Suspected", "Family_History", "Other"]
    # maxItems goes to the provider schema only (not validated): constrained
    # decoding then cannot loop forever in "chunk_ids": [11, 11, 11, ...].
    chunk_ids: list[int] = Field(min_length=1, json_schema_extra={"maxItems": 8})
    # No character offsets: the pipeline locates evidence_text in the cited
    # chunks (phentrieve.llm.evidence_offsets).

    _normalize_temporality = field_validator("temporality", mode="before")(
        _coerce_temporality
    )


class LLMGroundedExtractedPhenotypes(BaseModel):
    phenotypes: list[LLMGroundedExtractedPhenotype] = Field(default_factory=list)


class LLMPhenotypeEvidence(BaseModel):
    phrase: str
    evidence_text: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)
    # Character offsets in the source document, computed in code from the
    # cited chunks (see ``phentrieve.llm.evidence_offsets``); ``offset_basis``
    # records how: evidence_in_chunk | evidence_in_chunk_hull | chunk_span |
    # unresolved.
    start_char: int | None = None
    end_char: int | None = None
    offset_basis: str | None = None
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
    family_history_findings: list[LLMPhenotype] = Field(default_factory=list)
    # Findings of another person who is not a relative (experiencer ``other``).
    other_person_findings: list[LLMPhenotype] = Field(default_factory=list)
    meta: LLMMeta


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
    # Phase-1 rendering of the chunk including source text that cleanup dropped
    # before it (e.g. a leading "no"); None when nothing was dropped.
    prompt_text: str | None = None


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
