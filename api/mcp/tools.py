from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_NUM_RESULTS,
    MIN_SIMILARITY_THRESHOLD,
)


class ExtractHpoTermsRequest(BaseModel):
    text: str = Field(
        description=(
            "Clinical or biomedical research text to map to HPO term suggestions. "
            "Do not submit identifiable patient data to public demo instances."
        )
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code. Use null to let Phentrieve detect it.",
    )
    include_details: bool = Field(
        default=True,
        description="Include HPO definitions and synonyms.",
    )
    include_chunk_positions: bool = Field(
        default=True,
        description="Include source character offsets for evidence chunks.",
    )
    num_results_per_chunk: int = Field(
        default=DEFAULT_NUM_RESULTS,
        ge=1,
        le=50,
        description="Maximum HPO candidates per chunk.",
    )
    chunk_retrieval_threshold: float = Field(
        default=DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum chunk-level retrieval similarity.",
    )


class ExtractHpoTermsLlmRequest(ExtractHpoTermsRequest):
    llm_model: str = Field(
        description="LLM model name, optionally provider-prefixed.",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Provider name such as openai, anthropic, gemini, or ollama.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Optional provider base URL.",
    )
    llm_mode: Literal["two_phase"] = "two_phase"
    llm_internal_mode: Literal["whole_document_legacy", "whole_document_grounded"] = (
        "whole_document_grounded"
    )
    allow_standard_fallback: bool = Field(
        default=False,
        description="Fall back to standard extraction if production quota is exhausted.",
    )


class SearchHpoTermsRequest(BaseModel):
    text: str = Field(
        description=(
            "Phenotype phrase or short clinical or biomedical research snippet. "
            "Do not submit identifiable patient data to public demo instances."
        )
    )
    language: str | None = None
    num_results: int = Field(default=DEFAULT_NUM_RESULTS, ge=1, le=50)
    similarity_threshold: float = Field(
        default=MIN_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    include_details: bool = True


class CompareHpoTermsRequest(BaseModel):
    term1_id: str = Field(pattern=r"^HP:\d{7}$")
    term2_id: str = Field(pattern=r"^HP:\d{7}$")
    formula: Literal["hybrid", "simple_resnik_like"] = "hybrid"
