from typing import Any, cast

from pydantic import BaseModel, Field

from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)


class TextProcessingRequest(BaseModel):
    text_content: str = Field(..., description="The raw clinical text to process.")
    language: str | None = Field(
        default=DEFAULT_LANGUAGE,
        description="ISO 639-1 language code of the text (e.g., 'en', 'de'). If None, language detection might be attempted.",
    )

    # Chunking Configuration
    chunking_strategy: str = Field(
        default=DEFAULT_CHUNKING_STRATEGY,
        description="Predefined chunking strategy (e.g., 'simple', 'semantic', 'detailed', 'sliding_window_cleaned', 'sliding_window_punct_cleaned', 'sliding_window_punct_conj_cleaned'). See Phentrieve documentation for details.",
        json_schema_extra={"example": "sliding_window_punct_conj_cleaned"},
    )

    # Sliding window chunking parameters
    window_size: int | None = Field(
        default=DEFAULT_WINDOW_SIZE_TOKENS,
        ge=1,
        description="Sliding window size in tokens.",
    )
    step_size: int | None = Field(
        default=DEFAULT_STEP_SIZE_TOKENS,
        ge=1,
        description="Sliding window step size in tokens.",
    )
    split_threshold: float | None = Field(
        default=DEFAULT_SPLITTING_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for splitting (0-1).",
    )
    min_segment_length: int | None = Field(
        default=DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
        ge=1,
        description="Minimum segment length in words for sliding window.",
    )

    # Model Configuration
    semantic_model_name: str | None = Field(
        default=DEFAULT_MODEL,  # Will be handled in router logic to default to retrieval_model_name
        description=f"Model for semantic-based chunking strategies. If None, defaults to retrieval_model_name or Phentrieve's DEFAULT_MODEL ('{DEFAULT_MODEL}').",
    )
    retrieval_model_name: str | None = Field(
        default=DEFAULT_MODEL,
        description=f"Embedding model for HPO term retrieval (default: '{DEFAULT_MODEL}').",
    )
    trust_remote_code: bool | None = Field(
        default=False,
        description="Trust remote code when loading models from Hugging Face Hub (use with caution).",
    )

    # Retrieval & Reranking Parameters
    chunk_retrieval_threshold: float | None = Field(
        default=DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for HPO matches per chunk.",
    )
    num_results_per_chunk: int | None = Field(
        default=10, ge=1, description="Max HPO terms to consider from each chunk."
    )
    enable_reranker: bool | None = Field(
        default=False,
        deprecated=True,
        description="Deprecated: reranking is not currently supported by the text-processing endpoint and this parameter is ignored.",
    )
    reranker_model_name: str | None = Field(
        default=DEFAULT_RERANKER_MODEL,
        deprecated=True,
        description="Deprecated: reranking is not currently supported by the text-processing endpoint and this parameter is ignored.",
    )
    rerank_count_per_chunk: int | None = Field(
        default=50,
        ge=1,
        deprecated=True,
        description="Deprecated: reranking is not currently supported by the text-processing endpoint and this parameter is ignored.",
    )

    # Assertion Detection
    no_assertion_detection: bool | None = Field(
        default=False, description="Disable assertion detection."
    )
    assertion_preference: str | None = Field(
        default=cast(str, DEFAULT_ASSERTION_CONFIG.get("preference", "dependency")),
        description="Assertion detection preference ('dependency' or 'keyword').",
    )

    # Aggregation
    aggregated_term_confidence: float | None = Field(
        default=DEFAULT_MIN_CONFIDENCE_AGGREGATED,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for an aggregated HPO term.",
        json_schema_extra={"example": 0.75},
    )
    top_term_per_chunk_for_aggregation: bool | None = Field(
        default=False,
        description="Consider only the top HPO term from each chunk during final aggregation.",
    )

    # HPO Term Details
    include_details: bool | None = Field(
        default=True,
        description="Include HPO term definitions and synonyms in the response.",
    )

    # Position Tracking
    include_chunk_positions: bool = Field(
        default=False,
        description="Include character positions (start_char, end_char) for each chunk.",
    )


class HPOMatchInChunkAPI(BaseModel):
    hpo_id: str
    name: str
    score: float


class TextAttributionSpanAPI(BaseModel):
    chunk_id: int = Field(description="1-based ID of the source chunk for this span")
    start_char: int
    end_char: int
    matched_text_in_chunk: str


class ProcessedChunkAPI(BaseModel):
    chunk_id: int
    text: str
    status: str  # e.g., "affirmed", "negated" (string value of AssertionStatus enum)
    assertion_details: dict[str, Any] | None = None  # From AssertionDetector
    hpo_matches: list[HPOMatchInChunkAPI] = Field(
        default_factory=list,
        description="HPO terms identified as relevant to this specific chunk.",
    )
    start_char: int | None = Field(
        default=None,
        description="Start position in original document (0-indexed). None if not tracked.",
    )
    end_char: int | None = Field(
        default=None,
        description="End position in original document (exclusive). None if not tracked.",
    )


class AggregatedHPOTermAPI(BaseModel):
    id: str = Field(
        ..., alias="hpo_id"
    )  # Use alias for consistency if internal field is 'id'
    name: str
    confidence: float = Field(
        description="Average confidence score from all evidence chunks."
    )
    status: str = Field(
        description="Aggregated assertion status (e.g., 'affirmed', 'negated')."
    )
    evidence_count: int
    source_chunk_ids: list[int] = Field(
        description="List of 1-based chunk_ids that provide evidence."
    )
    max_score_from_evidence: float | None = Field(
        None,
        description="Highest raw score from any single evidence chunk for this term.",
    )
    top_evidence_chunk_id: int | None = Field(
        None,
        description="1-based ID of the chunk providing the max_score_from_evidence.",
    )
    text_attributions: list[TextAttributionSpanAPI] = Field(
        default_factory=list,
        description="Text spans in source chunks attributed to this HPO term.",
    )
    # HPO term details (populated when include_details=True)
    definition: str | None = Field(
        None, description="Definition of the HPO term (when include_details=True)."
    )
    synonyms: list[str] | None = Field(
        None,
        description="List of synonyms for the HPO term (when include_details=True).",
    )
    # Keeping these for backward compatibility
    score: float | None = None  # Max bi-encoder score from evidence
    reranker_score: float | None = (
        None  # Max reranker score from evidence (if applicable)
    )


class TextProcessingResponseAPI(BaseModel):
    meta: dict[str, Any] = Field(
        ..., description="Metadata about the request and processing parameters used."
    )
    processed_chunks: list[ProcessedChunkAPI] = Field(
        ..., description="List of text chunks after processing and assertion detection."
    )
    aggregated_hpo_terms: list[AggregatedHPOTermAPI] = Field(
        ...,
        description="Final list of aggregated HPO terms extracted from the document.",
    )
