from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from phentrieve.config import (  # Import defaults from config.py
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_RERANKER_MODE,
    DEFAULT_RERANKER_MODEL,
)


class TextProcessingRequest(BaseModel):
    text_content: str = Field(..., description="The raw clinical text to process.")
    language: Optional[str] = Field(
        default=DEFAULT_LANGUAGE,
        description="ISO 639-1 language code of the text (e.g., 'en', 'de'). If None, language detection might be attempted.",
    )

    # Chunking Configuration
    chunking_strategy: Optional[str] = Field(
        default="sliding_window_punct_conj_cleaned",  # Updated to the new default strategy
        description="Predefined chunking strategy (e.g., 'simple', 'semantic', 'detailed', 'sliding_window_cleaned', 'sliding_window_punct_cleaned', 'sliding_window_punct_conj_cleaned'). See Phentrieve documentation for details.",
        json_schema_extra={"example": "sliding_window_punct_conj_cleaned"},
    )

    # Sliding window chunking parameters
    window_size: Optional[int] = Field(
        default=2, ge=1, description="Sliding window size in tokens."
    )
    step_size: Optional[int] = Field(
        default=1, ge=1, description="Sliding window step size in tokens."
    )
    split_threshold: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for splitting (0-1).",
    )
    min_segment_length: Optional[int] = Field(
        default=1,
        ge=1,
        description="Minimum segment length in words for sliding window.",
    )

    # Model Configuration
    semantic_model_name: Optional[str] = Field(
        default=DEFAULT_MODEL,  # Will be handled in router logic to default to retrieval_model_name
        description=f"Model for semantic-based chunking strategies. If None, defaults to retrieval_model_name or Phentrieve's DEFAULT_MODEL ('{DEFAULT_MODEL}').",
    )
    retrieval_model_name: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description=f"Embedding model for HPO term retrieval (default: '{DEFAULT_MODEL}').",
    )
    trust_remote_code: Optional[bool] = Field(
        default=False,
        description="Trust remote code when loading models from Hugging Face Hub (use with caution).",
    )

    # Retrieval & Reranking Parameters
    chunk_retrieval_threshold: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for HPO matches per chunk.",
    )
    num_results_per_chunk: Optional[int] = Field(
        default=10, ge=1, description="Max HPO terms to consider from each chunk."
    )
    enable_reranker: Optional[bool] = Field(
        default=False, description="Enable cross-encoder reranking."
    )
    reranker_model_name: Optional[str] = Field(
        default=DEFAULT_RERANKER_MODEL,
        description="Multilingual/Cross-lingual reranker model.",
    )
    monolingual_reranker_model_name: Optional[str] = Field(
        default=DEFAULT_MONOLINGUAL_RERANKER_MODEL,
        description="Language-specific monolingual reranker model.",
    )
    reranker_mode: Optional[str] = Field(
        default=DEFAULT_RERANKER_MODE,
        description="Reranking mode: 'cross-lingual' or 'monolingual'.",
    )
    rerank_count_per_chunk: Optional[int] = Field(
        default=50, ge=1, description="Number of candidates to rerank per chunk."
    )

    # Assertion Detection
    no_assertion_detection: Optional[bool] = Field(
        default=False, description="Disable assertion detection."
    )
    assertion_preference: Optional[str] = Field(
        default=cast(str, DEFAULT_ASSERTION_CONFIG.get("preference", "dependency")),
        description="Assertion detection preference ('dependency' or 'keyword').",
    )

    # Aggregation
    aggregated_term_confidence: Optional[float] = Field(
        default=0.35,
        ge=0.0,
        le=1.0,  # Changed from 0.0 to 0.35 to align with CLI
        description="Minimum confidence score for an aggregated HPO term.",
        json_schema_extra={"example": 0.35},
    )
    top_term_per_chunk_for_aggregation: Optional[bool] = Field(
        default=False,
        description="Consider only the top HPO term from each chunk during final aggregation.",
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
    assertion_details: Optional[dict[str, Any]] = None  # From AssertionDetector
    hpo_matches: list[HPOMatchInChunkAPI] = Field(
        default_factory=list,
        description="HPO terms identified as relevant to this specific chunk.",
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
    max_score_from_evidence: Optional[float] = Field(
        None,
        description="Highest raw score from any single evidence chunk for this term.",
    )
    top_evidence_chunk_id: Optional[int] = Field(
        None,
        description="1-based ID of the chunk providing the max_score_from_evidence.",
    )
    text_attributions: list[TextAttributionSpanAPI] = Field(
        default_factory=list,
        description="Text spans in source chunks attributed to this HPO term.",
    )
    # Keeping these for backward compatibility
    score: Optional[float] = None  # Max bi-encoder score from evidence
    reranker_score: Optional[float] = (
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
