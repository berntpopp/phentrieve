from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from phentrieve.config import (  # Import defaults from config.py
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_TRANSLATIONS_SUBDIR,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_RERANKER_MODE,
)


class TextProcessingRequest(BaseModel):
    text_content: str = Field(..., description="The raw clinical text to process.")
    language: Optional[str] = Field(
        default=DEFAULT_LANGUAGE,
        description="ISO 639-1 language code of the text (e.g., 'en', 'de'). If None, language detection might be attempted.",
    )

    # Chunking Configuration
    chunking_strategy: Optional[str] = Field(
        default="semantic",
        description="Predefined chunking strategy (e.g., 'simple', 'semantic', 'detailed', 'sliding_window'). See Phentrieve documentation for details.",
    )

    # Model Configuration
    semantic_model_name: Optional[str] = Field(
        default=None,
        description=f"Model for semantic-based chunking strategies. Defaults to retrieval_model_name or Phentrieve's DEFAULT_MODEL ('{DEFAULT_MODEL}') if retrieval_model_name is also None.",
    )
    retrieval_model_name: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description=f"Embedding model for HPO term retrieval (default: '{DEFAULT_MODEL}').",
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
        default=DEFAULT_ASSERTION_CONFIG.get("preference", "dependency"),
        description="Assertion detection preference ('dependency' or 'keyword').",
    )

    # Aggregation
    aggregated_term_confidence: Optional[float] = Field(
        default=0.0, ge=0.0, description="Minimum confidence for aggregated HPO terms."
    )
    top_term_per_chunk_for_aggregation: Optional[bool] = Field(
        default=False,
        description="Consider only the top HPO term from each chunk during final aggregation.",
    )


class ProcessedChunkAPI(BaseModel):
    chunk_id: int
    text: str
    status: str  # e.g., "affirmed", "negated" (string value of AssertionStatus enum)
    assertion_details: Optional[Dict[str, Any]] = None  # From AssertionDetector


class AggregatedHPOTermAPI(BaseModel):
    id: str = Field(
        ..., alias="hpo_id"
    )  # Use alias for consistency if internal field is 'id'
    name: str
    confidence: float
    status: str  # Aggregated assertion status
    evidence_count: int
    score: Optional[float] = None  # Max bi-encoder score from evidence
    reranker_score: Optional[float] = (
        None  # Max reranker score from evidence (if applicable)
    )


class TextProcessingResponseAPI(BaseModel):
    meta: Dict[str, Any] = Field(
        ..., description="Metadata about the request and processing parameters used."
    )
    processed_chunks: List[ProcessedChunkAPI] = Field(
        ..., description="List of text chunks after processing and assertion detection."
    )
    aggregated_hpo_terms: List[AggregatedHPOTermAPI] = Field(
        ...,
        description="Final list of aggregated HPO terms extracted from the document.",
    )
