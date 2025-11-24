from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# For assertion status type


class QueryRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, description="Clinical text to query for HPO terms."
    )
    model_name: Optional[str] = Field(
        None,
        description="Embedding model for HPO retrieval (e.g., 'FremyCompany/BioLORD-2023-M'). If None, uses default from Phentrieve config.",
    )
    language: Optional[str] = Field(
        None,
        description="ISO 639-1 language code (e.g., 'en', 'de'). If None, language will be auto-detected.",
    )
    num_results: int = Field(
        10, gt=0, le=50, description="Number of HPO terms to return."
    )
    similarity_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for dense retrieval results.",
    )
    include_details: bool = Field(
        False,
        description="Include HPO term definitions and synonyms in results. When enabled, num_results is capped at 20 for performance.",
    )

    enable_reranker: bool = Field(False, description="Enable cross-encoder reranking.")
    reranker_model: Optional[str] = Field(
        None,
        description="Cross-encoder model for cross-lingual reranking (e.g., 'BAAI/bge-reranker-v2-m3'). Uses default if None and reranking enabled.",
    )
    monolingual_reranker_model: Optional[str] = Field(
        None,
        description="Cross-encoder model for monolingual reranking (e.g., 'ml6team/cross-encoder-mmarco-german-distilbert-base').",
    )
    reranker_mode: Literal["cross-lingual", "monolingual"] = Field(
        "cross-lingual", description="Reranking mode."
    )
    translation_dir_name: Optional[str] = Field(
        None,
        description="Name of the language-specific subdirectory within the main translations directory (e.g., 'de', 'es') for monolingual reranking. If None, uses default structure.",
    )
    rerank_count: int = Field(
        10,
        gt=0,
        le=100,
        description="Number of top dense results to pass to the reranker.",
    )

    # Assertion detection parameters
    detect_query_assertion: bool = Field(
        True, description="Whether to detect assertions (negations) in the query text."
    )
    query_assertion_language: Optional[str] = Field(
        None,
        description="Language for assertion detection (e.g., 'en', 'de'). If None, uses the query language.",
    )
    query_assertion_preference: Literal["dependency", "keyword", "any_negative"] = (
        Field(
            "dependency",
            description="Assertion detection strategy (dependency-based, keyword-based, or any negative detection).",
        )
    )

    # Add sentence_mode if you want to expose it directly, default to False for whole text processing by the API
    # sentence_mode: bool = Field(False, description="Process text sentence by sentence. If false, whole input text is processed as one query.")

    @model_validator(mode="after")
    def validate_num_results_with_details(self) -> "QueryRequest":
        """Validate that num_results doesn't exceed limit when include_details is enabled."""
        if self.include_details and self.num_results > 20:
            raise ValueError(
                f"Maximum 20 results allowed when include_details=true. "
                f"Requested: {self.num_results}. "
                f"Reduce num_results or disable include_details."
            )
        return self


class HPOResultItem(BaseModel):
    hpo_id: str
    label: str
    similarity: Optional[float] = None  # Score from dense retriever
    cross_encoder_score: Optional[float] = None  # Score from reranker
    original_rank: Optional[int] = None  # Rank before reranking
    definition: Optional[str] = None  # HPO term definition (when include_details=True)
    synonyms: Optional[list[str]] = (
        None  # HPO term synonyms (when include_details=True)
    )


class QueryResponseSegment(BaseModel):  # If processing in segments (e.g. sentences)
    segment_text: str
    hpo_results: list[HPOResultItem]


class QueryResponse(BaseModel):
    query_text_received: str  # The original full text query from the user
    language_detected: Optional[str] = None
    model_used_for_retrieval: str
    reranker_used: Optional[str] = None
    # Assertion status from query text
    query_assertion_status: Optional[str] = None
    # If processing as a single block (initial approach):
    results: list[HPOResultItem]
    # If supporting sentence_mode and returning per-sentence results:
    # processed_segments: List[QueryResponseSegment]
    # For now, implement the simpler `results: List[HPOResultItem]` for the whole query
