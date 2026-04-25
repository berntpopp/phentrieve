from typing import Literal

from pydantic import BaseModel, Field, model_validator

from phentrieve.config import DEFAULT_MULTI_VECTOR

# For assertion status type


class QueryRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, description="Clinical text to query for HPO terms."
    )
    model_name: str | None = Field(
        None,
        description="Embedding model for HPO retrieval (e.g., 'FremyCompany/BioLORD-2023-M'). If None, uses default from Phentrieve config.",
    )
    language: str | None = Field(
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

    # Multi-vector parameters (Issue #136)
    multi_vector: bool = Field(
        default=DEFAULT_MULTI_VECTOR,
        description="Use multi-vector index with component-level aggregation.",
    )
    aggregation_strategy: (
        Literal[
            "label_only",
            "label_synonyms_min",
            "label_synonyms_max",
            "all_weighted",
            "all_max",
            "all_min",
            "custom",
        ]
        | None
    ) = Field(
        "label_synonyms_max",
        description="Aggregation strategy for multi-vector results.",
    )
    component_weights: dict[str, float] | None = Field(
        None,
        description="Component weights for 'all_weighted' strategy. Example: {'label': 0.5, 'synonyms': 0.3, 'definition': 0.2}",
    )
    custom_formula: str | None = Field(
        None,
        description="Custom aggregation formula for 'custom' strategy. Example: '0.5 * max(label, max(synonyms)) + 0.5 * definition'",
    )

    # Assertion detection parameters
    detect_query_assertion: bool = Field(
        True, description="Whether to detect assertions (negations) in the query text."
    )
    query_assertion_language: str | None = Field(
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


class ComponentScores(BaseModel):
    """Component scores for multi-vector results."""

    label: float | None = None
    synonyms: list[float] | None = None
    definition: float | None = None


class HPOResultItem(BaseModel):
    hpo_id: str
    label: str
    similarity: float | None = None  # Score from dense retriever
    definition: str | None = None  # HPO term definition (when include_details=True)
    synonyms: list[str] | None = None  # HPO term synonyms (when include_details=True)
    # Multi-vector component scores (Issue #136)
    component_scores: ComponentScores | None = (
        None  # Component scores (when multi_vector=True)
    )


class QueryResponseSegment(BaseModel):  # If processing in segments (e.g. sentences)
    segment_text: str
    hpo_results: list[HPOResultItem]


class QueryResponse(BaseModel):
    query_text_received: str  # The original full text query from the user
    language_detected: str | None = None
    model_used_for_retrieval: str
    # Assertion status from query text
    query_assertion_status: str | None = None
    # If processing as a single block (initial approach):
    results: list[HPOResultItem]
    # If supporting sentence_mode and returning per-sentence results:
    # processed_segments: List[QueryResponseSegment]
    # For now, implement the simpler `results: List[HPOResultItem]` for the whole query
