from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal


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

    enable_reranker: bool = Field(False, description="Enable cross-encoder reranking.")
    reranker_model: Optional[str] = Field(
        None,
        description="Cross-encoder model for cross-lingual reranking (e.g., 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'). Uses default if None and reranking enabled.",
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

    # Add sentence_mode if you want to expose it directly, default to False for whole text processing by the API
    # sentence_mode: bool = Field(False, description="Process text sentence by sentence. If false, whole input text is processed as one query.")


class HPOResultItem(BaseModel):
    hpo_id: str
    label: str
    similarity: Optional[float] = None  # Score from dense retriever
    cross_encoder_score: Optional[float] = None  # Score from reranker
    original_rank: Optional[int] = None  # Rank before reranking


class QueryResponseSegment(BaseModel):  # If processing in segments (e.g. sentences)
    segment_text: str
    hpo_results: List[HPOResultItem]


class QueryResponse(BaseModel):
    query_text_received: str  # The original full text query from the user
    language_detected: Optional[str] = None
    model_used_for_retrieval: str
    reranker_used: Optional[str] = None
    # If processing as a single block (initial approach):
    results: List[HPOResultItem]
    # If supporting sentence_mode and returning per-sentence results:
    # processed_segments: List[QueryResponseSegment]
    # For now, implement the simpler `results: List[HPOResultItem]` for the whole query
