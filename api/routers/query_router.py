import logging
from typing import Literal, Optional, cast

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_cross_encoder_dependency,
    get_dense_retriever_dependency,
)
from api.schemas.query_schemas import QueryRequest, QueryResponse
from phentrieve.config import (
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
)
from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.utils import detect_language
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter()  # Prefix will be added in main.py


def _resolve_query_language(
    text: str,
    language: Optional[str] = None,
    default_language: str = DEFAULT_LANGUAGE,
) -> str:
    """Resolve the language for a query text.

    If language is provided, use it. Otherwise auto-detect.
    Falls back to default_language if detection fails.

    Args:
        text: The query text to detect language for
        language: Explicitly provided language code (optional)
        default_language: Fallback language if detection fails

    Returns:
        ISO 639-1 language code
    """
    if language:
        return language

    try:
        detected_lang = detect_language(text, default_lang=default_language)
        logger.info("Auto-detected language: %s", _sanitize(detected_lang))
        return detected_lang
    except Exception as e:
        logger.warning(
            "Language detection failed: %s. Using default: %s",
            _sanitize(e),
            default_language,
        )
        return default_language


# Helper dependency to extract model_name from request for the retriever
async def get_retriever_for_request(request: QueryRequest) -> DenseRetriever:
    """Extract model name from request and get retriever dependency"""
    model_name_to_use = request.model_name or DEFAULT_MODEL
    return await get_dense_retriever_dependency(
        sbert_model_name_for_retriever=model_name_to_use
    )


# Helper function for GET params to get retriever for model name
async def get_retriever_for_get_params(
    model_name: str = DEFAULT_MODEL,
) -> DenseRetriever:
    """Get retriever dependency for GET request's model name parameter"""
    return await get_dense_retriever_dependency(
        sbert_model_name_for_retriever=model_name or DEFAULT_MODEL
    )


@router.get("/", response_model=QueryResponse)
async def run_hpo_query_get(
    text: str,
    model_name: Optional[str] = DEFAULT_MODEL,
    language: Optional[str] = None,  # Will auto-detect if not provided
    num_results: int = 10,
    similarity_threshold: float = 0.3,
    include_details: bool = False,
    enable_reranker: bool = False,
    detect_query_assertion: bool = True,
    query_assertion_language: Optional[str] = None,
    query_assertion_preference: str = "dependency",
    retriever: DenseRetriever = Depends(
        lambda model=DEFAULT_MODEL: get_retriever_for_get_params(model)
    ),
):
    """Simple GET endpoint for HPO term queries.

    - text: Clinical text to query for HPO terms (required)
    - model_name: Embedding model for HPO retrieval
    - language: ISO 639-1 language code (auto-detected if not provided)
    - num_results: Number of HPO terms to return (capped at 20 when include_details=True)
    - similarity_threshold: Minimum similarity score
    - include_details: Include HPO term definitions and synonyms in results
    - enable_reranker: Enable cross-encoder reranking
    """
    # Create a QueryRequest object with the provided parameters
    request = QueryRequest(
        text=text,
        model_name=model_name,
        language=language,
        num_results=num_results,
        similarity_threshold=similarity_threshold,
        include_details=include_details,
        enable_reranker=enable_reranker,
        reranker_model=DEFAULT_RERANKER_MODEL,
        rerank_count=10,
        detect_query_assertion=detect_query_assertion,
        query_assertion_language=query_assertion_language,
        query_assertion_preference=cast(
            Literal["dependency", "keyword", "any_negative"],
            query_assertion_preference,
        ),
    )

    # Reuse the POST endpoint logic
    return await run_hpo_query(request=request, retriever=retriever)


@router.post("/", response_model=QueryResponse)
async def run_hpo_query(
    request: QueryRequest,
    retriever: DenseRetriever = Depends(get_retriever_for_request),
):
    """Execute an HPO term query with full control over parameters.

    This endpoint accepts a JSON body with various options for fine-tuning the query.
    For a simpler interface, use the GET endpoint.
    """
    sbert_model_to_use_for_retrieval = (
        request.model_name or DEFAULT_MODEL
    )  # Use requested or default for retrieval SBERT

    # Log information about the retriever we're using
    if hasattr(retriever, "model_name"):
        logger.info(
            "API: Using retriever '%s' for request '%s'",
            _sanitize(retriever.model_name),
            _sanitize(sbert_model_to_use_for_retrieval),
        )

    # Ensure retriever is using the correct SBERT model
    if (
        not retriever
        or not hasattr(retriever, "model_name")
        or retriever.model_name != sbert_model_to_use_for_retrieval
    ):
        logger.warning(
            "Retriever mismatch for %s. Re-initializing.",
            _sanitize(sbert_model_to_use_for_retrieval),
        )

        # Initialize retriever using environment-variable based resolution
        # This will use PHENTRIEVE_INDEX_DIR or PHENTRIEVE_DATA_ROOT_DIR/indexes
        retriever = await get_dense_retriever_dependency(
            sbert_model_name_for_retriever=sbert_model_to_use_for_retrieval
        )

        if not retriever:
            raise HTTPException(
                status_code=503,
                detail=f"Retriever for {sbert_model_to_use_for_retrieval} could not be initialized.",
            )

    # Language detection for the main query
    language_to_use = _resolve_query_language(
        text=request.text,
        language=request.language,
        default_language=DEFAULT_LANGUAGE,
    )

    # For assertion detection, explicitly use the query_assertion_language if provided
    # otherwise keep using the detected/specified language
    assertion_language = request.query_assertion_language
    if assertion_language:
        logger.info(
            "Using explicit assertion language: %s (overriding detected language: %s)",
            _sanitize(assertion_language),
            _sanitize(language_to_use),
        )
    else:
        logger.info(
            "No explicit assertion language provided, using query language: %s",
            _sanitize(language_to_use),
        )

    cross_encoder_instance = None
    actual_reranker_model_name = request.reranker_model or DEFAULT_RERANKER_MODEL

    if request.enable_reranker:
        cross_encoder_instance = await get_cross_encoder_dependency(
            reranker_model_name=actual_reranker_model_name,
            device_override=DEFAULT_DEVICE,
        )
        if not cross_encoder_instance:
            logger.warning(
                "Reranking enabled but cross-encoder %s failed to load. Proceeding without reranking.",
                _sanitize(actual_reranker_model_name),
            )

    # Call the core HPO retrieval logic
    query_results_dict = await execute_hpo_retrieval_for_api(
        text=request.text,
        language=language_to_use,
        retriever=retriever,
        num_results=request.num_results,
        similarity_threshold=request.similarity_threshold,
        enable_reranker=request.enable_reranker
        and (cross_encoder_instance is not None),
        cross_encoder=cross_encoder_instance,
        rerank_count=request.rerank_count,
        include_details=request.include_details,
        detect_query_assertion=request.detect_query_assertion,
        query_assertion_language=request.query_assertion_language,
        query_assertion_preference=request.query_assertion_preference,
        debug=False,
    )

    # Convert results to QueryResponse schema
    return QueryResponse(
        query_text_received=request.text,
        language_detected=language_to_use,
        model_used_for_retrieval=sbert_model_to_use_for_retrieval,
        reranker_used=(
            actual_reranker_model_name
            if request.enable_reranker and cross_encoder_instance
            else None
        ),
        query_assertion_status=query_results_dict.get(
            "original_query_assertion_status"
        ),
        results=query_results_dict.get("results", []),
    )
