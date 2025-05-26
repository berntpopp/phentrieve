from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
import os
from pathlib import Path
import logging

from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api
from phentrieve.retrieval.dense_retriever import DenseRetriever
from api.schemas.query_schemas import QueryRequest, QueryResponse
from api.dependencies import (
    get_dense_retriever_dependency,
    get_cross_encoder_dependency,
)
from phentrieve.config import (
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_TRANSLATIONS_SUBDIR,
    DEFAULT_LANGUAGE,
    DEFAULT_DEVICE,
)
from phentrieve.utils import detect_language

logger = logging.getLogger(__name__)
router = APIRouter()  # Prefix will be added in main.py


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
    enable_reranker: bool = False,
    reranker_mode: str = "cross-lingual",
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
    - num_results: Number of HPO terms to return
    - similarity_threshold: Minimum similarity score
    - enable_reranker: Enable cross-encoder reranking
    - reranker_mode: Either "cross-lingual" or "monolingual"
    """
    # Create a QueryRequest object with the provided parameters
    request = QueryRequest(
        text=text,
        model_name=model_name,
        language=language,
        num_results=num_results,
        similarity_threshold=similarity_threshold,
        enable_reranker=enable_reranker,
        reranker_mode=reranker_mode,
        detect_query_assertion=detect_query_assertion,
        query_assertion_language=query_assertion_language,
        query_assertion_preference=query_assertion_preference,
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
            f"API: Using retriever '{retriever.model_name}' "
            f"for request '{sbert_model_to_use_for_retrieval}'"
        )

    # Ensure retriever is using the correct SBERT model
    if (
        not retriever
        or not hasattr(retriever, "model_name")
        or retriever.model_name != sbert_model_to_use_for_retrieval
    ):
        logger.warning(
            f"Retriever mismatch for {sbert_model_to_use_for_retrieval}. "
            f"Re-initializing."
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
    language_to_use = request.language
    if not language_to_use:
        try:
            language_to_use = detect_language(
                request.text, default_lang=DEFAULT_LANGUAGE
            )
            logger.info(f"Auto-detected language: {language_to_use}")
        except Exception as e:  # Catch if langdetect fails
            logger.warning(
                f"Language detection failed: {e}. " f"Using default: {DEFAULT_LANGUAGE}"
            )
            language_to_use = DEFAULT_LANGUAGE
            
    # For assertion detection, explicitly use the query_assertion_language if provided
    # otherwise keep using the detected/specified language
    assertion_language = request.query_assertion_language
    if assertion_language:
        logger.info(f"Using explicit assertion language: {assertion_language} (overriding detected language: {language_to_use})")
    else:
        logger.info(f"No explicit assertion language provided, using query language: {language_to_use}")

    cross_encoder_instance = None
    actual_reranker_model_name = None
    if request.enable_reranker:
        if request.reranker_mode == "monolingual":
            actual_reranker_model_name = (
                request.monolingual_reranker_model or DEFAULT_MONOLINGUAL_RERANKER_MODEL
            )
        else:  # cross-lingual
            actual_reranker_model_name = (
                request.reranker_model or DEFAULT_RERANKER_MODEL
            )

        if actual_reranker_model_name:
            cross_encoder_instance = await get_cross_encoder_dependency(
                reranker_model_name=actual_reranker_model_name,
                device_override=DEFAULT_DEVICE,
            )
            if not cross_encoder_instance:
                logger.warning(
                    f"Reranking enabled but cross-encoder {actual_reranker_model_name} failed to load. Proceeding without reranking."
                )
        else:
            logger.warning(
                "Reranking enabled but no reranker model specified for the mode. Proceeding without reranking."
            )

    resolved_translation_dir = None
    if request.enable_reranker and request.reranker_mode == "monolingual":
        # For monolingual reranking mode, determine the translation directory path
        base_trans_dir = Path("data") / DEFAULT_TRANSLATIONS_SUBDIR
        resolved_translation_dir = str(
            base_trans_dir / (request.translation_dir_name or language_to_use)
        )
        if not os.path.exists(resolved_translation_dir):
            logger.warning(
                f"Monolingual reranking: Translation directory {resolved_translation_dir} not found. Reranking may be suboptimal or fail for translations."
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
        reranker_mode=request.reranker_mode,
        translation_dir_path=resolved_translation_dir,
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
