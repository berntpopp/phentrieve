import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from api.dependencies import (
    get_cross_encoder_dependency,
    get_dense_retriever_dependency,
    get_sbert_model_dependency,
)
from api.schemas.text_processing_schemas import (
    AggregatedHPOTermAPI,
    HPOMatchInChunkAPI,
    ProcessedChunkAPI,
    TextAttributionSpanAPI,
    TextProcessingRequest,
    TextProcessingResponseAPI,
)
from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.utils import detect_language
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/text", tags=["Text Processing and HPO Extraction"])


def _get_chunking_config_for_api(
    request: TextProcessingRequest,
) -> list[dict[str, Any]]:
    """
    Get chunking configuration based on request strategy and parameters.

    This is an API-specific wrapper around the shared config resolver.

    Args:
        request: Text processing request with strategy and parameters

    Returns:
        Chunking pipeline configuration list
    """
    from phentrieve.text_processing.config_resolver import resolve_chunking_config

    # Pydantic schema ensures chunking_strategy always has a default value
    strategy_name = request.chunking_strategy.lower()

    # Extract parameters with defaults from config
    ws = (
        request.window_size
        if request.window_size is not None
        else DEFAULT_WINDOW_SIZE_TOKENS
    )
    ss = (
        request.step_size if request.step_size is not None else DEFAULT_STEP_SIZE_TOKENS
    )
    th = (
        request.split_threshold
        if request.split_threshold is not None
        else DEFAULT_SPLITTING_THRESHOLD
    )
    msl = (
        request.min_segment_length
        if request.min_segment_length is not None
        else DEFAULT_MIN_SEGMENT_LENGTH_WORDS
    )

    logger.debug(
        "API: Building config for '%s': ws=%s, ss=%s, th=%s, msl=%s",
        _sanitize(strategy_name),
        _sanitize(ws),
        _sanitize(ss),
        _sanitize(th),
        _sanitize(msl),
    )

    # Use shared resolver
    return resolve_chunking_config(
        strategy_name=strategy_name,
        config_file=None,  # API doesn't support config files
        window_size=ws,
        step_size=ss,
        threshold=th,
        min_segment_length=msl,
    )


def _validate_response_chunk_references(
    processed_chunks: list[ProcessedChunkAPI],
    aggregated_terms: list[AggregatedHPOTermAPI],
) -> None:
    """
    Validate chunk ID references in API response for internal consistency.

    This function checks invariants that must hold for a valid response:
    1. Chunk IDs are sequential and 1-based
    2. All source_chunk_ids reference existing chunks
    3. All text_attribution chunk_ids reference existing chunks
    4. top_evidence_chunk_id references an existing chunk (if present)

    This is called under __debug__ (Python assertions enabled) to catch
    bugs during development/testing without production overhead.

    Args:
        processed_chunks: List of processed chunks with chunk_id
        aggregated_terms: List of aggregated HPO terms with chunk references

    Raises:
        AssertionError: If any invariant is violated
    """
    total_chunks = len(processed_chunks)
    chunk_ids = {chunk.chunk_id for chunk in processed_chunks}

    # Invariant 1: Chunk IDs are sequential 1-based
    expected_ids = set(range(1, total_chunks + 1))
    assert chunk_ids == expected_ids, (
        f"Chunk IDs not sequential 1-based. Expected {expected_ids}, got {chunk_ids}"
    )

    # Invariant 2: All source_chunk_ids reference existing chunks
    for term in aggregated_terms:
        invalid_source_ids = set(term.source_chunk_ids) - chunk_ids
        assert not invalid_source_ids, (
            f"HPO term {term.id} has invalid source_chunk_ids: "
            f"{invalid_source_ids} (valid range: 1-{total_chunks})"
        )

    # Invariant 3: All text_attribution chunk_ids reference existing chunks
    for term in aggregated_terms:
        for attr in term.text_attributions:
            assert attr.chunk_id in chunk_ids, (
                f"HPO term {term.id} has text_attribution with invalid "
                f"chunk_id {attr.chunk_id} (valid range: 1-{total_chunks})"
            )

    # Invariant 4: top_evidence_chunk_id references existing chunk (if present)
    for term in aggregated_terms:
        if term.top_evidence_chunk_id is not None:
            assert term.top_evidence_chunk_id in chunk_ids, (
                f"HPO term {term.id} has invalid top_evidence_chunk_id "
                f"{term.top_evidence_chunk_id} (valid range: 1-{total_chunks})"
            )


@router.post("/process", response_model=TextProcessingResponseAPI)
async def process_text_extract_hpo(request: TextProcessingRequest):
    """
    Process clinical text to extract Human Phenotype Ontology (HPO) terms.

    This endpoint replicates the functionality of the `phentrieve text process` CLI command,
    accepting raw clinical text input along with various processing configurations.
    It returns processed text chunks with assertion statuses and aggregated HPO terms.

    Heavy NLP operations are executed asynchronously to prevent blocking the API server.

    Includes adaptive timeout based on text length to prevent frontend disconnects.
    """
    logger.info(
        "API: Received request to process text. Language: %s, Strategy: %s",
        _sanitize(request.language),
        _sanitize(request.chunking_strategy),
    )

    # Calculate adaptive timeout based on text length
    text_length = len(request.text_content)
    if text_length < 500:
        timeout_seconds = 30
    elif text_length < 2000:
        timeout_seconds = 60
    elif text_length < 5000:
        timeout_seconds = 120
    else:
        timeout_seconds = 180

    logger.info(
        "API: Processing %s chars with %ss timeout", text_length, timeout_seconds
    )

    try:
        # Wrap processing with timeout protection
        return await asyncio.wait_for(
            _process_text_internal(request), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error(
            "API: Request timed out after %ss (text length: %s chars)",
            timeout_seconds,
            text_length,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Text processing timed out after {timeout_seconds} seconds. "
                f"Text length: {text_length} characters. "
                f"Suggestions: (1) reduce text length, "
                f"(2) use 'simple' chunking strategy, or "
                f"(3) disable reranker."
            ),
        )


async def _process_text_internal(request: TextProcessingRequest):
    """
    Internal processing function for text extraction.

    Separated from the main endpoint to enable timeout wrapping.
    Uses cached model dependencies for improved performance.
    """

    try:
        # Determine language
        actual_language = request.language
        if not actual_language or actual_language.lower() == "auto":
            try:
                actual_language = await run_in_threadpool(
                    detect_language, request.text_content, default_lang=DEFAULT_LANGUAGE
                )
                logger.info(
                    "API: Auto-detected language: %s", _sanitize(actual_language)
                )
            except Exception as lang_e:
                logger.warning(
                    "API: Language detection failed: %s. Defaulting to %s.",
                    _sanitize(lang_e),
                    DEFAULT_LANGUAGE,
                )
                actual_language = DEFAULT_LANGUAGE

        # Model loading using cached dependencies (much faster than direct loading!)
        # Determine which models to load, with dynamic defaulting for semantic model
        retrieval_model_name_to_load = request.retrieval_model_name or DEFAULT_MODEL
        sbert_for_chunking_name_to_load = (
            request.semantic_model_name or retrieval_model_name_to_load
        )

        logger.info(
            "API: Effective retrieval model: %s",
            _sanitize(retrieval_model_name_to_load),
        )
        logger.info(
            "API: Effective semantic model for chunking: %s",
            _sanitize(sbert_for_chunking_name_to_load),
        )

        # Get cached retrieval model (will load only once per server lifecycle)
        retrieval_sbert_model = await get_sbert_model_dependency(
            model_name_requested=retrieval_model_name_to_load,
            trust_remote_code=request.trust_remote_code or False,
        )

        # Determine whether we need a separate model for chunking
        if sbert_for_chunking_name_to_load != retrieval_model_name_to_load:
            logger.info(
                "API: Using separate semantic model for chunking: %s",
                _sanitize(sbert_for_chunking_name_to_load),
            )
            sbert_for_chunking = await get_sbert_model_dependency(
                model_name_requested=sbert_for_chunking_name_to_load,
                trust_remote_code=request.trust_remote_code or False,
            )
        else:
            # Reuse retrieval model for chunking
            sbert_for_chunking = retrieval_sbert_model

        # Get cached retriever (initializes only once per model)
        retriever = await get_dense_retriever_dependency(
            sbert_model_name_for_retriever=retrieval_model_name_to_load
        )

        # Get cached cross-encoder if reranking is enabled
        cross_enc = None
        if request.enable_reranker and request.reranker_model_name:
            logger.info(
                "API: Using reranker model: %s", _sanitize(request.reranker_model_name)
            )
            cross_enc = await get_cross_encoder_dependency(
                reranker_model_name=request.reranker_model_name
            )
            if not cross_enc:
                logger.warning(
                    "API: Reranker %s not available, proceeding without reranking.",
                    _sanitize(request.reranker_model_name),
                )

        # Prepare pipeline configuration
        chunking_pipeline_cfg = _get_chunking_config_for_api(request)

        assertion_cfg = dict(DEFAULT_ASSERTION_CONFIG)
        assertion_cfg["disable"] = request.no_assertion_detection
        assertion_cfg["preference"] = request.assertion_preference

        # Important: We need to ensure the language is explicitly set in assertion_cfg
        # This matches how the CLI explicitly passes language to the pipeline
        assertion_cfg["language"] = actual_language

        logger.info("API: Using assertion configuration: %s", assertion_cfg)

        text_pipeline = TextProcessingPipeline(
            language=actual_language,
            chunking_pipeline_config=chunking_pipeline_cfg,
            assertion_config=assertion_cfg,
            sbert_model_for_semantic_chunking=sbert_for_chunking,
        )

        # Run the pipeline to get chunks and initial assertion statuses
        logger.info("API: Processing text through pipeline...")
        processed_chunks_list = await run_in_threadpool(
            text_pipeline.process,
            request.text_content,
            include_positions=request.include_chunk_positions,
        )

        api_processed_chunks: list[ProcessedChunkAPI] = []
        text_chunks_for_orchestrator: list[str] = []
        assertion_statuses_for_orchestrator: list[str | None] = []

        for idx, p_chunk in enumerate(processed_chunks_list):
            api_processed_chunks.append(
                ProcessedChunkAPI(
                    chunk_id=idx + 1,  # 1-based for display/API
                    text=p_chunk["text"],
                    status=p_chunk["status"].value,  # Convert Enum to string
                    assertion_details=p_chunk.get("assertion_details"),
                    start_char=p_chunk.get("start_char"),
                    end_char=p_chunk.get("end_char"),
                )
            )
            text_chunks_for_orchestrator.append(p_chunk["text"])
            assertion_statuses_for_orchestrator.append(p_chunk["status"].value)

        logger.info(
            "API: Running HPO extraction orchestrator on %s chunks.",
            len(text_chunks_for_orchestrator),
        )
        # Call the core orchestrator function (this is synchronous, so wrap it)
        (
            aggregated_hpo_terms_internal,
            detailed_chunk_results_internal,
        ) = await run_in_threadpool(
            orchestrate_hpo_extraction,
            text_chunks=text_chunks_for_orchestrator,
            assertion_statuses=assertion_statuses_for_orchestrator,
            retriever=retriever,
            cross_encoder=cross_enc,
            language=actual_language,
            chunk_retrieval_threshold=request.chunk_retrieval_threshold
            or DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
            num_results_per_chunk=request.num_results_per_chunk or 10,
            min_confidence_for_aggregated=request.aggregated_term_confidence
            or DEFAULT_MIN_CONFIDENCE_AGGREGATED,
            top_term_per_chunk=request.top_term_per_chunk_for_aggregation or False,
            include_details=request.include_details or False,
        )

        # Add HPO matches to each processed chunk from the detailed chunk results
        chunk_id_to_processed_chunk = {
            chunk.chunk_id: chunk for chunk in api_processed_chunks
        }
        for chunk_result in detailed_chunk_results_internal:
            chunk_idx = chunk_result.get("chunk_idx", 0)
            chunk_id = chunk_idx + 1  # Convert 0-based to 1-based for API

            if chunk_id in chunk_id_to_processed_chunk:
                processed_chunk = chunk_id_to_processed_chunk[chunk_id]
                # Add HPO matches to the processed chunk
                for match in chunk_result.get("matches", []):
                    processed_chunk.hpo_matches.append(
                        HPOMatchInChunkAPI(
                            hpo_id=match.get("id"),
                            name=match.get("name"),
                            score=match.get("score", 0.0),
                        )
                    )

        # Convert internal aggregated results to API schema
        api_aggregated_hpo_terms: list[AggregatedHPOTermAPI] = []
        for term_data in aggregated_hpo_terms_internal:
            # Create text attribution spans
            text_attributions = []
            for attribution in term_data.get("text_attributions", []):
                text_attributions.append(
                    TextAttributionSpanAPI(
                        chunk_id=attribution.get("chunk_idx", 0)
                        + 1,  # Convert 0-based to 1-based
                        start_char=attribution.get("start_char", 0),
                        end_char=attribution.get("end_char", 0),
                        matched_text_in_chunk=attribution.get(
                            "matched_text_in_chunk", ""
                        ),
                    )
                )

            # Convert internal 0-based chunk indices to 1-based API indices
            source_chunk_ids = [
                chunk_idx + 1 for chunk_idx in term_data.get("chunks", [])
            ]
            top_evidence_chunk_id = None
            top_evidence_chunk_idx = term_data.get("top_evidence_chunk_idx")
            if top_evidence_chunk_idx is not None:
                top_evidence_chunk_id = top_evidence_chunk_idx + 1

            api_aggregated_hpo_terms.append(
                AggregatedHPOTermAPI(
                    hpo_id=term_data["id"],  # ID should always be present
                    name=term_data["name"],  # Name should always be present
                    confidence=term_data.get("confidence", 0.0),
                    status=term_data.get("status", "unknown"),
                    evidence_count=term_data.get("evidence_count", 1),
                    source_chunk_ids=source_chunk_ids,
                    max_score_from_evidence=term_data.get("score", 0.0),
                    top_evidence_chunk_id=top_evidence_chunk_id,
                    text_attributions=text_attributions,
                    definition=term_data.get("definition"),  # Include when available
                    synonyms=term_data.get("synonyms"),  # Include when available
                    # Keep these for backward compatibility
                    score=term_data.get("score", 0.0),
                    reranker_score=term_data.get("reranker_score"),
                )
            )

        # Construct meta information for the response
        response_meta = {
            "request_parameters": request.dict(
                exclude_none=True
            ),  # Using dict for Pydantic v1.x compatibility
            "effective_language": actual_language,
            "effective_chunking_strategy_config": chunking_pipeline_cfg,
            "effective_retrieval_model": retriever.model_name if retriever else None,
            "effective_reranker_model": (
                request.reranker_model_name if cross_enc else None
            ),
            "num_processed_chunks": len(api_processed_chunks),
            "num_aggregated_hpo_terms": len(api_aggregated_hpo_terms),
        }

        # Validate response invariants (only when assertions enabled)
        if __debug__:
            _validate_response_chunk_references(
                api_processed_chunks, api_aggregated_hpo_terms
            )

        return TextProcessingResponseAPI(
            meta=response_meta,
            processed_chunks=api_processed_chunks,
            aggregated_hpo_terms=api_aggregated_hpo_terms,
        )

    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(
            "API: Bad request for text processing: %s", _sanitize(ve), exc_info=True
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid input parameter: {str(ve)}"
        )
    except FileNotFoundError as fnfe:
        logger.error(
            "API: Missing data file during text processing: %s",
            _sanitize(fnfe),
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable due to missing data. Details: {str(fnfe)}",
        )
    except Exception as e:
        error_type = type(e).__name__
        logger.error(
            "API: Unhandled internal server error during text processing: %s - %s",
            _sanitize(error_type),
            _sanitize(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal server error occurred ({error_type}). Please check server logs.",
        )
