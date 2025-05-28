from fastapi import APIRouter, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from api.schemas.text_processing_schemas import (
    TextProcessingRequest,
    TextProcessingResponseAPI,
    ProcessedChunkAPI,
    AggregatedHPOTermAPI,
)
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.text_processing.assertion_detection import AssertionStatus
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.reranker import load_cross_encoder
from phentrieve.embeddings import load_embedding_model
from phentrieve.config import (
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_ASSERTION_CONFIG,
    get_simple_chunking_config,
    get_semantic_chunking_config,
    get_detailed_chunking_config,
    get_sliding_window_config_with_params,
    get_sliding_window_cleaned_config,
    get_sliding_window_punct_cleaned_config,
    get_sliding_window_punct_conj_cleaned_config,
    DEFAULT_TRANSLATIONS_SUBDIR,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
)
from phentrieve.utils import detect_language, resolve_data_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/text", tags=["Text Processing and HPO Extraction"])


# Helper to get chunking config based on strategy name from request
def _get_chunking_config_for_api(
    request: TextProcessingRequest,
) -> List[Dict[str, Any]]:
    strategy_name = (
        request.chunking_strategy.lower()
    )  # Already defaults to sliding_window_cleaned in schema

    # Use defaults from phentrieve.config as fallback if not provided in request
    # These values are for get_sliding_window_config_with_params and similar internal configs
    cfg_window_size = request.window_size if request.window_size is not None else 7
    cfg_step_size = request.step_size if request.step_size is not None else 1
    cfg_split_threshold = (
        request.split_threshold if request.split_threshold is not None else 0.5
    )
    cfg_min_segment_length = (
        request.min_segment_length if request.min_segment_length is not None else 3
    )

    logger.debug(
        f"API: Building chunking config for strategy '{strategy_name}' with params: "
        f"ws={cfg_window_size}, ss={cfg_step_size}, th={cfg_split_threshold}, msl={cfg_min_segment_length}"
    )

    if strategy_name == "simple":
        return get_simple_chunking_config()
    elif strategy_name == "semantic":
        config = get_semantic_chunking_config()
        for component in config:
            if component.get("type") == "sliding_window":
                component["config"].update(
                    {
                        "window_size_tokens": cfg_window_size,
                        "step_size_tokens": cfg_step_size,
                        "splitting_threshold": cfg_split_threshold,
                        "min_split_segment_length_words": cfg_min_segment_length,
                    }
                )
        return config
    elif strategy_name == "detailed":
        config = get_detailed_chunking_config()
        for component in config:
            if component.get("type") == "sliding_window":
                component["config"].update(
                    {
                        "window_size_tokens": cfg_window_size,
                        "step_size_tokens": cfg_step_size,
                        "splitting_threshold": cfg_split_threshold,
                        "min_split_segment_length_words": cfg_min_segment_length,
                    }
                )
        return config
    elif strategy_name == "sliding_window":
        return get_sliding_window_config_with_params(
            window_size=cfg_window_size,
            step_size=cfg_step_size,
            threshold=cfg_split_threshold,
            min_segment_length=cfg_min_segment_length,
        )
    elif strategy_name == "sliding_window_cleaned":
        config = get_sliding_window_cleaned_config()
        for component in config:
            if component.get("type") == "sliding_window":
                component["config"].update(
                    {
                        "window_size_tokens": cfg_window_size,
                        "step_size_tokens": cfg_step_size,
                        "splitting_threshold": cfg_split_threshold,
                        "min_split_segment_length_words": cfg_min_segment_length,
                    }
                )
        return config
    elif strategy_name == "sliding_window_punct_cleaned":
        config = get_sliding_window_punct_cleaned_config()
        for component in config:
            if component.get("type") == "sliding_window":
                component["config"].update(
                    {
                        "window_size_tokens": cfg_window_size,
                        "step_size_tokens": cfg_step_size,
                        "splitting_threshold": cfg_split_threshold,
                        "min_split_segment_length_words": cfg_min_segment_length,
                    }
                )
        return config
    elif strategy_name == "sliding_window_punct_conj_cleaned":
        config = get_sliding_window_punct_conj_cleaned_config()
        for component in config:
            if component.get("type") == "sliding_window":
                component["config"].update(
                    {
                        "window_size_tokens": cfg_window_size,
                        "step_size_tokens": cfg_step_size,
                        "splitting_threshold": cfg_split_threshold,
                        "min_split_segment_length_words": cfg_min_segment_length,
                    }
                )
        return config
    else:  # Fallback
        logger.warning(
            f"API: Unknown chunking strategy '{strategy_name}'. Defaulting to sliding_window_punct_conj_cleaned."
        )
        # Use sliding_window_punct_conj_cleaned as the default fallback
        config = get_sliding_window_punct_conj_cleaned_config()
        for component in config:
            if component.get("type") == "sliding_window":
                component["config"].update(
                    {
                        "window_size_tokens": cfg_window_size,
                        "step_size_tokens": cfg_step_size,
                        "splitting_threshold": cfg_split_threshold,
                        "min_split_segment_length_words": cfg_min_segment_length,
                    }
                )
        return config


@router.post("/process", response_model=TextProcessingResponseAPI)
async def process_text_extract_hpo(request: TextProcessingRequest):
    """
    Process clinical text to extract Human Phenotype Ontology (HPO) terms.

    This endpoint replicates the functionality of the `phentrieve text process` CLI command,
    accepting raw clinical text input along with various processing configurations.
    It returns processed text chunks with assertion statuses and aggregated HPO terms.

    Heavy NLP operations are executed asynchronously to prevent blocking the API server.
    """
    logger.info(
        f"API: Received request to process text. Language: {request.language}, Strategy: {request.chunking_strategy}"
    )

    try:
        # Determine language
        actual_language = request.language
        if not actual_language or actual_language.lower() == "auto":
            try:
                actual_language = await run_in_threadpool(
                    detect_language, request.text_content, default_lang=DEFAULT_LANGUAGE
                )
                logger.info(f"API: Auto-detected language: {actual_language}")
            except Exception as lang_e:
                logger.warning(
                    f"API: Language detection failed: {lang_e}. Defaulting to {DEFAULT_LANGUAGE}."
                )
                actual_language = DEFAULT_LANGUAGE

        # Resolve translation directory if needed for monolingual reranking
        translation_dir = resolve_data_path(DEFAULT_TRANSLATIONS_SUBDIR)

        # Model loading (run in threadpool as it's blocking)
        # Determine which models to load, with dynamic defaulting for semantic model
        retrieval_model_name_to_load = request.retrieval_model_name or DEFAULT_MODEL
        sbert_for_chunking_name_to_load = (
            request.semantic_model_name or retrieval_model_name_to_load
        )

        logger.info(f"API: Effective retrieval model: {retrieval_model_name_to_load}")
        logger.info(
            f"API: Effective semantic model for chunking: {sbert_for_chunking_name_to_load}"
        )

        # Load the retrieval model
        retrieval_sbert_model = await run_in_threadpool(
            load_embedding_model,
            model_name=retrieval_model_name_to_load,
            trust_remote_code=request.trust_remote_code,
        )

        # Determine whether we need a separate model for chunking
        sbert_for_chunking = retrieval_sbert_model
        if sbert_for_chunking_name_to_load != retrieval_model_name_to_load:
            logger.info(
                f"API: Loading separate semantic model for chunking: {sbert_for_chunking_name_to_load}"
            )
            sbert_for_chunking = await run_in_threadpool(
                load_embedding_model,
                model_name=sbert_for_chunking_name_to_load,
                trust_remote_code=request.trust_remote_code,
            )

        retriever = await run_in_threadpool(
            DenseRetriever.from_model_name,
            model=retrieval_sbert_model,
            model_name=retrieval_model_name_to_load,
            min_similarity=request.chunk_retrieval_threshold,
        )
        if not retriever:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize retriever for model {retrieval_model_name_to_load}. Index might be missing.",
            )

        cross_enc = None
        if request.enable_reranker:
            reranker_to_load = request.reranker_model_name
            if request.reranker_mode == "monolingual" and actual_language != "en":
                reranker_to_load = request.monolingual_reranker_model_name
            if reranker_to_load:
                logger.info(f"API: Loading reranker model: {reranker_to_load}")
                cross_enc = await run_in_threadpool(
                    load_cross_encoder, model_name=reranker_to_load
                )
            if not cross_enc:
                logger.warning(
                    f"API: Failed to load reranker {reranker_to_load}, proceeding without reranking."
                )

        # Prepare pipeline configuration
        chunking_pipeline_cfg = _get_chunking_config_for_api(request)

        assertion_cfg = dict(DEFAULT_ASSERTION_CONFIG)
        assertion_cfg["disable"] = request.no_assertion_detection
        assertion_cfg["preference"] = request.assertion_preference
        
        # Important: We need to ensure the language is explicitly set in assertion_cfg
        # This matches how the CLI explicitly passes language to the pipeline
        assertion_cfg["language"] = actual_language
        
        logger.info(f"API: Using assertion configuration: {assertion_cfg}")
        
        text_pipeline = TextProcessingPipeline(
            language=actual_language,
            chunking_pipeline_config=chunking_pipeline_cfg,
            assertion_config=assertion_cfg,
            sbert_model_for_semantic_chunking=sbert_for_chunking,
        )

        # Run the pipeline to get chunks and initial assertion statuses
        logger.info("API: Processing text through pipeline...")
        processed_chunks_list = await run_in_threadpool(
            text_pipeline.process, request.text_content
        )

        api_processed_chunks: List[ProcessedChunkAPI] = []
        text_chunks_for_orchestrator: List[str] = []
        assertion_statuses_for_orchestrator: List[str] = []

        for idx, p_chunk in enumerate(processed_chunks_list):
            api_processed_chunks.append(
                ProcessedChunkAPI(
                    chunk_id=idx + 1,  # 1-based for display/API
                    text=p_chunk["text"],
                    status=p_chunk["status"].value,  # Convert Enum to string
                    assertion_details=p_chunk.get("assertion_details"),
                )
            )
            text_chunks_for_orchestrator.append(p_chunk["text"])
            assertion_statuses_for_orchestrator.append(p_chunk["status"].value)

        logger.info(
            f"API: Running HPO extraction orchestrator on {len(text_chunks_for_orchestrator)} chunks."
        )
        # Call the core orchestrator function (this is synchronous, so wrap it)
        aggregated_hpo_terms_internal, _ = await run_in_threadpool(
            orchestrate_hpo_extraction,
            text_chunks=text_chunks_for_orchestrator,
            assertion_statuses=assertion_statuses_for_orchestrator,
            retriever=retriever,
            cross_encoder=cross_enc,
            language=actual_language,
            chunk_retrieval_threshold=request.chunk_retrieval_threshold,
            num_results_per_chunk=request.num_results_per_chunk,
            reranker_mode=request.reranker_mode,
            translation_dir_path=translation_dir,
            min_confidence_for_aggregated=request.aggregated_term_confidence,
            top_term_per_chunk=request.top_term_per_chunk_for_aggregation,
        )

        # Convert internal aggregated results to API schema
        api_aggregated_hpo_terms: List[AggregatedHPOTermAPI] = []
        for term_data in aggregated_hpo_terms_internal:
            api_aggregated_hpo_terms.append(
                AggregatedHPOTermAPI(
                    hpo_id=term_data["id"],  # ID should always be present
                    name=term_data["name"],  # Name should always be present
                    confidence=term_data.get("confidence", 0.0),
                    status=term_data.get("status", "unknown"),
                    evidence_count=term_data.get("evidence_count", 1),
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

        return TextProcessingResponseAPI(
            meta=response_meta,
            processed_chunks=api_processed_chunks,
            aggregated_hpo_terms=api_aggregated_hpo_terms,
        )

    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"API: Bad request for text processing: {ve}", exc_info=True)
        raise HTTPException(
            status_code=400, detail=f"Invalid input parameter: {str(ve)}"
        )
    except FileNotFoundError as fnfe:
        logger.error(
            f"API: Missing data file during text processing: {fnfe}", exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable due to missing data. Details: {str(fnfe)}",
        )
    except Exception as e:
        error_type = type(e).__name__
        logger.error(
            f"API: Unhandled internal server error during text processing: {error_type} - {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal server error occurred ({error_type}). Please check server logs.",
        )
