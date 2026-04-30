import logging
from typing import Any

from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

from api.dependencies import (
    get_dense_retriever_dependency,
    get_sbert_model_dependency,
)
from api.schemas.text_processing_schemas import TextProcessingRequest
from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)
from phentrieve.retrieval.model_policy import resolve_retrieval_model_policy
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.utils import detect_language
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)


def validate_model_name(field_name: str, model_name: str | None) -> str:
    """Validate a caller-supplied model name against the server allowlist."""
    try:
        return resolve_retrieval_model_policy(model_name).model_name
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported {field_name}: {model_name}. {exc}",
        ) from exc


def get_trust_remote_code_for_model(model_name: str) -> bool:
    """Return the server-owned trust policy for an allowed text-processing model."""
    return resolve_retrieval_model_policy(model_name).trust_remote_code


def get_chunking_config_for_api(
    request: TextProcessingRequest,
) -> list[dict[str, Any]]:
    """
    Get chunking configuration based on request strategy and parameters.

    This is an API-specific wrapper around the shared config resolver.
    """
    from phentrieve.text_processing.config_resolver import resolve_chunking_config

    strategy_name = request.chunking_strategy.lower()

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

    return resolve_chunking_config(
        strategy_name=strategy_name,
        config_file=None,
        window_size=ws,
        step_size=ss,
        threshold=th,
        min_segment_length=msl,
    )


async def prepare_standard_text_processing_context(
    request: TextProcessingRequest,
) -> dict[str, Any]:
    """Prepare standard-backend dependencies and config for text processing."""
    actual_language = request.language
    if not actual_language or actual_language.lower() == "auto":
        try:
            actual_language = await run_in_threadpool(
                detect_language, request.text, default_lang=DEFAULT_LANGUAGE
            )
            logger.info("API: Auto-detected language: %s", _sanitize(actual_language))
        except Exception as lang_e:  # noqa: BLE001
            logger.warning(
                "API: Language detection failed: %s. Defaulting to %s.",
                _sanitize(lang_e),
                DEFAULT_LANGUAGE,
            )
            actual_language = DEFAULT_LANGUAGE

    retrieval_model_name_to_load = validate_model_name(
        "retrieval_model_name", request.retrieval_model_name
    )
    sbert_for_chunking_name_to_load = validate_model_name(
        "semantic_model_name",
        request.semantic_model_name
        if request.semantic_model_name is not None
        else retrieval_model_name_to_load,
    )

    logger.info(
        "API: Effective retrieval model: %s",
        _sanitize(retrieval_model_name_to_load),
    )
    logger.info(
        "API: Effective semantic model for chunking: %s",
        _sanitize(sbert_for_chunking_name_to_load),
    )

    retrieval_sbert_model = await get_sbert_model_dependency(
        model_name_requested=retrieval_model_name_to_load,
        trust_remote_code=get_trust_remote_code_for_model(retrieval_model_name_to_load),
    )

    if sbert_for_chunking_name_to_load != retrieval_model_name_to_load:
        logger.info(
            "API: Using separate semantic model for chunking: %s",
            _sanitize(sbert_for_chunking_name_to_load),
        )
        sbert_for_chunking = await get_sbert_model_dependency(
            model_name_requested=sbert_for_chunking_name_to_load,
            trust_remote_code=get_trust_remote_code_for_model(
                sbert_for_chunking_name_to_load
            ),
        )
    else:
        sbert_for_chunking = retrieval_sbert_model

    retriever = await get_dense_retriever_dependency(
        sbert_model_name_for_retriever=retrieval_model_name_to_load
    )

    chunking_pipeline_cfg = get_chunking_config_for_api(request)

    assertion_cfg = dict(DEFAULT_ASSERTION_CONFIG)
    assertion_cfg["disable"] = request.no_assertion_detection
    assertion_cfg["preference"] = request.assertion_preference
    assertion_cfg["language"] = actual_language

    logger.info("API: Using assertion configuration: %s", assertion_cfg)

    text_pipeline = TextProcessingPipeline(
        language=actual_language,
        chunking_pipeline_config=chunking_pipeline_cfg,
        assertion_config=assertion_cfg,
        sbert_model_for_semantic_chunking=sbert_for_chunking,
    )

    return {
        "actual_language": actual_language,
        "retrieval_model_name": retrieval_model_name_to_load,
        "chunking_pipeline_config": chunking_pipeline_cfg,
        "retriever": retriever,
        "text_pipeline": text_pipeline,
    }
