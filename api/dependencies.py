import asyncio
import logging
from typing import Literal, Optional

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import CrossEncoder, SentenceTransformer

from api.config import CROSS_ENCODER_LOAD_TIMEOUT, SBERT_LOAD_TIMEOUT
from phentrieve.config import DEFAULT_DEVICE, DEFAULT_MODEL, DEFAULT_MULTI_VECTOR

# Core loader functions
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.reranker import load_cross_encoder as load_ce_model
from phentrieve.utils import sanitize_log_value

logger = logging.getLogger(__name__)


# Alias for brevity in this module
_sanitize = sanitize_log_value

# Global cache for models and retrievers
# Key: model_name (or unique identifier), Value: loaded instance
LOADED_SBERT_MODELS: dict[str, SentenceTransformer] = {}
# Key is now only model name, no need for index_dir in key
LOADED_RETRIEVERS: dict[str, DenseRetriever] = {}
LOADED_CROSS_ENCODERS: dict[str, Optional[CrossEncoder]] = {}

# Model loading status tracking
ModelLoadStatus = Literal["not_loaded", "loading", "loaded", "failed"]
MODEL_LOADING_STATUS: dict[str, ModelLoadStatus] = {}
MODEL_LOAD_LOCKS: dict[str, asyncio.Lock] = {}
# Track active loading tasks to enable awaiting
MODEL_LOADING_TASKS: dict[str, asyncio.Task] = {}


def _get_lock_for_model(model_name: str) -> asyncio.Lock:
    """Get or create a lock for a given model name."""
    if model_name not in MODEL_LOAD_LOCKS:
        MODEL_LOAD_LOCKS[model_name] = asyncio.Lock()
    return MODEL_LOAD_LOCKS[model_name]


async def _load_model_in_background(
    model_name: str, is_sbert: bool, trust_remote_code: bool, device: Optional[str]
):
    """Load the specified model (SBERT or CrossEncoder) in background using run_in_threadpool.

    Updates global cache and loading status. Handles exceptions.
    """
    model_type = "SBERT" if is_sbert else "CrossEncoder"
    logger.info(
        "Background task started: Loading %s model '%s' on device '%s'.",
        model_type,
        _sanitize(model_name),
        _sanitize(device),
    )
    actual_device = device or DEFAULT_DEVICE
    try:
        if is_sbert:
            model_instance = await run_in_threadpool(
                load_embedding_model,
                model_name=model_name,
                trust_remote_code=trust_remote_code,
                device=actual_device,
            )
            LOADED_SBERT_MODELS[model_name] = model_instance
        else:
            # Explicit type annotation for cross-encoder loading
            ce_model_instance: CrossEncoder | None = await run_in_threadpool(
                load_ce_model, model_name=model_name, device=actual_device
            )
            LOADED_CROSS_ENCODERS[model_name] = ce_model_instance

        MODEL_LOADING_STATUS[model_name] = "loaded"
        logger.info(
            "Background task success: Model '%s' loaded and cached.",
            _sanitize(model_name),
        )
    except Exception as e:
        MODEL_LOADING_STATUS[model_name] = "failed"
        logger.error(
            "Background task error: Failed to load model '%s': %s",
            _sanitize(model_name),
            _sanitize(e),
            exc_info=True,
        )
        # Clear from cache if partially added
        if is_sbert and model_name in LOADED_SBERT_MODELS:
            del LOADED_SBERT_MODELS[model_name]
        if not is_sbert and model_name in LOADED_CROSS_ENCODERS:
            del LOADED_CROSS_ENCODERS[model_name]
    finally:
        # Clean up the task from tracking dict
        if model_name in MODEL_LOADING_TASKS:
            del MODEL_LOADING_TASKS[model_name]


async def get_sbert_model_dependency(
    model_name_requested: Optional[str] = None,
    trust_remote_code: bool = False,  # From API config
    device_override: Optional[str] = None,  # From config/request
) -> SentenceTransformer:
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE

    if model_name in LOADED_SBERT_MODELS:
        logger.debug("API: Returning cached SBERT model: %s", _sanitize(model_name))
        return LOADED_SBERT_MODELS[model_name]

    lock = _get_lock_for_model(model_name)
    async with lock:
        # Re-check after acquiring lock
        if model_name in LOADED_SBERT_MODELS:
            logger.debug(
                "API: Returning cached SBERT model (post-lock): %s",
                _sanitize(model_name),
            )
            return LOADED_SBERT_MODELS[model_name]

        current_status = MODEL_LOADING_STATUS.get(model_name, "not_loaded")

        if current_status == "loading":
            # Model is loading - wait for it with timeout
            logger.info(
                "API: Model '%s' is loading. Waiting up to %ss...",
                _sanitize(model_name),
                SBERT_LOAD_TIMEOUT,
            )
            if model_name in MODEL_LOADING_TASKS:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(MODEL_LOADING_TASKS[model_name]),
                        timeout=SBERT_LOAD_TIMEOUT,
                    )
                    # Loading completed successfully, return the model
                    if model_name in LOADED_SBERT_MODELS:
                        logger.info(
                            "API: Model '%s' finished loading, returning it.",
                            _sanitize(model_name),
                        )
                        return LOADED_SBERT_MODELS[model_name]
                except asyncio.TimeoutError:
                    logger.warning(
                        "API: Model '%s' loading timeout (%ss). Loading continues in background.",
                        _sanitize(model_name),
                        SBERT_LOAD_TIMEOUT,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model '{model_name}' is taking longer than expected to load. Please try again in 30 seconds.",
                        headers={"Retry-After": "30"},
                    )
            # Task not found but status is loading - fallback to old behavior
            logger.warning(
                "API: Model '%s' status is 'loading' but no task found. Advise retry.",
                _sanitize(model_name),
            )
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' is currently being prepared. Please try again in 30 seconds.",
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error(
                "API: Model '%s' failed to load previously.", _sanitize(model_name)
            )
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' failed to load and is unavailable.",
            )

        # If 'not_loaded', initiate loading
        logger.info(
            "API: Initiating background load for SBERT model: %s", _sanitize(model_name)
        )
        MODEL_LOADING_STATUS[model_name] = "loading"

        # Create task and store it for awaiting
        task = asyncio.create_task(
            _load_model_in_background(model_name, True, trust_remote_code, device)
        )
        MODEL_LOADING_TASKS[model_name] = task

        # Wait for the model to load with timeout
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=SBERT_LOAD_TIMEOUT)
            # Loading completed successfully, return the model
            if model_name in LOADED_SBERT_MODELS:
                logger.info(
                    "API: Model '%s' loaded successfully on first request.",
                    _sanitize(model_name),
                )
                return LOADED_SBERT_MODELS[model_name]
            # Edge case: task completed but model not in cache (should not happen)
            logger.error(
                "API: Model '%s' loading task completed but model not found in cache.",
                _sanitize(model_name),
            )
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model_name}' failed to load due to an internal error.",
            )
        except asyncio.TimeoutError:
            logger.warning(
                "API: Model '%s' loading timeout (%ss) on first request. Loading continues in background.",
                _sanitize(model_name),
                SBERT_LOAD_TIMEOUT,
            )
            # Task continues in background - don't cancel it
            # Inform client to retry
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' is taking longer than expected to load. Please try again in 30 seconds.",
                headers={"Retry-After": "30"},
            )


async def get_dense_retriever_dependency(
    sbert_model_name_for_retriever: str,  # SBERT model name for retriever
    multi_vector: bool = DEFAULT_MULTI_VECTOR,  # Use multi-vector index by default
) -> DenseRetriever:
    # Include multi_vector in cache key to support both index types
    retriever_cache_key = (
        f"retriever_for_{sbert_model_name_for_retriever}_multi={multi_vector}"
    )

    if retriever_cache_key not in LOADED_RETRIEVERS:
        logger.info(
            "API: Initializing DenseRetriever for model: %s (multi_vector=%s)",
            _sanitize(sbert_model_name_for_retriever),
            multi_vector,
        )
        sbert_instance = await get_sbert_model_dependency(
            model_name_requested=sbert_model_name_for_retriever
        )

        # Uses internal logic (resolve_data_path -> get_default_index_dir)
        # to find the index based on environment variables.
        retriever = DenseRetriever.from_model_name(
            model=sbert_instance,
            model_name=sbert_model_name_for_retriever,
            multi_vector=multi_vector,  # Pass multi_vector flag to retriever
            # No index_dir is passed here.
        )

        if not retriever:
            logger.error(
                "API: Failed to init DenseRetriever for %s. "
                "Ensure index is built and env vars are set correctly.",
                _sanitize(sbert_model_name_for_retriever),
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Retriever for model '{sbert_model_name_for_retriever}' "
                    "is unavailable. Check environment variables and indexes."
                ),
            )

        index_path = getattr(retriever, "index_base_path", "Path not set")
        logger.info(
            "API: DenseRetriever initialized for %s. Index: %s",
            _sanitize(sbert_model_name_for_retriever),
            _sanitize(index_path),
        )
        LOADED_RETRIEVERS[retriever_cache_key] = retriever
    else:
        logger.debug(
            "API: Using cached DenseRetriever for %s",
            _sanitize(sbert_model_name_for_retriever),
        )
    return LOADED_RETRIEVERS[retriever_cache_key]


async def get_cross_encoder_dependency(
    reranker_model_name: Optional[str] = None, device_override: Optional[str] = None
) -> Optional[CrossEncoder]:
    if not reranker_model_name:
        return None

    device = device_override or DEFAULT_DEVICE

    if (
        reranker_model_name in LOADED_CROSS_ENCODERS
        and LOADED_CROSS_ENCODERS[reranker_model_name] is not None
    ):
        logger.debug(
            "API: Returning cached CrossEncoder: %s", _sanitize(reranker_model_name)
        )
        return LOADED_CROSS_ENCODERS[reranker_model_name]

    lock = _get_lock_for_model(reranker_model_name)
    async with lock:
        # Re-check after acquiring lock
        if (
            reranker_model_name in LOADED_CROSS_ENCODERS
            and LOADED_CROSS_ENCODERS[reranker_model_name] is not None
        ):
            logger.debug(
                "API: Returning cached CrossEncoder (post-lock): %s",
                _sanitize(reranker_model_name),
            )
            return LOADED_CROSS_ENCODERS[reranker_model_name]

        current_status = MODEL_LOADING_STATUS.get(reranker_model_name, "not_loaded")

        if current_status == "loading":
            # Model is loading - wait for it with timeout
            logger.info(
                "API: CrossEncoder '%s' is loading. Waiting up to %ss...",
                _sanitize(reranker_model_name),
                CROSS_ENCODER_LOAD_TIMEOUT,
            )
            if reranker_model_name in MODEL_LOADING_TASKS:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(MODEL_LOADING_TASKS[reranker_model_name]),
                        timeout=CROSS_ENCODER_LOAD_TIMEOUT,
                    )
                    # Loading completed successfully, return the model
                    if (
                        reranker_model_name in LOADED_CROSS_ENCODERS
                        and LOADED_CROSS_ENCODERS[reranker_model_name] is not None
                    ):
                        logger.info(
                            "API: CrossEncoder '%s' finished loading, returning it.",
                            _sanitize(reranker_model_name),
                        )
                        return LOADED_CROSS_ENCODERS[reranker_model_name]
                except asyncio.TimeoutError:
                    logger.warning(
                        "API: CrossEncoder '%s' loading timeout (%ss). Loading continues in background.",
                        _sanitize(reranker_model_name),
                        CROSS_ENCODER_LOAD_TIMEOUT,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"CrossEncoder '{reranker_model_name}' is taking longer than expected to load. Please try again in 10 seconds.",
                        headers={"Retry-After": "10"},
                    )
            # Task not found but status is loading - fallback to old behavior
            logger.warning(
                "API: CrossEncoder '%s' status is 'loading' but no task found. Advise retry.",
                _sanitize(reranker_model_name),
            )
            raise HTTPException(
                status_code=503,
                detail=f"CrossEncoder '{reranker_model_name}' is currently being prepared. Please try again in 30 seconds.",
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error(
                "API: CrossEncoder '%s' failed to load previously.",
                _sanitize(reranker_model_name),
            )
            raise HTTPException(
                status_code=503,
                detail=f"CrossEncoder '{reranker_model_name}' failed to load and is unavailable.",
            )

        # If 'not_loaded', initiate loading
        logger.info(
            "API: Initiating background load for CrossEncoder: %s",
            _sanitize(reranker_model_name),
        )
        MODEL_LOADING_STATUS[reranker_model_name] = "loading"

        # Create task and store it for awaiting
        task = asyncio.create_task(
            _load_model_in_background(reranker_model_name, False, False, device)
        )
        MODEL_LOADING_TASKS[reranker_model_name] = task

        # Wait for the model to load with timeout
        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=CROSS_ENCODER_LOAD_TIMEOUT
            )
            # Loading completed successfully, return the model
            if (
                reranker_model_name in LOADED_CROSS_ENCODERS
                and LOADED_CROSS_ENCODERS[reranker_model_name] is not None
            ):
                logger.info(
                    "API: CrossEncoder '%s' loaded successfully on first request.",
                    _sanitize(reranker_model_name),
                )
                return LOADED_CROSS_ENCODERS[reranker_model_name]
            # Edge case: task completed but model not in cache (should not happen)
            logger.error(
                "API: CrossEncoder '%s' loading task completed but model not found in cache.",
                _sanitize(reranker_model_name),
            )
            raise HTTPException(
                status_code=500,
                detail=f"CrossEncoder '{reranker_model_name}' failed to load due to an internal error.",
            )
        except asyncio.TimeoutError:
            logger.warning(
                "API: CrossEncoder '%s' loading timeout (%ss) on first request. "
                "Loading continues in background.",
                _sanitize(reranker_model_name),
                CROSS_ENCODER_LOAD_TIMEOUT,
            )
            # Task continues in background - don't cancel it
            # Inform client to retry
            raise HTTPException(
                status_code=503,
                detail=f"CrossEncoder '{reranker_model_name}' is taking longer than expected to load. Please try again in 10 seconds.",
                headers={"Retry-After": "10"},
            )
