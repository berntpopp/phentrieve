import asyncio
import logging
import threading
from typing import Any, Literal, cast

from cachetools import TTLCache
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from phentrieve.retrieval.reranker import load_cross_encoder as load_ce_model
from sentence_transformers import CrossEncoder, SentenceTransformer

from api.config import CROSS_ENCODER_LOAD_TIMEOUT, SBERT_LOAD_TIMEOUT
from phentrieve.config import DEFAULT_DEVICE, DEFAULT_MODEL, DEFAULT_MULTI_VECTOR

# Core loader functions
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.utils import sanitize_log_value

logger = logging.getLogger(__name__)


# Alias for brevity in this module
_sanitize = sanitize_log_value

# Bounded model caches with 1-hour TTL and max 10 models
# TTL prevents stale models in long-running processes
_cache_lock = threading.Lock()
LOADED_SBERT_MODELS: TTLCache = TTLCache(maxsize=10, ttl=3600)
LOADED_RETRIEVERS: TTLCache = TTLCache(maxsize=10, ttl=3600)
LOADED_CROSS_ENCODERS: TTLCache = TTLCache(maxsize=10, ttl=3600)

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
    model_name: str, is_sbert: bool, trust_remote_code: bool, device: str | None
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


async def _load_model_with_status_tracking(
    model_name: str,
    cache_dict: Any,  # dict or TTLCache — both support __contains__/getitem/setitem
    is_sbert: bool,
    trust_remote_code: bool,
    device: str | None,
    timeout: float,
    model_type_label: str,
) -> Any:
    """Shared model loading with status tracking, locking, and timeout.

    Replaces duplicated logic in get_sbert_model_dependency and
    get_cross_encoder_dependency.
    """
    # Cache hit (fast path)
    if model_name in cache_dict and cache_dict[model_name] is not None:
        logger.debug(
            "API: Returning cached %s: %s", model_type_label, _sanitize(model_name)
        )
        return cache_dict[model_name]

    lock = _get_lock_for_model(model_name)
    async with lock:
        # Re-check after acquiring lock
        if model_name in cache_dict and cache_dict[model_name] is not None:
            logger.debug(
                "API: Returning cached %s (post-lock): %s",
                model_type_label,
                _sanitize(model_name),
            )
            return cache_dict[model_name]

        current_status = MODEL_LOADING_STATUS.get(model_name, "not_loaded")

        if current_status == "loading":
            logger.info(
                "API: %s '%s' is loading. Waiting up to %ss...",
                model_type_label,
                _sanitize(model_name),
                timeout,
            )
            if model_name in MODEL_LOADING_TASKS:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(MODEL_LOADING_TASKS[model_name]),
                        timeout=timeout,
                    )
                    if model_name in cache_dict and cache_dict[model_name] is not None:
                        logger.info(
                            "API: %s '%s' finished loading, returning it.",
                            model_type_label,
                            _sanitize(model_name),
                        )
                        return cache_dict[model_name]
                except asyncio.TimeoutError:
                    logger.warning(
                        "API: %s '%s' loading timeout (%ss).",
                        model_type_label,
                        _sanitize(model_name),
                        timeout,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"{model_type_label} '{model_name}' is taking longer "
                            "than expected to load. Please try again."
                        ),
                        headers={"Retry-After": "30"},
                    )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{model_type_label} '{model_name}' is currently being prepared. "
                    "Please try again."
                ),
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error(
                "API: %s '%s' failed to load.",
                model_type_label,
                _sanitize(model_name),
            )
            raise HTTPException(
                status_code=503,
                detail=f"{model_type_label} '{model_name}' failed to load and is unavailable.",
            )

        # not_loaded: initiate loading
        logger.info(
            "API: Initiating load for %s: %s",
            model_type_label,
            _sanitize(model_name),
        )
        MODEL_LOADING_STATUS[model_name] = "loading"
        task = asyncio.create_task(
            _load_model_in_background(model_name, is_sbert, trust_remote_code, device)
        )
        MODEL_LOADING_TASKS[model_name] = task

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            if model_name in cache_dict and cache_dict[model_name] is not None:
                logger.info(
                    "API: %s '%s' loaded successfully.",
                    model_type_label,
                    _sanitize(model_name),
                )
                return cache_dict[model_name]
            raise HTTPException(
                status_code=500,
                detail=(
                    f"{model_type_label} '{model_name}' failed to load "
                    "due to an internal error."
                ),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "API: %s '%s' loading timeout (%ss) on first request.",
                model_type_label,
                _sanitize(model_name),
                timeout,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{model_type_label} '{model_name}' is taking longer "
                    "than expected to load. Please try again."
                ),
                headers={"Retry-After": "30"},
            )


async def get_sbert_model_dependency(
    model_name_requested: str | None = None,
    trust_remote_code: bool = False,
    device_override: str | None = None,
) -> SentenceTransformer:
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE
    return cast(
        SentenceTransformer,
        await _load_model_with_status_tracking(
            model_name=model_name,
            cache_dict=LOADED_SBERT_MODELS,
            is_sbert=True,
            trust_remote_code=trust_remote_code,
            device=device,
            timeout=SBERT_LOAD_TIMEOUT,
            model_type_label="SBERT model",
        ),
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
    return cast(DenseRetriever, LOADED_RETRIEVERS[retriever_cache_key])


async def get_cross_encoder_dependency(
    reranker_model_name: str | None = None,
    device_override: str | None = None,
) -> CrossEncoder | None:
    if not reranker_model_name:
        return None
    device = device_override or DEFAULT_DEVICE
    result = await _load_model_with_status_tracking(
        model_name=reranker_model_name,
        cache_dict=LOADED_CROSS_ENCODERS,
        is_sbert=False,
        trust_remote_code=False,
        device=device,
        timeout=CROSS_ENCODER_LOAD_TIMEOUT,
        model_type_label="CrossEncoder",
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail=f"CrossEncoder '{reranker_model_name}' failed to load.",
        )
    return cast(CrossEncoder, result)


async def cleanup_model_caches() -> None:
    """Cancel outstanding loading tasks and clear all model caches.

    Called during app shutdown via lifespan. Cancels any in-flight
    background model loads before clearing caches to prevent
    repopulation after cleanup.
    """
    # Cancel outstanding loading tasks first
    for model_name, task in list(MODEL_LOADING_TASKS.items()):
        if not task.done():
            task.cancel()
            logger.info("API: Cancelled loading task for %s", model_name)
    MODEL_LOADING_TASKS.clear()

    with _cache_lock:
        LOADED_SBERT_MODELS.clear()
        LOADED_RETRIEVERS.clear()
        LOADED_CROSS_ENCODERS.clear()
        MODEL_LOADING_STATUS.clear()
        MODEL_LOAD_LOCKS.clear()
    logger.info("API: All model caches cleared.")
