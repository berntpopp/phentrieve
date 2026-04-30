import asyncio
import logging
import threading
from typing import Literal, cast

from cachetools import TTLCache  # type: ignore[import-untyped]
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer

from api.config import SBERT_LOAD_TIMEOUT
from phentrieve.config import DEFAULT_DEVICE, DEFAULT_MODEL, DEFAULT_MULTI_VECTOR

# Core loader functions
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.model_policy import resolve_retrieval_model_policy
from phentrieve.utils import sanitize_log_value

logger = logging.getLogger(__name__)


# Alias for brevity in this module
_sanitize = sanitize_log_value

# Bounded caches prevent unbounded model growth in long-running API processes.
_cache_lock = threading.Lock()
LOADED_SBERT_MODELS: TTLCache = TTLCache(maxsize=10, ttl=3600)
LOADED_RETRIEVERS: TTLCache = TTLCache(maxsize=10, ttl=3600)

# Model loading status tracking
ModelLoadStatus = Literal["not_loaded", "loading", "loaded", "failed"]
MODEL_LOADING_STATUS: TTLCache = TTLCache(maxsize=50, ttl=3600)
MODEL_LOAD_LOCKS: TTLCache = TTLCache(maxsize=50, ttl=3600)
# Track active loading tasks to enable awaiting
MODEL_LOADING_TASKS: dict[str, asyncio.Task] = {}


def _get_lock_for_model(model_name: str) -> asyncio.Lock:
    """Get or create a lock for a given model name."""
    existing = MODEL_LOAD_LOCKS.get(model_name)
    if existing is not None:
        return cast(asyncio.Lock, existing)
    with _cache_lock:
        existing = MODEL_LOAD_LOCKS.get(model_name)
        if existing is not None:
            return cast(asyncio.Lock, existing)
        new_lock = asyncio.Lock()
        MODEL_LOAD_LOCKS[model_name] = new_lock
        return new_lock


async def _load_model_in_background(
    model_name: str, trust_remote_code: bool, device: str | None
):
    """Load the specified SBERT model in background using run_in_threadpool.

    Updates global cache and loading status. Handles exceptions.
    """
    logger.info(
        "Background task started: Loading SBERT model '%s' on device '%s'.",
        _sanitize(model_name),
        _sanitize(device),
    )
    actual_device = device or DEFAULT_DEVICE
    try:
        model_instance = await run_in_threadpool(
            load_embedding_model,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            device=actual_device,
        )
        with _cache_lock:
            LOADED_SBERT_MODELS[model_name] = model_instance

        with _cache_lock:
            MODEL_LOADING_STATUS[model_name] = "loaded"
        logger.info(
            "Background task success: Model '%s' loaded and cached.",
            _sanitize(model_name),
        )
    except Exception as e:
        with _cache_lock:
            MODEL_LOADING_STATUS[model_name] = "failed"
        logger.error(
            "Background task error: Failed to load model '%s': %s",
            _sanitize(model_name),
            _sanitize(e),
            exc_info=True,
        )
        # Clear from cache if partially added
        with _cache_lock:
            LOADED_SBERT_MODELS.pop(model_name, None)
    finally:
        # Clean up the task from tracking dict
        if model_name in MODEL_LOADING_TASKS:
            del MODEL_LOADING_TASKS[model_name]


async def get_sbert_model_dependency(
    model_name_requested: str | None = None,
    trust_remote_code: bool = False,  # From API config
    device_override: str | None = None,  # From config/request
) -> SentenceTransformer:
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE

    if model_name in LOADED_SBERT_MODELS:
        logger.debug("API: Returning cached SBERT model: %s", _sanitize(model_name))
        return cast(SentenceTransformer, LOADED_SBERT_MODELS[model_name])

    lock = _get_lock_for_model(model_name)
    async with lock:
        # Re-check after acquiring lock
        if model_name in LOADED_SBERT_MODELS:
            logger.debug(
                "API: Returning cached SBERT model (post-lock): %s",
                _sanitize(model_name),
            )
            return cast(SentenceTransformer, LOADED_SBERT_MODELS[model_name])

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
                        return cast(
                            SentenceTransformer, LOADED_SBERT_MODELS[model_name]
                        )
                except asyncio.exceptions.TimeoutError:
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
        with _cache_lock:
            MODEL_LOADING_STATUS[model_name] = "loading"

        # Create task and store it for awaiting
        task = asyncio.create_task(
            _load_model_in_background(model_name, trust_remote_code, device)
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
                return cast(SentenceTransformer, LOADED_SBERT_MODELS[model_name])
            # Edge case: task completed but model not in cache (should not happen)
            logger.error(
                "API: Model '%s' loading task completed but model not found in cache.",
                _sanitize(model_name),
            )
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model_name}' failed to load due to an internal error.",
            )
        except asyncio.exceptions.TimeoutError:
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
    try:
        model_policy = resolve_retrieval_model_policy(sbert_model_name_for_retriever)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sbert_model_name_for_retriever = model_policy.model_name

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
            model_name_requested=sbert_model_name_for_retriever,
            trust_remote_code=model_policy.trust_remote_code,
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
        with _cache_lock:
            LOADED_RETRIEVERS[retriever_cache_key] = retriever
    else:
        logger.debug(
            "API: Using cached DenseRetriever for %s",
            _sanitize(sbert_model_name_for_retriever),
        )
    return cast(DenseRetriever, LOADED_RETRIEVERS[retriever_cache_key])


async def cleanup_model_caches() -> None:
    """Clear cached models and cancel any in-flight background loading tasks."""
    pending_tasks = []
    for task in MODEL_LOADING_TASKS.values():
        if not task.done():
            task.cancel()
            pending_tasks.append(task)
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    MODEL_LOADING_TASKS.clear()
    MODEL_LOADING_STATUS.clear()
    MODEL_LOAD_LOCKS.clear()
    LOADED_SBERT_MODELS.clear()
    LOADED_RETRIEVERS.clear()
