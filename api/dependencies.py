import asyncio
import logging
import threading
from typing import Literal, cast

from cachetools import TTLCache
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer

from api.config import SBERT_LOAD_TIMEOUT
from phentrieve.config import DEFAULT_DEVICE, DEFAULT_MODEL, DEFAULT_MULTI_VECTOR

# Core loader functions
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.utils import sanitize_log_value

logger = logging.getLogger(__name__)


# Alias for brevity in this module
_sanitize = sanitize_log_value

# Bounded model caches with 1-hour TTL and max 10 models
# TTL prevents stale models in long-running processes.
# _cache_lock serializes mutations across threads — TTLCache itself is not
# thread-safe, though in this FastAPI app writes happen on the event loop.
_cache_lock = threading.Lock()
LOADED_SBERT_MODELS: TTLCache = TTLCache(maxsize=10, ttl=3600)
LOADED_RETRIEVERS: TTLCache = TTLCache(maxsize=10, ttl=3600)

# Model loading status tracking — also bounded to prevent unbounded growth
# when many distinct model names are requested over time. maxsize is larger
# than the model cache because status/lock entries are cheap and we want
# them to outlive model evictions for a short grace window.
ModelLoadStatus = Literal["not_loaded", "loading", "loaded", "failed"]
MODEL_LOADING_STATUS: TTLCache = TTLCache(maxsize=50, ttl=3600)
MODEL_LOAD_LOCKS: TTLCache = TTLCache(maxsize=50, ttl=3600)
# Track active loading tasks to enable awaiting. Cleaned up in the finally
# block of _load_sbert_in_background, so bounded by concurrency, not by
# distinct model names over time.
MODEL_LOADING_TASKS: dict[str, asyncio.Task] = {}


def _get_lock_for_model(model_name: str) -> asyncio.Lock:
    """Get or create a lock for a given model name.

    Uses TTLCache.get() + locked write to avoid a race between membership
    check and insertion when the entry has expired.
    """
    existing = MODEL_LOAD_LOCKS.get(model_name)
    if existing is not None:
        return cast(asyncio.Lock, existing)
    with _cache_lock:
        # Re-check under the lock in case another thread inserted it first.
        existing = MODEL_LOAD_LOCKS.get(model_name)
        if existing is not None:
            return cast(asyncio.Lock, existing)
        new_lock = asyncio.Lock()
        MODEL_LOAD_LOCKS[model_name] = new_lock
        return new_lock


async def _load_sbert_in_background(
    model_name: str, trust_remote_code: bool, device: str | None
) -> None:
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
            MODEL_LOADING_STATUS[model_name] = "loaded"
        logger.info(
            "Background task success: Model '%s' loaded and cached.",
            _sanitize(model_name),
        )
    except Exception as e:
        with _cache_lock:
            MODEL_LOADING_STATUS[model_name] = "failed"
            # Clear from cache if partially added. pop() with default is
            # TTL-safe; no need for membership check that could race.
            LOADED_SBERT_MODELS.pop(model_name, None)
        logger.error(
            "Background task error: Failed to load model '%s': %s",
            _sanitize(model_name),
            _sanitize(e),
            exc_info=True,
        )
    finally:
        # Clean up the task from tracking dict
        MODEL_LOADING_TASKS.pop(model_name, None)


async def get_sbert_model_dependency(
    model_name_requested: str | None = None,
    trust_remote_code: bool = False,
    device_override: str | None = None,
) -> SentenceTransformer:
    """Return a loaded SBERT model, using the shared cache with status tracking.

    Handles cache hits, concurrent loading via a per-model asyncio.Lock, and
    loading timeouts. Uses ``cache.get()`` rather than ``in`` + ``[]`` because
    the underlying ``TTLCache`` can expire entries between those two calls.
    """
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE
    timeout = SBERT_LOAD_TIMEOUT

    # Cache hit (fast path).
    cached = LOADED_SBERT_MODELS.get(model_name)
    if cached is not None:
        logger.debug("API: Returning cached SBERT model: %s", _sanitize(model_name))
        return cast(SentenceTransformer, cached)

    lock = _get_lock_for_model(model_name)
    async with lock:
        # Re-check after acquiring lock
        cached = LOADED_SBERT_MODELS.get(model_name)
        if cached is not None:
            logger.debug(
                "API: Returning cached SBERT model (post-lock): %s",
                _sanitize(model_name),
            )
            return cast(SentenceTransformer, cached)

        current_status = MODEL_LOADING_STATUS.get(model_name, "not_loaded")

        if current_status == "loading":
            logger.info(
                "API: SBERT model '%s' is loading. Waiting up to %ss...",
                _sanitize(model_name),
                timeout,
            )
            if model_name in MODEL_LOADING_TASKS:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(MODEL_LOADING_TASKS[model_name]),
                        timeout=timeout,
                    )
                    cached = LOADED_SBERT_MODELS.get(model_name)
                    if cached is not None:
                        logger.info(
                            "API: SBERT model '%s' finished loading, returning it.",
                            _sanitize(model_name),
                        )
                        return cast(SentenceTransformer, cached)
                except asyncio.TimeoutError:
                    logger.warning(
                        "API: SBERT model '%s' loading timeout (%ss).",
                        _sanitize(model_name),
                        timeout,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"SBERT model '{model_name}' is taking longer "
                            "than expected to load. Please try again."
                        ),
                        headers={"Retry-After": "30"},
                    )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"SBERT model '{model_name}' is currently being prepared. "
                    "Please try again."
                ),
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error(
                "API: SBERT model '%s' failed to load.",
                _sanitize(model_name),
            )
            raise HTTPException(
                status_code=503,
                detail=f"SBERT model '{model_name}' failed to load and is unavailable.",
            )

        # not_loaded: initiate loading
        logger.info(
            "API: Initiating load for SBERT model: %s",
            _sanitize(model_name),
        )
        with _cache_lock:
            MODEL_LOADING_STATUS[model_name] = "loading"
        task = asyncio.create_task(
            _load_sbert_in_background(model_name, trust_remote_code, device)
        )
        MODEL_LOADING_TASKS[model_name] = task

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            cached = LOADED_SBERT_MODELS.get(model_name)
            if cached is not None:
                logger.info(
                    "API: SBERT model '%s' loaded successfully.",
                    _sanitize(model_name),
                )
                return cast(SentenceTransformer, cached)
            raise HTTPException(
                status_code=500,
                detail=(
                    f"SBERT model '{model_name}' failed to load "
                    "due to an internal error."
                ),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "API: SBERT model '%s' loading timeout (%ss) on first request.",
                _sanitize(model_name),
                timeout,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"SBERT model '{model_name}' is taking longer "
                    "than expected to load. Please try again."
                ),
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

    # Use .get() to avoid TTLCache race between `in` check and __getitem__.
    cached_retriever = LOADED_RETRIEVERS.get(retriever_cache_key)
    if cached_retriever is not None:
        logger.debug(
            "API: Using cached DenseRetriever for %s",
            _sanitize(sbert_model_name_for_retriever),
        )
        return cast(DenseRetriever, cached_retriever)

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
    with _cache_lock:
        LOADED_RETRIEVERS[retriever_cache_key] = retriever
    return retriever


async def cleanup_model_caches() -> None:
    """Cancel outstanding loading tasks and clear all model caches.

    Called during app shutdown via lifespan. Cancels any in-flight
    background model loads before clearing caches to prevent
    repopulation after cleanup.
    """
    # Cancel outstanding loading tasks and await their completion
    pending_tasks = []
    for model_name, task in list(MODEL_LOADING_TASKS.items()):
        if not task.done():
            task.cancel()
            pending_tasks.append(task)
            logger.info("API: Cancelled loading task for %s", model_name)
    if pending_tasks:
        # Wait for all cancelled tasks to finish (deterministic shutdown)
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    MODEL_LOADING_TASKS.clear()

    with _cache_lock:
        LOADED_SBERT_MODELS.clear()
        LOADED_RETRIEVERS.clear()
        MODEL_LOADING_STATUS.clear()
        MODEL_LOAD_LOCKS.clear()
    logger.info("API: All model caches cleared.")
