import asyncio
import logging
from typing import Literal, Optional

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import CrossEncoder, SentenceTransformer

from phentrieve.config import DEFAULT_DEVICE, DEFAULT_MODEL

# Core loader functions
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.retrieval.reranker import load_cross_encoder as load_ce_model

logger = logging.getLogger(__name__)

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
    logger.info(
        f"Background task started: Loading {'SBERT' if is_sbert else 'CrossEncoder'} model '{model_name}' on device '{device}'."
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
            model_instance = await run_in_threadpool(
                load_ce_model, model_name=model_name, device=actual_device
            )
            LOADED_CROSS_ENCODERS[model_name] = model_instance

        MODEL_LOADING_STATUS[model_name] = "loaded"
        logger.info(f"Background task success: Model '{model_name}' loaded and cached.")
    except Exception as e:
        MODEL_LOADING_STATUS[model_name] = "failed"
        logger.error(
            f"Background task error: Failed to load model '{model_name}': {e}",
            exc_info=True,
        )
        # Clear from cache if partially added
        if is_sbert and model_name in LOADED_SBERT_MODELS:
            del LOADED_SBERT_MODELS[model_name]
        if not is_sbert and model_name in LOADED_CROSS_ENCODERS:
            del LOADED_CROSS_ENCODERS[model_name]


async def get_sbert_model_dependency(
    model_name_requested: Optional[str] = None,
    trust_remote_code: bool = False,  # From API config
    device_override: Optional[str] = None,  # From config/request
) -> SentenceTransformer:
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE

    if model_name in LOADED_SBERT_MODELS:
        logger.debug(f"API: Returning cached SBERT model: {model_name}")
        return LOADED_SBERT_MODELS[model_name]

    lock = _get_lock_for_model(model_name)
    async with lock:
        # Re-check after acquiring lock
        if model_name in LOADED_SBERT_MODELS:
            logger.debug(f"API: Returning cached SBERT model (post-lock): {model_name}")
            return LOADED_SBERT_MODELS[model_name]

        current_status = MODEL_LOADING_STATUS.get(model_name, "not_loaded")

        if current_status == "loading":
            logger.info(f"API: Model '{model_name}' is already loading. Advise retry.")
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' is currently being prepared. Please try again in 30 seconds.",
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error(f"API: Model '{model_name}' failed to load previously.")
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' failed to load and is unavailable.",
            )

        # If 'not_loaded', initiate loading
        logger.info(f"API: Initiating background load for SBERT model: {model_name}")
        MODEL_LOADING_STATUS[model_name] = "loading"

        # Using asyncio.create_task as this specific request will return 503 anyway
        # and the task is not tied to this response finishing
        asyncio.create_task(
            _load_model_in_background(model_name, True, trust_remote_code, device)
        )

        # Inform client to retry
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' is being prepared. Please try again in 30 seconds.",
            headers={"Retry-After": "30"},
        )


async def get_dense_retriever_dependency(
    sbert_model_name_for_retriever: str,  # SBERT model name for retriever
) -> DenseRetriever:
    retriever_cache_key = f"retriever_for_{sbert_model_name_for_retriever}"

    if retriever_cache_key not in LOADED_RETRIEVERS:
        logger.info(
            f"API: Initializing DenseRetriever for model: {sbert_model_name_for_retriever}"
        )
        sbert_instance = await get_sbert_model_dependency(
            model_name_requested=sbert_model_name_for_retriever
        )

        # Uses internal logic (resolve_data_path -> get_default_index_dir)
        # to find the index based on environment variables.
        retriever = DenseRetriever.from_model_name(
            model=sbert_instance,
            model_name=sbert_model_name_for_retriever,
            # No index_dir is passed here.
        )

        if not retriever:
            logger.error(
                f"API: Failed to init DenseRetriever for {sbert_model_name_for_retriever}. "
                "Ensure index is built and env vars are set correctly."
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Retriever for model '{sbert_model_name_for_retriever}' "
                    "is unavailable. Check environment variables and indexes."
                ),
            )

        logger.info(
            f"API: DenseRetriever initialized for {sbert_model_name_for_retriever}. "
            f"Index: {getattr(retriever, 'index_base_path', 'Path not set')}"
        )
        LOADED_RETRIEVERS[retriever_cache_key] = retriever
    else:
        logger.debug(
            f"API: Using cached DenseRetriever for {sbert_model_name_for_retriever}"
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
        logger.debug(f"API: Returning cached CrossEncoder: {reranker_model_name}")
        return LOADED_CROSS_ENCODERS[reranker_model_name]

    lock = _get_lock_for_model(reranker_model_name)
    async with lock:
        # Re-check after acquiring lock
        if (
            reranker_model_name in LOADED_CROSS_ENCODERS
            and LOADED_CROSS_ENCODERS[reranker_model_name] is not None
        ):
            logger.debug(
                f"API: Returning cached CrossEncoder (post-lock): {reranker_model_name}"
            )
            return LOADED_CROSS_ENCODERS[reranker_model_name]

        current_status = MODEL_LOADING_STATUS.get(reranker_model_name, "not_loaded")

        if current_status == "loading":
            logger.info(
                f"API: CrossEncoder '{reranker_model_name}' is already loading. Advise retry."
            )
            raise HTTPException(
                status_code=503,
                detail=f"CrossEncoder '{reranker_model_name}' is currently being prepared. Please try again in 30 seconds.",
                headers={"Retry-After": "30"},
            )

        if current_status == "failed":
            logger.error(
                f"API: CrossEncoder '{reranker_model_name}' failed to load previously."
            )
            raise HTTPException(
                status_code=503,
                detail=f"CrossEncoder '{reranker_model_name}' failed to load and is unavailable.",
            )

        # If 'not_loaded', initiate loading
        logger.info(
            f"API: Initiating background load for CrossEncoder: {reranker_model_name}"
        )
        MODEL_LOADING_STATUS[reranker_model_name] = "loading"

        # Using asyncio.create_task as this specific request will return 503 anyway
        asyncio.create_task(
            _load_model_in_background(reranker_model_name, False, False, device)
        )

        # Inform client to retry
        raise HTTPException(
            status_code=503,
            detail=f"CrossEncoder '{reranker_model_name}' is being prepared. Please try again in 30 seconds.",
            headers={"Retry-After": "30"},
        )
