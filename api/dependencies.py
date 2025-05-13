import logging
from fastapi import HTTPException
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder

# Core loader functions
from phentrieve.embeddings import load_embedding_model
from phentrieve.retrieval.reranker import load_cross_encoder as load_ce_model
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.config import DEFAULT_MODEL, DEFAULT_DEVICE

logger = logging.getLogger(__name__)

# Global cache for models and retrievers
# Key: model_name (or unique identifier), Value: loaded instance
LOADED_SBERT_MODELS: Dict[str, SentenceTransformer] = {}
# Key is now only model name, no need for index_dir in key
LOADED_RETRIEVERS: Dict[str, DenseRetriever] = {}
LOADED_CROSS_ENCODERS: Dict[str, Optional[CrossEncoder]] = {}


async def get_sbert_model_dependency(
    model_name_requested: Optional[str] = None,
    trust_remote_code: bool = False,  # From API config
    device_override: Optional[str] = None,  # From config/request
) -> SentenceTransformer:
    model_name = model_name_requested or DEFAULT_MODEL
    device = device_override or DEFAULT_DEVICE

    if model_name not in LOADED_SBERT_MODELS:
        logger.info(f"Loading SBERT model for API: {model_name}")
        try:
            LOADED_SBERT_MODELS[model_name] = load_embedding_model(
                model_name=model_name,
                trust_remote_code=trust_remote_code,
                device=device,
            )
        except ValueError as e:
            logger.error(f"API: Failed to load SBERT model {model_name}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"SBERT model '{model_name}' is currently unavailable.",
            )
    return LOADED_SBERT_MODELS[model_name]


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

    if reranker_model_name not in LOADED_CROSS_ENCODERS:
        logger.info(f"Loading CrossEncoder model for API: {reranker_model_name}")
        try:
            # Use the existing load_cross_encoder from phentrieve.retrieval.reranker
            model_instance = load_ce_model(
                model_name=reranker_model_name, device=device
            )
            LOADED_CROSS_ENCODERS[reranker_model_name] = model_instance  # May be None
        except Exception as e:  # Catch any exception during loading
            logger.error(f"API: Failed to load CrossEncoder {reranker_model_name}: {e}")
            LOADED_CROSS_ENCODERS[reranker_model_name] = None  # Cache failure

    # Check if it was cached as None (meaning previous load attempt failed)
    if (
        LOADED_CROSS_ENCODERS.get(reranker_model_name) is None
        and reranker_model_name in LOADED_CROSS_ENCODERS
    ):
        logger.warning(
            f"CrossEncoder '{reranker_model_name}' was previously unavailable."
        )

    return LOADED_CROSS_ENCODERS.get(reranker_model_name)
