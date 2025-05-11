import logging
from fastapi import HTTPException
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
import os

from phentrieve.embeddings import load_embedding_model  # Core Phentrieve loader
from phentrieve.retrieval.reranker import (
    load_cross_encoder as load_ce_model,
)  # Core Phentrieve loader
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.config import (
    DEFAULT_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_MONOLINGUAL_RERANKER_MODEL,
    DEFAULT_DEVICE,
)

logger = logging.getLogger(__name__)

# Global cache for models and retrievers
# Key: model_name (or unique identifier), Value: loaded instance
LOADED_SBERT_MODELS: Dict[str, SentenceTransformer] = {}
LOADED_RETRIEVERS: Dict[str, DenseRetriever] = {}  # Key might be retriever_config_hash
LOADED_CROSS_ENCODERS: Dict[str, Optional[CrossEncoder]] = {}


async def get_sbert_model_dependency(
    model_name_requested: Optional[str] = None,
    trust_remote_code: bool = False,  # Could come from API config
    device_override: Optional[str] = None,  # Could come from API config or request
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
    sbert_model_name_for_retriever: str,  # Name of the SBERT model to base retriever on
    index_dir: Optional[str] = None,  # Custom index directory path
    # min_similarity_from_request: float # This will come from the QueryRequest directly
) -> DenseRetriever:
    # Key for retriever cache should be based on the SBERT model it uses and index location
    retriever_cache_key = f"retriever_for_{sbert_model_name_for_retriever}_{index_dir}"

    if retriever_cache_key not in LOADED_RETRIEVERS:
        logger.info(
            f"Initializing DenseRetriever for API with SBERT model: {sbert_model_name_for_retriever}"
        )
        # Get the actual SBERT model instance using the other dependency
        sbert_instance = await get_sbert_model_dependency(
            model_name_requested=sbert_model_name_for_retriever
        )

        try:
            # Try to debug available index paths
            import os
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_index_paths = [
                os.path.join(project_dir, "data", "indexes"),  # Project data/indexes directory
                os.path.join(project_dir, "hpo_chroma_index"),  # Project directory index
                os.path.expanduser("~/.phentrieve/hpo_chroma_index"),  # User home directory index
                index_dir or ""  # Custom index path if provided
            ]

            for path in possible_index_paths:
                if path and os.path.exists(path):
                    logger.info(f"Found potential index directory: {path}")
                    if os.path.isdir(path):
                        # List contents for debugging
                        contents = os.listdir(path)
                        logger.info(f"Index directory contents: {contents}")

            # Use the first available index path or the provided one
            found_index_path = None
            for path in possible_index_paths:
                if path and os.path.exists(path):
                    found_index_path = path
                    logger.info(f"Using index directory: {path}")
                    break
                    
            # Pass the found index path or fallback to the provided one
            retriever = DenseRetriever.from_model_name(
                model=sbert_instance,
                model_name=sbert_model_name_for_retriever,
                index_dir=found_index_path or index_dir
                # min_similarity is applied per-query from request, not on retriever instantiation
            )

            if not retriever:
                logger.error(
                    f"API: Failed to initialize DenseRetriever for {sbert_model_name_for_retriever}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Retriever for SBERT model '{sbert_model_name_for_retriever}' is unavailable.",
                )

            LOADED_RETRIEVERS[retriever_cache_key] = retriever

        except Exception as e:
            logger.error(
                f"Error initializing retriever for {sbert_model_name_for_retriever}: {e}"
            )
            raise HTTPException(
                status_code=503,
                detail=f"Retriever for SBERT model '{sbert_model_name_for_retriever}' could not be initialized: {str(e)}",
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
            LOADED_CROSS_ENCODERS[reranker_model_name] = (
                model_instance  # Stores None if loading failed
            )
        except Exception as e:  # Catch any exception during loading
            logger.error(f"API: Failed to load CrossEncoder {reranker_model_name}: {e}")
            LOADED_CROSS_ENCODERS[reranker_model_name] = None  # Cache failure

    # Check if it was cached as None (meaning previous load attempt failed)
    if (
        LOADED_CROSS_ENCODERS.get(reranker_model_name) is None
        and reranker_model_name in LOADED_CROSS_ENCODERS
    ):
        logger.warning(
            f"CrossEncoder '{reranker_model_name}' was previously reported as unavailable or failed to load."
        )

    return LOADED_CROSS_ENCODERS.get(reranker_model_name)
