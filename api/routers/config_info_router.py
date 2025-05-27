"""
API Router for Phentrieve configuration and information.

This module provides an API endpoint that returns information about the available models,
default settings, and system status to clients (e.g., frontend applications).
"""

from fastapi import APIRouter, HTTPException
import logging

# Import constants and getters from Phentrieve's config
from phentrieve import config as phentrieve_config
from api.schemas.config_info_schemas import (
    PhentrieveConfigInfoResponseAPI,
    ModelInfo,
    DefaultParametersAPI,
    ChunkingConfig,
    HPODataStatusAPI,
)
from phentrieve.evaluation.metrics import load_hpo_graph_data

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Configuration & Information"])


@router.get(
    "/info",
    response_model=PhentrieveConfigInfoResponseAPI,
    summary="Get Phentrieve Configuration and Model Information",
    description="Provides lists of available models, default settings, and status of critical data components.",
)
async def get_phentrieve_info():
    """Get Phentrieve configuration and model information.

    This endpoint returns comprehensive information about:
    - Available embedding and reranker models
    - Default operational parameters
    - Text chunking strategies
    - HPO data status

    This is useful for frontend applications to dynamically populate UI elements
    and configure default settings based on the server's configuration.
    """
    logger.info("API: Request for Phentrieve configuration information.")

    try:
        # Build embedding models list
        embedding_models = []
        for model_id in phentrieve_config.BENCHMARK_MODELS:
            description = (
                "Domain-specific biomedical model"
                if "BioLORD" in model_id
                else "Multilingual embedding model"
            )
            if "jina" in model_id.lower():
                description = "Language-specific embedding model (German)"

            embedding_models.append(
                ModelInfo(
                    id=model_id,
                    description=description,
                    is_default=(model_id == phentrieve_config.DEFAULT_MODEL),
                )
            )

        # Available reranker models
        # In a real implementation, this should be sourced from phentrieve_config if available
        available_rerankers = [
            ModelInfo(
                id=phentrieve_config.DEFAULT_RERANKER_MODEL,
                description="Multilingual cross-encoder for reranking",
                is_default=True,
            )
        ]

        # Available monolingual reranker models
        available_mono_rerankers = [
            ModelInfo(
                id=phentrieve_config.DEFAULT_MONOLINGUAL_RERANKER_MODEL,
                description="German-specific reranker model",
                is_default=True,
            )
        ]

        # Default parameters
        default_params = DefaultParametersAPI(
            similarity_threshold=phentrieve_config.DEFAULT_SIMILARITY_THRESHOLD,
            reranker_mode=phentrieve_config.DEFAULT_RERANKER_MODE,
            top_k=phentrieve_config.DEFAULT_TOP_K,
            enable_reranker=phentrieve_config.DEFAULT_ENABLE_RERANKER,
            rerank_candidate_count=phentrieve_config.DEFAULT_RERANK_CANDIDATE_COUNT,
            similarity_formula=phentrieve_config.DEFAULT_SIMILARITY_FORMULA,
            language=phentrieve_config.DEFAULT_LANGUAGE,
        )

        # Get chunking strategies
        chunking_strategies = [
            "simple",  # Simple paragraph -> sentence
            "semantic",  # Paragraph -> sentence -> semantic splitting
            "detailed",  # Paragraph -> sentence -> punctuation -> semantic split
            "sliding_window",  # Sliding window semantic splitting
            "sliding_window_cleaned",  # Sliding window with cleaning
            "sliding_window_punct_cleaned",  # Sliding window with punctuation cleaning
            "sliding_window_punct_conj_cleaned",  # Sliding window with punctuation and conjunction cleaning
        ]

        chunking_config = ChunkingConfig(
            available_strategies=chunking_strategies,
            default_strategy="sliding_window_punct_conj_cleaned",  # Updated default strategy
        )

        # HPO Data Status
        hpo_data_status_dict = {"ancestors_loaded": False, "depths_loaded": False}
        try:
            ancestors, depths = load_hpo_graph_data()
            hpo_data_status_dict["ancestors_loaded"] = bool(ancestors)
            hpo_data_status_dict["depths_loaded"] = bool(depths)
        except Exception as e:
            logger.warning(f"API: Unable to check HPO data status: {e}")

        hpo_data_status = HPODataStatusAPI(**hpo_data_status_dict)
        logger.info(f"API: HPO data status: {hpo_data_status_dict}")

        # Construct and return the full response
        return PhentrieveConfigInfoResponseAPI(
            available_embedding_models=embedding_models,
            default_embedding_model=phentrieve_config.DEFAULT_MODEL,
            available_reranker_models=available_rerankers,
            default_reranker_model=phentrieve_config.DEFAULT_RERANKER_MODEL,
            available_monolingual_reranker_models=available_mono_rerankers,
            default_monolingual_reranker_model=phentrieve_config.DEFAULT_MONOLINGUAL_RERANKER_MODEL,
            default_parameters=default_params,
            chunking_config=chunking_config,
            hpo_data_status=hpo_data_status,
        )

    except Exception as e:
        logger.error(
            f"API Error: Failed to retrieve Phentrieve configuration info: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve configuration information."
        )
