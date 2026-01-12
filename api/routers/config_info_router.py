"""
API Router for Phentrieve configuration and information.

This module provides an API endpoint that returns information about the available models,
default settings, and system status to clients (e.g., frontend applications).
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas.config_info_schemas import (
    ChunkingConfig,
    DefaultParametersAPI,
    HPODataStatusAPI,
    ModelInfo,
    PhentrieveConfigInfoResponseAPI,
)

# Import constants and getters from Phentrieve's config
from phentrieve import config as phentrieve_config
from phentrieve.evaluation.metrics import load_hpo_graph_data

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Configuration & Information"])


def _get_hpo_metadata() -> dict[str, Optional[str | int]]:
    """Load HPO metadata from database.

    Returns:
        Dictionary with version, download_date, and term_count.
        Values are None if database is unavailable.
    """
    result: dict[str, Optional[str | int]] = {
        "version": None,
        "download_date": None,
        "term_count": None,
    }

    try:
        from phentrieve.config import DEFAULT_HPO_DB_FILENAME
        from phentrieve.data_processing.hpo_database import HPODatabase
        from phentrieve.utils import get_default_data_dir

        # Try to find HPO database
        db_path = get_default_data_dir() / DEFAULT_HPO_DB_FILENAME
        if not db_path.exists():
            # Fallback to relative path for Docker/dev environments
            db_path = Path("data") / DEFAULT_HPO_DB_FILENAME

        if not db_path.exists():
            logger.debug("HPO database not found at %s", db_path)
            return result

        with HPODatabase(db_path) as db:
            result["version"] = db.get_metadata("hpo_version")
            result["download_date"] = db.get_metadata("hpo_download_date")
            result["term_count"] = db.get_term_count()

    except Exception as e:
        logger.warning("Failed to load HPO metadata: %s", e)

    return result


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
        available_rerankers = [
            ModelInfo(
                id=phentrieve_config.DEFAULT_RERANKER_MODEL,
                description="Multilingual cross-encoder for reranking",
                is_default=True,
            )
        ]

        # Default parameters
        default_params = DefaultParametersAPI(
            similarity_threshold=phentrieve_config.DEFAULT_SIMILARITY_THRESHOLD,
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

        # HPO Data Status - load graph data and metadata
        hpo_data_status_dict: dict = {"ancestors_loaded": False, "depths_loaded": False}
        try:
            ancestors, depths = load_hpo_graph_data()
            hpo_data_status_dict["ancestors_loaded"] = bool(ancestors)
            hpo_data_status_dict["depths_loaded"] = bool(depths)
        except Exception as e:
            logger.warning(f"API: Unable to check HPO graph data status: {e}")

        # Load HPO metadata from database
        hpo_metadata = _get_hpo_metadata()
        hpo_data_status_dict["version"] = hpo_metadata["version"]
        hpo_data_status_dict["download_date"] = hpo_metadata["download_date"]
        hpo_data_status_dict["term_count"] = hpo_metadata["term_count"]

        hpo_data_status = HPODataStatusAPI(**hpo_data_status_dict)
        logger.info(f"API: HPO data status: {hpo_data_status_dict}")

        # Construct and return the full response
        return PhentrieveConfigInfoResponseAPI(
            available_embedding_models=embedding_models,
            default_embedding_model=phentrieve_config.DEFAULT_MODEL,
            available_reranker_models=available_rerankers,
            default_reranker_model=phentrieve_config.DEFAULT_RERANKER_MODEL,
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
