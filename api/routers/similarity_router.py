"""
API Router for HPO term similarity calculations.

This module provides API endpoints for calculating semantic similarity
between HPO terms using the Human Phenotype Ontology graph structure.
"""

from fastapi import APIRouter, HTTPException, Path as FastApiPath, Query
from typing import Optional, Dict
import logging
from functools import lru_cache

# Core Phentrieve logic imports
from phentrieve.evaluation.metrics import (
    calculate_semantic_similarity,
    find_lowest_common_ancestor,
    load_hpo_graph_data,
    SimilarityFormula,
)
from phentrieve.utils import normalize_id
from phentrieve.data_processing.document_creator import load_hpo_terms
from api.schemas.similarity_schemas import HPOTermSimilarityResponseAPI, LCADetailAPI
from phentrieve.config import DEFAULT_SIMILARITY_FORMULA

logger = logging.getLogger(__name__)
router = APIRouter()  # Prefix will be added in main.py


@lru_cache(maxsize=1)  # Cache the HPO labels once loaded
def _get_hpo_label_map_api() -> Dict[str, str]:
    """
    Initialize and cache HPO label mapping for the API.

    Returns:
        Dictionary mapping HPO IDs to their English labels
    """
    logger.info("API Router: Initializing HPO label map cache...")
    # This is a synchronous call. If load_hpo_terms() is very slow,
    # consider FastAPI's application startup events or a dependency injection for this.
    hpo_terms_data = load_hpo_terms()
    if not hpo_terms_data:
        logger.warning("API Router: No HPO terms data loaded for labels.")
        return {}
    label_map = {
        term_data["id"]: term_data["label"]
        for term_data in hpo_terms_data
        if term_data.get("id") and term_data.get("label")
    }
    logger.info(
        f"API Router: HPO label map cache initialized with {len(label_map)} terms."
    )
    return label_map


# Ensure HPO graph data is pre-loaded or loaded efficiently on first use by the API
# The load_hpo_graph_data function itself uses caching.
load_hpo_graph_data()


@router.get(
    "/{term1_id}/{term2_id}",
    response_model=HPOTermSimilarityResponseAPI,
    summary="Calculate Semantic Similarity Between Two HPO Terms",
    description="Calculates the semantic similarity score between two HPO terms using either the 'hybrid' or 'simple_resnik_like' formula. Also provides Lowest Common Ancestor (LCA) details.",
)
async def get_hpo_term_similarity(
    term1_id: str = FastApiPath(
        ...,
        description="The first HPO Term ID (e.g., HP:0001197). Must be in HP:####### format.",
        examples=["HP:0001197"],
        pattern=r"^HP:\d{7}$",  # Basic HPO ID pattern validation
    ),
    term2_id: str = FastApiPath(
        ...,
        description="The second HPO Term ID (e.g., HP:0000750). Must be in HP:####### format.",
        examples=["HP:0000750"],
        pattern=r"^HP:\d{7}$",
    ),
    formula: Optional[str] = Query(
        default=DEFAULT_SIMILARITY_FORMULA,
        description="The semantic similarity formula to apply ('hybrid' or 'simple_resnik_like').",
    ),
):
    """
    Calculate semantic similarity between two HPO terms.

    This endpoint determines how semantically related two phenotypic terms are
    based on their positions in the HPO ontology hierarchy. The similarity score
    ranges from 0.0 (completely unrelated) to 1.0 (identical terms).
    """
    # Convert string formula parameter to SimilarityFormula enum
    try:
        formula_enum = SimilarityFormula(formula)
    except ValueError:
        logger.warning(
            f"API: Invalid formula '{formula}'. Falling back to default '{DEFAULT_SIMILARITY_FORMULA}'."
        )
        formula_enum = SimilarityFormula(DEFAULT_SIMILARITY_FORMULA)

    logger.info(
        f"API: Similarity request for T1='{term1_id}', T2='{term2_id}', Formula='{formula_enum.value}'"
    )

    ancestors, depths = load_hpo_graph_data()  # Relies on its internal cache
    if not ancestors or not depths:
        logger.error(
            "API Error: HPO graph data (ancestors or depths) is critically unavailable."
        )
        raise HTTPException(
            status_code=503,
            detail="Ontology data is currently unavailable. Please try again later or contact support.",
        )

    labels = _get_hpo_label_map_api()  # Relies on its internal cache

    norm_term1 = normalize_id(term1_id)
    norm_term2 = normalize_id(term2_id)

    label1 = labels.get(norm_term1)
    label2 = labels.get(norm_term2)

    response_kwargs = {
        "term1_id": norm_term1,
        "term1_label": label1,
        "term2_id": norm_term2,
        "term2_label": label2,
        "formula_used": formula_enum,  # FastAPI will use enum's value
        "similarity_score": 0.0,  # Default in case of early exit
    }

    # Check if terms exist in the ontology
    if norm_term1 not in depths or norm_term2 not in depths:
        missing = []
        if norm_term1 not in depths:
            missing.append(norm_term1)
        if norm_term2 not in depths:
            missing.append(norm_term2)
        error_detail = f"One or both HPO terms not found in ontology data: {', '.join(missing)}. Ensure terms are valid and data is prepared."
        logger.warning(f"API: {error_detail}")
        response_kwargs["error_message"] = error_detail
        # For term not found, 404 is appropriate
        raise HTTPException(status_code=404, detail=response_kwargs)

    try:
        similarity_score = calculate_semantic_similarity(
            norm_term1, norm_term2, formula=formula_enum
        )
        response_kwargs["similarity_score"] = similarity_score

        lca_id, lca_depth = find_lowest_common_ancestor(
            norm_term1, norm_term2, ancestors_dict=ancestors
        )

        if lca_id and lca_depth != -1:
            lca_label = labels.get(lca_id)
            response_kwargs["lca_details"] = LCADetailAPI(
                id=lca_id, label=lca_label, depth=lca_depth
            )

        return HPOTermSimilarityResponseAPI(**response_kwargs)

    except (
        ValueError
    ) as ve:  # Catch specific errors from core logic if they signal bad input
        logger.warning(f"API: Value error during similarity calculation: {ve}")
        response_kwargs["error_message"] = str(ve)
        raise HTTPException(status_code=400, detail=response_kwargs)
    except Exception as e:
        logger.error(
            f"API: Unexpected error during similarity calculation for {norm_term1}, {norm_term2}: {e}",
            exc_info=True,
        )
        response_kwargs["error_message"] = "An internal server error occurred."
        raise HTTPException(status_code=500, detail=response_kwargs)
