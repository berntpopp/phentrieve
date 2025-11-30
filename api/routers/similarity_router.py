"""
API Router for HPO term similarity calculations.

This module provides API endpoints for calculating semantic similarity
between HPO terms using the Human Phenotype Ontology graph structure.
"""

import logging
from functools import lru_cache
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as FastApiPath

from api.schemas.similarity_schemas import HPOTermSimilarityResponseAPI, LCADetailAPI
from phentrieve.config import DEFAULT_SIMILARITY_FORMULA
from phentrieve.data_processing.document_creator import load_hpo_terms

# Core Phentrieve logic imports
from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
    find_lowest_common_ancestor,
    load_hpo_graph_data,
)
from phentrieve.utils import normalize_id
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter()  # Prefix will be added in main.py


@lru_cache(maxsize=1)  # Cache the HPO labels once loaded
def _get_hpo_label_map_api() -> dict[str, str]:
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
        "API Router: HPO label map cache initialized with %s terms.", len(label_map)
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
            "API: Invalid formula '%s'. Falling back to default '%s'.",
            _sanitize(formula),
            DEFAULT_SIMILARITY_FORMULA,
        )
        formula_enum = SimilarityFormula(DEFAULT_SIMILARITY_FORMULA)

    logger.info(
        "API: Similarity request for T1='%s', T2='%s', Formula='%s'",
        _sanitize(term1_id),
        _sanitize(term2_id),
        _sanitize(formula_enum.value),
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

    response_kwargs: dict[str, Any] = {
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
        sanitized_missing = [_sanitize(m) for m in missing]
        error_detail = f"One or both HPO terms not found in ontology data: {', '.join(sanitized_missing)}. Ensure terms are valid and data is prepared."
        logger.warning("API: %s", _sanitize(error_detail))
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
                id=lca_id, label=lca_label, depth=int(lca_depth)
            )

        return HPOTermSimilarityResponseAPI(**response_kwargs)

    except (
        ValueError
    ) as ve:  # Catch specific errors from core logic if they signal bad input
        logger.warning(
            "API: Value error during similarity calculation: %s", _sanitize(ve)
        )
        response_kwargs["error_message"] = str(ve)
        raise HTTPException(status_code=400, detail=response_kwargs)
    except Exception as e:
        logger.error(
            "API: Unexpected error during similarity calculation for %s, %s: %s",
            _sanitize(norm_term1),
            _sanitize(norm_term2),
            _sanitize(e),
            exc_info=True,
        )
        response_kwargs["error_message"] = "An internal server error occurred."
        raise HTTPException(status_code=500, detail=response_kwargs)
