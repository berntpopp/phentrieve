"""
API Router for HPO term similarity calculations.

This module provides API endpoints for calculating semantic similarity
between HPO terms using the Human Phenotype Ontology graph structure.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as FastApiPath

from api.schemas.similarity_schemas import HPOTermSimilarityResponseAPI, LCADetailAPI
from phentrieve.config import DEFAULT_SIMILARITY_FORMULA
from phentrieve.evaluation.metrics import SimilarityFormula

# Shared similarity computation (also used by the MCP compare tool, #415)
from phentrieve.evaluation.term_similarity import (
    HPODataUnavailableError,
    HPOTermNotFoundError,
    compute_term_similarity,
    load_hpo_labels,
)
from phentrieve.utils import normalize_id
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter()  # Prefix will be added in main.py


@router.get(
    "/{term1_id}/{term2_id}",
    response_model=HPOTermSimilarityResponseAPI,
    operation_id="calculate_term_similarity",
    summary="Calculate semantic similarity between HPO terms",
    description="Calculate semantic similarity between two HPO terms using ontology structure.",
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
    formula: str | None = Query(
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

    norm_term1 = normalize_id(term1_id)
    norm_term2 = normalize_id(term2_id)

    try:
        result = compute_term_similarity(
            norm_term1, norm_term2, formula_enum, with_lca=True
        )
    except HPODataUnavailableError:
        logger.error(
            "API Error: HPO graph data (ancestors or depths) is critically unavailable."
        )
        raise HTTPException(
            status_code=503,
            detail="Ontology data is currently unavailable. Please try again later or contact support.",
        )
    except HPOTermNotFoundError as nf:
        sanitized_missing = [_sanitize(m) for m in nf.missing]
        error_detail = f"One or both HPO terms not found in ontology data: {', '.join(sanitized_missing)}. Ensure terms are valid and data is prepared."
        logger.warning("API: %s", _sanitize(error_detail))
        # For term not found, 404 is appropriate
        raise HTTPException(
            status_code=404,
            detail=_error_detail(norm_term1, norm_term2, formula_enum, error_detail),
        )
    except ValueError as ve:  # Core logic signalling bad input
        logger.warning(
            "API: Value error during similarity calculation: %s", _sanitize(ve)
        )
        raise HTTPException(
            status_code=400,
            detail=_error_detail(norm_term1, norm_term2, formula_enum, str(ve)),
        )
    except Exception as e:
        logger.error(
            "API: Unexpected error during similarity calculation for %s, %s: %s",
            _sanitize(norm_term1),
            _sanitize(norm_term2),
            _sanitize(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                norm_term1,
                norm_term2,
                formula_enum,
                "An internal server error occurred.",
            ),
        )

    lca_details = None
    if result.lca_id is not None:
        lca_details = LCADetailAPI(
            id=result.lca_id, label=result.lca_label, depth=result.lca_depth
        )
    return HPOTermSimilarityResponseAPI(
        term1_id=result.term1_id,
        term1_label=result.term1_label,
        term2_id=result.term2_id,
        term2_label=result.term2_label,
        formula_used=result.formula,
        similarity_score=result.similarity_score,
        lca_details=lca_details,
    )


def _error_detail(
    term1_id: str,
    term2_id: str,
    formula: SimilarityFormula,
    error_message: str,
) -> dict[str, Any]:
    """Error payload: the response fields known so far plus ``error_message``."""
    labels = load_hpo_labels([term1_id, term2_id])
    return {
        "term1_id": term1_id,
        "term1_label": labels.get(term1_id),
        "term2_id": term2_id,
        "term2_label": labels.get(term2_id),
        # .value: an Enum in the detail dict is not JSON serializable
        "formula_used": formula.value,
        "similarity_score": 0.0,
        "error_message": error_message,
    }
