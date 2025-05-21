"""
Pydantic models for the HPO term similarity API.

This module defines the data structures used for the similarity API endpoint
responses, ensuring consistent and well-documented API contracts.
"""

from typing import Optional
from pydantic import BaseModel, Field
from phentrieve.evaluation.metrics import (
    SimilarityFormula,
)  # For type safety and OpenAPI docs


class LCADetailAPI(BaseModel):
    """Details of the Lowest Common Ancestor between two HPO terms."""

    id: Optional[str] = Field(
        default=None,
        description="HPO ID of the Lowest Common Ancestor (LCA). Example: HP:0000005",
    )
    label: Optional[str] = Field(
        default=None,
        description="English label of the LCA. Example: Mode of inheritance",
    )
    depth: Optional[int] = Field(
        default=None, description="Depth of the LCA in the HPO graph. Example: 3"
    )


class HPOTermSimilarityResponseAPI(BaseModel):
    """Response model for the HPO term similarity calculation API endpoint."""

    term1_id: str = Field(
        ...,
        description="First HPO Term ID provided in the request (normalized).",
        example="HP:0001197",
    )
    term1_label: Optional[str] = Field(
        default=None,
        description="English label for HPO Term 1. Example: Abnormality of the genitourinary system",
    )
    term2_id: str = Field(
        ...,
        description="Second HPO Term ID provided in the request (normalized).",
        example="HP:0000750",
    )
    term2_label: Optional[str] = Field(
        default=None,
        description="English label for HPO Term 2. Example: Delayed speech and language development",
    )
    formula_used: SimilarityFormula = Field(
        ..., description="The semantic similarity formula applied for the calculation."
    )
    similarity_score: float = Field(
        ...,
        description="Calculated semantic similarity score, ranging from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
    lca_details: Optional[LCADetailAPI] = Field(
        default=None,
        description="Details of the Lowest Common Ancestor, if one is found and applicable.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Provides details if an error occurred during processing (e.g., term not found).",
    )

    class Config:
        use_enum_values = (
            True  # Ensures enum values (strings) are used in the schema and response
        )
