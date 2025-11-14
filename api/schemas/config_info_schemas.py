"""
Pydantic models for the Phentrieve configuration information API.

This module defines the data structures used for the configuration/info API
endpoint responses, ensuring consistent and well-documented API contracts.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """Information about a single model available in Phentrieve."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "FremyCompany/BioLORD-2023-M",
                    "description": "Domain-specific biomedical model",
                    "is_default": True,
                }
            ]
        }
    )

    id: str = Field(description="Model identifier")
    description: str = Field(description="Brief description of the model")
    is_default: bool = Field(description="Whether this is the default model")


class DefaultParametersAPI(BaseModel):
    """Default operational parameters used by Phentrieve."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "similarity_threshold": 0.1,
                    "reranker_mode": "cross-lingual",
                    "top_k": 10,
                    "enable_reranker": False,
                    "rerank_candidate_count": 30,
                    "similarity_formula": "hybrid",
                    "language": "en",
                }
            ]
        }
    )

    similarity_threshold: float = Field(
        description="Minimum similarity score for retrieval"
    )
    reranker_mode: str = Field(
        description="Reranker mode (cross-lingual or monolingual)"
    )
    top_k: int = Field(description="Number of top results to return")
    enable_reranker: bool = Field(description="Whether reranking is enabled by default")
    rerank_candidate_count: int = Field(description="Number of candidates to rerank")
    similarity_formula: str = Field(description="Formula for calculating similarity")
    language: str = Field(description="Default language for queries")


class ChunkingConfig(BaseModel):
    """Available text chunking strategies."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "available_strategies": [
                        "simple",
                        "semantic",
                        "detailed",
                        "sliding_window",
                    ],
                    "default_strategy": "sliding_window",
                }
            ]
        }
    )

    available_strategies: list[str] = Field(description="Available chunking strategies")
    default_strategy: str = Field(description="Default chunking strategy")


class HPODataStatusAPI(BaseModel):
    """Status of HPO data files."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"ancestors_loaded": True, "depths_loaded": True}]
        }
    )

    ancestors_loaded: bool = Field(
        description="Whether the HPO ancestor data is loaded"
    )
    depths_loaded: bool = Field(description="Whether the HPO depth data is loaded")


class PhentrieveConfigInfoResponseAPI(BaseModel):
    """Response model for the Phentrieve configuration information API endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "available_embedding_models": [
                        {
                            "id": "FremyCompany/BioLORD-2023-M",
                            "description": "Domain-specific biomedical model",
                            "is_default": True,
                        }
                    ],
                    "default_embedding_model": "FremyCompany/BioLORD-2023-M",
                    "available_reranker_models": [
                        {
                            "id": "FremyCompany/BioLORD-2023-M",
                            "description": "Domain-specific biomedical model",
                            "is_default": True,
                        }
                    ],
                    "default_reranker_model": "FremyCompany/BioLORD-2023-M",
                    "available_monolingual_reranker_models": [
                        {
                            "id": "FremyCompany/BioLORD-2023-M",
                            "description": "Domain-specific biomedical model",
                            "is_default": True,
                        }
                    ],
                    "default_monolingual_reranker_model": "FremyCompany/BioLORD-2023-M",
                    "default_parameters": {
                        "similarity_threshold": 0.1,
                        "reranker_mode": "cross-lingual",
                        "top_k": 10,
                        "enable_reranker": False,
                        "rerank_candidate_count": 30,
                        "similarity_formula": "hybrid",
                        "language": "en",
                    },
                    "chunking_config": {
                        "available_strategies": [
                            "simple",
                            "semantic",
                            "detailed",
                            "sliding_window",
                        ],
                        "default_strategy": "sliding_window",
                    },
                    "hpo_data_status": {
                        "ancestors_loaded": True,
                        "depths_loaded": True,
                    },
                }
            ]
        }
    )

    available_embedding_models: list[ModelInfo] = Field(
        description="List of available embedding models"
    )
    default_embedding_model: str = Field(description="Default embedding model ID")
    available_reranker_models: list[ModelInfo] = Field(
        description="List of available reranker models"
    )
    default_reranker_model: str = Field(description="Default reranker model ID")
    available_monolingual_reranker_models: list[ModelInfo] = Field(
        description="List of available monolingual reranker models"
    )
    default_monolingual_reranker_model: Optional[str] = Field(
        description="Default monolingual reranker model ID"
    )
    default_parameters: DefaultParametersAPI = Field(
        description="Default operational parameters"
    )
    chunking_config: ChunkingConfig = Field(description="Text chunking configuration")
    hpo_data_status: HPODataStatusAPI = Field(description="Status of HPO data loading")
