"""
Pydantic models for the Phentrieve configuration information API.

This module defines the data structures used for the configuration/info API
endpoint responses, ensuring consistent and well-documented API contracts.
"""

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
                    "top_k": 10,
                    "similarity_formula": "hybrid",
                    "language": "en",
                }
            ]
        }
    )

    similarity_threshold: float = Field(
        description="Minimum similarity score for retrieval"
    )
    top_k: int = Field(description="Number of top results to return")
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
    """Status and metadata of HPO data."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ancestors_loaded": True,
                    "depths_loaded": True,
                    "version": "v2025-03-03",
                    "download_date": "2025-12-08T10:12:25+00:00",
                    "term_count": 19534,
                }
            ]
        }
    )

    ancestors_loaded: bool = Field(
        description="Whether the HPO ancestor data is loaded"
    )
    depths_loaded: bool = Field(description="Whether the HPO depth data is loaded")
    version: str | None = Field(
        default=None, description="HPO ontology version (e.g., 'v2025-03-03')"
    )
    download_date: str | None = Field(
        default=None, description="ISO timestamp when HPO data was downloaded"
    )
    term_count: int | None = Field(
        default=None, description="Total number of HPO terms in the database"
    )


class PublicLLMTargetAPI(BaseModel):
    """Read-only public LLM target advertised by the server."""

    provider: str = Field(description="Server-owned LLM provider identifier")
    model: str = Field(description="Server-owned LLM model identifier")
    display_name: str = Field(description="Human-readable LLM target name")


class PublicLLMCapabilitiesAPI(BaseModel):
    """Read-only public LLM capabilities advertised by the server."""

    default_llm_provider: str = Field(description="Default public LLM provider")
    default_llm_model: str = Field(description="Default public LLM model")
    configured_llm_models: list[str] = Field(
        description="Read-only list of configured public LLM models"
    )
    allowed_llm_targets: list[PublicLLMTargetAPI] = Field(
        description="Read-only public LLM targets"
    )
    llm_modes: list[str] = Field(description="Publicly supported LLM extraction modes")
    research_use_only: bool = Field(
        description="Whether public LLM use is restricted to research use"
    )


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
                    "default_parameters": {
                        "similarity_threshold": 0.1,
                        "top_k": 10,
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
                        "version": "v2025-03-03",
                        "download_date": "2025-12-08T10:12:25+00:00",
                        "term_count": 19534,
                    },
                }
            ]
        }
    )

    available_embedding_models: list[ModelInfo] = Field(
        description="List of available embedding models"
    )
    default_embedding_model: str = Field(description="Default embedding model ID")
    default_parameters: DefaultParametersAPI = Field(
        description="Default operational parameters"
    )
    chunking_config: ChunkingConfig = Field(description="Text chunking configuration")
    hpo_data_status: HPODataStatusAPI = Field(description="Status of HPO data loading")
    public_llm_capabilities: PublicLLMCapabilitiesAPI = Field(
        description=(
            "Read-only public LLM configuration. Clients cannot mutate these values."
        )
    )
