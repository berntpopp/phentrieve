"""Profile data model and resolution for Phentrieve CLI configuration profiles.

Implements Spec A (.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Forward reference - defined in Plan B (phentrieve/retrieval/adaptive_rechunker.py).
# Until Plan B lands, use a permissive dict shape.
AdaptiveRechunkingProfileBlock = dict[str, Any]


class Profile(BaseModel):
    """A named bundle of CLI option defaults loaded from phentrieve.yaml.

    All fields are optional. None means "this profile does not preset this
    option" - the resolution chain falls through to YAML / fallback constants.
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    command: str | None = None  # e.g. "text process", "query", "text interactive"

    # Shared option keys
    language: str | None = None
    model_name: str | None = None
    semantic_chunker_model: str | None = None
    retrieval_model: str | None = None
    similarity_threshold: float | None = None
    chunk_retrieval_threshold: float | None = None
    aggregated_term_confidence: float | None = None
    num_results: int | None = None
    chunking_strategy: str | None = None
    window_size: int | None = None
    step_size: int | None = None
    split_threshold: float | None = None
    min_segment_length: int | None = None
    output_format: str | None = None
    assertion_preference: str | None = None
    no_assertion_detection: bool | None = None
    multi_vector: bool | None = None
    aggregation_strategy: str | None = None
    extraction_backend: Literal["standard", "llm"] | None = None

    # Adaptive rechunking block - strict shape defined by Plan B; permissive here.
    adaptive_rechunking: AdaptiveRechunkingProfileBlock | None = None


class ProfilesFile(BaseModel):
    """Top-level YAML model.

    Only the `profiles:` key is consumed by Plan A; other top-level keys
    (data_dir, default_model, etc.) are ignored at this layer.
    """

    model_config = ConfigDict(extra="ignore")
    profiles: dict[str, Profile] = Field(default_factory=dict)
