"""Shared annotated argument types and helpers for the Phentrieve MCP tools."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_NUM_RESULTS,
    MIN_SIMILARITY_THRESHOLD,
)

ResponseMode = Annotated[
    Literal["minimal", "compact", "standard", "full"],
    Field(description="Verbosity / token budget: minimal | compact | standard | full."),
]

TextArg = Annotated[
    str,
    Field(
        description="Clinical or biomedical research text. Do not submit "
        "identifiable patient data to public demo instances.",
        examples=["progressive muscle weakness and seizures"],
    ),
]

LanguageArg = Annotated[
    str | None,
    Field(
        description="ISO 639-1 language code (en, de, es, fr, nl). Null = autodetect."
    ),
]

NumResults = Annotated[
    int,
    Field(ge=1, le=50, description="Maximum HPO candidates to return."),
]

SimilarityThreshold = Annotated[
    float,
    Field(ge=0.0, le=1.0, description="Minimum retrieval similarity to include."),
]

IncludeDetails = Annotated[
    bool,
    Field(
        description="Include HPO definitions and synonyms (subject to response_mode)."
    ),
]

IncludeChunkPositions = Annotated[
    bool,
    Field(description="Include source character offsets for evidence chunks."),
]

NumResultsPerChunk = Annotated[
    int,
    Field(ge=1, le=50, description="Maximum HPO candidates per chunk."),
]

ChunkRetrievalThreshold = Annotated[
    float,
    Field(ge=0.0, le=1.0, description="Minimum chunk-level retrieval similarity."),
]

HpoIdArg = Annotated[
    str,
    Field(
        pattern=r"^HP:\d{7}$",
        description="HPO term id, e.g. HP:0001250.",
        examples=["HP:0001250"],
    ),
]

SimilarityFormulaArg = Annotated[
    Literal["hybrid", "simple_resnik_like"],
    Field(description="Ontology similarity formula."),
]

ResearchAck = Annotated[
    bool,
    Field(
        description="Set true to acknowledge research-use-only terms. Required by "
        "public-hosted instances before extraction tools will run.",
    ),
]

# Re-exported defaults so tool signatures can reference them.
DEFAULT_NUM_RESULTS = DEFAULT_NUM_RESULTS
DEFAULT_CHUNK_RETRIEVAL_THRESHOLD = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
MIN_SIMILARITY_THRESHOLD = MIN_SIMILARITY_THRESHOLD


def require_research_ack(acknowledged: bool) -> None:
    """Raise McpToolError when a hosted instance needs an unmet research-use ack."""
    from api.mcp.envelope import McpToolError
    from api.research_use import is_research_ack_required

    if is_research_ack_required() and not acknowledged:
        raise McpToolError(
            "invalid_input",
            "This public hosted instance is research-use-only and requires "
            "research_use_acknowledged=true for extraction tools. Present the "
            "research-use limitation to the user, then retry with the flag set.",
            details={"field": "research_use_acknowledged"},
        )
