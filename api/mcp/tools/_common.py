"""Shared annotated argument types and helpers for the Phentrieve MCP tools."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, Field


def _reject_blank_text(value: str) -> str:
    """Reject empty / whitespace-only text (defect L1: garbage-in was accepted)."""
    if not value or not value.strip():
        raise ValueError("text must not be empty or whitespace-only.")
    return value


# Deterministic extraction defaults to the single best match per phrase. The
# whole top-N candidate list (mutually exclusive HPO siblings) is not co-occurring
# evidence; raise num_results_per_chunk to surface sibling candidates (defect H1).
DEFAULT_EXTRACT_NUM_RESULTS = 1

ResponseMode = Annotated[
    Literal["minimal", "compact", "standard", "full"],
    Field(description="Verbosity / token budget: minimal | compact | standard | full."),
]

# Enumerated chunking strategies. Kept in sync with
# phentrieve.text_processing.config_resolver.KNOWN_CHUNK_STRATEGIES by
# test_mcp_chunk_strategy_enum (a drift guard), since a Literal needs literal
# members at definition time.
ChunkStrategy = Annotated[
    Literal[
        "simple",
        "detailed",
        "semantic",
        "sliding_window",
        "sliding_window_cleaned",
        "sliding_window_punct_cleaned",
        "sliding_window_punct_conj_cleaned",
    ],
    Field(
        description="Predefined chunking strategy. Null = 'simple' (paragraph + "
        "sentence). Semantic/sliding_window strategies need the embedding model."
    ),
]

TextArg = Annotated[
    str,
    Field(
        min_length=1,
        description="Clinical or biomedical research text. Do not submit "
        "identifiable patient data to public demo instances.",
        examples=["progressive muscle weakness and seizures"],
    ),
    AfterValidator(_reject_blank_text),
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

IncludeUnmatchedChunks = Annotated[
    bool,
    Field(
        description="Include processed chunks that produced no HPO matches. Off by "
        "default to save tokens; turn on to inspect coverage."
    ),
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
