"""Retrieval and extraction tools: search, extract (standard + LLM), chunk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import anyio

from api.mcp.annotations import READ_ONLY_OPEN_WORLD
from api.mcp.envelope import McpErrorContext, run_mcp_tool
from api.mcp.next_commands import after_chunk, after_extract, after_search
from api.mcp.projection import project_extract_payload
from api.mcp.resources import recommended_citation
from api.mcp.schemas import CHUNK_SCHEMA, EXTRACT_SCHEMA, SEARCH_SCHEMA
from api.mcp.service_adapters import (
    chunk_text_service,
    extract_hpo_terms_llm_service,
    extract_hpo_terms_service,
    search_hpo_terms_service,
)
from api.mcp.shaping import apply_response_mode, enforce_budget, resolve_mode
from api.mcp.tools._common import (
    DEFAULT_EXTRACT_NUM_RESULTS,
    ChunkRetrievalThreshold,
    IncludeChunkPositions,
    IncludeDetails,
    IncludeUnmatchedChunks,
    LanguageArg,
    NumResults,
    NumResultsPerChunk,
    ResearchAck,
    ResponseMode,
    SimilarityThreshold,
    TextArg,
    require_research_ack,
)
from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_NUM_RESULTS,
    MIN_SIMILARITY_THRESHOLD,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP


def _maybe_citation(meta: dict[str, Any], mode: str) -> None:
    if mode in ("standard", "full"):
        meta["recommended_citation"] = recommended_citation()


def register_retrieval_tools(mcp: FastMCP) -> None:
    """Register search, extract, extract_llm, and chunk tools."""

    @mcp.tool(
        name="phentrieve_search_hpo_terms",
        title="Search HPO Terms",
        annotations=READ_ONLY_OPEN_WORLD,
        output_schema=SEARCH_SCHEMA,
        description=(
            "Map a short research phenotype phrase to ranked HPO candidates by "
            "dense retrieval. Use for single phrases, not documents. Research use "
            "only; not for diagnosis, treatment, triage, patient management, or "
            "clinical decision support. Signature: "
            "phentrieve_search_hpo_terms(text, language=, num_results=, "
            "similarity_threshold=, include_details=, response_mode=)."
        ),
    )
    async def search_hpo_terms(
        text: TextArg,
        language: LanguageArg = None,
        num_results: NumResults = DEFAULT_NUM_RESULTS,
        similarity_threshold: SimilarityThreshold = MIN_SIMILARITY_THRESHOLD,
        include_details: IncludeDetails = True,
        response_mode: ResponseMode = "compact",
    ) -> dict[str, Any]:
        mode = resolve_mode(response_mode)

        async def call() -> dict[str, Any]:
            raw = await search_hpo_terms_service(
                text=text,
                language=language,
                num_results=num_results,
                similarity_threshold=similarity_threshold,
                include_details=include_details,
            )
            shaped = apply_response_mode(raw, mode)
            shaped, trunc = enforce_budget(shaped, mode, list_field="results")
            meta: dict[str, Any] = {
                "next_commands": after_search(shaped.get("results", []))
            }
            if trunc:
                meta["truncated"] = trunc
            _maybe_citation(meta, mode)
            shaped["_meta"] = meta
            return shaped

        return await run_mcp_tool(
            "phentrieve_search_hpo_terms",
            call,
            response_mode=mode,
            context=McpErrorContext("phentrieve_search_hpo_terms", {"text": text}),
        )

    @mcp.tool(
        name="phentrieve_extract_hpo_terms",
        title="Extract HPO Terms",
        annotations=READ_ONLY_OPEN_WORLD,
        output_schema=EXTRACT_SCHEMA,
        description=(
            "Deterministic retrieval-backed extraction of HPO terms from "
            "multi-sentence text (chunking + assertion detection + aggregation), "
            "no LLM calls. For full abstracts or syndrome/eponym-heavy text prefer "
            "phentrieve_extract_hpo_terms_llm. Research use only; not for clinical "
            "use. Signature: phentrieve_extract_hpo_terms(text, language=, "
            "include_details=, include_chunk_positions=, num_results_per_chunk=, "
            "chunk_retrieval_threshold=, research_use_acknowledged=, response_mode=)."
        ),
    )
    async def extract_hpo_terms(
        text: TextArg,
        language: LanguageArg = None,
        include_details: IncludeDetails = False,
        include_chunk_positions: IncludeChunkPositions = True,
        include_unmatched_chunks: IncludeUnmatchedChunks = False,
        num_results_per_chunk: NumResultsPerChunk = DEFAULT_EXTRACT_NUM_RESULTS,
        chunk_retrieval_threshold: ChunkRetrievalThreshold = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
        research_use_acknowledged: ResearchAck = False,
        response_mode: ResponseMode = "compact",
    ) -> dict[str, Any]:
        mode = resolve_mode(response_mode)

        async def call() -> dict[str, Any]:
            require_research_ack(research_use_acknowledged)
            raw = await anyio.to_thread.run_sync(
                lambda: extract_hpo_terms_service(
                    text=text,
                    language=language,
                    include_details=include_details,
                    include_chunk_positions=include_chunk_positions,
                    num_results_per_chunk=num_results_per_chunk,
                    chunk_retrieval_threshold=chunk_retrieval_threshold,
                )
            )
            raw = project_extract_payload(
                raw, include_unmatched_chunks=include_unmatched_chunks
            )
            shaped = apply_response_mode(raw, mode)
            shaped, trunc = enforce_budget(
                shaped, mode, list_field="aggregated_hpo_terms"
            )
            meta: dict[str, Any] = {
                "next_commands": after_extract(shaped.get("aggregated_hpo_terms", []))
            }
            if trunc:
                meta["truncated"] = trunc
            _maybe_citation(meta, mode)
            shaped["_meta"] = meta
            return shaped

        return await run_mcp_tool(
            "phentrieve_extract_hpo_terms",
            call,
            response_mode=mode,
            context=McpErrorContext("phentrieve_extract_hpo_terms", {"text": text}),
        )

    @mcp.tool(
        name="phentrieve_extract_hpo_terms_llm",
        title="Extract HPO Terms With LLM",
        annotations=READ_ONLY_OPEN_WORLD,
        output_schema=EXTRACT_SCHEMA,
        description=(
            "LLM-assisted two-phase extraction for full abstracts, publication-"
            "style annotation, syndrome/eponym-heavy text, and review workflows. "
            "Uses only the server-configured LLM target; clients cannot override "
            "provider/model/base URL. Subject to a daily quota in hosted mode; set "
            "allow_standard_fallback=true to fall back to deterministic extraction. "
            "Research use only; not for clinical use. Signature: "
            "phentrieve_extract_hpo_terms_llm(text, language=, llm_internal_mode=, "
            "allow_standard_fallback=, include_details=, include_chunk_positions=, "
            "num_results_per_chunk=, chunk_retrieval_threshold=, "
            "research_use_acknowledged=, response_mode=)."
        ),
    )
    async def extract_hpo_terms_llm(
        text: TextArg,
        language: LanguageArg = None,
        llm_internal_mode: Literal[
            "whole_document_grounded", "whole_document_legacy"
        ] = "whole_document_grounded",
        allow_standard_fallback: bool = False,
        include_details: IncludeDetails = False,
        include_chunk_positions: IncludeChunkPositions = True,
        include_unmatched_chunks: IncludeUnmatchedChunks = False,
        num_results_per_chunk: NumResultsPerChunk = DEFAULT_NUM_RESULTS,
        chunk_retrieval_threshold: ChunkRetrievalThreshold = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
        research_use_acknowledged: ResearchAck = False,
        response_mode: ResponseMode = "compact",
    ) -> dict[str, Any]:
        mode = resolve_mode(response_mode)

        async def call() -> dict[str, Any]:
            require_research_ack(research_use_acknowledged)
            raw = await anyio.to_thread.run_sync(
                lambda: extract_hpo_terms_llm_service(
                    text=text,
                    language=language,
                    include_details=include_details,
                    include_chunk_positions=include_chunk_positions,
                    num_results_per_chunk=num_results_per_chunk,
                    chunk_retrieval_threshold=chunk_retrieval_threshold,
                    llm_mode="two_phase",
                    llm_internal_mode=llm_internal_mode,
                    allow_standard_fallback=allow_standard_fallback,
                )
            )
            raw = project_extract_payload(
                raw, include_unmatched_chunks=include_unmatched_chunks
            )
            shaped = apply_response_mode(raw, mode)
            shaped, trunc = enforce_budget(
                shaped, mode, list_field="aggregated_hpo_terms"
            )
            meta: dict[str, Any] = {
                "next_commands": after_extract(shaped.get("aggregated_hpo_terms", []))
            }
            if trunc:
                meta["truncated"] = trunc
            _maybe_citation(meta, mode)
            shaped["_meta"] = meta
            return shaped

        return await run_mcp_tool(
            "phentrieve_extract_hpo_terms_llm",
            call,
            response_mode=mode,
            context=McpErrorContext("phentrieve_extract_hpo_terms_llm", {"text": text}),
        )

    @mcp.tool(
        name="phentrieve_chunk_text",
        title="Chunk Text",
        annotations=READ_ONLY_OPEN_WORLD,
        output_schema=CHUNK_SCHEMA,
        description=(
            "Split text into chunks without HPO retrieval, for clients driving "
            "their own loop. Defaults to the 'simple' (paragraph + sentence) "
            "strategy. Research use only. Signature: "
            "phentrieve_chunk_text(text, language=, strategy=, response_mode=)."
        ),
    )
    async def chunk_text(
        text: TextArg,
        language: LanguageArg = None,
        strategy: str | None = None,
        response_mode: ResponseMode = "compact",
    ) -> dict[str, Any]:
        mode = resolve_mode(response_mode)

        async def call() -> dict[str, Any]:
            raw = await anyio.to_thread.run_sync(
                lambda: chunk_text_service(
                    text=text, language=language, strategy=strategy
                )
            )
            shaped = apply_response_mode(raw, mode)
            shaped, trunc = enforce_budget(shaped, mode, list_field="chunks")
            meta: dict[str, Any] = {
                "next_commands": after_chunk(shaped.get("chunks", []))
            }
            if trunc:
                meta["truncated"] = trunc
            shaped["_meta"] = meta
            return shaped

        return await run_mcp_tool(
            "phentrieve_chunk_text",
            call,
            response_mode=mode,
            context=McpErrorContext("phentrieve_chunk_text", {"text": text}),
        )
