from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from api.mcp.tools import (
    ExtractHpoTermsLlmRequest,
    ExtractHpoTermsRequest,
)
from phentrieve.text_processing.full_text_service import run_full_text_service

RESEARCH_USE_INSTRUCTIONS = (
    "Phentrieve maps clinical or biomedical research text to Human Phenotype "
    "Ontology term suggestions for research, benchmarking, education, and "
    "research data standardisation only. It is not for diagnosis, treatment, "
    "triage, patient management, or clinical decision support. Do not submit "
    "identifiable patient data to public demo instances. Use the LLM tool only "
    "when the user asks for research-only LLM-assisted full-text extraction or "
    "document-level phenotype annotation."
)


def create_phentrieve_mcp() -> FastMCP:
    mcp = FastMCP(
        name="phentrieve",
        instructions=RESEARCH_USE_INSTRUCTIONS,
    )

    @mcp.tool(
        name="phentrieve.extract_hpo_terms",
        title="Extract HPO Terms",
    )
    def extract_hpo_terms(request: ExtractHpoTermsRequest) -> dict[str, Any]:
        """Use this when research text should be mapped to HPO term suggestions without LLM calls. Research use only; not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return run_full_text_service(
            text=request.text,
            extraction_backend="standard",
            language=request.language,
            include_details=request.include_details,
            include_positions=request.include_chunk_positions,
            num_results_per_chunk=request.num_results_per_chunk,
            chunk_retrieval_threshold=request.chunk_retrieval_threshold,
        )

    @mcp.tool(
        name="phentrieve.extract_hpo_terms_llm",
        title="Extract HPO Terms With LLM",
    )
    def extract_hpo_terms_llm(request: ExtractHpoTermsLlmRequest) -> dict[str, Any]:
        """Use this when research-only full-text LLM extraction should identify phenotype mentions and map them to grounded HPO term suggestions. Not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return run_full_text_service(
            text=request.text,
            extraction_backend="llm",
            language=request.language,
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            llm_base_url=request.llm_base_url,
            llm_mode=request.llm_mode,
            llm_internal_mode=request.llm_internal_mode,
            include_details=request.include_details,
            include_positions=request.include_chunk_positions,
        )

    @mcp.tool(
        name="phentrieve.get_server_capabilities",
        title="Get Phentrieve Capabilities",
    )
    def get_server_capabilities() -> dict[str, Any]:
        """Use this when a client needs supported languages, backends, examples, and research-use limits."""
        return {
            "server": "phentrieve",
            "transports": ["streamable_http"],
            "extraction_backends": ["standard", "llm"],
            "llm_modes": ["two_phase"],
            "llm_internal_modes": [
                "whole_document_grounded",
                "whole_document_legacy",
            ],
            "languages": ["en", "de", "es", "fr", "nl"],
            "intended_use": (
                "Research, benchmarking, education, and research data "
                "standardisation only."
            ),
            "prohibited_uses": [
                "diagnosis",
                "treatment",
                "triage",
                "patient management",
                "clinical decision support",
                "identifiable patient data in public demo instances",
            ],
        }

    return mcp
