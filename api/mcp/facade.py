"""MCP facade for Phentrieve (Streamable HTTP, FastMCP v3).

``create_phentrieve_mcp`` assembles a FastMCP server with the eight read-only
Phentrieve tools, the ``phentrieve://`` resources, the research prompts, and the
argument-validation middleware. Tools reuse the ``phentrieve.*`` service layer
directly via api.mcp.service_adapters; the envelope/_meta and error handling are
provided by api.mcp.envelope.run_mcp_tool.
"""

from __future__ import annotations

from fastmcp import FastMCP

from api.mcp.middleware import ArgValidationMiddleware
from api.mcp.prompts import (
    annotate_research_text_prompt,
    extract_research_case_phenotypes_prompt,
    review_hpo_research_annotations_prompt,
)
from api.mcp.resources import SERVER_INSTRUCTIONS, register_resources
from api.mcp.tools import (
    register_discovery_tools,
    register_phenopacket_tools,
    register_retrieval_tools,
    register_similarity_tools,
)


def create_phentrieve_mcp() -> FastMCP:
    """Build the Phentrieve FastMCP server (tools, resources, prompts, middleware)."""
    mcp = FastMCP(
        name="phentrieve",
        instructions=SERVER_INSTRUCTIONS,
        mask_error_details=True,
    )

    register_retrieval_tools(mcp)
    register_similarity_tools(mcp)
    register_phenopacket_tools(mcp)
    register_discovery_tools(mcp)
    register_resources(mcp)
    _register_prompts(mcp)
    mcp.add_middleware(ArgValidationMiddleware())

    return mcp


def _register_prompts(mcp: FastMCP) -> None:
    @mcp.prompt(name="annotate_research_text", title="Annotate Research Text")
    def annotate_research_text(language: str = "en") -> str:
        return annotate_research_text_prompt(language=language)

    @mcp.prompt(
        name="review_hpo_research_annotations",
        title="Review HPO Research Annotations",
    )
    def review_hpo_research_annotations() -> str:
        return review_hpo_research_annotations_prompt()

    @mcp.prompt(
        name="extract_research_case_phenotypes",
        title="Extract Research Case Phenotypes",
    )
    def extract_research_case_phenotypes(language: str = "en") -> str:
        return extract_research_case_phenotypes_prompt(language=language)
