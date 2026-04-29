from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from api.mcp.prompts import (
    annotate_research_text_prompt,
    extract_research_case_phenotypes_prompt,
    review_hpo_research_annotations_prompt,
)
from api.mcp.resources import (
    get_capabilities_resource,
    get_extraction_profiles_resource,
    get_languages_resource,
    get_research_use_resource,
)
from api.mcp.tools import (
    CompareHpoTermsRequest,
    ExtractHpoTermsLlmRequest,
    ExtractHpoTermsRequest,
    SearchHpoTermsRequest,
)
from phentrieve.config import DEFAULT_LANGUAGE, DEFAULT_MODEL
from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
    load_hpo_graph_data,
)
from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api
from phentrieve.text_processing.full_text_service import run_full_text_service
from phentrieve.utils import normalize_id

RESEARCH_USE_INSTRUCTIONS = (
    "Phentrieve maps clinical or biomedical research text to Human Phenotype "
    "Ontology term suggestions for research, benchmarking, education, and "
    "research data standardisation only. It is not for diagnosis, treatment, "
    "triage, patient management, or clinical decision support. Do not submit "
    "identifiable patient data to public demo instances. Use the LLM tool only "
    "when the user asks for research-only LLM-assisted full-text extraction or "
    "document-level phenotype annotation."
)

McpResult = dict[str, Any]
SyncMcpService = Callable[..., McpResult]


def extract_hpo_terms_impl(
    request: ExtractHpoTermsRequest,
    *,
    service: SyncMcpService = run_full_text_service,
) -> McpResult:
    return service(
        text=request.text,
        extraction_backend="standard",
        language=request.language,
        include_details=request.include_details,
        include_positions=request.include_chunk_positions,
        num_results_per_chunk=request.num_results_per_chunk,
        chunk_retrieval_threshold=request.chunk_retrieval_threshold,
    )


def extract_hpo_terms_llm_impl(
    request: ExtractHpoTermsLlmRequest,
    *,
    service: SyncMcpService = run_full_text_service,
) -> McpResult:
    return service(
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


def search_hpo_terms_impl(
    request: SearchHpoTermsRequest,
    *,
    search: SyncMcpService,
) -> McpResult:
    return search(
        text=request.text,
        language=request.language,
        num_results=request.num_results,
        similarity_threshold=request.similarity_threshold,
        include_details=request.include_details,
    )


def compare_hpo_terms_impl(
    request: CompareHpoTermsRequest,
    *,
    compare: SyncMcpService,
) -> McpResult:
    return compare(
        term1_id=request.term1_id,
        term2_id=request.term2_id,
        formula=request.formula,
    )


async def _search_hpo_terms_service(**kwargs: Any) -> dict[str, Any]:
    from api.dependencies import get_dense_retriever_dependency

    retriever = await get_dense_retriever_dependency(
        sbert_model_name_for_retriever=DEFAULT_MODEL,
        multi_vector=False,
    )
    language = kwargs["language"] or DEFAULT_LANGUAGE
    result = await execute_hpo_retrieval_for_api(
        text=kwargs["text"],
        language=language,
        retriever=retriever,
        num_results=kwargs["num_results"],
        similarity_threshold=kwargs["similarity_threshold"],
        include_details=kwargs["include_details"],
        detect_query_assertion=False,
        debug=False,
    )
    return {"results": result.get("results", [])}


def _compare_hpo_terms_service(**kwargs: Any) -> dict[str, Any]:
    term1_id = normalize_id(kwargs["term1_id"])
    term2_id = normalize_id(kwargs["term2_id"])
    formula = SimilarityFormula(kwargs["formula"])
    _ancestors, depths = load_hpo_graph_data()
    if term1_id not in depths or term2_id not in depths:
        return {
            "term1_id": term1_id,
            "term2_id": term2_id,
            "formula_used": formula.value,
            "similarity_score": 0.0,
            "error_message": "One or both HPO terms were not found in ontology data.",
        }
    return {
        "term1_id": term1_id,
        "term2_id": term2_id,
        "formula_used": formula.value,
        "similarity_score": calculate_semantic_similarity(
            term1_id,
            term2_id,
            formula=formula,
        ),
    }


def create_phentrieve_mcp(*, streamable_http_path: str = "/mcp") -> FastMCP:
    mcp = FastMCP(
        name="phentrieve",
        instructions=RESEARCH_USE_INSTRUCTIONS,
        streamable_http_path=streamable_http_path,
        json_response=True,
    )

    @mcp.tool(
        name="phentrieve.extract_hpo_terms",
        title="Extract HPO Terms",
    )
    def extract_hpo_terms(request: ExtractHpoTermsRequest) -> dict[str, Any]:
        """Use this when research text should be mapped to HPO term suggestions without LLM calls. Research use only; not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return extract_hpo_terms_impl(request)

    @mcp.tool(
        name="phentrieve.extract_hpo_terms_llm",
        title="Extract HPO Terms With LLM",
    )
    def extract_hpo_terms_llm(request: ExtractHpoTermsLlmRequest) -> dict[str, Any]:
        """Use this when research-only full-text LLM extraction should identify phenotype mentions and map them to grounded HPO term suggestions. Not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return extract_hpo_terms_llm_impl(request)

    @mcp.tool(
        name="phentrieve.search_hpo_terms",
        title="Search HPO Terms",
    )
    async def search_hpo_terms(request: SearchHpoTermsRequest) -> dict[str, Any]:
        """Use this when a short research phenotype phrase should be mapped to candidate HPO terms. Not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return await _search_hpo_terms_service(
            text=request.text,
            language=request.language,
            num_results=request.num_results,
            similarity_threshold=request.similarity_threshold,
            include_details=request.include_details,
        )

    @mcp.tool(
        name="phentrieve.compare_hpo_terms",
        title="Compare HPO Terms",
    )
    def compare_hpo_terms(request: CompareHpoTermsRequest) -> dict[str, Any]:
        """Use this when two HPO IDs should be compared for research similarity analysis. Not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return compare_hpo_terms_impl(request, compare=_compare_hpo_terms_service)

    @mcp.resource("phentrieve://capabilities")
    def capabilities() -> dict[str, Any]:
        return get_capabilities_resource()

    @mcp.resource("phentrieve://hpo/languages")
    def languages() -> dict[str, Any]:
        return get_languages_resource()

    @mcp.resource("phentrieve://hpo/extraction-profiles")
    def extraction_profiles() -> dict[str, Any]:
        return get_extraction_profiles_resource()

    @mcp.resource("phentrieve://compliance/research-use")
    def research_use() -> dict[str, Any]:
        return get_research_use_resource()

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
