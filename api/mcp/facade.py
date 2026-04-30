from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import api.config as api_config
from api.llm_quota import (
    DailyQuotaStore,
    QuotaExceededError,
    QuotaStatus,
    hash_subject_key,
    quota_reset_at_iso,
)
from api.mcp.prompts import (
    annotate_research_text_prompt,
    extract_research_case_phenotypes_prompt,
    review_hpo_research_annotations_prompt,
)
from api.mcp.resources import (
    get_capabilities_resource,
    get_extraction_profiles_resource,
    get_languages_resource,
    get_llm_capability_defaults,
    get_research_use_resource,
)
from api.mcp.tools import (
    CompareHpoTermsRequest,
    ExtractHpoTermsLlmRequest,
    ExtractHpoTermsRequest,
    SearchHpoTermsRequest,
)
from phentrieve.config import (
    DEFAULT_ASSERTION_CONFIG,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_MULTI_VECTOR,
    get_default_chunk_pipeline_config,
)
from phentrieve.evaluation.metrics import (
    SimilarityFormula,
    calculate_semantic_similarity,
    load_hpo_graph_data,
)
from phentrieve.llm.security_policy import resolve_public_llm_target
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


def _is_production_environment() -> bool:
    return api_config.PHENTRIEVE_ENV.strip().lower() == "production"


def _get_llm_quota_store() -> DailyQuotaStore:
    return DailyQuotaStore(
        db_path=Path(api_config.PHENTRIEVE_LLM_QUOTA_DB_PATH),
        daily_limit=api_config.PHENTRIEVE_LLM_DAILY_LIMIT,
    )


def _check_mcp_llm_quota_or_raise() -> QuotaStatus:
    usage_date_utc = datetime.now(UTC).date().isoformat()
    quota_status = _get_llm_quota_store().get_status(
        subject_key=hash_subject_key("mcp:streamable-http"),
        usage_date_utc=usage_date_utc,
    )
    if quota_status.quota_remaining <= 0:
        raise QuotaExceededError(
            quota_used=quota_status.quota_used,
            quota_limit=quota_status.quota_limit,
            quota_remaining=quota_status.quota_remaining,
            usage_date_utc=quota_status.usage_date_utc,
        )
    return quota_status


def _record_mcp_llm_quota_success(quota_status: QuotaStatus) -> QuotaStatus:
    return _get_llm_quota_store().record_success(
        subject_key=quota_status.subject_key,
        usage_date_utc=quota_status.usage_date_utc,
    )


def _standard_fallback_result(
    request: ExtractHpoTermsLlmRequest,
    *,
    service: SyncMcpService,
    fallback_meta: dict[str, Any],
) -> McpResult:
    result = service(
        text=request.text,
        extraction_backend="standard",
        language=request.language,
        include_details=request.include_details,
        include_positions=request.include_chunk_positions,
        num_results_per_chunk=request.num_results_per_chunk,
        chunk_retrieval_threshold=request.chunk_retrieval_threshold,
    )
    result.setdefault("meta", {})
    result["meta"].update(fallback_meta)
    return result


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
    target = resolve_public_llm_target()
    actual_language = request.language or DEFAULT_LANGUAGE
    llm_kwargs = {
        "text": request.text,
        "extraction_backend": "llm",
        "language": actual_language,
        "llm_provider": target.provider,
        "llm_model": target.model,
        "llm_base_url": target.base_url,
        "llm_mode": request.llm_mode,
        "llm_internal_mode": request.llm_internal_mode,
        "include_details": request.include_details,
        "include_positions": request.include_chunk_positions,
        "num_results_per_chunk": request.num_results_per_chunk,
        "chunk_retrieval_threshold": request.chunk_retrieval_threshold,
        "chunking_pipeline_config": get_default_chunk_pipeline_config(),
        "assertion_config": {
            **DEFAULT_ASSERTION_CONFIG,
            "disable": False,
            "language": actual_language,
        },
        "retrieval_model_name": DEFAULT_MODEL,
    }
    quota_status: QuotaStatus | None = None
    if _is_production_environment():
        try:
            quota_status = _check_mcp_llm_quota_or_raise()
        except QuotaExceededError as exc:
            if not request.allow_standard_fallback:
                raise
            return _standard_fallback_result(
                request,
                service=service,
                fallback_meta={
                    "fallback_reason": "llm_quota_exhausted",
                    "llm_quota_limit": exc.quota_limit,
                    "llm_quota_reset_at": quota_reset_at_iso(exc.usage_date_utc),
                },
            )

    try:
        result = service(**llm_kwargs)
    except Exception as exc:
        if not request.allow_standard_fallback:
            raise
        return _standard_fallback_result(
            request,
            service=service,
            fallback_meta={
                "fallback_reason": "llm_backend_error",
                "fallback_error": str(exc),
            },
        )

    if quota_status is not None:
        updated_quota_status = _record_mcp_llm_quota_success(quota_status)
        result.setdefault("meta", {})
        result["meta"].update(
            {
                "quota_limit": updated_quota_status.quota_limit,
                "quota_remaining": updated_quota_status.quota_remaining,
                "quota_reset_at": quota_reset_at_iso(
                    updated_quota_status.usage_date_utc
                ),
            }
        )
    return result


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
        multi_vector=DEFAULT_MULTI_VECTOR,
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
        multi_vector=DEFAULT_MULTI_VECTOR,
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
        """Use this for quick deterministic research screening without LLM calls. For full abstracts, publication-style annotation, syndrome/eponym-heavy text, or review work where retrieval-only noise should be suppressed, prefer phentrieve.extract_hpo_terms_llm. Research use only; not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
        return extract_hpo_terms_impl(request)

    @mcp.tool(
        name="phentrieve.extract_hpo_terms_llm",
        title="Extract HPO Terms With LLM",
    )
    def extract_hpo_terms_llm(request: ExtractHpoTermsLlmRequest) -> dict[str, Any]:
        """Prefer this for full abstracts, publication-style annotation, syndrome/eponym-heavy text, and review workflows where retrieval-only noise should be suppressed. Uses only the server-configured LLM provider/model; clients cannot override model, provider, or base URL. Not for diagnosis, treatment, triage, patient management, clinical decision support, or identifiable patient data in public demo instances."""
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
            **get_llm_capability_defaults(),
        }

    return mcp
