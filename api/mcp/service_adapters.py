"""Domain service adapters for the Phentrieve MCP tools.

Each function reuses the existing ``phentrieve.*`` / ``api.*`` service layer
directly (no HTTP round-trip) and returns a plain dict. Verbosity shaping,
``_meta`` injection, and error enveloping happen in the tool layer; these
adapters hold only the domain logic (including the LLM quota / fallback policy).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import api.config as api_config
from api.llm_quota import (
    DailyQuotaStore,
    QuotaExceededError,
    QuotaStatus,
    hash_subject_key,
    quota_reset_at_iso,
)
from api.mcp.envelope import McpToolError
from api.mcp.projection import cap_response_synonyms
from phentrieve.assertion_vocab import is_excluded
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
    find_lowest_common_ancestor,
    load_hpo_graph_data,
)
from phentrieve.llm.security_policy import resolve_public_llm_target
from phentrieve.retrieval.api_helpers import execute_hpo_retrieval_for_api
from phentrieve.text_processing.config_resolver import resolve_chunking_config
from phentrieve.text_processing.full_text_service import run_full_text_service
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.utils import detect_language, normalize_id

McpResult = dict[str, Any]


# --------------------------------------------------------------------------- #
# LLM quota helpers (ported from the previous facade)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Retrieval / extraction
# --------------------------------------------------------------------------- #
async def search_hpo_terms_service(
    *,
    text: str,
    language: str | None,
    num_results: int,
    similarity_threshold: float,
    include_details: bool,
) -> McpResult:
    from api.dependencies import get_dense_retriever_dependency

    retriever = await get_dense_retriever_dependency(
        sbert_model_name_for_retriever=DEFAULT_MODEL,
        multi_vector=DEFAULT_MULTI_VECTOR,
    )
    result = await execute_hpo_retrieval_for_api(
        text=text,
        language=language or DEFAULT_LANGUAGE,
        retriever=retriever,
        num_results=num_results,
        similarity_threshold=similarity_threshold,
        include_details=include_details,
        detect_query_assertion=False,
        debug=False,
        multi_vector=DEFAULT_MULTI_VECTOR,
    )
    results = result.get("results", [])
    # R3: cap each result's synonym list in the response (the attribution layer
    # keeps the full list); include_details otherwise dumps it uncapped.
    for record in results:
        if isinstance(record, dict):
            cap_response_synonyms(record)
    return {"results": results}


def extract_hpo_terms_service(
    *,
    text: str,
    language: str | None,
    include_details: bool,
    include_chunk_positions: bool,
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    service: Any = run_full_text_service,
) -> McpResult:
    result: McpResult = service(
        text=text,
        extraction_backend="standard",
        # language=None means "autodetect" per the tool contract; resolve it here
        # because the chunking pipeline requires a concrete language string.
        language=language or detect_language(text, default_lang=DEFAULT_LANGUAGE),
        include_details=include_details,
        include_positions=include_chunk_positions,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
    )
    return result


def _standard_fallback_result(
    *,
    text: str,
    language: str | None,
    include_details: bool,
    include_chunk_positions: bool,
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    fallback_meta: dict[str, Any],
    service: Any,
) -> McpResult:
    result: McpResult = service(
        text=text,
        extraction_backend="standard",
        language=language,
        include_details=include_details,
        include_positions=include_chunk_positions,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
    )
    result.setdefault("meta", {})
    result["meta"].update(fallback_meta)
    return result


def extract_hpo_terms_llm_service(
    *,
    text: str,
    language: str | None,
    include_details: bool,
    include_chunk_positions: bool,
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    llm_mode: str,
    llm_internal_mode: str,
    allow_standard_fallback: bool,
    service: Any = run_full_text_service,
) -> McpResult:
    target = resolve_public_llm_target()
    actual_language = language or detect_language(text, default_lang=DEFAULT_LANGUAGE)
    fallback_kwargs = {
        "text": text,
        "language": actual_language,
        "include_details": include_details,
        "include_chunk_positions": include_chunk_positions,
        "num_results_per_chunk": num_results_per_chunk,
        "chunk_retrieval_threshold": chunk_retrieval_threshold,
        "service": service,
    }
    llm_kwargs = {
        "text": text,
        "extraction_backend": "llm",
        "language": actual_language,
        "llm_provider": target.provider,
        "llm_model": target.model,
        "llm_base_url": target.base_url,
        "llm_mode": llm_mode,
        "llm_internal_mode": llm_internal_mode,
        "include_details": include_details,
        "include_positions": include_chunk_positions,
        "num_results_per_chunk": num_results_per_chunk,
        "chunk_retrieval_threshold": chunk_retrieval_threshold,
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
            if not allow_standard_fallback:
                raise
            return _standard_fallback_result(
                fallback_meta={
                    "fallback_reason": "llm_quota_exhausted",
                    "llm_quota_limit": exc.quota_limit,
                    "llm_quota_reset_at": quota_reset_at_iso(exc.usage_date_utc),
                },
                **fallback_kwargs,
            )

    try:
        result: McpResult = service(**llm_kwargs)
    except Exception as exc:
        if not allow_standard_fallback:
            raise
        return _standard_fallback_result(
            fallback_meta={
                "fallback_reason": "llm_backend_error",
                "fallback_error": str(exc),
            },
            **fallback_kwargs,
        )

    if quota_status is not None:
        updated = _record_mcp_llm_quota_success(quota_status)
        result.setdefault("meta", {})
        result["meta"].update(
            {
                "quota_limit": updated.quota_limit,
                "quota_remaining": updated.quota_remaining,
                "quota_reset_at": quota_reset_at_iso(updated.usage_date_utc),
            }
        )
    return result


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #
def _hpo_label_map(ids: list[str]) -> dict[str, str]:
    """Best-effort id -> label lookup; empty dict if the HPO DB is unavailable."""
    try:
        from phentrieve.config import DEFAULT_HPO_DB_FILENAME
        from phentrieve.data_processing.hpo_database import HPODatabase
        from phentrieve.utils import get_default_data_dir, resolve_data_path

        db_path = (
            resolve_data_path(None, "data_dir", get_default_data_dir)
            / DEFAULT_HPO_DB_FILENAME
        )
        if not db_path.exists():
            return {}
        db = HPODatabase(db_path)
        terms = db.get_terms_by_ids([i for i in ids if i])
        db.close()
        return {hid: t.get("label", "") for hid, t in terms.items()}
    except Exception:  # pragma: no cover - label is a non-essential enrichment
        return {}


def _build_lca_details(t1: str, t2: str, depths: Mapping[str, float]) -> dict[str, Any]:
    """Explain a similarity score: MICA, per-term depth + normalized depth, path.

    D2: the per-term structural proxy is depth/max_depth -- normalized ontology
    depth, NOT corpus information content. It is labelled ``normalized_depth`` so
    consumers do not mistake it for a Resnik-style IC (the bundle ships no corpus
    to compute true IC).
    """
    lca, lca_depth = find_lowest_common_ancestor(t1, t2)
    max_depth = max(depths.values()) if depths else 0
    d1 = depths.get(t1)
    d2 = depths.get(t2)
    labels = _hpo_label_map([t1, t2, lca] if lca else [t1, t2])

    def _normalized_depth(depth: float | None) -> float | None:
        if depth is None or not max_depth:
            return None
        return round(depth / max_depth, 4)

    path_length: int | None = None
    if d1 is not None and d2 is not None and lca_depth is not None and lca_depth >= 0:
        path_length = int((d1 - lca_depth) + (d2 - lca_depth))

    return {
        "mica": {
            "hpo_id": lca,
            "label": labels.get(lca) if lca else None,
            "depth": lca_depth if lca else None,
        },
        "lca_depth": lca_depth if lca else None,
        "term1": {
            "hpo_id": t1,
            "label": labels.get(t1),
            "depth": d1,
            "normalized_depth": _normalized_depth(d1),
        },
        "term2": {
            "hpo_id": t2,
            "label": labels.get(t2),
            "depth": d2,
            "normalized_depth": _normalized_depth(d2),
        },
        "path_length": path_length,
    }


def compare_hpo_terms_service(
    *, term1_id: str, term2_id: str, formula: str, include_lca_details: bool = False
) -> McpResult:
    t1 = normalize_id(term1_id)
    t2 = normalize_id(term2_id)
    similarity_formula = SimilarityFormula(formula)
    _ancestors, depths = load_hpo_graph_data()
    if t1 not in depths or t2 not in depths:
        missing = [t for t in (t1, t2) if t not in depths]
        raise McpToolError(
            "not_found",
            f"HPO term(s) not found in ontology data: {', '.join(missing)}.",
            details={"term1_id": t1, "term2_id": t2},
        )
    result: McpResult = {
        "term1_id": t1,
        "term2_id": t2,
        "formula_used": similarity_formula.value,
        "similarity_score": calculate_semantic_similarity(
            t1, t2, formula=similarity_formula
        ),
    }
    # At standard/full, explain the score with the MICA, per-term IC and the
    # subsumer path so the number is not a bare scalar (defect D5).
    if include_lca_details:
        result["lca_details"] = _build_lca_details(t1, t2, depths)
    return result


# --------------------------------------------------------------------------- #
# Phenopacket export (reuses the REST router's mapping + bundle logic)
# --------------------------------------------------------------------------- #
def _coerce_export_phenotype(request_cls: Any, p: dict[str, Any], idx: int) -> Any:
    """Map a phenotype dict onto ExportPhenotypeRequest, accepting both the
    canonical {hpo_id, label, assertion} shape and the extractor's
    {id, name, assertion_status} shape, and carrying score -> confidence.

    Raises a typed validation_failed error (not a raw KeyError) when the id is
    missing, with a did-you-mean mapping hint (defects M2, H3, M3).
    """
    hpo_id = p.get("hpo_id") or p.get("id")
    if not hpo_id:
        raise McpToolError(
            "validation_failed",
            f"phenotypes[{idx}] missing 'hpo_id' (got keys: {sorted(p)}); map "
            "id->hpo_id, name->label, assertion_status->assertion. Hand the "
            "aggregated_hpo_terms from an extract call directly.",
            details={"field": f"phenotypes[{idx}].hpo_id"},
        )
    assertion = p.get("assertion") or p.get("status") or p.get("assertion_status")
    # A family-history mention is not a proband phenotypic feature; never fold it
    # into an affirmed feature on the subject (LLM-1). Drop it from the packet.
    if str(assertion).strip().lower() == "family_history":
        return None
    confidence = p.get("score")
    if confidence is None:
        confidence = p.get("confidence", p.get("max_score_from_evidence"))
    chunk_ids = p.get("source_chunk_ids") or p.get("chunk_ids") or []
    # Honest provenance: a phenotype handed to export without extractor
    # provenance is client-supplied, not a dictionary match. Preserve any
    # provenance the caller did pass through (defect D11).
    match_method = p.get("match_method") or "client_supplied"
    source_mode = p.get("source_mode") or "unknown"
    return request_cls(
        hpo_id=hpo_id,
        label=p.get("label") or p.get("name") or hpo_id,
        assertion_status="negated" if is_excluded(assertion) else "affirmed",
        confidence=confidence,
        source_chunk_ids=[c for c in chunk_ids if isinstance(c, int)],
        match_method=match_method,
        source_mode=source_mode,
    )


# --------------------------------------------------------------------------- #
def export_phenopacket_service(
    *,
    case_id: str,
    case_label: str | None,
    input_text: str | None,
    subject: dict[str, Any] | None,
    phenotypes: list[dict[str, Any]],
    include_annotation_sidecar: bool,
) -> McpResult:
    from api.routers.phenopacket_router import (
        _apply_request_metadata_to_bundle,
        _map_phenotype_for_export,
        export_phenopacket_bundle,
    )
    from api.schemas.phenopacket_schemas import (
        ExportPhenotypeRequest,
        ExportSubjectRequest,
        PhenopacketExportRequest,
        PhenopacketExportResponse,
    )

    export_phenotypes = [
        coerced
        for idx, p in enumerate(phenotypes)
        if (coerced := _coerce_export_phenotype(ExportPhenotypeRequest, p, idx))
        is not None
    ]
    export_request = PhenopacketExportRequest(
        case_id=case_id,
        case_label=case_label,
        input_text=input_text,
        subject=ExportSubjectRequest(**subject) if subject else None,
        include_annotation_sidecar=include_annotation_sidecar,
        phenotypes=export_phenotypes,
    )
    aggregated = [_map_phenotype_for_export(p) for p in export_request.phenotypes]
    bundle = export_phenopacket_bundle(
        aggregated_results=aggregated,
        input_text=export_request.input_text,
        include_annotation_sidecar=export_request.include_annotation_sidecar,
    )
    bundle = _apply_request_metadata_to_bundle(bundle, export_request)
    result = PhenopacketExportResponse.model_validate(bundle).model_dump(
        exclude_none=True
    )
    # Return the phenopacket as a native JSON object (MCP/Anthropic guidance:
    # structured content should be real JSON, not a stringified blob). The
    # serialized ``phenopacket_json`` string is kept for backwards
    # compatibility (defect D4).
    raw_json = result.get("phenopacket_json")
    if isinstance(raw_json, str):
        try:
            result["phenopacket"] = json.loads(raw_json)
        except (TypeError, ValueError):
            # Non-JSON/invalid payload: intentionally keep only the serialized
            # ``phenopacket_json`` string (backward compat) and skip the parsed
            # ``phenopacket`` field rather than failing the whole response.
            pass
    return result


# --------------------------------------------------------------------------- #
# Chunking (no retrieval, no assertion)
# --------------------------------------------------------------------------- #
def chunk_text_service(
    *, text: str, language: str | None, strategy: str | None
) -> McpResult:
    from phentrieve.text_processing.config_resolver import KNOWN_CHUNK_STRATEGIES

    # Reject unknown strategies explicitly (config_resolver silently falls back to
    # the default for unknown names, so without this an invalid value would be
    # accepted on a model-loaded server -- defect L4).
    if strategy is not None and strategy.lower() not in KNOWN_CHUNK_STRATEGIES:
        raise McpToolError(
            "validation_failed",
            f"Unknown chunking strategy '{strategy}'. Valid strategies: "
            f"{', '.join(KNOWN_CHUNK_STRATEGIES)}.",
            details={
                "field": "strategy",
                "allowed_values": list(KNOWN_CHUNK_STRATEGIES),
            },
        )
    try:
        config = resolve_chunking_config(strategy or "simple")
    except Exception as exc:  # invalid strategy name (when it raises)
        raise McpToolError(
            "invalid_input",
            f"Unknown chunking strategy '{strategy}'. Use a predefined strategy "
            "such as 'simple', 'detailed', or 'sliding_window_punct_conj_cleaned'.",
            details={"field": "strategy"},
        ) from exc

    # B2: 6 of 7 strategies need a semantic model (any config with a
    # sliding_window stage). Lazy-load the cached embedding singleton -- the same
    # instance search/extract warm -- instead of hard-failing, restoring parity
    # with the documented lazy-load latency contract.
    needs_model = any(
        isinstance(stage, dict) and stage.get("type") == "sliding_window"
        for stage in config
    )
    sbert_model = None
    if needs_model:
        from phentrieve.embeddings import load_embedding_model

        try:
            sbert_model = load_embedding_model(DEFAULT_MODEL)
        except Exception as exc:
            # A genuine load failure is a transient server condition, not a bad
            # argument: surface it as retryable rather than blaming the strategy.
            raise McpToolError(
                "temporarily_unavailable",
                "The embedding model required for this chunking strategy could "
                "not be loaded; retry shortly or use the 'simple' strategy.",
                details={"field": "strategy"},
            ) from exc

    try:
        pipeline = TextProcessingPipeline(
            language=language or DEFAULT_LANGUAGE,
            chunking_pipeline_config=config,
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=sbert_model,
        )
        processed = pipeline.process(text, include_positions=True)
    except Exception as exc:  # genuine processing failure, not a bad argument
        raise McpToolError(
            "internal_error",
            "Failed to process text with the requested chunking strategy.",
            details={"field": "strategy"},
        ) from exc

    chunks = [
        {
            "chunk_id": index + 1,
            "text": chunk.get("text", ""),
            "start_char": chunk.get("start_char"),
            "end_char": chunk.get("end_char"),
        }
        for index, chunk in enumerate(processed)
    ]
    return {"chunks": chunks, "chunk_count": len(chunks)}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _probe_ontology_data() -> str:
    try:
        _ancestors, depths = load_hpo_graph_data()
        return "ok" if depths else "error"
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return "error"


def _probe_embedding_model() -> str:
    """Report loaded | loading | cold | error for the default embedding model.

    D3: membership tests on the live caches (no load, no lock). The search path
    warms api.dependencies.LOADED_SBERT_MODELS; extract/chunk warm the standalone
    phentrieve.embeddings registry -- either counts as loaded.
    """
    try:
        from api.dependencies import LOADED_SBERT_MODELS, MODEL_LOADING_STATUS
        from phentrieve.embeddings import _MODEL_REGISTRY

        if DEFAULT_MODEL in LOADED_SBERT_MODELS or DEFAULT_MODEL in _MODEL_REGISTRY:
            return "loaded"
        status = MODEL_LOADING_STATUS.get(DEFAULT_MODEL)
        if status == "loading":
            return "loading"
        if status == "failed":
            return "error"
        return "cold"
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return "error"


def _probe_vector_index() -> str:
    """Report loaded | cold | error for the dense retriever cache (D3)."""
    try:
        from api.dependencies import LOADED_RETRIEVERS

        return "loaded" if len(LOADED_RETRIEVERS) > 0 else "cold"
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return "error"


def diagnostics_service() -> McpResult:
    from api.mcp.capabilities import capabilities_version
    from api.mcp.envelope import get_recent_errors

    subsystems = {
        "ontology_data": _probe_ontology_data(),
        "embedding_model": _probe_embedding_model(),
        "llm_backend": "configured",
        "vector_index": _probe_vector_index(),
    }
    status = "ok" if all(v != "error" for v in subsystems.values()) else "degraded"
    return {
        "status": status,
        "subsystems": subsystems,
        "recent_errors": get_recent_errors()[-10:],
        "minimum_workflow": [
            "phentrieve_search_hpo_terms",
            "phentrieve_extract_hpo_terms",
            "phentrieve_export_phenopacket",
        ],
        "capabilities_version": capabilities_version(),
    }
