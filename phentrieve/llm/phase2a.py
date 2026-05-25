from __future__ import annotations

import logging
from typing import Any

from phentrieve.llm.pipeline_phase2 import (
    build_grounded_context,
    downstream_dedupe_key,
    extract_first_result_list,
    hybrid_select_candidates,
    prepare_retrieval_queries,
)

logger = logging.getLogger(__name__)


def retrieve_candidates(
    *,
    actionable: list[dict[str, Any]],
    grounded_chunks: list[dict[str, Any]],
    language: str,
    tool_executor: Any,
    n_results_per_phrase: int,
    max_unique_candidates: int,
    min_unique_candidates: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    unique_actionable: list[dict[str, Any]] = []
    actionable_groups: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for item in actionable:
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        actionable_groups.setdefault(dedupe_key, []).append(item)
        if len(actionable_groups[dedupe_key]) == 1:
            unique_actionable.append(item)

    expanded_queries: list[str] = []
    expanded_query_keys: list[tuple[str, str, tuple[str, ...]]] = []
    for item in unique_actionable:
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        query_variants = prepare_retrieval_queries(str(item["phrase"]))
        if not query_variants:
            query_variants = [str(item["phrase"])]
        for query in query_variants:
            expanded_queries.append(query)
            expanded_query_keys.append(dedupe_key)

    batched_variant_results = tool_executor.query_batch_hpo_terms(
        phrases=expanded_queries,
        language=language,
        n_results=n_results_per_phrase,
    )

    shared_results: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    grouped_variant_results: dict[
        tuple[str, str, tuple[str, ...]], list[tuple[str, dict[str, Any]]]
    ] = {}
    for index, dedupe_key in enumerate(expanded_query_keys):
        query = expanded_queries[index]
        batch_result = (
            batched_variant_results[index]
            if index < len(batched_variant_results)
            else {}
        )
        grouped_variant_results.setdefault(dedupe_key, []).append(
            (query, dict(batch_result) if isinstance(batch_result, dict) else {})
        )

    for item in unique_actionable:
        phrase = str(item["phrase"])
        category = str(item["category"])
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        merged_candidates: dict[str, dict[str, Any]] = {}
        for query, batch_result in grouped_variant_results.get(dedupe_key, []):
            if "candidates" in batch_result:
                candidates = [
                    {**dict(candidate), "retrieval_query": query}
                    for candidate in batch_result.get("candidates", [])
                    if isinstance(candidate, dict)
                ]
            else:
                metadatas = extract_first_result_list(batch_result, "metadatas")
                similarities = extract_first_result_list(batch_result, "similarities")
                candidates = [
                    {**candidate, "retrieval_query": query}
                    for candidate in hybrid_select_candidates(
                        phrase=query,
                        metadatas=metadatas,
                        similarities=similarities,
                        max_unique_candidates=max_unique_candidates,
                        min_unique_candidates=min_unique_candidates,
                        similarity_threshold=similarity_threshold,
                    )
                ]

            for candidate in candidates:
                hpo_id = str(candidate.get("hpo_id", "")).strip()
                if not hpo_id:
                    continue
                existing = merged_candidates.get(hpo_id)
                if existing is None or float(
                    candidate.get("score", 0.0) or 0.0
                ) > float(existing.get("score", 0.0) or 0.0):
                    merged_candidates[hpo_id] = candidate

        merged = sorted(
            merged_candidates.values(),
            key=lambda candidate: float(candidate.get("score", 0.0) or 0.0),
            reverse=True,
        )[:n_results_per_phrase]
        shared_results[dedupe_key] = {
            "phrase": phrase,
            "category": category,
            "candidates": merged,
        }
        logger.debug(
            "Phase 2A candidate retrieval: phrase=%r candidates=%d",
            phrase,
            len(merged),
        )

    results: list[dict[str, Any]] = []
    for item in actionable:
        dedupe_key = downstream_dedupe_key(item, include_candidate_ids=False)
        shared_result = dict(shared_results.get(dedupe_key, {}))
        shared_result.setdefault("phrase", str(item["phrase"]))
        shared_result.setdefault("category", str(item["category"]))
        shared_result["chunk_ids"] = list(item.get("chunk_ids", []))
        shared_result["evidence_text"] = item.get("evidence_text")
        shared_result["start_char"] = item.get("start_char")
        shared_result["end_char"] = item.get("end_char")
        shared_result["grounded_context"] = build_grounded_context(
            item=item,
            grounded_chunks=grounded_chunks,
        )
        shared_result["candidates"] = list(shared_result.get("candidates", []))
        results.append(shared_result)
    return results
