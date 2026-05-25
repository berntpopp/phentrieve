from __future__ import annotations

import logging
import re
from typing import Any

from phentrieve.llm.pipeline_phase2 import (
    build_grounded_context,
    downstream_dedupe_key,
    extract_first_result_list,
    hybrid_select_candidates,
    prepare_retrieval_queries,
)

logger = logging.getLogger(__name__)

CONTEXT_HEAD_NOUNS = (
    "seizures",
    "seizure",
    "headache",
    "tumour",
    "tumor",
    "hairs",
    "hair",
)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped


def _contains_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def _phrase_can_use_context_head(phrase: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", phrase.lower())
    if not tokens or len(tokens) > 2:
        return False
    phrase_heads = {token.rstrip("s") for token in tokens}
    context_heads = {head.rstrip("s") for head in CONTEXT_HEAD_NOUNS}
    return not bool(phrase_heads & context_heads)


def prepare_contextual_retrieval_queries(
    *,
    phrase: str,
    grounded_context: dict[str, Any],
) -> list[str]:
    queries = prepare_retrieval_queries(phrase)
    if not _phrase_can_use_context_head(phrase):
        return queries

    context_parts = [
        str(grounded_context.get("primary_chunk_text", "") or ""),
        *[
            str(text or "")
            for text in grounded_context.get("neighbor_chunk_texts", [])
            if isinstance(text, str)
        ],
    ]
    context_text = " ".join(context_parts)
    for head in CONTEXT_HEAD_NOUNS:
        if _contains_word(context_text, head):
            queries.append(f"{phrase} {head}")
            break
    return _dedupe_preserving_order(queries)


def _candidate_match_text(candidate: dict[str, Any]) -> str:
    matched_text = str(candidate.get("matched_text", "") or "").strip()
    if matched_text:
        return matched_text
    return str(candidate.get("term_name", "") or "")


def _filter_raw_modifier_candidates(
    *,
    phrase: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    phrase_tokens = re.findall(r"[A-Za-z]+", phrase.lower())
    if not phrase_tokens or len(phrase_tokens) > 2:
        return candidates
    phrase_clean = " ".join(phrase.lower().split())
    has_context_variant = any(
        str(candidate.get("retrieval_query", "") or "").strip().lower()
        not in {"", phrase_clean}
        for candidate in candidates
    )
    if not has_context_variant:
        return candidates
    return [
        candidate
        for candidate in candidates
        if not (
            str(candidate.get("retrieval_query", "") or "").strip().lower()
            == phrase_clean
            and " ".join(_candidate_match_text(candidate).lower().split())
            == phrase_clean
        )
    ]


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
        grounded_context = build_grounded_context(
            item=item,
            grounded_chunks=grounded_chunks,
        )
        query_variants = prepare_contextual_retrieval_queries(
            phrase=str(item["phrase"]),
            grounded_context=grounded_context,
        )
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
        merged = _filter_raw_modifier_candidates(phrase=phrase, candidates=merged)
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
