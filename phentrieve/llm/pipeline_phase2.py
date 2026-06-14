from __future__ import annotations

import json
from typing import Any

from phentrieve.llm.config import (
    NEGATED_ASSERTION,
    PRESENT_ASSERTION,
    UNCERTAIN_ASSERTION,
)
from phentrieve.llm.pipeline_phase1 import (
    UNIT_TOKEN_PATTERN,
    clean_text,
    experiencer_for_category,
    normalize_category,
    tokenize,
)
from phentrieve.llm.prompts.loader import get_batch_mapping_prompt, get_mapping_prompt
from phentrieve.llm.types import (
    LLMBatchMappingSelections,
    LLMMappingSelection,
    LLMPhenotype,
    LLMPhenotypeEvidence,
)
from phentrieve.llm.utils import token_sort_similarity

CATEGORY_TO_ASSERTION = {
    "abnormal": PRESENT_ASSERTION,
    "normal": NEGATED_ASSERTION,
    "suspected": UNCERTAIN_ASSERTION,
    "family_history": "family_history",
    "other": "other",
}


def prepare_retrieval_queries(phrase: str) -> list[str]:
    original = " ".join(str(phrase or "").split()).strip()
    if not original:
        return []

    variants = [original]
    stripped_units = " ".join(UNIT_TOKEN_PATTERN.sub(" ", original).split())
    if stripped_units and stripped_units != original:
        variants.append(stripped_units)
    return variants


def candidate_match_text(candidate: dict[str, Any]) -> str:
    matched_text = str(candidate.get("matched_text", "") or "").strip()
    if matched_text:
        return matched_text
    return str(candidate.get("term_name", "") or "")


def normalize_mapping_phrase_key(text: str) -> str:
    return clean_text(text)


def candidate_id_tuple(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(candidate.get("hpo_id", "")).strip()
                for candidate in item.get("candidates", [])
                if str(candidate.get("hpo_id", "")).strip()
            }
        )
    )


def downstream_dedupe_key(
    item: dict[str, Any],
    *,
    include_candidate_ids: bool,
) -> tuple[str, str, tuple[str, ...]]:
    return (
        normalize_mapping_phrase_key(str(item.get("phrase", ""))),
        normalize_category(str(item.get("category", ""))),
        candidate_id_tuple(item) if include_candidate_ids else (),
    )


def normalize_grounded_text(text: Any) -> str:
    return str(text or "").strip()


def mapping_batch_item_id(index: int) -> str:
    return f"item_{index + 1}"


def compact_mapping_item(
    item: dict[str, Any],
    *,
    item_id: str | None = None,
) -> dict[str, Any]:
    grounded_context = dict(item.get("grounded_context", {}) or {})
    neighbor_texts = [
        normalize_grounded_text(text)
        for text in grounded_context.get("neighbor_chunk_texts", [])
        if normalize_grounded_text(text)
    ]
    compact_item: dict[str, Any] = {
        "primary_chunk_text": normalize_grounded_text(
            grounded_context.get("primary_chunk_text")
        ),
        "neighbor_chunk_texts": neighbor_texts,
        "phrase": str(item["phrase"]).lower().replace("-", " ").strip(),
        "category": item["category"],
        "candidates": [],
    }
    for candidate in item["candidates"]:
        compact_candidate = {
            "id": candidate["hpo_id"],
            "term": candidate["term_name"],
            "retrieval_score": candidate.get("score"),
        }
        if candidate.get("retrieval_query"):
            compact_candidate["retrieval_query"] = candidate.get("retrieval_query")
        if candidate.get("matched_text"):
            compact_candidate["matched_text"] = candidate.get("matched_text")
        if candidate.get("matched_component"):
            compact_candidate["matched_component"] = candidate.get("matched_component")
        compact_item["candidates"].append(compact_candidate)
    if item_id is not None:
        compact_item["item_id"] = item_id
    return compact_item


def extract_first_result_list(
    batch_result: dict[str, Any],
    key: str,
) -> list[Any]:
    values = batch_result.get(key)
    if not isinstance(values, list) or not values:
        return []
    first = values[0]
    return first if isinstance(first, list) else []


def build_grounded_context(
    *,
    item: dict[str, Any],
    grounded_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in grounded_chunks}
    chunk_ids = [int(chunk_id) for chunk_id in item.get("chunk_ids", [])]
    primary = chunk_lookup.get(chunk_ids[0]) if chunk_ids else None
    neighbor_ids = [chunk_ids[0] - 1, chunk_ids[0] + 1] if chunk_ids else []
    neighbors = [
        chunk_lookup[cid]
        for cid in neighbor_ids
        if cid in chunk_lookup and chunk_lookup[cid] is not primary
    ]
    return {
        "chunk_ids": chunk_ids,
        "primary_chunk_text": (
            primary.get("text", "")
            if primary
            else item.get("evidence_text", item.get("phrase", ""))
        ),
        "neighbor_chunk_texts": [chunk.get("text", "") for chunk in neighbors],
    }


def hybrid_select_candidates(
    *,
    phrase: str,
    metadatas: list[dict[str, Any]],
    similarities: list[float],
    max_unique_candidates: int,
    min_unique_candidates: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    phrase_tokens = tokenize(phrase)

    for index, metadata in enumerate(metadatas):
        if len(selected) >= max_unique_candidates:
            break

        hpo_id = str(metadata.get("hpo_id", "")).strip()
        if not hpo_id or hpo_id in seen_ids:
            continue

        label = str(metadata.get("label", "")).strip()
        similarity = float(similarities[index]) if index < len(similarities) else 0.0
        label_tokens = tokenize(label)

        has_token_overlap = bool(phrase_tokens & label_tokens)
        meets_threshold = similarity >= similarity_threshold
        requires_fill = len(selected) < min_unique_candidates

        if has_token_overlap or meets_threshold or requires_fill:
            seen_ids.add(hpo_id)
            selected.append(
                {
                    "hpo_id": hpo_id,
                    "term_name": label,
                    "score": similarity,
                }
            )
    return selected


def run_mapping_batch(
    *,
    provider,
    batch: list[dict[str, Any]],
    mapping_prompt,
) -> tuple[LLMMappingSelection | LLMBatchMappingSelections, dict[str, int]]:
    response_model: type[LLMMappingSelection] | type[LLMBatchMappingSelections]
    if len(batch) == 1:
        batch_mapping_prompt = get_mapping_prompt(mapping_prompt.language)
        candidate_payload = json.dumps(
            compact_mapping_item(batch[0]),
            ensure_ascii=False,
        )
        response_model = LLMMappingSelection
    else:
        batch_mapping_prompt = get_batch_mapping_prompt(mapping_prompt.language)
        candidate_payload = json.dumps(
            {
                "items": [
                    compact_mapping_item(
                        item,
                        item_id=mapping_batch_item_id(index),
                    )
                    for index, item in enumerate(batch)
                ]
            },
            ensure_ascii=False,
        )
        response_model = LLMBatchMappingSelections
    response = provider.run_structured_prompt(
        system_prompt=batch_mapping_prompt.render_system_prompt(
            language=batch_mapping_prompt.language
        ),
        user_prompt=batch_mapping_prompt.render_user_prompt(
            candidate_payload,
            language=batch_mapping_prompt.language,
        ),
        response_model=response_model,
    )
    usage = dict(getattr(provider, "last_usage", {}) or {})
    return response, usage


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def phenotype_from_candidate(
    *,
    item: dict[str, Any],
    candidate: dict[str, Any],
    match_method: str = "local",
) -> LLMPhenotype:
    evidence = LLMPhenotypeEvidence(
        phrase=str(item["phrase"]),
        evidence_text=item.get("evidence_text"),
        chunk_ids=list(item.get("chunk_ids", [])),
        start_char=item.get("start_char"),
        end_char=item.get("end_char"),
        match_method=match_method,
    )
    return LLMPhenotype(
        term_id=candidate["hpo_id"],
        label=candidate["term_name"],
        evidence=item["phrase"],
        assertion=CATEGORY_TO_ASSERTION.get(
            normalize_category(str(item.get("category", ""))),
            PRESENT_ASSERTION,
        ),
        experiencer=experiencer_for_category(str(item.get("category", ""))),
        negated_qualifier=(item.get("negated_qualifier") or None),
        category=normalize_category(str(item.get("category", ""))),
        confidence=(
            optional_float(candidate.get("confidence"))
            if candidate.get("confidence") is not None
            else float(candidate.get("score", 0.0) or 0.0)
        ),
        score=float(candidate.get("score", 0.0) or 0.0),
        evidence_records=[evidence],
    )


def select_candidate_id(
    *,
    mapping_response: LLMMappingSelection,
    candidates: list[dict[str, Any]],
) -> str | None:
    candidate_ids = {candidate["hpo_id"] for candidate in candidates}
    return mapping_response.hpo_id if mapping_response.hpo_id in candidate_ids else None


def select_candidate_ids(
    *,
    mapping_response: LLMMappingSelection | LLMBatchMappingSelections,
    batch: list[dict[str, Any]],
) -> dict[str, str | None]:
    if len(batch) == 1:
        item = batch[0]
        if not isinstance(mapping_response, LLMMappingSelection):
            return {mapping_batch_item_id(0): None}
        return {
            mapping_batch_item_id(0): select_candidate_id(
                mapping_response=mapping_response,
                candidates=item["candidates"],
            )
        }

    selected: dict[str, str | None] = {
        mapping_batch_item_id(index): None for index, _ in enumerate(batch)
    }
    if not isinstance(mapping_response, LLMBatchMappingSelections):
        return selected

    candidates_by_item_id = {
        mapping_batch_item_id(index): {
            candidate["hpo_id"] for candidate in item["candidates"]
        }
        for index, item in enumerate(batch)
    }
    for mapping in mapping_response.mappings:
        item_id = mapping.item_id
        selected_id = mapping.hpo_id
        if (
            isinstance(selected_id, str)
            and item_id in candidates_by_item_id
            and selected_id in candidates_by_item_id[item_id]
        ):
            selected[item_id] = selected_id
    return selected


def deduplicate_terms(terms: list[LLMPhenotype]) -> list[LLMPhenotype]:
    deduplicated: dict[tuple[str, str, str], LLMPhenotype] = {}
    for term in terms:
        key = (term.term_id, term.experiencer, term.assertion)
        if key not in deduplicated:
            deduplicated[key] = term
            continue
        deduplicated[key].evidence_records.extend(term.evidence_records)
        if not deduplicated[key].evidence and term.evidence:
            deduplicated[key].evidence = term.evidence
    return list(deduplicated.values())


def local_match(
    *,
    phrase: str,
    candidates: list[dict[str, Any]],
    local_match_threshold: float,
) -> dict[str, Any] | None:
    phrase_clean = clean_text(phrase)
    phrase_tokens = tokenize(phrase)

    for candidate in candidates:
        match_tokens = tokenize(candidate_match_text(candidate))
        if phrase_tokens and match_tokens and phrase_tokens == match_tokens:
            return candidate

    for candidate in candidates:
        if clean_text(candidate_match_text(candidate)) == phrase_clean:
            return candidate

    if len(phrase_tokens) > 1:
        for candidate in candidates:
            match_clean = clean_text(candidate_match_text(candidate))
            if match_clean and re_search_word(match_clean, phrase_clean):
                return candidate

    best_candidate: dict[str, Any] | None = None
    best_score = 0.0
    for candidate in candidates:
        score = token_sort_similarity(
            phrase_clean, clean_text(candidate_match_text(candidate))
        )
        if score >= local_match_threshold and score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate


def re_search_word(needle: str, haystack: str) -> bool:
    import re

    return bool(re.search(rf"\b{re.escape(needle)}\b", haystack))
