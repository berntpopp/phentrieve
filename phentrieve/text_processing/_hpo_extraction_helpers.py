"""Focused helpers for hpo_extraction_orchestrator.

Split out from the 298-LOC god function orchestrate_hpo_extraction()
so each concern can be tested, read, and evolved independently.

The public API surface remains orchestrate_hpo_extraction(); these
helpers are a private implementation detail (leading underscore on
the module name). Tests exercise them via the public function.
"""

import logging
from collections import Counter, defaultdict
from typing import Any

from phentrieve.config import DEFAULT_HPO_DB_FILENAME
from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.retrieval.text_attribution import get_text_attributions
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)


def process_chunk_matches(
    text_chunks: list[str],
    all_query_results: list[dict[str, Any]],
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    top_term_per_chunk: bool,
    assertion_statuses: list[str | None] | None,
    chunk_negated_scope_texts: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Convert batched retriever results into per-chunk match lists.

    Applies chunk_retrieval_threshold, the num_results_per_chunk cap,
    the top_term_per_chunk filter, and propagates assertion_statuses
    from the parallel list onto each match. The chunk-level status is the
    fallback; ``build_evidence_map`` refines it per match using
    ``negated_scope_texts`` (assessment defect D1).

    Returns one dict per input chunk with keys: chunk_idx, chunk_text,
    matches (list of {id, name, score, assertion_status}), and
    negated_scope_texts.
    """
    chunk_results: list[dict[str, Any]] = []
    for chunk_idx, chunk_text in enumerate(text_chunks):
        try:
            query_results = all_query_results[chunk_idx]
            current_hpo_matches: list[dict[str, Any]] = []
            metadatas_entry = query_results.get("metadatas") or [[]]
            similarities_entry = query_results.get("similarities") or [[]]
            metadatas_list = metadatas_entry[0] if metadatas_entry else []
            similarities_list = similarities_entry[0] if similarities_entry else []

            matches_added = 0
            for i, metadata in enumerate(metadatas_list):
                if matches_added >= num_results_per_chunk:
                    break
                similarity = similarities_list[i] if i < len(similarities_list) else 0.0
                if similarity < chunk_retrieval_threshold:
                    continue
                hpo_id = metadata.get("id") or metadata.get("hpo_id")
                name = metadata.get("label") or metadata.get("name")
                if not (hpo_id and name):
                    continue
                current_hpo_matches.append(
                    {
                        "id": hpo_id,
                        "name": name,
                        "score": similarity,
                        "assertion_status": (
                            assertion_statuses[chunk_idx]
                            if assertion_statuses
                            else None
                        ),
                    }
                )
                matches_added += 1

            logger.info(
                f"Found {len(current_hpo_matches)} matches for chunk {chunk_idx + 1}"
            )

            if top_term_per_chunk and current_hpo_matches:
                current_hpo_matches = [current_hpo_matches[0]]

            chunk_results.append(
                {
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_text,
                    "matches": current_hpo_matches,
                    "negated_scope_texts": (
                        chunk_negated_scope_texts[chunk_idx]
                        if chunk_negated_scope_texts
                        and chunk_idx < len(chunk_negated_scope_texts)
                        else []
                    ),
                }
            )
        except Exception:
            logger.exception("Failed to process chunk %s", chunk_idx + 1)
            continue
    return chunk_results


def load_term_details(
    all_hpo_ids: set[str],
    include_details: bool,
) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    """Batch-load synonyms (and optionally definitions) for all HPO IDs.

    Returns (synonyms_cache, definitions_cache). definitions_cache is
    always returned but is only populated when include_details is True.
    Both caches are empty dicts if the HPO database file is not found
    or loading fails; the caller treats that as "no enrichment available".
    """
    hpo_synonyms_cache: dict[str, list[str]] = {}
    hpo_definitions_cache: dict[str, str | None] = {}
    if not all_hpo_ids:
        return hpo_synonyms_cache, hpo_definitions_cache

    try:
        data_dir = resolve_data_path(None, "data_dir", get_default_data_dir)
        db_path = data_dir / DEFAULT_HPO_DB_FILENAME
        if not db_path.exists():
            logger.warning(
                f"HPO database not found: {db_path}. Skipping synonym lookup."
            )
            return hpo_synonyms_cache, hpo_definitions_cache

        logger.debug(
            f"Loading {'synonyms and definitions' if include_details else 'synonyms'} "
            f"for {len(all_hpo_ids)} unique HPO terms"
        )
        db = HPODatabase(db_path)
        terms_map = db.get_terms_by_ids(list(all_hpo_ids))
        db.close()

        for hpo_id, term_data in terms_map.items():
            hpo_synonyms_cache[hpo_id] = term_data.get("synonyms", [])
            if include_details:
                hpo_definitions_cache[hpo_id] = term_data.get("definition")

        logger.info(
            f"Loaded {'synonyms and definitions' if include_details else 'synonyms'} "
            f"for {len(hpo_synonyms_cache)} HPO terms"
        )
    except Exception:
        logger.warning("Failed to batch-load HPO term data", exc_info=True)

    return hpo_synonyms_cache, hpo_definitions_cache


def _find_negated_scope_spans(
    chunk_text: str, scope_texts: list[str]
) -> list[tuple[int, int]]:
    """Locate each negated scope concept inside the chunk text.

    The detector reports the negated concept text (e.g. "regression"); we
    re-find it in the chunk to get chunk-relative spans, which is robust to the
    detector running over a restored-context frame. Falls back to the scope's
    first content word when punctuation/whitespace differs.
    """
    spans: list[tuple[int, int]] = []
    low = chunk_text.lower()
    for raw in scope_texts:
        scope = (raw or "").strip().lower()
        if not scope:
            continue
        idx = low.find(scope)
        if idx >= 0:
            spans.append((idx, idx + len(scope)))
            continue
        first = scope.replace(",", " ").split()
        if first:
            word = first[0]
            widx = low.find(word)
            if widx >= 0:
                spans.append((widx, widx + len(word)))
    return spans


def _resolve_match_assertion(
    chunk_status: str | None,
    chunk_text: str,
    scope_texts: list[str],
    attributions: list[dict[str, Any]],
) -> str | None:
    """Refine a chunk-level negation to the matches it actually scopes.

    Only the NEGATED case is span-gated (the over-negation defect D1); AFFIRMED,
    NORMAL and UNCERTAIN pass through unchanged. When the negated scope cannot be
    localized in the chunk, or the match has no attribution span, fall back to
    the chunk-level status so cue-stripped single-concept chunks stay negated
    (preserves the C1 behaviour).
    """
    if chunk_status != "negated" or not scope_texts:
        return chunk_status
    neg_spans = _find_negated_scope_spans(chunk_text, scope_texts)
    if not neg_spans or not attributions:
        return chunk_status
    for attribution in attributions:
        a_start = attribution.get("start_char")
        a_end = attribution.get("end_char")
        if a_start is None or a_end is None:
            continue
        for n_start, n_end in neg_spans:
            if a_start < n_end and n_start < a_end:
                return "negated"
    return "affirmed"


def build_evidence_map(
    chunk_results: list[dict[str, Any]],
    hpo_synonyms_cache: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Group match evidence by HPO ID, attaching text attributions.

    One list per HPO ID, each element a dict with: score, chunk_idx,
    text, status, name, attributions_in_chunk. The per-match ``status`` is
    refined from the chunk-level assertion using span overlap (defect D1).
    """
    evidence_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_result in chunk_results:
        chunk_idx: Any = chunk_result["chunk_idx"]
        chunk_text: Any = chunk_result["chunk_text"]
        matches: Any = chunk_result["matches"]
        scope_texts: list[str] = chunk_result.get("negated_scope_texts") or []
        for term in matches:
            hpo_id = term["id"]
            synonyms = hpo_synonyms_cache.get(hpo_id, [])
            attributions_in_chunk = get_text_attributions(
                source_chunk_text=chunk_text,
                hpo_term_label=term["name"],
                hpo_term_synonyms=synonyms,
                hpo_term_id=hpo_id,
            )
            resolved_status = _resolve_match_assertion(
                term.get("assertion_status"),
                chunk_text,
                scope_texts,
                attributions_in_chunk,
            )
            evidence_map[hpo_id].append(
                {
                    "score": term["score"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "status": resolved_status,
                    "name": term["name"],
                    "attributions_in_chunk": attributions_in_chunk,
                }
            )
    return evidence_map


def aggregate_and_rank(
    evidence_map: dict[str, list[dict[str, Any]]],
    min_confidence_for_aggregated: float,
    hpo_synonyms_cache: dict[str, list[str]],
    hpo_definitions_cache: dict[str, str | None],
    include_details: bool,
) -> list[dict[str, Any]]:
    """Collapse the evidence map into a ranked list of aggregated HPO terms.

    Each output dict mirrors the contract held by the existing API
    response shape: id, name, score (max), count, evidence_count,
    avg_score, confidence, chunks, top_evidence_chunk_idx,
    text_attributions, assertion_status, status, and optionally
    definition + synonyms when include_details is True. Ranks start at 1.
    """
    aggregated_list: list[dict[str, Any]] = []
    for hpo_id, evidence_list in evidence_map.items():
        if not evidence_list:
            continue
        total_score = sum(e["score"] for e in evidence_list)
        avg_score = total_score / len(evidence_list)
        if avg_score < min_confidence_for_aggregated:
            continue
        max_score = max(e["score"] for e in evidence_list)
        top_evidence_chunk_idx = next(
            e["chunk_idx"] for e in evidence_list if e["score"] == max_score
        )
        status_counter = Counter([e["status"] for e in evidence_list if e["status"]])
        assertion_status = (
            status_counter.most_common(1)[0][0] if status_counter else None
        )

        text_attributions: list[dict[str, Any]] = []
        for e in evidence_list:
            for attribution in e.get("attributions_in_chunk", []):
                enriched = attribution.copy()
                enriched["chunk_idx"] = e["chunk_idx"]
                text_attributions.append(enriched)

        term: dict[str, Any] = {
            "id": hpo_id,
            "name": evidence_list[0]["name"],
            "score": max_score,
            "count": len(evidence_list),
            "evidence_count": len(evidence_list),
            "avg_score": avg_score,
            "confidence": avg_score,
            "chunks": sorted({e["chunk_idx"] for e in evidence_list}),
            "top_evidence_chunk_idx": top_evidence_chunk_idx,
            "text_attributions": text_attributions,
            "assertion_status": assertion_status,
            "status": assertion_status,
        }
        if include_details:
            term["definition"] = hpo_definitions_cache.get(hpo_id)
            term["synonyms"] = hpo_synonyms_cache.get(hpo_id, [])
        aggregated_list.append(term)

    aggregated_list.sort(key=lambda x: (-x["avg_score"], -x["count"]))
    for idx, term in enumerate(aggregated_list):
        term["rank"] = idx + 1
    return aggregated_list
