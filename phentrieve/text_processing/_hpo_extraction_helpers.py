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
) -> list[dict[str, Any]]:
    """Convert batched retriever results into per-chunk match lists.

    Applies chunk_retrieval_threshold, the num_results_per_chunk cap,
    the top_term_per_chunk filter, and propagates assertion_statuses
    from the parallel list onto each match.

    Returns one dict per input chunk with keys: chunk_idx, chunk_text,
    matches (list of {id, name, score, assertion_status}).
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
                }
            )
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_idx + 1}: {e}")
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
    except Exception as e:
        logger.warning(f"Failed to batch-load HPO term data: {e}")

    return hpo_synonyms_cache, hpo_definitions_cache


def build_evidence_map(
    chunk_results: list[dict[str, Any]],
    hpo_synonyms_cache: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Group match evidence by HPO ID, attaching text attributions.

    One list per HPO ID, each element a dict with: score, chunk_idx,
    text, status, name, attributions_in_chunk.
    """
    evidence_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_result in chunk_results:
        chunk_idx: Any = chunk_result["chunk_idx"]
        chunk_text: Any = chunk_result["chunk_text"]
        matches: Any = chunk_result["matches"]
        for term in matches:
            hpo_id = term["id"]
            synonyms = hpo_synonyms_cache.get(hpo_id, [])
            attributions_in_chunk = get_text_attributions(
                source_chunk_text=chunk_text,
                hpo_term_label=term["name"],
                hpo_term_synonyms=synonyms,
                hpo_term_id=hpo_id,
            )
            evidence_map[hpo_id].append(
                {
                    "score": term["score"],
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "status": term.get("assertion_status"),
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
