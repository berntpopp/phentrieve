from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

FUZZY_MATCH_RATIO_THRESHOLD = 90.0


@dataclass
class EvidenceValidationReport:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    repairs: list[dict[str, Any]]
    status: str = "validated"


def validate_phase1_evidence(
    *,
    extracted: list[dict[str, Any]],
    grounded_chunks: list[dict[str, Any]],
    fuzzy_threshold: float = FUZZY_MATCH_RATIO_THRESHOLD,
) -> EvidenceValidationReport:
    if not grounded_chunks:
        return EvidenceValidationReport(
            kept=[dict(item) for item in extracted],
            dropped=[],
            repairs=[],
            status="skipped_no_grounded_chunks",
        )

    chunks_by_id = {int(chunk["chunk_id"]): chunk for chunk in grounded_chunks}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    for raw_item in extracted:
        item = dict(raw_item)
        phrase = str(item.get("phrase", "") or "").strip()
        chunk_ids = _coerce_chunk_ids(item.get("chunk_ids"))
        item["chunk_ids"] = chunk_ids

        if not chunk_ids:
            dropped.append(
                {"phrase": phrase, "reason": "empty_chunk_ids", "chunk_ids": chunk_ids}
            )
            continue
        if any(chunk_id not in chunks_by_id for chunk_id in chunk_ids):
            dropped.append(
                {"phrase": phrase, "reason": "unknown_chunk_id", "chunk_ids": chunk_ids}
            )
            continue

        referenced_chunks = [chunks_by_id[chunk_id] for chunk_id in chunk_ids]
        haystack = " ".join(
            str(chunk.get("text", "") or "") for chunk in referenced_chunks
        )
        evidence = str(item.get("evidence_text") or "").strip()
        if not evidence and phrase and _find_case_insensitive(phrase, haystack):
            item["evidence_text"] = phrase
            evidence = phrase
            repairs.append({"phrase": phrase, "kind": "evidence_text_repair"})

        if not evidence:
            dropped.append(
                {"phrase": phrase, "reason": "empty_evidence", "chunk_ids": chunk_ids}
            )
            continue

        exact_span = _find_case_insensitive(evidence, haystack)
        if exact_span is not None:
            repaired = _repair_offsets(
                item=item,
                evidence=evidence,
                exact_span=exact_span,
                referenced_chunks=referenced_chunks,
            )
            if repaired["kind"]:
                repairs.append({"phrase": phrase, "kind": repaired["kind"]})
            kept.append(repaired["item"])
            continue

        ratio = _best_window_ratio(evidence, haystack)
        if ratio >= fuzzy_threshold:
            downgraded = {**item, "start_char": None, "end_char": None}
            repairs.append({"phrase": phrase, "kind": "fuzzy_evidence_downgrade"})
            kept.append(downgraded)
            continue

        dropped.append(
            {
                "phrase": phrase,
                "reason": "evidence_not_grounded",
                "chunk_ids": chunk_ids,
            }
        )

    return EvidenceValidationReport(kept=kept, dropped=dropped, repairs=repairs)


def validation_report_summary(report: EvidenceValidationReport) -> dict[str, Any]:
    downgraded_count = sum(
        1
        for repair in report.repairs
        if str(repair.get("kind", "")).endswith("_downgrade")
    )
    return {
        "status": report.status,
        "kept_count": len(report.kept),
        "dropped_count": len(report.dropped),
        "repair_count": len(report.repairs),
        "downgraded_count": downgraded_count,
        "dropped": list(report.dropped),
        "repairs": list(report.repairs),
    }


def _coerce_chunk_ids(value: Any) -> list[int]:
    ids: list[int] = []
    for chunk_id in value or []:
        try:
            ids.append(int(chunk_id))
        except (TypeError, ValueError):
            continue
    return ids


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _find_case_insensitive(needle: str, haystack: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.IGNORECASE)
    match = pattern.search(haystack)
    if match is None:
        return None
    return match.start(), match.end()


def _best_window_ratio(needle: str, haystack: str) -> float:
    needle_tokens = _normalize_text(needle).split()
    haystack_tokens = _normalize_text(haystack).split()
    if not needle_tokens or not haystack_tokens:
        return 0.0
    window_size = max(1, len(needle_tokens))
    best = 0.0
    for size in {window_size - 1, window_size, window_size + 1}:
        if size < 1:
            continue
        for start in range(0, max(len(haystack_tokens) - size + 1, 1)):
            window = " ".join(haystack_tokens[start : start + size])
            best = max(
                best,
                SequenceMatcher(None, _normalize_text(needle), window).ratio() * 100.0,
            )
    return best


def _repair_offsets(
    *,
    item: dict[str, Any],
    evidence: str,
    exact_span: tuple[int, int] | None,
    referenced_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(referenced_chunks) != 1:
        return {
            "item": {**item, "start_char": None, "end_char": None},
            "kind": "multi_chunk_offset_downgrade",
        }

    chunk = referenced_chunks[0]
    chunk_text = str(chunk.get("text", "") or "")
    start = item.get("start_char")
    end = item.get("end_char")
    has_offset_value = start is not None or end is not None

    if isinstance(start, int) and isinstance(end, int):
        if _span_matches_evidence(chunk_text, start, end, evidence):
            return {"item": item, "kind": ""}
        chunk_start = chunk.get("start_char")
        if isinstance(chunk_start, int):
            local_start = start - chunk_start
            local_end = end - chunk_start
            if _span_matches_evidence(chunk_text, local_start, local_end, evidence):
                return {
                    "item": {**item, "start_char": local_start, "end_char": local_end},
                    "kind": "offset_coordinate_repair",
                }

    if exact_span is not None and has_offset_value:
        repaired = {**item, "start_char": exact_span[0], "end_char": exact_span[1]}
        return {"item": repaired, "kind": "offset_repair"}
    if exact_span is not None:
        return {"item": item, "kind": ""}
    downgraded = {**item, "start_char": None, "end_char": None}
    return {"item": downgraded, "kind": "offset_downgrade"}


def _span_matches_evidence(text: str, start: int, end: int, evidence: str) -> bool:
    if not (0 <= start < end <= len(text)):
        return False
    if text[start:end].lower() != evidence.lower():
        return False
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (not before.isalnum() and before != "_") and (
        not after.isalnum() and after != "_"
    )
