from __future__ import annotations

from typing import Any, cast

from phentrieve.llm.pipeline_phase1 import ACTIONABLE_CATEGORIES, normalize_category
from phentrieve.llm.pipeline_retry import LLMPipelinePhaseError
from phentrieve.llm.types import Phase1FailureClass, Phase1Mode


def sum_usage_dicts(*usage_dicts: dict[str, int]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usage_dicts:
        for key, value in usage.items():
            totals[key] = int(totals.get(key, 0) or 0) + int(value or 0)
    return totals


def phase1_extracted_trace_items(
    extracted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "phrase": str(item["phrase"]),
            "category": normalize_category(str(item["category"])),
            "chunk_ids": list(item.get("chunk_ids", [])),
            "evidence_text": item.get("evidence_text"),
            "actionable": normalize_category(str(item["category"]))
            in ACTIONABLE_CATEGORIES,
        }
        for item in extracted
    ]


def build_phase1_attempt_trace(
    *,
    mode: Phase1Mode,
    status: str,
    groups_trace: list[dict[str, Any]],
    request_count: int,
    elapsed_seconds: float,
    failure_class: Phase1FailureClass = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": status,
        "request_count": request_count,
        "elapsed_seconds": elapsed_seconds,
        "failure_class": failure_class,
        "groups": list(groups_trace),
    }


def build_phase1_trace(
    *,
    extracted: list[dict[str, Any]],
    groups_trace: list[dict[str, Any]],
    attempts_trace: list[dict[str, Any]],
    initial_mode: Phase1Mode,
    final_mode: Phase1Mode,
    fallback_count: int,
    failure_class: Phase1FailureClass,
) -> dict[str, Any]:
    return {
        "extracted": phase1_extracted_trace_items(extracted),
        "groups": groups_trace,
        "attempts": attempts_trace,
        "initial_mode": initial_mode,
        "final_mode": final_mode,
        "fallback_triggered": fallback_count > 0,
        "failure_class": failure_class,
    }


def phase1_failure_class_from_groups(
    phase1_groups_trace: list[dict[str, Any]],
) -> Phase1FailureClass:
    for group in phase1_groups_trace:
        failure_class = group.get("failure_class")
        if isinstance(failure_class, str):
            return cast(Phase1FailureClass, failure_class)
    return None


def attach_phase1_failure_context(
    *,
    exc: LLMPipelinePhaseError,
    initial_mode: Phase1Mode,
    final_mode: Phase1Mode,
    fallback_count: int,
    failure_class: Phase1FailureClass,
    final_groups_trace: list[dict[str, Any]],
    attempts_trace: list[dict[str, Any]],
) -> None:
    exc.initial_mode = initial_mode
    exc.final_mode = final_mode
    exc.fallback_triggered = fallback_count > 0
    exc.failure_class = failure_class
    exc.phase1_trace = {
        "initial_mode": initial_mode,
        "final_mode": final_mode,
        "fallback_triggered": fallback_count > 0,
        "failure_class": failure_class,
        "groups": list(final_groups_trace),
        "attempts": list(attempts_trace),
    }
    all_groups = [
        group for attempt in attempts_trace for group in attempt.get("groups", [])
    ]
    exc.phase_counts = {
        "phase1_fallbacks": fallback_count,
        "phase1_completed_groups": sum(
            1 for group in all_groups if group.get("status") == "completed"
        ),
        "phase1_failed_groups": sum(
            1 for group in all_groups if group.get("status") == "failed"
        ),
    }
    exc.phase_counts["phase1_partial_failures"] = int(
        exc.phase_counts["phase1_completed_groups"] > 0
        and exc.phase_counts["phase1_failed_groups"] > 0
    )
    exc.phase_request_counts = {
        "phase1_requests": sum(
            int(attempt.get("request_count", 0) or 0) for attempt in attempts_trace
        )
    }


def build_phase2a_trace(
    phrase_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_sets": [
            {
                "phrase": str(item.get("phrase", "")),
                "category": normalize_category(str(item.get("category", ""))),
                "grounded_context": dict(item.get("grounded_context", {})),
                "candidates": [
                    {
                        "term_id": str(candidate.get("hpo_id", "")),
                        "label": str(candidate.get("term_name", "")),
                        "score": float(candidate.get("score", 0.0) or 0.0),
                        "matched_text": candidate.get("matched_text"),
                        "matched_component": candidate.get("matched_component"),
                        "retrieval_query": candidate.get("retrieval_query"),
                    }
                    for candidate in item.get("candidates", [])
                ],
            }
            for item in phrase_candidates
        ]
    }


def build_phase2b_local_trace(
    *,
    locally_resolved,
    unresolved: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "resolved": [
            {
                "phrase": term.evidence,
                "term_id": term.term_id,
                "label": term.label,
                "assertion": term.assertion,
                "category": term.category,
                "match_method": "local",
            }
            for term in locally_resolved
        ],
        "unresolved": [
            {
                "phrase": str(item.get("phrase", "")),
                "category": normalize_category(str(item.get("category", ""))),
            }
            for item in unresolved
        ],
    }


def annotate_mapping_trace_with_provenance(
    trace_entry: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    if not trace_entry:
        return trace_entry
    annotated = dict(trace_entry)
    if item.get("evidence_text") is not None:
        annotated["evidence_text"] = item.get("evidence_text")
    if item.get("chunk_ids") is not None:
        annotated["chunk_ids"] = list(item.get("chunk_ids", []))
    if item.get("start_char") is not None:
        annotated["start_char"] = item.get("start_char")
    if item.get("end_char") is not None:
        annotated["end_char"] = item.get("end_char")
    return annotated
