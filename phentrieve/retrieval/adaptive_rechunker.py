"""Adaptive re-chunking for poor-quality retrieval results.

Implements Spec B (.planning/specs/2026-04-25-adaptive-rechunking-spec.md).

This module owns the runtime configuration shape (`AdaptiveRechunkingConfig`)
and the Pydantic profile block schema (`AdaptiveRechunkingProfileBlock`) that
Plan A's `Profile` model imports. Keeping the canonical block here keeps
`phentrieve.profiles` free of feature-specific schema and avoids circular
imports - this module does not import from `phentrieve.profiles`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveRechunkingConfig:
    """End-to-end configuration carried through the pipeline.

    Frozen so it can be safely passed across function boundaries without
    risk of mutation. Defaults are calibrated for BioLORD-class biomedical
    encoders; users on other encoders should retune via YAML or CLI flags.
    """

    enabled: bool = False
    quality_threshold: float = 0.55
    margin_threshold: float = 0.03
    use_ontology_coherence: bool = False  # reserved, inert in v1
    max_depth: int = 2
    min_chunk_chars: int = 30
    max_sentences_per_subchunk: int = 3
    overlap_sentences: int = 1
    score_improvement_gate: float = 0.05


class AdaptiveRechunkingProfileBlock(BaseModel):
    """Pydantic block on `Profile.adaptive_rechunking`.

    `extra="forbid"` so YAML typos error at load time. All fields are
    optional - `None` means "this profile does not preset this knob" and
    the resolution chain falls through to YAML / built-in defaults.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    quality_threshold: float | None = None
    margin_threshold: float | None = None
    use_ontology_coherence: bool | None = None
    max_depth: int | None = None
    min_chunk_chars: int | None = None
    max_sentences_per_subchunk: int | None = None
    overlap_sentences: int | None = None
    score_improvement_gate: float | None = None


def adaptive_config_from_profile_block(
    block: AdaptiveRechunkingProfileBlock | None,
    yaml_block: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AdaptiveRechunkingConfig:
    """Resolve an `AdaptiveRechunkingConfig` with CLI > profile > YAML > defaults.

    Args:
        block: Profile-level block (typically `Profile.adaptive_rechunking`).
            `None` means no profile contribution.
        yaml_block: Raw YAML mapping under `extraction.adaptive_rechunking`.
            `None` means no YAML contribution.
        cli_overrides: Mapping of CLI-supplied values keyed by config field
            name. `None` (or values of `None`) means no CLI contribution
            for the affected fields.

    Returns:
        A frozen `AdaptiveRechunkingConfig` populated by walking the
        precedence stack for each field.
    """
    defaults = AdaptiveRechunkingConfig()
    fields = set(defaults.__dataclass_fields__)

    resolved: dict[str, Any] = {}
    for name in fields:
        cli_value = (cli_overrides or {}).get(name)
        if cli_value is not None:
            resolved[name] = cli_value
            continue
        profile_value = getattr(block, name, None) if block is not None else None
        if profile_value is not None:
            resolved[name] = profile_value
            continue
        yaml_value = (yaml_block or {}).get(name)
        if yaml_value is not None:
            resolved[name] = yaml_value
            continue
        resolved[name] = getattr(defaults, name)
    return AdaptiveRechunkingConfig(**resolved)


@dataclass(frozen=True)
class ChunkQualitySignals:
    """Quality assessment of one chunk's retrieval result."""

    chunk_idx: int
    top_1: float | None
    top_2: float | None
    margin: float | None
    n_matches_above_threshold: int
    is_poor: bool
    reason: str  # "low_score" | "low_margin" | "no_matches" | "ok"


def assess_chunk_quality(
    raw_query_result: dict[str, Any],
    chunk_idx: int,
    chunk_retrieval_threshold: float,
    config: AdaptiveRechunkingConfig,
) -> ChunkQualitySignals:
    """Read top-K from raw query_batch output, decide if the chunk is poor.

    Reads from ``raw_query_result["similarities"][0]`` (the unfiltered list of
    top-K similarity scores from ``query_batch``). Crucially, this is NOT the
    threshold-filtered ``chunk_results`` - see Spec B Architecture for why.

    Trigger conjunction:
        is_poor = top_1 < quality_threshold AND
                  (margin < margin_threshold OR top_2 is None)
    """
    similarities_outer = raw_query_result.get("similarities") or []
    similarities = similarities_outer[0] if similarities_outer else []

    if not similarities:
        return ChunkQualitySignals(
            chunk_idx=chunk_idx,
            top_1=None,
            top_2=None,
            margin=None,
            n_matches_above_threshold=0,
            is_poor=True,
            reason="no_matches",
        )

    top_1 = similarities[0]
    top_2 = similarities[1] if len(similarities) > 1 else None
    margin = (top_1 - top_2) if top_2 is not None else None
    n_above = sum(1 for s in similarities if s >= chunk_retrieval_threshold)

    score_low = top_1 < config.quality_threshold
    margin_low_or_unknown = top_2 is None or (
        margin is not None and margin < config.margin_threshold
    )

    is_poor = score_low and margin_low_or_unknown
    if not is_poor:
        reason = "ok"
    elif top_2 is None:
        reason = "low_score"  # only one match, score too low
    else:
        reason = "low_margin"

    return ChunkQualitySignals(
        chunk_idx=chunk_idx,
        top_1=top_1,
        top_2=top_2,
        margin=margin,
        n_matches_above_threshold=n_above,
        is_poor=is_poor,
        reason=reason,
    )


def subdivide_parent_chunk(
    parent_chunk: dict[str, Any],
    language: str,
    config: AdaptiveRechunkingConfig,
    depth: int,
) -> list[dict[str, Any]]:
    """Generate sentence-bounded sub-chunks from a parent chunk.

    Returns sub-chunks in the same dict shape as the upstream
    ``TextProcessingPipeline`` output. Sub-chunks inherit the parent's
    ``status`` / ``assertion_details`` (no re-detection in v1 - subdividing
    a NEGATED parent must not flip non-negated tail clauses to AFFIRMED).
    Returns ``[]`` if no useful subdivision is possible (single-sentence
    parent, empty text, or every candidate would be dropped).

    Each sub-chunk's ``source_indices.processing_stages`` gets the parent's
    stages plus ``"adaptive_rechunker_depth_<N>"`` for traceability, and its
    ``start_char`` / ``end_char`` are computed by linear search inside the
    parent text and offset by the parent's ``start_char`` for absolute
    document positions.
    """
    # Lazy import - SentenceChunker pulls in pysbd which is heavy.
    from phentrieve.text_processing.chunkers import SentenceChunker

    parent_text = parent_chunk.get("text", "")
    if not parent_text:
        return []

    chunker = SentenceChunker(language=language)
    sentences = chunker.chunk([parent_text])
    if len(sentences) <= 1:
        return []  # Single sentence - no useful subdivision.

    # Sliding window: window=max_sentences_per_subchunk, step=window-overlap.
    window = max(1, config.max_sentences_per_subchunk)
    overlap = max(0, min(config.overlap_sentences, window - 1))
    step = window - overlap
    if step <= 0:
        step = 1

    parent_start = parent_chunk.get("start_char", 0)
    parent_status = parent_chunk.get("status")
    parent_details = parent_chunk.get("assertion_details")
    parent_stages = list(
        parent_chunk.get("source_indices", {}).get("processing_stages", [])
    )
    parent_text_stripped = parent_text.strip()

    children: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for i in range(0, len(sentences), step):
        group = sentences[i : i + window]
        if not group:
            continue
        sub_text = " ".join(group).strip()
        if not sub_text:
            continue
        if sub_text == parent_text_stripped:
            continue  # Identical to parent - no useful subdivision.
        if sub_text in seen_texts:
            continue
        seen_texts.add(sub_text)
        if len(sub_text) < config.min_chunk_chars:
            continue

        # Locate within the parent text to compute spans.
        idx = parent_text.find(sub_text)
        if idx < 0:
            # Fallback: anchor on the first sentence of the group.
            idx = parent_text.find(group[0]) if group else -1
            if idx < 0:
                idx = 0
        start_char = parent_start + idx
        end_char = start_char + len(sub_text)

        children.append(
            {
                "text": sub_text,
                "status": parent_status,
                "assertion_details": parent_details,
                "source_indices": {
                    "processing_stages": parent_stages
                    + [f"adaptive_rechunker_depth_{depth}"],
                },
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return children


def apply_score_improvement_gate(
    parent_to_children: dict[int, list[int]],
    parent_top_1: dict[int, float],
    child_top_1: dict[int, float],
    config: AdaptiveRechunkingConfig,
) -> tuple[set[int], set[int]]:
    """Decide per parent whether subdivision improved retrieval enough to keep.

    Returns:
        ``(revert_parents, keep_parents)`` - indices of parents whose
        children should be reverted (sub-chunks dropped, parent restored)
        vs kept (sub-chunks replace the parent in the final flat list).

    A parent is kept iff at least one of its children's ``top_1`` is at
    least ``parent_top_1 + config.score_improvement_gate``. Parents with
    no children, no recorded ``parent_top_1``, or no recorded child scores
    are skipped (no children) or reverted (children but no scores).

    Operates on raw ``top_1`` values from ``raw_query_results`` (parent)
    and from this recursion level's child query (children). No
    re-aggregation runs before the gate decision.
    """
    revert: set[int] = set()
    keep: set[int] = set()
    for parent_idx, children in parent_to_children.items():
        if not children:
            continue
        parent_t1 = parent_top_1.get(parent_idx)
        if parent_t1 is None:
            continue
        child_scores = [
            score
            for score in (child_top_1.get(c) for c in children)
            if score is not None
        ]
        if not child_scores:
            revert.add(parent_idx)
            continue
        best = max(child_scores)
        if best >= parent_t1 + config.score_improvement_gate:
            keep.add(parent_idx)
        else:
            revert.add(parent_idx)
    return revert, keep


@dataclass(frozen=True)
class AdaptiveRechunkingResult:
    """Return value of ``run_adaptive_rechunking``.

    Mirrors the legacy 2-tuple ``(aggregated_results, chunk_results)`` that
    callers expect from ``orchestrate_hpo_extraction`` while also exposing
    the (possibly expanded / curated) ``processed_chunks`` and a ``meta``
    dict with bookkeeping for telemetry / logging.
    """

    processed_chunks: list[dict[str, Any]]
    aggregated_results: list[dict[str, Any]]
    chunk_results: list[dict[str, Any]]
    meta: dict[str, Any]


def _placeholder_raw() -> dict[str, Any]:
    """Empty raw query slot used while child slots are reserved before
    the depth-N batch query has run.
    """
    return {
        "ids": [[]],
        "metadatas": [[]],
        "similarities": [[]],
        "distances": [[]],
        "documents": [[]],
    }


def _no_op_result(
    processed_chunks: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    meta: dict[str, Any],
) -> AdaptiveRechunkingResult:
    """Return a result that mirrors the inputs without doing anything.

    ``aggregated_results`` is left empty - the caller already holds the
    aggregated output from the initial extraction pass and does not need
    the rechunker to recompute it when adaptive re-chunking is disabled
    or no chunks were flagged.
    """
    return AdaptiveRechunkingResult(
        processed_chunks=list(processed_chunks),
        aggregated_results=[],
        chunk_results=list(chunk_results),
        meta=meta,
    )


def _aggregate(
    chunks: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    retriever: Any,
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    language: str,
    min_confidence_for_aggregated: float,
    include_details: bool,
    assertion_statuses: list[str | None] | None,
) -> Any:
    """Re-aggregate via ``orchestrate_hpo_extraction`` with precomputed raw
    results so retrieval is skipped (cost-model invariant).
    """
    # Lazy import to avoid an import cycle: the orchestrator depends on
    # the retriever module, and this module is imported from CLI / API
    # paths that may not need the orchestrator on every call.
    from phentrieve.text_processing.hpo_extraction_orchestrator import (
        orchestrate_hpo_extraction,
    )

    text_chunks = [c.get("text", "") for c in chunks]
    return orchestrate_hpo_extraction(
        text_chunks=text_chunks,
        retriever=retriever,
        num_results_per_chunk=num_results_per_chunk,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        language=language,
        min_confidence_for_aggregated=min_confidence_for_aggregated,
        assertion_statuses=assertion_statuses,
        include_details=include_details,
        precomputed_query_results=raw,  # KEY: skips query_batch.
    )


def run_adaptive_rechunking(
    processed_chunks: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
    raw_query_results: list[dict[str, Any]],
    retriever: Any,
    language: str,
    config: AdaptiveRechunkingConfig,
    num_results_per_chunk: int,
    chunk_retrieval_threshold: float,
    min_confidence_for_aggregated: float,
    include_details: bool,
    assertion_statuses: list[str | None] | None = None,
) -> AdaptiveRechunkingResult:
    """Top-level adaptive re-chunking orchestration.

    Cost contract: at most ``config.max_depth`` additional
    ``retriever.query_batch`` calls (one per recursion level where chunks
    flag as poor). Re-aggregation uses
    ``orchestrate_hpo_extraction(precomputed_query_results=...)`` so
    existing chunk results are not re-queried.

    The function:
      1. If disabled, returns inputs unchanged with no retrieval calls.
      2. Detects poor chunks via ``assess_chunk_quality`` over the raw
         (unfiltered) query results.
      3. Subdivides each poor parent via ``subdivide_parent_chunk``.
      4. Issues a single ``retriever.query_batch`` for all children at
         this depth.
      5. Applies the score-improvement gate to keep / revert per parent.
      6. Rebuilds the flat chunk list (originals for non-flagged or
         reverted parents, accepted children for kept parents).
      7. Recurses on the new list up to ``config.max_depth``.
      8. Re-aggregates once at the end via
         ``orchestrate_hpo_extraction(precomputed_query_results=...)``.
    """
    meta: dict[str, Any] = {
        "enabled": config.enabled,
        "trigger_count": 0,
        "subdivided_count": 0,
        "reverted_count": 0,
        "max_depth_reached": 0,
        "extra_chunks_added": 0,
    }

    if not config.enabled:
        return _no_op_result(processed_chunks, chunk_results, meta)

    # Detect poor chunks at depth 0 (the initial pass already done by the
    # caller). Used only for the trigger_count meta and the early-exit
    # path - the recursion loop re-assesses against ``current_raw``.
    initial_quality = [
        assess_chunk_quality(raw, idx, chunk_retrieval_threshold, config)
        for idx, raw in enumerate(raw_query_results)
    ]
    initial_poor = [s.chunk_idx for s in initial_quality if s.is_poor]
    meta["trigger_count"] = len(initial_poor)

    if not initial_poor:
        # Nothing to subdivide - keep inputs as-is and skip aggregation
        # (the caller already has aggregated_results from the initial
        # extraction pass).
        return _no_op_result(processed_chunks, chunk_results, meta)

    # Recursion loop. Each iteration performs at most ONE query_batch call.
    current_chunks: list[dict[str, Any]] = list(processed_chunks)
    current_raw: list[dict[str, Any]] = list(raw_query_results)
    current_assertions: list[str | None] = list(
        assertion_statuses
        if assertion_statuses is not None
        else [c.get("status") for c in current_chunks]
    )

    any_kept = False
    for depth in range(1, config.max_depth + 1):
        # Identify chunks still flagged as poor at this depth, against the
        # raw scores attached to the *current* flat list.
        depth_quality = [
            assess_chunk_quality(raw, idx, chunk_retrieval_threshold, config)
            for idx, raw in enumerate(current_raw)
        ]
        depth_poor = {s.chunk_idx for s in depth_quality if s.is_poor}
        if not depth_poor:
            break

        # Subdivide each poor parent and reserve slots for its children
        # in the new flat list. Non-flagged chunks are copied through.
        new_chunks: list[dict[str, Any]] = []
        new_assertions: list[str | None] = []
        new_raw: list[dict[str, Any]] = []
        children_texts: list[str] = []
        # Maps original-index-in-current_chunks -> new-flat-list slots.
        child_slots: dict[int, list[int]] = {}

        for idx, parent in enumerate(current_chunks):
            if idx not in depth_poor:
                new_chunks.append(parent)
                new_assertions.append(current_assertions[idx])
                new_raw.append(current_raw[idx])
                continue
            children = subdivide_parent_chunk(parent, language, config, depth)
            if not children:
                # Subdivision impossible - keep parent as-is.
                new_chunks.append(parent)
                new_assertions.append(current_assertions[idx])
                new_raw.append(current_raw[idx])
                continue
            slot_indices: list[int] = []
            for child in children:
                slot_indices.append(len(new_chunks))
                new_chunks.append(child)
                new_assertions.append(child.get("status"))
                new_raw.append(_placeholder_raw())
                children_texts.append(child["text"])
            child_slots[idx] = slot_indices

        if not children_texts:
            # No useful subdivision was possible at this depth.
            break

        # ONE query_batch call per recursion level (the cost-model
        # invariant). Returned list is one entry per child text.
        child_raw = retriever.query_batch(
            texts=children_texts,
            n_results=num_results_per_chunk,
            include_similarities=True,
        )
        meta["max_depth_reached"] = depth

        # Fill placeholders with real raw results in the same order they
        # were appended to ``children_texts``.
        ci = 0
        for slot_indices in child_slots.values():
            for slot in slot_indices:
                new_raw[slot] = child_raw[ci]
                ci += 1

        # Score-improvement gate decides per parent: keep children, or
        # revert to the original parent.
        parent_top_1: dict[int, float] = {}
        for parent_idx in child_slots:
            sims_outer = current_raw[parent_idx].get("similarities") or []
            sims = sims_outer[0] if sims_outer else []
            if sims:
                parent_top_1[parent_idx] = sims[0]

        # Re-key child top_1 onto the parent_idx -> child slot mapping
        # used by ``apply_score_improvement_gate``.
        child_top_1: dict[int, float] = {}
        for slot_indices in child_slots.values():
            for slot in slot_indices:
                sims_outer = new_raw[slot].get("similarities") or []
                sims = sims_outer[0] if sims_outer else []
                if sims:
                    child_top_1[slot] = sims[0]

        revert, keep = apply_score_improvement_gate(
            child_slots, parent_top_1, child_top_1, config
        )

        if revert:
            # Rebuild flat list: restore reverted parents, drop their
            # children. Non-flagged chunks and kept-parent children pass
            # through. Iterate ``current_chunks`` so we preserve original
            # ordering for parents and pull children from ``new_*`` for
            # kept parents.
            keep_chunks: list[dict[str, Any]] = []
            keep_assertions: list[str | None] = []
            keep_raw: list[dict[str, Any]] = []
            for idx, parent in enumerate(current_chunks):
                if idx in child_slots and idx in revert:
                    # Flagged + subdivided + reverted: restore parent.
                    keep_chunks.append(parent)
                    keep_assertions.append(current_assertions[idx])
                    keep_raw.append(current_raw[idx])
                elif idx in child_slots:
                    # Flagged + subdivided + kept: append children.
                    for slot in child_slots[idx]:
                        keep_chunks.append(new_chunks[slot])
                        keep_assertions.append(new_assertions[slot])
                        keep_raw.append(new_raw[slot])
                else:
                    # Not flagged or subdivision failed: copy parent.
                    keep_chunks.append(parent)
                    keep_assertions.append(current_assertions[idx])
                    keep_raw.append(current_raw[idx])
            new_chunks, new_assertions, new_raw = (
                keep_chunks,
                keep_assertions,
                keep_raw,
            )

        meta["subdivided_count"] += len(keep)
        meta["reverted_count"] += len(revert)
        if keep:
            any_kept = True

        current_chunks = new_chunks
        current_assertions = new_assertions
        current_raw = new_raw

        if not keep:
            # All subdivisions reverted - no point recursing further.
            break

    meta["extra_chunks_added"] = len(current_chunks) - len(processed_chunks)

    if not any_kept:
        # Nothing accepted - the flat list is unchanged from the input.
        # Skip re-aggregation; caller already has the original aggregate.
        return _no_op_result(processed_chunks, chunk_results, meta)

    # Final re-aggregation using ``precomputed_query_results`` - no
    # retrieval call.
    final = _aggregate(
        current_chunks,
        current_raw,
        retriever,
        num_results_per_chunk,
        chunk_retrieval_threshold,
        language,
        min_confidence_for_aggregated,
        include_details,
        current_assertions,
    )

    return AdaptiveRechunkingResult(
        processed_chunks=current_chunks,
        aggregated_results=final.aggregated_results,
        chunk_results=final.chunk_results,
        meta=meta,
    )


def dump_quality_report(
    raw_query_results: list[dict[str, Any]],
    chunk_retrieval_threshold: float,
    config: AdaptiveRechunkingConfig,
) -> str:
    """Library-only helper for users tuning thresholds.

    Returns a per-chunk quality report as a human-readable string.
    """
    lines = ["chunk_idx  is_poor  reason       top_1  top_2  margin"]
    for idx, raw in enumerate(raw_query_results):
        s = assess_chunk_quality(raw, idx, chunk_retrieval_threshold, config)
        t1 = f"{s.top_1:.3f}" if s.top_1 is not None else "  -  "
        t2 = f"{s.top_2:.3f}" if s.top_2 is not None else "  -  "
        m = f"{s.margin:.3f}" if s.margin is not None else "  -  "
        lines.append(f"{idx:>9}  {s.is_poor!s:<6}  {s.reason:<11}  {t1}  {t2}  {m}")
    return "\n".join(lines)
