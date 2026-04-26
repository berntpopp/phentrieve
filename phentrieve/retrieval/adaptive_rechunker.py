"""Adaptive re-chunking for poor-quality retrieval results.

Implements Spec B (.planning/specs/2026-04-25-adaptive-rechunking-spec.md).

This module owns the runtime configuration shape (`AdaptiveRechunkingConfig`)
and the Pydantic profile block schema (`AdaptiveRechunkingProfileBlock`) that
Plan A's `Profile` model imports. Keeping the canonical block here keeps
`phentrieve.profiles` free of feature-specific schema and avoids circular
imports - this module does not import from `phentrieve.profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


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
