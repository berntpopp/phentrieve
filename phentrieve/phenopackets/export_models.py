"""Normalized export models shared by Phenopacket and sidecar exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self

AssertionValue = Literal[
    "affirmed",
    "negated",
    "uncertain",
    "family_history",
    "unknown",
]

_ASSERTION_ALIASES: dict[str, AssertionValue] = {
    "present": "affirmed",
    "affirmed": "affirmed",
    "negated": "negated",
    "absent": "negated",
    "uncertain": "uncertain",
    "suspected": "uncertain",
    "family_history": "family_history",
    "family history": "family_history",
    "unknown": "unknown",
}


def _normalize_assertion(value: str | None) -> AssertionValue:
    normalized = (value or "affirmed").strip().lower()
    return _ASSERTION_ALIASES.get(normalized, "unknown")


def _normalize_chunk_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple):
        return tuple(int(chunk_id) for chunk_id in value)
    if isinstance(value, list):
        return tuple(int(chunk_id) for chunk_id in value)
    return (int(value),)


def _span_payload(span: NormalizedSpan) -> dict[str, Any]:
    return {
        "evidence_text": span.evidence_text,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "chunk_ids": list(span.chunk_ids),
    }


def _record_payload(record: NormalizedPhenotypeExportRecord) -> dict[str, Any]:
    return {
        "hpo_id": record.hpo_id,
        "label": record.label,
        "assertion": record.assertion,
        "confidence": record.confidence,
        "evidence_text": record.evidence_text,
        "spans": [_span_payload(span) for span in record.spans],
        "chunk_ids": list(record.chunk_ids),
        "source_mode": record.source_mode,
        "match_method": record.match_method,
    }


def _build_sidecar_linkage_key(
    record: NormalizedPhenotypeExportRecord,
) -> str:
    payload = json.dumps(
        _record_payload(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"phentrieve:{digest}"


@dataclass(frozen=True, slots=True)
class NormalizedSpan:
    """Normalized span for a mapped phenotype mention."""

    evidence_text: str
    start_char: int
    end_char: int
    chunk_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        object.__setattr__(self, "chunk_ids", _normalize_chunk_ids(self.chunk_ids))

    @classmethod
    def from_legacy_dict(cls, legacy: Mapping[str, Any]) -> Self:
        evidence_text = (
            legacy.get("evidence_text")
            or legacy.get("text")
            or legacy.get("chunk_text")
        )
        if not isinstance(evidence_text, str) or not evidence_text:
            raise ValueError("legacy span input must include evidence_text")

        start_char = legacy.get("start_char")
        end_char = legacy.get("end_char")
        if start_char is None or end_char is None:
            raise ValueError("legacy span input must include start_char and end_char")

        chunk_ids = legacy.get("chunk_ids")
        if chunk_ids is None and "chunk_refs" in legacy:
            chunk_ids = legacy["chunk_refs"]
        if chunk_ids is None and "chunk_idx" in legacy:
            chunk_ids = legacy["chunk_idx"]
        if chunk_ids is None and "chunk_id" in legacy:
            chunk_ids = legacy["chunk_id"]

        return cls(
            evidence_text=evidence_text,
            start_char=int(start_char),
            end_char=int(end_char),
            chunk_ids=_normalize_chunk_ids(chunk_ids),
        )


@dataclass(frozen=True, slots=True)
class NormalizedPhenotypeExportRecord:
    """Normalized phenotype record for export and sidecar linkage."""

    hpo_id: str
    label: str
    assertion: AssertionValue
    confidence: float | None = None
    evidence_text: str | None = None
    spans: tuple[NormalizedSpan, ...] = field(default_factory=tuple)
    chunk_ids: tuple[int, ...] = field(default_factory=tuple)
    source_mode: str | None = None
    match_method: str | None = None
    sidecar_linkage_key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assertion", _normalize_assertion(self.assertion))
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(self, "chunk_ids", _normalize_chunk_ids(self.chunk_ids))

        normalized_spans = tuple(
            span
            if isinstance(span, NormalizedSpan)
            else NormalizedSpan.from_legacy_dict(span)
            for span in self.spans
        )
        object.__setattr__(self, "spans", normalized_spans)

        if self.evidence_text is None and normalized_spans:
            object.__setattr__(self, "evidence_text", normalized_spans[0].evidence_text)

        if not self.chunk_ids and normalized_spans:
            span_chunk_ids: list[int] = []
            for span in normalized_spans:
                for chunk_id in span.chunk_ids:
                    if chunk_id not in span_chunk_ids:
                        span_chunk_ids.append(chunk_id)
            object.__setattr__(self, "chunk_ids", tuple(span_chunk_ids))

        object.__setattr__(
            self, "sidecar_linkage_key", _build_sidecar_linkage_key(self)
        )

    @classmethod
    def from_legacy_dict(cls, legacy: Mapping[str, Any]) -> Self:
        hpo_id = legacy.get("hpo_id") or legacy.get("id")
        label = legacy.get("label") or legacy.get("name") or legacy.get("term_name")
        if not hpo_id or not label:
            raise ValueError("legacy phenotype input must include hpo_id and label")

        confidence = legacy.get("confidence")
        if confidence is None:
            confidence = legacy.get("score")

        evidence_text = (
            legacy.get("evidence_text")
            or legacy.get("text")
            or legacy.get("chunk_text")
        )

        raw_spans = legacy.get("spans") or ()
        spans = tuple(
            span
            if isinstance(span, NormalizedSpan)
            else NormalizedSpan.from_legacy_dict(span)
            for span in raw_spans
        )

        chunk_ids = legacy.get("chunk_ids")
        if chunk_ids is None and "chunk_refs" in legacy:
            chunk_ids = legacy["chunk_refs"]
        if chunk_ids is None and "chunk_idx" in legacy:
            chunk_ids = legacy["chunk_idx"]
        if chunk_ids is None and "chunk_id" in legacy:
            chunk_ids = legacy["chunk_id"]

        source_mode = legacy.get("source_mode")
        if source_mode is None:
            source_mode = (
                "chunk"
                if (
                    "chunk_idx" in legacy
                    or "chunk_id" in legacy
                    or "chunk_ids" in legacy
                    or "chunk_refs" in legacy
                    or "chunk_text" in legacy
                )
                else "aggregated"
            )

        match_method = legacy.get("match_method") or "legacy_dict"

        return cls(
            hpo_id=str(hpo_id),
            label=str(label),
            assertion=_normalize_assertion(
                str(
                    legacy.get("assertion")
                    or legacy.get("assertion_status")
                    or "affirmed"
                )
            ),
            confidence=float(confidence) if confidence is not None else None,
            evidence_text=str(evidence_text) if evidence_text is not None else None,
            spans=spans,
            chunk_ids=_normalize_chunk_ids(chunk_ids),
            source_mode=str(source_mode) if source_mode is not None else None,
            match_method=str(match_method) if match_method is not None else None,
        )


__all__ = [
    "AssertionValue",
    "NormalizedPhenotypeExportRecord",
    "NormalizedSpan",
]
