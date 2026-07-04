"""Normalized export models shared by Phenopacket and sidecar exports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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


def _resolve_legacy_assertion(legacy: Mapping[str, Any]) -> str:
    """Pick the assertion source from a legacy dict (B0 export contract).

    The canonical export fields (``assertion`` / ``assertion_status``) win when
    present, but the shared full-text service emits aggregated terms keyed on the
    raw pipeline ``status`` (present | negated | normal | ...) plus a derived
    ``excluded`` bool, and carries NEITHER canonical field. Reading ``status``
    and then the ``excluded`` flag here ensures a ruled-out finding routed
    through the aggregated phenopacket/sidecar path exports as ``negated`` rather
    than silently defaulting to ``affirmed`` (present)."""
    explicit = (
        legacy.get("assertion")
        or legacy.get("assertion_status")
        or legacy.get("status")
    )
    if explicit:
        return str(explicit)
    if legacy.get("excluded"):
        return "negated"
    return "affirmed"


def _normalize_chunk_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, tuple):
        return [int(chunk_id) for chunk_id in value]
    if isinstance(value, list):
        return [int(chunk_id) for chunk_id in value]
    return [int(value)]


@dataclass(frozen=True, slots=True, init=False)
class NormalizedSpan:
    """Normalized span for a mapped phenotype mention."""

    evidence_text: str
    start_char: int
    end_char: int
    chunk_ids: list[int] = field(default_factory=list)

    def __init__(
        self,
        *,
        text: str | None = None,
        evidence_text: str | None = None,
        start_char: int,
        end_char: int,
        chunk_ids: Sequence[int] | int | None = None,
        chunk_refs: Sequence[int] | int | None = None,
    ) -> None:
        resolved_text = evidence_text if evidence_text is not None else text
        if not isinstance(resolved_text, str) or not resolved_text:
            raise ValueError("NormalizedSpan requires text or evidence_text")
        if start_char < 0:
            raise ValueError("start_char must be non-negative")
        if end_char < start_char:
            raise ValueError("end_char must be greater than or equal to start_char")

        resolved_chunk_ids = _normalize_chunk_ids(
            chunk_ids if chunk_ids is not None else chunk_refs
        )
        object.__setattr__(self, "evidence_text", resolved_text)
        object.__setattr__(self, "start_char", int(start_char))
        object.__setattr__(self, "end_char", int(end_char))
        object.__setattr__(self, "chunk_ids", resolved_chunk_ids)

    @property
    def text(self) -> str:
        return self.evidence_text

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
            text=evidence_text,
            start_char=int(start_char),
            end_char=int(end_char),
            chunk_ids=_normalize_chunk_ids(chunk_ids),
        )


@dataclass(frozen=True, slots=True, init=False)
class NormalizedPhenotypeExportRecord:
    """Normalized phenotype record for export and sidecar linkage."""

    hpo_id: str
    label: str
    assertion: AssertionValue
    certainty: str | None = None
    confidence: float | None = None
    evidence_text: str | None = None
    spans: list[NormalizedSpan] = field(default_factory=list)
    chunk_refs: list[int] = field(default_factory=list)
    source_mode: str | None = None
    match_method: str | None = None

    def __init__(
        self,
        *,
        hpo_id: str,
        label: str,
        assertion: AssertionValue | str,
        certainty: str | None = None,
        confidence: float | None = None,
        evidence_text: str | None = None,
        spans: Iterable[NormalizedSpan | Mapping[str, Any]] | None = None,
        chunk_refs: Sequence[int] | int | None = None,
        chunk_ids: Sequence[int] | int | None = None,
        source_mode: str | None = None,
        match_method: str | None = None,
    ) -> None:
        if not hpo_id or not label:
            raise ValueError(
                "NormalizedPhenotypeExportRecord requires hpo_id and label"
            )

        normalized_spans = [
            span
            if isinstance(span, NormalizedSpan)
            else NormalizedSpan.from_legacy_dict(span)
            for span in (spans or ())
        ]
        resolved_chunk_refs = _normalize_chunk_ids(
            chunk_refs if chunk_refs is not None else chunk_ids
        )
        if not resolved_chunk_refs and normalized_spans:
            span_chunk_refs: list[int] = []
            for span in normalized_spans:
                for chunk_ref in span.chunk_ids:
                    if chunk_ref not in span_chunk_refs:
                        span_chunk_refs.append(chunk_ref)
            resolved_chunk_refs = span_chunk_refs

        resolved_evidence_text = evidence_text
        if resolved_evidence_text is None and normalized_spans:
            resolved_evidence_text = normalized_spans[0].evidence_text

        object.__setattr__(self, "hpo_id", str(hpo_id))
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "assertion", _normalize_assertion(str(assertion)))
        object.__setattr__(self, "certainty", certainty)
        object.__setattr__(
            self, "confidence", float(confidence) if confidence is not None else None
        )
        object.__setattr__(self, "evidence_text", resolved_evidence_text)
        object.__setattr__(self, "spans", normalized_spans)
        object.__setattr__(self, "chunk_refs", resolved_chunk_refs)
        object.__setattr__(self, "source_mode", source_mode)
        object.__setattr__(self, "match_method", match_method)

    @property
    def chunk_ids(self) -> list[int]:
        return list(self.chunk_refs)

    @classmethod
    def from_legacy_dict(cls, legacy: Mapping[str, Any]) -> Self:
        hpo_id = legacy.get("hpo_id") or legacy.get("id")
        label = legacy.get("label") or legacy.get("name") or legacy.get("term_name")
        if not hpo_id or not label:
            raise ValueError("legacy phenotype input must include hpo_id and label")

        confidence = legacy.get("confidence")
        if confidence is None:
            confidence = legacy.get("score")
        certainty = legacy.get("certainty")

        evidence_text = (
            legacy.get("evidence_text")
            or legacy.get("text")
            or legacy.get("chunk_text")
        )

        raw_spans = legacy.get("spans") or ()
        spans = [
            span
            if isinstance(span, NormalizedSpan)
            else NormalizedSpan.from_legacy_dict(span)
            for span in raw_spans
        ]

        chunk_refs = legacy.get("chunk_refs")
        if chunk_refs is None and "chunk_ids" in legacy:
            chunk_refs = legacy["chunk_ids"]
        if chunk_refs is None and "chunk_idx" in legacy:
            chunk_refs = legacy["chunk_idx"]
        if chunk_refs is None and "chunk_id" in legacy:
            chunk_refs = legacy["chunk_id"]

        source_mode = legacy.get("source_mode")
        if source_mode is None:
            source_mode = (
                "chunk"
                if (
                    "chunk_idx" in legacy
                    or "chunk_id" in legacy
                    or "chunk_refs" in legacy
                    or "chunk_ids" in legacy
                    or "chunk_text" in legacy
                )
                else "aggregated"
            )

        match_method = legacy.get("match_method") or "legacy_dict"

        return cls(
            hpo_id=str(hpo_id),
            label=str(label),
            assertion=_normalize_assertion(_resolve_legacy_assertion(legacy)),
            certainty=str(certainty) if certainty is not None else None,
            confidence=float(confidence) if confidence is not None else None,
            evidence_text=str(evidence_text) if evidence_text is not None else None,
            spans=spans,
            chunk_refs=_normalize_chunk_ids(chunk_refs),
            source_mode=str(source_mode) if source_mode is not None else None,
            match_method=str(match_method) if match_method is not None else None,
        )


__all__ = [
    "AssertionValue",
    "NormalizedPhenotypeExportRecord",
    "NormalizedSpan",
]
