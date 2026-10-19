"""Document offsets for phase-1 evidence, computed in code.

The model quotes ``evidence_text`` and cites ``chunk_ids``; character offsets
it reports itself are unreliable. The span is the evidence text located inside
the cited chunks, whose document positions are known from preprocessing.
Matching tolerates typography a model may alter (quotes, dashes, repeated or
non-breaking whitespace); offsets always refer to the original document.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

OffsetBasis = Literal[
    "evidence_in_chunk", "evidence_in_chunk_hull", "chunk_span", "unresolved"
]

_TYPOGRAPHY = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


@dataclass(frozen=True)
class EvidenceSpan:
    start: int | None
    end: int | None
    basis: OffsetBasis


def _normalized(text: str) -> tuple[str, list[int], list[int]]:
    """Normalized text; character i covers original [starts[i], ends[i])."""
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            characters.append(" ")
            starts.append(index)
            ends.append(end)
            index = end
            continue
        characters.append(text[index].translate(_TYPOGRAPHY))
        starts.append(index)
        ends.append(index + 1)
        index += 1
    return "".join(characters), starts, ends


def _find(document: str, evidence: str, start: int, end: int) -> tuple[int, int] | None:
    """First match of ``evidence`` lying within document[start:end]."""
    haystack, starts, ends = _normalized(document[start:end])
    needle = _normalized(unicodedata.normalize("NFC", evidence).strip())[0]
    if not needle:
        return None
    position = haystack.find(needle)
    if position == -1:
        return None
    return start + starts[position], start + ends[position + len(needle) - 1]


def locate_evidence(
    document: str,
    evidence_text: str | None,
    chunk_ids: Sequence[int],
    chunk_spans: Mapping[int, tuple[int, int]],
) -> EvidenceSpan:
    """Document span of one piece of evidence.

    Tried in order: the evidence inside one cited chunk (first in chunk-id
    order), inside the hull of all cited chunks (evidence that crosses chunk
    borders), and finally the hull of the cited chunks themselves.
    """
    cited = [chunk_spans[c] for c in sorted(set(chunk_ids)) if c in chunk_spans]
    if not cited:
        return EvidenceSpan(None, None, "unresolved")
    hull = (min(s for s, _ in cited), max(e for _, e in cited))
    if evidence_text:
        for start, end in cited:
            found = _find(document, evidence_text, start, end)
            if found:
                return EvidenceSpan(*found, "evidence_in_chunk")
        found = _find(document, evidence_text, *hull)
        if found:
            return EvidenceSpan(*found, "evidence_in_chunk_hull")
    return EvidenceSpan(*hull, "chunk_span")
