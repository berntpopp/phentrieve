"""Document offsets for phase-1 evidence, computed in code.

The model quotes ``evidence_text`` and cites ``chunk_ids``; character offsets
it reports itself are unreliable. The span is the evidence text located inside
the cited chunks, whose document positions are known from preprocessing.
Matching tolerates typography a model may alter (quotes, dashes, repeated or
non-breaking whitespace, letter case, Unicode composition); offsets always
refer to the original document.
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
        # A base character and its combining marks are composed and case-folded
        # together; every resulting character maps to the whole cluster.
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = unicodedata.normalize("NFC", text[index:end]).casefold()
        for character in cluster.translate(_TYPOGRAPHY):
            characters.append(character)
            starts.append(index)
            ends.append(end)
        index = end
    return "".join(characters), starts, ends


def _is_located(span: tuple[int, int]) -> bool:
    """Chunks the text pipeline could not locate carry ``-1`` positions."""
    return 0 <= span[0] <= span[1]


def _adjacent_runs(chunk_ids: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for chunk_id in chunk_ids:
        if runs and chunk_id == runs[-1][-1] + 1:
            runs[-1].append(chunk_id)
        else:
            runs.append([chunk_id])
    return runs


def _find(document: str, evidence: str, start: int, end: int) -> tuple[int, int] | None:
    """First match of ``evidence`` lying within document[start:end]."""
    haystack, starts, ends = _normalized(document[start:end])
    needle = _normalized(evidence.strip())[0]
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
    *,
    search_spans: Mapping[int, tuple[int, int]] | None = None,
) -> EvidenceSpan:
    """Document span of one piece of evidence.

    Tried in order: the evidence inside one cited chunk (first in chunk-id
    order), inside the hull of a run of adjacent cited chunks (evidence that
    crosses chunk borders; a hull never spans an uncited chunk), and finally
    the first run of adjacent cited chunks themselves. ``search_spans`` may
    widen a chunk's search region (for example by text that cleanup dropped
    before it); the fallback always uses ``chunk_spans``. Chunks without a
    located position are ignored.
    """
    search_spans = search_spans or chunk_spans
    ids = [
        c
        for c in sorted(set(chunk_ids))
        if c in chunk_spans and _is_located(chunk_spans[c])
    ]
    if not ids:
        return EvidenceSpan(None, None, "unresolved")
    runs = _adjacent_runs(ids)
    if evidence_text:
        regions = {c: search_spans.get(c, chunk_spans[c]) for c in ids}
        for chunk_id in ids:
            found = _find(document, evidence_text, *regions[chunk_id])
            if found:
                return EvidenceSpan(*found, "evidence_in_chunk")
        for run in runs:
            if len(run) < 2:
                continue
            hull = (
                min(regions[c][0] for c in run),
                max(regions[c][1] for c in run),
            )
            found = _find(document, evidence_text, *hull)
            if found:
                return EvidenceSpan(*found, "evidence_in_chunk_hull")
    run = runs[0]
    return EvidenceSpan(chunk_spans[run[0]][0], chunk_spans[run[-1]][1], "chunk_span")
