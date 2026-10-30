"""Document offsets for phase-1 evidence, computed in code.

The model quotes ``evidence_text`` and cites ``chunk_ids``; character offsets
it reports itself are unreliable. The span is the evidence text located inside
the cited chunks, whose document positions are known from preprocessing.
Matching tolerates typography a model may alter (quotes, dashes, repeated or
non-breaking whitespace, letter case, Unicode composition, soft hyphens and
hyphenation at line breaks); offsets always refer to the original document.
"""

from __future__ import annotations

import re
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


# A hyphen that splits a word across a line break ("Hämo-\nglobin"), as in
# text pasted from PDFs; it is dropped together with the line break.
_LINE_BREAK_HYPHEN = re.compile(r"[-\u2010\u2011\u00ad][^\S\r\n]*[\r\n]\s*")
_SOFT_HYPHEN = "\u00ad"
_QUOTED_LINE_BREAK_HYPHEN = re.compile(r"(?<=\w)- (?=\w)")

# Separators the phase-1 chunk index shows after the chunk they follow.
_SEPARATORS = " \t\r\n.,;:!?)]"


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
        if index > 0 and (
            text[index - 1].isalnum() or unicodedata.combining(text[index - 1])
        ):
            hyphenation = _LINE_BREAK_HYPHEN.match(text, index)
            if hyphenation:
                index = hyphenation.end()
                continue
        if text[index] == _SOFT_HYPHEN:
            index += 1
            continue
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


def is_located(start: object, end: object) -> bool:
    """Whether a chunk has a document position.

    Chunks the text pipeline could not locate carry ``-1`` (or no) positions.
    """
    return isinstance(start, int) and isinstance(end, int) and 0 <= start <= end


@dataclass(frozen=True)
class RenderedRegion:
    """Document region the phase-1 chunk index shows for one located chunk.

    ``index`` is the chunk's position in the input sequence. The region runs
    from the source text that cleanup dropped before the chunk (``start``) to
    the separators that follow it (``end``); ``chunk_start``/``chunk_end`` are
    the chunk's own positions.
    """

    index: int
    start: int
    end: int
    chunk_start: int
    chunk_end: int


def rendered_regions(
    text: str, spans: Sequence[tuple[object, object]]
) -> list[RenderedRegion]:
    """Walk the gaps between located chunks, given as ``(start, end)`` pairs.

    The dropped text before a chunk is what lies after the previous located
    chunk, without the separators that are shown after that chunk instead. A
    chunk that follows an unlocated chunk gets no dropped text, because that
    gap still holds the unlocated chunk's own text.
    """
    located: list[tuple[int, int, int]] = [
        (index, start, end)
        for index, (start, end) in enumerate(spans)
        if isinstance(start, int) and isinstance(end, int) and is_located(start, end)
    ]
    located_indices = {index for index, _, _ in located}
    regions: list[RenderedRegion] = []
    previous_end = 0
    for position, (index, start, end) in enumerate(located):
        if index > 0 and index - 1 not in located_indices:
            region_start = start
        else:
            gap_start = min(previous_end, start)
            gap = text[gap_start:start]
            region_start = start - len(gap.lstrip(_SEPARATORS))
        next_start = (
            located[position + 1][1] if position + 1 < len(located) else len(text)
        )
        following = text[end : max(end, next_start)]
        region_end = end + len(following) - len(following.lstrip(_SEPARATORS))
        regions.append(RenderedRegion(index, region_start, region_end, start, end))
        previous_end = max(previous_end, end)
    return regions


def _adjacent_runs(chunk_ids: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for chunk_id in chunk_ids:
        if runs and chunk_id == runs[-1][-1] + 1:
            runs[-1].append(chunk_id)
        else:
            runs.append([chunk_id])
    return runs


class EvidenceLocator:
    """Locates evidence in one document.

    Each search region is normalized once and reused for every finding that
    cites it, so long notes with many findings do not re-normalize the same
    chunks over and over.
    """

    def __init__(
        self,
        document: str,
        chunk_spans: Mapping[int, tuple[int, int]],
        *,
        search_spans: Mapping[int, tuple[int, int]] | None = None,
    ) -> None:
        self._document = document
        self._chunk_spans = chunk_spans
        self._search_spans = search_spans or chunk_spans
        self._normalized_regions: dict[
            tuple[int, int], tuple[str, list[int], list[int]]
        ] = {}

    def _find(self, evidence: str, start: int, end: int) -> tuple[int, int] | None:
        """First match of ``evidence`` lying within document[start:end]."""
        needle = _normalized(evidence.strip())[0]
        if not needle:
            return None
        region = self._normalized_regions.get((start, end))
        if region is None:
            region = _normalized(self._document[start:end])
            self._normalized_regions[(start, end)] = region
        haystack, starts, ends = region
        position = haystack.find(needle)
        if position == -1:
            # The chunk index renders "hemo-\nglobin" as "hemo- globin", so a quote
            # can keep a hyphen that the normalized document no longer has.
            needle = _QUOTED_LINE_BREAK_HYPHEN.sub("", needle)
            position = haystack.find(needle)
        if position == -1:
            return None
        return start + starts[position], start + ends[position + len(needle) - 1]

    def locate(
        self, evidence_text: str | None, chunk_ids: Sequence[int]
    ) -> EvidenceSpan:
        """Document span of one piece of evidence (see ``locate_evidence``)."""
        chunk_spans = self._chunk_spans
        ids = [
            c
            for c in sorted(set(chunk_ids))
            if c in chunk_spans and is_located(*chunk_spans[c])
        ]
        if not ids:
            return EvidenceSpan(None, None, "unresolved")
        runs = _adjacent_runs(ids)
        if evidence_text:
            regions = {c: self._search_spans.get(c, chunk_spans[c]) for c in ids}
            for chunk_id in ids:
                found = self._find(evidence_text, *regions[chunk_id])
                if found:
                    return EvidenceSpan(*found, "evidence_in_chunk")
            for run in runs:
                if len(run) < 2:
                    continue
                hull = (
                    min(regions[c][0] for c in run),
                    max(regions[c][1] for c in run),
                )
                found = self._find(evidence_text, *hull)
                if found:
                    return EvidenceSpan(*found, "evidence_in_chunk_hull")
        run = runs[0]
        return EvidenceSpan(
            chunk_spans[run[0]][0], chunk_spans[run[-1]][1], "chunk_span"
        )


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
    located position are ignored. To locate many findings in one document,
    reuse one ``EvidenceLocator``.
    """
    return EvidenceLocator(document, chunk_spans, search_spans=search_spans).locate(
        evidence_text, chunk_ids
    )
