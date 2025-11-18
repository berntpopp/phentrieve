"""
Pure annotation-position-based chunking.

Independent of Phentrieve's semantic chunking. Uses only character spans
for territory calculation via Voronoi boundaries (midpoint algorithm).

This module provides ground-truth chunking variants for benchmarking
Phentrieve's semantic chunking strategies. The algorithm is purely geometric
and language-agnostic.
"""

from datetime import datetime
from typing import Any

# Version for provenance
CHUNKER_VERSION = "1.0.0"

# Default expansion ratios (KISS: 3 for alpha)
DEFAULT_EXPANSION_RATIOS = [0.0, 0.5, 1.0]


def compute_voronoi_boundaries(
    annotations: list[dict[str, Any]],
    text_length: int,
) -> list[tuple[int, int]]:
    """
    Calculate Voronoi territory boundaries for each annotation.

    Uses midpoint algorithm: each annotation's territory extends to the
    midpoint between its edges and neighboring annotations.

    Args:
        annotations: List of annotations with evidence_spans
        text_length: Total length of document text

    Returns:
        List of (left_boundary, right_boundary) tuples, one per annotation

    Examples:
        >>> annotations = [
        ...     {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        ...     {"evidence_spans": [{"start_char": 50, "end_char": 60}]}
        ... ]
        >>> compute_voronoi_boundaries(annotations, 100)
        [(0, 35), (35, 100)]
    """
    if not annotations:
        return []

    # Sort annotations by position
    sorted_anns = sorted(
        annotations,
        key=lambda a: a["evidence_spans"][0]["start_char"],
    )

    boundaries = []

    for i, ann in enumerate(sorted_anns):
        span = ann["evidence_spans"][0]  # Use first span

        # Left boundary
        if i == 0:
            left = 0
        else:
            prev_end = sorted_anns[i - 1]["evidence_spans"][0]["end_char"]
            left = (prev_end + span["start_char"]) // 2

        # Right boundary
        if i == len(sorted_anns) - 1:
            right = text_length
        else:
            next_start = sorted_anns[i + 1]["evidence_spans"][0]["start_char"]
            right = (span["end_char"] + next_start) // 2

        boundaries.append((left, right))

    return boundaries


def align_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Adjust span boundaries to align with word boundaries.

    Expands left boundary to start of word, and right boundary to end of word.
    Handles punctuation and whitespace correctly.

    Args:
        text: Full document text
        start: Initial start position
        end: Initial end position

    Returns:
        Tuple of (adjusted_start, adjusted_end)

    Examples:
        >>> text = "The quick brown fox jumps"
        >>> align_to_word_boundaries(text, 6, 11)  # "uick "
        (4, 11)  # "quick "
    """
    # Clamp to text boundaries
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))

    # Expand left: move backwards to start of word
    # Skip back through word characters
    while start > 0 and text[start - 1].isalnum():
        start -= 1

    # Skip any leading whitespace/punctuation to next word start
    while start < len(text) and start < end and not text[start].isalnum():
        start += 1

    # Expand right: move forward to end of word
    # Skip forward through word characters
    while end < len(text) and text[end].isalnum():
        end += 1

    # Include trailing whitespace/punctuation up to next word
    while end < len(text) and not text[end].isalnum():
        # Don't go past sentence boundaries
        if text[end] in ".!?":
            end += 1
            break
        end += 1

    return start, end


def generate_chunk_variants(
    doc: dict[str, Any],
    expansion_ratios: list[float] | None = None,
) -> dict[str, Any]:
    """
    Generate chunk variants for all annotations in document.

    Each annotation gets chunks at multiple expansion levels:
    - 0.0: Just the annotation text (minimal)
    - 0.5: 50% of available territory (balanced)
    - 1.0: Full territory to boundaries (maximal)

    Args:
        doc: Document with 'full_text' and 'annotations' fields
        expansion_ratios: Ratios for context expansion (default: [0.0, 0.5, 1.0])

    Returns:
        Dictionary with 'provenance' and 'chunks' fields

    Raises:
        ValueError: If annotations are invalid or overlap
    """
    if expansion_ratios is None:
        expansion_ratios = DEFAULT_EXPANSION_RATIOS

    text = doc["full_text"]
    annotations = doc["annotations"]

    # Validate annotations
    validate_annotations(annotations)

    # Compute boundaries
    boundaries = compute_voronoi_boundaries(annotations, len(text))

    # Generate chunks
    chunks = []

    for ann, (left_bound, right_bound) in zip(annotations, boundaries):
        span = ann["evidence_spans"][0]
        ann_start = span["start_char"]
        ann_end = span["end_char"]

        # Calculate available expansion space
        available_left = ann_start - left_bound
        available_right = right_bound - ann_end

        # Generate variants for each expansion ratio
        variants = {}
        for ratio in expansion_ratios:
            expand_left = int(available_left * ratio)
            expand_right = int(available_right * ratio)

            chunk_start = ann_start - expand_left
            chunk_end = ann_end + expand_right

            # Align to word boundaries
            aligned_start, aligned_end = align_to_word_boundaries(
                text, chunk_start, chunk_end
            )

            # Extract chunk text
            chunk_text = text[aligned_start:aligned_end]

            variants[f"{ratio:.2f}"] = {
                "text": chunk_text,
                "span": [aligned_start, aligned_end],
            }

        chunks.append(
            {
                "hpo_id": ann["hpo_id"],
                "annotation_span": [ann_start, ann_end],
                "variants": variants,
            }
        )

    # Return with provenance
    return {
        "provenance": {
            "script": "generate_chunking_variants.py",
            "script_version": CHUNKER_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "parameters": {
                "expansion_ratios": expansion_ratios,
                "strategy": "voronoi_midpoint",
            },
        },
        "chunks": chunks,
    }


def validate_annotations(annotations: list[dict[str, Any]]) -> None:
    """
    Validate annotation spans.

    Checks for:
    - Invalid spans (start >= end)
    - Negative positions
    - Overlapping annotations

    Args:
        annotations: List of annotations to validate

    Raises:
        ValueError: If annotations are invalid or overlap
    """
    for i, ann in enumerate(annotations):
        for span in ann["evidence_spans"]:
            if span["start_char"] >= span["end_char"]:
                raise ValueError(
                    f"Invalid span in annotation {i}: "
                    f"start={span['start_char']} >= end={span['end_char']}"
                )

            if span["start_char"] < 0:
                raise ValueError(
                    f"Negative position in annotation {i}: {span['start_char']}"
                )

    # Check for overlaps
    sorted_spans = sorted(
        [
            (s["start_char"], s["end_char"], i)
            for i, ann in enumerate(annotations)
            for s in ann["evidence_spans"]
        ]
    )

    for i in range(len(sorted_spans) - 1):
        if sorted_spans[i][1] > sorted_spans[i + 1][0]:
            raise ValueError(
                f"Overlapping annotations detected: "
                f"annotation {sorted_spans[i][2]} overlaps with "
                f"annotation {sorted_spans[i + 1][2]}"
            )
