"""Text span with position tracking for chunk localization.

This module provides the TextSpan dataclass for tracking character positions
of text chunks within their source document, enabling text highlighting and
benchmark comparisons with position-annotated datasets like PhenoBERT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Immutable text span with document position.

    Represents a substring of text along with its character positions
    in the source document. Positions follow Python slice conventions:
    start_char is inclusive, end_char is exclusive.

    Attributes:
        text: The text content of the span
        start_char: Start position in document (0-indexed, inclusive)
        end_char: End position in document (exclusive, like Python slicing)

    Example:
        >>> span = TextSpan("hello", start_char=0, end_char=5)
        >>> doc = "hello world"
        >>> doc[span.start_char:span.end_char]
        'hello'
    """

    text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        """Validate span positions."""
        if self.start_char < 0:
            raise ValueError(f"start_char must be >= 0, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be >= start_char ({self.start_char})"
            )

    def __str__(self) -> str:
        """Return the text content."""
        return self.text

    def __len__(self) -> int:
        """Return length of text content."""
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def find_span_in_text(
    needle: str,
    haystack: str,
    search_start: int = 0,
) -> TextSpan | None:
    """Find text in document and return its position as a TextSpan.

    Uses exact string matching first, then falls back to whitespace-normalized
    matching if exact match fails. This handles cases where chunking may have
    slightly different whitespace than the original document.

    Args:
        needle: The text to find
        haystack: The document to search in
        search_start: Position to start searching from (for handling duplicates)

    Returns:
        TextSpan with position if found, None otherwise

    Example:
        >>> span = find_span_in_text("world", "hello world")
        >>> span.start_char, span.end_char
        (6, 11)
    """
    if not needle or not haystack:
        return None

    # Try exact match first (most common case)
    pos = haystack.find(needle, search_start)
    if pos != -1:
        return TextSpan(
            text=haystack[pos : pos + len(needle)],
            start_char=pos,
            end_char=pos + len(needle),
        )

    # Whitespace-normalized fallback for edge cases
    # This handles cases where whitespace was modified during processing
    norm_needle = re.sub(r"\s+", " ", needle.strip())
    norm_haystack = re.sub(r"\s+", " ", haystack)

    pos = norm_haystack.find(norm_needle, search_start)
    if pos != -1:
        # Return original needle text but with normalized position
        return TextSpan(
            text=needle,
            start_char=pos,
            end_char=pos + len(norm_needle),
        )

    return None
