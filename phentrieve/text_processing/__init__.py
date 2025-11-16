"""
Phentrieve Text Processing Package

This module provides functionality for processing clinical text documents,
including text chunking, cleaning, and assertion detection.
"""

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
    TextChunker,
)
from phentrieve.text_processing.cleaners import (
    clean_internal_newlines_and_extra_spaces,
    normalize_line_endings,
)

__all__ = [
    # Cleaners
    "normalize_line_endings",
    "clean_internal_newlines_and_extra_spaces",
    # Chunkers
    "TextChunker",
    "NoOpChunker",
    "ParagraphChunker",
    "SentenceChunker",
    "FineGrainedPunctuationChunker",
    "SlidingWindowSemanticSplitter",
    "ConjunctionChunker",
]
