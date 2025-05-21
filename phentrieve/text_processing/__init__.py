"""
Phentrieve Text Processing Package

This module provides functionality for processing clinical text documents,
including text chunking, cleaning, and assertion detection.
"""

from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)
from phentrieve.text_processing.chunkers import (
    TextChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    FineGrainedPunctuationChunker,
)
from phentrieve.text_processing.sliding_window_chunker import (
    SlidingWindowSemanticSplitter,
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
]
