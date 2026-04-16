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
from phentrieve.text_processing.full_text_service import (
    FullTextService,
    adapt_full_text_response,
    adapt_standard_response,
    run_full_text_service,
    run_llm_backend,
    run_standard_backend,
)
from phentrieve.text_processing.spans import TextSpan, find_span_in_text

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
    # Full-text service
    "FullTextService",
    "adapt_full_text_response",
    "adapt_standard_response",
    "run_full_text_service",
    "run_llm_backend",
    "run_standard_backend",
    # Spans
    "TextSpan",
    "find_span_in_text",
]
