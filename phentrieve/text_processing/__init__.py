"""
Phentrieve Text Processing Package

This module provides functionality for processing clinical text documents,
including text chunking, cleaning, assertion detection, graph-based reasoning,
and mention-level HPO extraction.

Note: Mention-level extraction modules (mention.py, mention_extractor.py, etc.)
are not exported at the package level to avoid heavy imports (spacy/torch) at
import time. Import them directly when needed:

    from phentrieve.text_processing.mention_extraction_orchestrator import (
        MentionExtractionOrchestrator,
        orchestrate_mention_extraction,
    )
"""

# Graph-based assertion propagation (Phase 2)
from phentrieve.text_processing.assertion_propagation import (
    AssertionPropagator,
    PropagationConfig,
    PropagationResult,
    propagate_assertions,
)

# Graph-based reasoning modules (Phase 1)
from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    affirmed_vector,
    negated_vector,
    normal_vector,
    uncertain_vector,
)
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
from phentrieve.text_processing.semantic_graph import (
    ChunkNode,
    GraphEdge,
    SemanticDocumentGraph,
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
    # Spans
    "TextSpan",
    "find_span_in_text",
    # Graph-based assertion representation (Phase 1)
    "AssertionVector",
    "affirmed_vector",
    "negated_vector",
    "normal_vector",
    "uncertain_vector",
    # Semantic document graph (Phase 1)
    "ChunkNode",
    "GraphEdge",
    "SemanticDocumentGraph",
    # Assertion propagation (Phase 2)
    "AssertionPropagator",
    "PropagationConfig",
    "PropagationResult",
    "propagate_assertions",
]
