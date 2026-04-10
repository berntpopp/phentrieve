"""Registry-based factory for chunker construction.

Each entry in CHUNKER_FACTORIES maps a chunker_type string to a callable
``(language, chunker_config, sbert_model) -> TextChunker``. Legacy aliases
(e.g. "sliding_window_semantic" -> "sliding_window") are resolved via
CHUNKER_ALIASES before lookup.

Split out from pipeline.py::_create_chunkers() so the factory logic is
straight-line, table-driven, and testable, with no duplicate branches
(the prior if/elif chain handled "sliding_window" twice — once reachable
and once dead).
"""

from collections.abc import Callable
from typing import Any

from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FinalChunkCleaner,
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
    TextChunker,
)

# Legacy aliases resolved before the main lookup
CHUNKER_ALIASES: dict[str, str] = {
    "sliding_window_semantic": "sliding_window",
}


def _make_simple(cls: type[TextChunker]) -> Callable[..., TextChunker]:
    """Return a factory that ignores config and instantiates ``cls(language=...)``."""

    def factory(
        language: str,
        chunker_config: dict[str, Any],
        sbert_model: SentenceTransformer | None,
    ) -> TextChunker:
        del chunker_config, sbert_model
        return cls(language=language)

    return factory


def _make_sliding_window(
    language: str,
    chunker_config: dict[str, Any],
    sbert_model: SentenceTransformer | None,
) -> TextChunker:
    if sbert_model is None:
        raise ValueError(
            "SentenceTransformer model required for sliding window semantic "
            "splitting but none was provided"
        )
    return SlidingWindowSemanticSplitter(
        language=language,
        model=sbert_model,
        window_size_tokens=chunker_config.get("window_size_tokens", 4),
        step_size_tokens=chunker_config.get("step_size_tokens", 2),
        splitting_threshold=chunker_config.get("splitting_threshold", 0.5),
        min_split_segment_length_words=chunker_config.get(
            "min_split_segment_length_words", 50
        ),
    )


def _make_final_chunk_cleaner(
    language: str,
    chunker_config: dict[str, Any],
    sbert_model: SentenceTransformer | None,
) -> TextChunker:
    del sbert_model
    params: dict[str, Any] = {
        "language": language,
        "min_cleaned_chunk_length_chars": chunker_config.get(
            "min_cleaned_chunk_length_chars", 1
        ),
        "filter_short_low_value_chunks_max_words": chunker_config.get(
            "filter_short_low_value_chunks_max_words", 2
        ),
        "max_cleanup_passes": chunker_config.get("max_cleanup_passes", 3),
    }
    # Optional custom lists — only forward when explicitly provided so the
    # FinalChunkCleaner falls back to loaded resources when absent.
    for key in (
        "custom_leading_words_to_remove",
        "custom_trailing_words_to_remove",
        "custom_leading_punctuation",
        "custom_trailing_punctuation",
        "custom_low_value_words",
    ):
        value = chunker_config.get(key)
        if value is not None:
            params[key] = value
    return FinalChunkCleaner(**params)


ChunkerFactory = Callable[
    [str, dict[str, Any], SentenceTransformer | None], TextChunker
]

CHUNKER_FACTORIES: dict[str, ChunkerFactory] = {
    "paragraph": _make_simple(ParagraphChunker),
    "sentence": _make_simple(SentenceChunker),
    "fine_grained_punctuation": _make_simple(FineGrainedPunctuationChunker),
    "conjunction": _make_simple(ConjunctionChunker),
    "noop": _make_simple(NoOpChunker),
    "sliding_window": _make_sliding_window,
    "final_chunk_cleaner": _make_final_chunk_cleaner,
}


def build_chunker(
    chunker_type: str,
    chunker_config: dict[str, Any],
    language: str,
    sbert_model: SentenceTransformer | None,
) -> TextChunker | None:
    """Build a chunker from its type string, or return None for unknown types.

    Aliases are resolved before lookup. The caller (pipeline.py) is
    responsible for logging the unknown-type warning and for the
    empty-pipeline NoOp fallback.
    """
    resolved_type = CHUNKER_ALIASES.get(chunker_type, chunker_type)
    factory = CHUNKER_FACTORIES.get(resolved_type)
    if factory is None:
        return None
    return factory(language, chunker_config, sbert_model)
