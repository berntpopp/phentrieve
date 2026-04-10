"""Characterization tests for TextProcessingPipeline._create_chunkers.

Locks the factory behavior before the Task 7 refactor to a registry.
"""

from unittest.mock import MagicMock

import pytest

from phentrieve.text_processing.chunkers import (
    ConjunctionChunker,
    FinalChunkCleaner,
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowSemanticSplitter,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline

pytestmark = pytest.mark.unit


def _build(pipeline_config, model=None):
    return TextProcessingPipeline(
        language="en",
        chunking_pipeline_config=pipeline_config,
        assertion_config={"disable": True},
        sbert_model_for_semantic_chunking=model,
    )


class TestBasicChunkerTypes:
    def test_paragraph(self):
        pipe = _build([{"type": "paragraph"}])
        assert len(pipe.chunkers) == 1
        assert isinstance(pipe.chunkers[0], ParagraphChunker)

    def test_sentence(self):
        pipe = _build([{"type": "sentence"}])
        assert isinstance(pipe.chunkers[0], SentenceChunker)

    def test_fine_grained_punctuation(self):
        pipe = _build([{"type": "fine_grained_punctuation"}])
        assert isinstance(pipe.chunkers[0], FineGrainedPunctuationChunker)

    def test_conjunction(self):
        pipe = _build([{"type": "conjunction"}])
        assert isinstance(pipe.chunkers[0], ConjunctionChunker)

    def test_noop(self):
        pipe = _build([{"type": "noop"}])
        assert isinstance(pipe.chunkers[0], NoOpChunker)


class TestSlidingWindow:
    def test_sliding_window_requires_model(self):
        with pytest.raises(ValueError, match="SentenceTransformer model required"):
            _build([{"type": "sliding_window"}], model=None)

    def test_sliding_window_with_model(self):
        mock_model = MagicMock()
        pipe = _build(
            [
                {
                    "type": "sliding_window",
                    "config": {
                        "window_size_tokens": 4,
                        "step_size_tokens": 2,
                        "splitting_threshold": 0.5,
                        "min_split_segment_length_words": 50,
                    },
                }
            ],
            model=mock_model,
        )
        assert isinstance(pipe.chunkers[0], SlidingWindowSemanticSplitter)

    def test_sliding_window_semantic_alias(self):
        """sliding_window_semantic is a legacy alias for sliding_window."""
        mock_model = MagicMock()
        pipe = _build([{"type": "sliding_window_semantic"}], model=mock_model)
        assert isinstance(pipe.chunkers[0], SlidingWindowSemanticSplitter)


class TestFinalChunkCleaner:
    def test_default_final_chunk_cleaner(self):
        pipe = _build([{"type": "final_chunk_cleaner"}])
        assert isinstance(pipe.chunkers[0], FinalChunkCleaner)


class TestMultiStage:
    def test_full_pipeline(self):
        mock_model = MagicMock()
        pipe = _build(
            [
                {"type": "paragraph"},
                {"type": "sliding_window"},
                {"type": "fine_grained_punctuation"},
                {"type": "final_chunk_cleaner"},
            ],
            model=mock_model,
        )
        assert len(pipe.chunkers) == 4
        assert isinstance(pipe.chunkers[0], ParagraphChunker)
        assert isinstance(pipe.chunkers[1], SlidingWindowSemanticSplitter)
        assert isinstance(pipe.chunkers[2], FineGrainedPunctuationChunker)
        assert isinstance(pipe.chunkers[3], FinalChunkCleaner)


class TestFallback:
    def test_unknown_chunker_type_skipped(self):
        """Unknown types log warning and are skipped; empty result falls back to NoOp."""
        pipe = _build([{"type": "does_not_exist"}])
        assert len(pipe.chunkers) == 1
        assert isinstance(pipe.chunkers[0], NoOpChunker)

    def test_empty_config_falls_back_to_noop(self):
        pipe = _build([])
        assert len(pipe.chunkers) == 1
        assert isinstance(pipe.chunkers[0], NoOpChunker)
