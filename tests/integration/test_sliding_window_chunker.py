"""Integration tests for sliding window semantic chunker (pytest style)."""

import logging
import string
import pytest
from sentence_transformers import SentenceTransformer

from phentrieve.config import get_sliding_window_config_with_params
from phentrieve.text_processing.chunkers import SlidingWindowSemanticSplitter
from phentrieve.text_processing.pipeline import TextProcessingPipeline


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def small_model():
    """Small embedding model for testing (module-scoped)."""
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")


@pytest.fixture(scope="module")
def biolord_model():
    """BioLORD model for clinical testing (module-scoped)."""
    logging.basicConfig(level=logging.ERROR)
    return SentenceTransformer("FremyCompany/BioLORD-2023-M")


class TestSlidingWindowSplitter:
    """Test cases for SlidingWindowSemanticSplitter directly."""

    @pytest.fixture(autouse=True)
    def setup(self, small_model):
        """Initialize resources for tests."""
        self.model = small_model
        self.chunker = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=5,
            step_size_tokens=1,
            splitting_threshold=0.7,
            min_split_segment_length_words=2,
        )

    def test_empty_input(self):
        """Test empty input produces empty output."""
        result = self.chunker.chunk([])
        assert result == []

        result = self.chunker.chunk([""])
        assert result == []

        result = self.chunker.chunk(["   "])
        assert result == []

    def test_short_segment(self):
        """Test short segments are returned as-is."""
        short_text = "Just a few words"  # 4 tokens, less than window_size
        result = self.chunker.chunk([short_text])
        assert result == [short_text]

    def test_semantic_splitting(self):
        """Test text with distinct topics is split appropriately."""
        mixed_topic_text = (
            "Python is a popular programming language used for web development,"
            " data science, and AI. The Sahara desert is the largest hot desert"
            " in the world spanning 11 countries in North Africa."
        )

        result = self.chunker.chunk([mixed_topic_text])

        assert len(result) > 1, "Should split into at least 2 chunks for distinct topics"

        combined_result = " ".join(result)
        translator = str.maketrans("", "", string.punctuation)
        result_text_clean = combined_result.lower().translate(translator)
        result_words = set(result_text_clean.split())

        for key_word in ["python", "programming", "sahara", "desert", "africa"]:
            assert key_word in result_words

    def test_multiple_segments(self):
        """Test multiple input segments are processed correctly."""
        segment1 = "Dogs are mammals with four legs and a tail."
        segment2 = "A novel is a long, fictional narrative."

        result = self.chunker.chunk([segment1, segment2])

        combined_result = " ".join(result)
        for key_word in ["dogs", "mammals", "novel", "fictional", "narrative"]:
            assert key_word.lower() in combined_result.lower()


class TestNegationAwareMerging:
    """Test cases for negation-aware merging."""

    @pytest.fixture(autouse=True)
    def setup(self, small_model):
        """Initialize resources for tests."""
        self.model = small_model
        self.splitter = SlidingWindowSemanticSplitter(
            model=self.model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,  # Very low to force splits
            min_split_segment_length_words=1,
            language="en",
        )

    def test_merge_negation_patterns(self, small_model):
        """Test negation patterns are properly merged."""
        splitter = SlidingWindowSemanticSplitter(
            model=small_model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,
            min_split_segment_length_words=1,
            language="en",
        )

        text_segments = [
            "Patient shows no response to stimuli and no eye contact with the examiner."
        ]
        result = splitter.chunk(text_segments)

        combined = " ".join(result).lower()
        assert "no response" in combined
        assert "no eye contact" in combined

    def test_german_negation_merging(self, small_model):
        """Test German negation patterns are properly merged."""
        german_splitter = SlidingWindowSemanticSplitter(
            model=small_model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,
            min_split_segment_length_words=1,
            language="de",
        )

        text_segments = ["kein Blickkontakt und keine Reaktion"]
        result = german_splitter.chunk(text_segments)

        assert any("kein Blickkontakt" in s for s in result)
        assert any("keine Reaktion" in s for s in result)

    def test_no_merge_after_connector(self, small_model):
        """Test we don't merge after connector words."""
        splitter = SlidingWindowSemanticSplitter(
            model=small_model,
            window_size_tokens=2,
            step_size_tokens=1,
            splitting_threshold=0.1,
            min_split_segment_length_words=1,
            language="en",
        )

        text_segments = ["Patient has no fever but has pain. No cough or cold."]
        result = splitter.chunk(text_segments)

        combined = " ".join(result).lower()
        assert "no fever" in combined
        assert "no cough" in combined
        assert "no but" not in combined

    def test_multiple_negations(self):
        """Test text with multiple negation patterns."""
        text_segments = ["no response, no eye contact, and no movement"]
        result = self.splitter.chunk(text_segments)

        assert any("no response" in s for s in result)
        assert any("no eye contact" in s for s in result)
        assert any("no movement" in s for s in result)


class TestSlidingWindowChunker:
    """Test cases for sliding window chunker functionality."""

    def test_german_clinical_note_chunking(self, biolord_model):
        """Test chunking of German clinical note."""
        text = """
        Trinkschwäche und Hypotonie. Fehlende Fixation und kein Blickkontakt.
        MRT: Kortikale Atrophie und Kleinhirnhypoplasie.
        Mikrozephalie und Krampfanfälle in der Vorgeschichte.
        """

        config = get_sliding_window_config_with_params(
            window_size=2, step_size=1, threshold=0.6, min_segment_length=1
        )

        pipeline = TextProcessingPipeline(
            language="de",
            chunking_pipeline_config=config,
            assertion_config={"disable": True},
            sbert_model_for_semantic_chunking=biolord_model,
        )

        chunks = pipeline.process(text)

        assert len(chunks) > 1, "Should split into multiple chunks"

        for chunk in chunks:
            assert isinstance(chunk, dict), "Chunks should be dictionary objects"
            assert "text" in chunk, "Chunks should contain 'text' key"
            assert len(chunk["text"]) > 0, "Chunk text should not be empty"

    def test_sliding_window_parameters(self, biolord_model):
        """Test sliding window chunker with different parameters."""
        text = "This is a test sentence. This is another sentence with more words."

        configs = [
            get_sliding_window_config_with_params(
                window_size=2, step_size=1, threshold=0.3
            ),
            get_sliding_window_config_with_params(
                window_size=4, step_size=2, threshold=0.7
            ),
        ]

        for i, config in enumerate(configs):
            pipeline = TextProcessingPipeline(
                language="en",
                chunking_pipeline_config=config,
                assertion_config={"disable": True},
                sbert_model_for_semantic_chunking=biolord_model,
            )
            chunks = pipeline.process(text)

            assert len(chunks) > 0, f"Configuration {i} produced no chunks"

            for chunk in chunks:
                assert "text" in chunk, "Chunk should have a 'text' key"
                assert len(chunk["text"]) > 0, "Chunk text should not be empty"
