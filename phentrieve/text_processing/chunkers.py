"""
Text chunking utilities for Phentrieve.

This module provides classes for breaking down text into processable chunks
using various strategies, from simple paragraph splitting to semantic grouping.
The primary semantic chunking is handled by SlidingWindowSemanticSplitter in a separate module.
"""

import re
import logging
import pysbd
from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)

# SlidingWindowSemanticSplitter is implemented in its own module
# to avoid circular imports

logger = logging.getLogger(__name__)


class TextChunker(ABC):
    """
    Abstract base class for text chunking strategies.

    All chunkers must implement a chunk method that takes a list of text
    segments and returns a list of potentially modified or split text segments.
    """

    def __init__(self, language: str = "en", **kwargs):
        """
        Initialize the chunker.

        Args:
            language: ISO language code ('en', 'de', etc.)
            **kwargs: Additional configuration parameters for specific chunker
                implementations
        """
        self.language = language

    @abstractmethod
    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Processes a list of input text segments and returns a new list of
        potentially modified, split, or filtered text segments.
        Each input segment can be split into multiple output segments by a chunker.

        Args:
            text_segments: List of input text segments to process

        Returns:
            List of text chunks
        """
        pass


class NoOpChunker(TextChunker):
    """
    Simple chunker that returns the text segments unchanged.

    This is useful as a baseline or when no chunking is desired.
    """

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Return the text segments after cleaning, without further splitting.

        Args:
            text_segments: List of input text segments

        Returns:
            List of cleaned text segments
        """
        all_output_segments = []
        for segment_str in text_segments:
            if not segment_str or not segment_str.strip():
                logger.debug("NoOpChunker: Skipping empty or whitespace-only segment.")
                continue

            cleaned_text = clean_internal_newlines_and_extra_spaces(segment_str.strip())
            if cleaned_text:
                all_output_segments.append(cleaned_text)

        return all_output_segments


class ParagraphChunker(TextChunker):
    """
    Chunker that splits text into paragraphs based on blank lines.
    """

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Split text segments into paragraphs by looking for blank lines.

        Args:
            text_segments: List of input text segments to split into paragraphs

        Returns:
            List of paragraph chunks
        """
        all_paragraph_chunks = []
        for segment_str in text_segments:
            if not segment_str or not segment_str.strip():
                logger.debug(
                    "ParagraphChunker: Skipping empty or whitespace-only segment."
                )
                continue

            # Apply original paragraph splitting logic to 'segment_str'
            normalized_text = normalize_line_endings(segment_str)
            paragraphs_from_segment = re.split(r"\n\s*\n+", normalized_text)
            # Filter out empty strings that can result from re.split
            all_paragraph_chunks.extend(
                [p.strip() for p in paragraphs_from_segment if p.strip()]
            )

        logger.debug(
            f"ParagraphChunker produced {len(all_paragraph_chunks)} segments from {len(text_segments)} input segments."
        )
        return all_paragraph_chunks


class SentenceChunker(TextChunker):
    """
    Chunker that splits text into individual sentences using pysbd.
    """

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Split text segments into sentences using pysbd with fallback method.

        Args:
            text_segments: List of text segments to split into sentences

        Returns:
            List of sentence chunks
        """
        all_sentences = []
        for segment_str in text_segments:
            if not segment_str or not segment_str.strip():
                logger.debug("SentenceChunker: Skipping empty segment.")
                continue

            sentences_from_segment = self._segment_into_sentences(
                segment_str, self.language
            )
            all_sentences.extend(sentences_from_segment)

        logger.debug(
            f"SentenceChunker produced {len(all_sentences)} sentences "
            f"from {len(text_segments)} input segments."
        )
        return all_sentences

    def _segment_into_sentences(self, text: str, lang: str = "en") -> List[str]:
        """
        Split text into sentences using pysbd with fallback to regex for errors.

        Args:
            text: Text to segment
            lang: Language code for pysbd

        Returns:
            List of sentence strings
        """
        if not text.strip():
            return []

        try:
            segmenter = pysbd.Segmenter(language=lang, clean=False)
            sentences = segmenter.segment(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(
                f"pysbd error for lang '{lang}' on text '{text[:50]}...': {e}. "
                "Using fallback sentence splitting."
            )
            # Fallback method for sentence splitting
            processed_lines = []
            for line in text.split("\n"):
                stripped_line = line.strip()
                if stripped_line:
                    if not stripped_line.endswith((".", "?", "!")):
                        processed_lines.append(stripped_line + ".")
                    else:
                        processed_lines.append(stripped_line)

            text_for_fallback_splitting = " ".join(processed_lines)
            sentences_raw = re.findall(
                r"[^.?!]+(?:[.?!]|$)", text_for_fallback_splitting
            )
            return [s.strip() for s in sentences_raw if s.strip()]


class FineGrainedPunctuationChunker(TextChunker):
    """
    Chunker that splits text at various punctuation marks while preserving
    special cases like decimal points.
    """

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Split text segments at punctuation marks like periods, commas, semicolons, etc.

        Args:
            text_segments: List of text segments to split

        Returns:
            List of fine-grained chunks
        """
        all_fine_grained_chunks = []
        for segment_str in text_segments:
            if not segment_str or not segment_str.strip():
                logger.debug("FineGrainedPunctuationChunker: Skipping empty segment.")
                continue

            # Note the non-capturing groups (?:...) for abbreviations, etc.
            # This prevents splitting on periods in abbreviations like Dr., Ms., etc.
            # or in decimal numbers like 3.14
            segments = re.split(
                r"(?<!\bDr|\bMs|\bMr|\bMrs|\bPh|\bed|\bp|\bie|\beg|\bcf|\bvs|\bSt|\bJr|\bSr|(?:\d+(\,\d+)*))"  # Don't split abbreviations
                r"(?<![A-Z]\.(?:[A-Z]\.)+)"  # Don't split acronyms like U.S.A.
                r"(?<![A-Za-z]\.)"  # Don't split initials like J.
                r"(?<!\d)"  # Don't split decimals like 3.14
                r"(?<!\.\d)"  # Don't split version numbers like 1.2.3
                r"(?<![\-\w])"  # Don't split within hyphenated words or inside words
                r"[.,:;?!]\s+",  # Split on these punctuation marks followed by whitespace
                segment_str,
            )

            # Remove empty strings and whitespace
            fine_grained_chunks = [s.strip() for s in segments if s.strip()]
            all_fine_grained_chunks.extend(fine_grained_chunks)

        logger.debug(
            f"FineGrainedPunctuationChunker produced {len(all_fine_grained_chunks)} chunks "
            f"from {len(text_segments)} input segments."
        )
        return all_fine_grained_chunks


class PreChunkSemanticGrouper(TextChunker):
    """
    Chunker that semantically groups pre-chunks (typically from FineGrainedPunctuationChunker)
    from a single paragraph using SBERT embeddings and cosine similarity.

    This chunker differs from standard TextChunkers as it expects a list of pre-chunks
    rather than a single text string. It maintains the TextChunker interface by
    overriding the chunk method but provides additional methods for handling
    lists of pre-chunks directly.
    """

    def __init__(
        self,
        language: str = "en",
        model: SentenceTransformer = None,
        similarity_threshold: float = 0.5,
        min_group_size: int = 1,
        max_group_size: int = 7,
        **kwargs,
    ):
        """
        Initialize the pre-chunk semantic grouper.

        Args:
            language: ISO language code
            model: Pre-loaded SentenceTransformer model (required)
            similarity_threshold: Cosine similarity threshold for grouping pre-chunks (0-1)
            min_group_size: Minimum number of pre-chunks in a final semantic chunk
            max_group_size: Maximum number of pre-chunks in a final semantic chunk
            **kwargs: Additional parameters for future extensibility
        """
        super().__init__(language=language, **kwargs)

        if model is None:
            raise ValueError(
                "A SentenceTransformer model must be provided to PreChunkSemanticGrouper."
            )

        self.model = model
        self.similarity_threshold = similarity_threshold
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size

        logger.info(
            f"Initialized PreChunkSemanticGrouper with similarity_threshold={similarity_threshold}, "
            f"min_group_size={min_group_size}, max_group_size={max_group_size}"
        )

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Process a list of text segments using semantic grouping.

        Note: All segments in the input list are treated as pre-chunks for semantic grouping.
        This method does not perform any sentence splitting prior to semantic grouping,
        unlike the SemanticChunker which first splits each segment into sentences.

        Args:
            text_segments: List of input text segments to process as pre-chunks

        Returns:
            List of semantically grouped chunks
        """
        if not text_segments:
            logger.debug("PreChunkSemanticGrouper: No input segments provided.")
            return []

        # Filter out empty segments
        valid_segments = [seg for seg in text_segments if seg and seg.strip()]
        if not valid_segments:
            logger.debug("PreChunkSemanticGrouper: All input segments were empty.")
            return []

        # Apply semantic grouping to the entire set of segments as pre-chunks
        result_chunks = self.chunk_pre_chunks(valid_segments)

        logger.debug(
            f"PreChunkSemanticGrouper processed {len(text_segments)} input segments "
            f"into {len(result_chunks)} output chunks."
        )
        return result_chunks

    def chunk_pre_chunks(self, pre_chunks_from_paragraph: List[str]) -> List[str]:
        """
        Split a list of pre-chunks into semantically coherent chunks.

        Args:
            pre_chunks_from_paragraph: List of pre-chunks from a single paragraph

        Returns:
            List of semantic chunks (each chunk is formed by joining one or more pre-chunks)
        """
        # A. Handle edge cases
        if not pre_chunks_from_paragraph:
            return []

        if len(pre_chunks_from_paragraph) == 1:
            return list(pre_chunks_from_paragraph)

        # B. Embed pre-chunks
        embeddings = self.model.encode(
            pre_chunks_from_paragraph, show_progress_bar=False, convert_to_numpy=True
        )

        # Another edge case check after encoding
        if len(embeddings) <= 1:
            return list(pre_chunks_from_paragraph)

        # C. Calculate pairwise similarities between adjacent pre-chunks
        similarities_with_next = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities_with_next.append(sim)

        # D. Identify breakpoints (low similarity between adjacent chunks)
        breakpoints = []
        for i, sim in enumerate(similarities_with_next):
            if sim < self.similarity_threshold:
                breakpoints.append(i)

        # E. Form chunks based on breakpoints
        final_chunks = []
        current_chunk_start_idx = 0

        # Process each segment between breakpoints
        for bp_idx in breakpoints:
            segment_pre_chunks = pre_chunks_from_paragraph[
                current_chunk_start_idx : bp_idx + 1
            ]

            # Check if this segment meets minimum group size
            if len(segment_pre_chunks) < self.min_group_size:
                # If we're not at the start, prefer to merge with previous segment
                if final_chunks and current_chunk_start_idx > 0:
                    # Merge with previous chunk by removing the last chunk and combining
                    prev_chunk = final_chunks.pop()
                    combined_chunk = prev_chunk + " " + " ".join(segment_pre_chunks)
                    final_chunks.append(combined_chunk)
                else:
                    # Otherwise, just add it
                    final_chunks.append(" ".join(segment_pre_chunks))
            else:
                # Segment meets minimum size requirement
                final_chunks.append(" ".join(segment_pre_chunks))

            current_chunk_start_idx = bp_idx + 1

        # Add the last segment
        remaining_pre_chunks = pre_chunks_from_paragraph[current_chunk_start_idx:]
        if remaining_pre_chunks:
            # Check if last segment meets minimum group size and we have previous chunks
            if len(remaining_pre_chunks) < self.min_group_size and final_chunks:
                # Merge with previous chunk
                prev_chunk = final_chunks.pop()
                combined = prev_chunk + " " + " ".join(remaining_pre_chunks)
                final_chunks.append(combined)
            else:
                final_chunks.append(" ".join(remaining_pre_chunks))

        # F. Handle max_group_size constraints
        # For chunks that exceed max_group_size, find optimal split points
        result_chunks = []
        for chunk in final_chunks:
            chunk_parts = chunk.split(" ")
            if len(chunk_parts) > self.max_group_size:
                # Log that we're splitting an oversized chunk
                logger.debug(
                    f"Splitting chunk with {len(chunk_parts)} parts "
                    f"exceeding max_group_size={self.max_group_size}"
                )

                # Split the oversized chunk into smaller chunks
                for i in range(0, len(chunk_parts), self.max_group_size):
                    sub_chunk = " ".join(chunk_parts[i : i + self.max_group_size])
                    if sub_chunk:
                        result_chunks.append(sub_chunk)
            else:
                result_chunks.append(chunk)

        return result_chunks
