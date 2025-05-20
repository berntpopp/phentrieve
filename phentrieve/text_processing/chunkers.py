"""
Text chunking utilities for Phentrieve.

This module provides classes for breaking down text into processable chunks
using various strategies, from simple paragraph splitting to semantic chunking.
"""

import re
import logging
import pysbd
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)


class TextChunker(ABC):
    """
    Abstract base class for text chunking strategies.

    All chunkers must implement a chunk method that takes a text
    string and returns a list of chunk strings.
    """

    def __init__(self, language: str = "en", **kwargs):
        """
        Initialize the chunker.

        Args:
            language: ISO language code ('en', 'de', etc.)
            **kwargs: Additional configuration parameters for specific chunker implementations
        """
        self.language = language

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks according to the chunker's strategy.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        pass


class NoOpChunker(TextChunker):
    """
    Simple chunker that returns the entire text as a single chunk.

    This is useful as a baseline or when no chunking is desired.
    """

    def chunk(self, text: str) -> List[str]:
        """
        Return the entire text as a single chunk after cleaning.

        Args:
            text: Input text

        Returns:
            List containing a single chunk or empty list if input is empty
        """
        if not text or not text.strip():
            return []

        cleaned_text = clean_internal_newlines_and_extra_spaces(text.strip())
        return [cleaned_text] if cleaned_text else []


class ParagraphChunker(TextChunker):
    """
    Chunker that splits text into paragraphs based on blank lines.
    """

    def chunk(self, text: str) -> List[str]:
        """
        Split text into paragraphs by looking for blank lines.

        Args:
            text: Input text to split into paragraphs

        Returns:
            List of paragraph chunks
        """
        if not text or not text.strip():
            return []

        normalized_text = normalize_line_endings(text)
        paragraphs = re.split(r"\n\s*\n+", normalized_text)
        return [p.strip() for p in paragraphs if p.strip()]


class SentenceChunker(TextChunker):
    """
    Chunker that splits text into individual sentences using pysbd.
    """

    def chunk(self, text: str) -> List[str]:
        """
        Split text into sentences using pysbd with fallback method.

        Args:
            text: Input text to split into sentences

        Returns:
            List of sentence chunks
        """
        return self._segment_into_sentences(text, self.language)

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


class SemanticChunker(TextChunker):
    """
    Chunker that groups sentences based on semantic similarity.

    This chunker uses a sentence transformer model to compute embeddings
    and groups sentences with similar meanings together.
    """

    def __init__(
        self,
        language: str = "en",
        model: SentenceTransformer = None,
        similarity_threshold: float = 0.4,
        min_chunk_sentences: int = 1,
        max_chunk_sentences: int = 5,
        **kwargs,
    ):
        """
        Initialize the semantic chunker.

        Args:
            language: ISO language code
            model: Pre-loaded SentenceTransformer model
            similarity_threshold: Cosine similarity threshold for grouping sentences (0-1)
            min_chunk_sentences: Minimum number of sentences per chunk
            max_chunk_sentences: Maximum number of sentences per chunk
            **kwargs: Additional parameters
        """
        super().__init__(language=language, **kwargs)
        self.model = model
        if not self.model:
            raise ValueError(
                "A SentenceTransformer model must be provided for SemanticChunker"
            )

        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.sentence_chunker = SentenceChunker(language=language)

    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text to be split

        Returns:
            List of semantic chunks
        """
        if not text or not text.strip():
            return []

        # 1. First split into sentences
        sentences = self.sentence_chunker.chunk(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences  # Single sentence is a single chunk

        # 2. Compute sentence embeddings
        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        # 3. Group semantically similar sentences
        chunks = []
        current_chunk_indices = [0]  # Start with the first sentence
        current_chunk_embedding = embeddings[0].reshape(1, -1)

        for i in range(1, len(sentences)):
            # Compare the current sentence to the average of the current chunk
            similarity = cosine_similarity(
                embeddings[i].reshape(1, -1), current_chunk_embedding
            )[0][0]

            # Check if we should add to the current chunk or start a new one
            if (
                similarity >= self.similarity_threshold
                and len(current_chunk_indices) < self.max_chunk_sentences
            ):
                # Add to current chunk
                current_chunk_indices.append(i)
                # Update the chunk embedding (average)
                chunk_embeddings = np.vstack(
                    [embeddings[idx] for idx in current_chunk_indices]
                )
                current_chunk_embedding = np.mean(chunk_embeddings, axis=0).reshape(
                    1, -1
                )
            else:
                # Finalize the current chunk if it meets minimum size
                if len(current_chunk_indices) >= self.min_chunk_sentences:
                    chunk_text = " ".join(
                        [sentences[idx] for idx in current_chunk_indices]
                    )
                    chunks.append(chunk_text)

                # Start a new chunk with the current sentence
                current_chunk_indices = [i]
                current_chunk_embedding = embeddings[i].reshape(1, -1)

        # Add the last chunk if it's not empty
        if current_chunk_indices:
            chunk_text = " ".join([sentences[idx] for idx in current_chunk_indices])
            chunks.append(chunk_text)

        return chunks


class FineGrainedPunctuationChunker(TextChunker):
    """
    Chunker that splits text at various punctuation marks while preserving
    special cases like decimal points.
    """

    def chunk(self, text: str) -> List[str]:
        """
        Split text at punctuation marks like periods, commas, semicolons, etc.

        Args:
            text: Text to split

        Returns:
            List of fine-grained chunks
        """
        if not text or not text.strip():
            return []

        # First pass: protect decimal numbers from splitting
        # Replace decimal points with a placeholder
        decimal_pattern = r"(?<!\d)(\d+)\.(\d+)(?!\d)"
        protected_text = re.sub(decimal_pattern, r"\1<DECIMAL_POINT>\2", text)

        # Split on various punctuation marks
        split_pattern = r"[.?!,;:]+"
        raw_chunks = re.split(split_pattern, protected_text)

        # Restore decimal points and clean chunks
        cleaned_chunks = []
        for chunk in raw_chunks:
            if chunk.strip():
                restored_chunk = chunk.replace("<DECIMAL_POINT>", ".")
                cleaned_chunks.append(restored_chunk.strip())

        return cleaned_chunks


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

    def chunk(self, text: str) -> List[str]:
        """
        Standard TextChunker interface method. This implementation treats the input text
        as a single pre-chunk and returns it without additional processing, as this
        chunker is designed to work on collections of pre-chunks, not single texts.

        Args:
            text: Input text (interpreted as a single pre-chunk)

        Returns:
            List containing the input text as a single chunk
        """
        logger.warning(
            "PreChunkSemanticGrouper.chunk() called with a single text string. "
            "This chunker is designed to work with lists of pre-chunks via "
            "chunk_pre_chunks(). Returning the input as a single chunk."
        )
        return [text] if text and text.strip() else []

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
