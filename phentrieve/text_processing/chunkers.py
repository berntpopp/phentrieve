"""
Text chunking utilities for Phentrieve.

This module provides classes for breaking down text into processable chunks
using various strategies, from simple paragraph splitting to semantic chunking.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional

import logging
import pysbd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
