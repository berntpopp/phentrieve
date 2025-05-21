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
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)

# SlidingWindowSemanticSplitter was originally in a separate module,
# but is now integrated here

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


class SlidingWindowSemanticSplitter(TextChunker):
    """
    Chunker that splits text segments using a sliding window of token embeddings to
    identify semantic boundaries. This enables more fine-grained semantic splitting
    at sub-sentence level.
    """

    def __init__(
        self,
        language: str = "en",
        model: SentenceTransformer = None,
        window_size_tokens: int = 7,
        # Determines overlap. step_size_tokens=window_size_tokens means no overlap.
        step_size_tokens: int = 1,
        splitting_threshold: float = 0.5,
        min_split_segment_length_words: int = 3,
        **kwargs,
    ):
        """
        Initialize the sliding window semantic splitter.

        Args:
            language: ISO language code
            model: Pre-loaded SentenceTransformer model (required)
            window_size_tokens: Number of tokens in each sliding window
            step_size_tokens: Number of tokens to step between windows (1 = maximum overlap)
            splitting_threshold: Cosine similarity threshold below which to split (0-1)
            min_split_segment_length_words: Minimum number of words in a split segment
            **kwargs: Additional parameters
        """
        super().__init__(language=language, **kwargs)
        if model is None:
            raise ValueError(
                "SentenceTransformer model is required for SlidingWindowSemanticSplitter."
            )
        self.model = model
        self.window_size_tokens = window_size_tokens
        self.step_size_tokens = max(1, step_size_tokens)  # Ensure step is at least 1
        self.splitting_threshold = splitting_threshold
        self.min_split_segment_length_words = min_split_segment_length_words

        # Simple whitespace tokenizer. Consider enhancing with spaCy for robustness if needed.
        # Simple tokenizer that removes empty tokens
        self.tokenizer = lambda text: [token for token in text.split() if token]

        logger.info(
            f"Initialized SlidingWindowSemanticSplitter: "
            f"window_size={self.window_size_tokens}, step={self.step_size_tokens}, "
            f"threshold={self.splitting_threshold}, min_seg_len={self.min_split_segment_length_words}"
        )

        # Log model info
        logger.debug(f"Using model: {self.model.__class__.__name__}")
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            logger.debug(
                f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Process a list of text segments and split each one semantically.

        Args:
            text_segments: List of text segments to process

        Returns:
            List of semantically split text segments
        """
        logger.debug(
            f"SlidingWindowSemanticSplitter received {len(text_segments)} segments to process"
        )
        output_segments: List[str] = []

        for i, segment in enumerate(text_segments):
            # Skip empty segments
            if not segment or not segment.strip():
                logger.debug(f"Skipping empty segment #{i}")
                continue

            logger.debug(f"Processing segment #{i} (length: {len(segment)} chars)")
            # Process each segment with the sliding window approach
            segment_splits = self._split_one_segment_by_sliding_window(segment)
            logger.debug(
                f"Segment #{i} was split into {len(segment_splits)} sub-segments"
            )
            output_segments.extend(segment_splits)

        logger.debug(
            f"SlidingWindowSemanticSplitter produced {len(output_segments)} total segments"
        )
        return output_segments

    def _split_one_segment_by_sliding_window(
        self, current_text_segment: str
    ) -> List[str]:
        """
        Split a single text segment into multiple parts based on semantic boundaries
        detected using a sliding window of token embeddings.

        Args:
            current_text_segment: Single text segment to process

        Returns:
            List of semantically split chunks derived from this segment
        """
        logger.debug(
            f"SlidingWindow: Attempting to split segment: "
            f'"{current_text_segment[:150]}..."'
        )
        tokens = self.tokenizer(current_text_segment)

        # Heuristic: if shorter than min words, don't try to split
        if (
            len(tokens) < self.window_size_tokens
            or len(tokens) < self.min_split_segment_length_words
        ):
            logger.debug(
                f"Segment too short ('{len(tokens)}' tokens) for window splitting, "
                f"returning as is."
            )
            return [current_text_segment] if current_text_segment.strip() else []

        window_texts: List[str] = []
        window_token_spans: List[Tuple[int, int]] = []

        for i in range(
            0, len(tokens) - self.window_size_tokens + 1, self.step_size_tokens
        ):
            start_token_idx = i
            end_token_idx = i + self.window_size_tokens
            current_window_tokens_list = tokens[start_token_idx:end_token_idx]
            window_texts.append(" ".join(current_window_tokens_list))
            window_token_spans.append((start_token_idx, end_token_idx))

        if len(window_texts) < 2:
            logger.debug(
                "Not enough windows generated from segment to perform similarity comparison."
            )
            return [current_text_segment] if current_text_segment.strip() else []

        logger.debug(f"Generated {len(window_texts)} sliding windows for the segment.")
        window_embeddings = self.model.encode(window_texts, show_progress_bar=False)

        similarities_between_windows: List[float] = []
        for j in range(len(window_embeddings) - 1):
            sim = cosine_similarity(
                window_embeddings[j].reshape(1, -1),
                window_embeddings[j + 1].reshape(1, -1),
            )[0][0]
            similarities_between_windows.append(sim)

        # Identify indices of the *first* window in a pair that has low similarity to the next one
        # These mark potential *ends* of semantic segments.
        potential_split_marker_indices: List[int] = []
        for k, sim_score in enumerate(similarities_between_windows):
            logger.debug(
                f"Similarity between window {k} (tokens {window_token_spans[k]}) and "
                f"window {k+1} (tokens {window_token_spans[k+1]}): {sim_score:.4f}"
            )
            if sim_score < self.splitting_threshold:
                potential_split_marker_indices.append(
                    k
                )  # k is the index of the first window in the dissimilar pair

        if not potential_split_marker_indices:
            logger.debug(
                "No semantic splits found based on threshold, returning original segment."
            )
            return [current_text_segment]

        final_segments: List[str] = []
        current_segment_start_token_idx = 0
        for marker_idx in potential_split_marker_indices:
            # The split occurs AFTER the window at marker_idx.
            # The segment includes all tokens up to the end of window_token_spans[marker_idx].
            split_after_token_idx = window_token_spans[marker_idx][
                1
            ]  # exclusive end index of this window

            segment_tokens = tokens[
                current_segment_start_token_idx:split_after_token_idx
            ]
            if segment_tokens:  # Ensure not empty
                segment_str = " ".join(segment_tokens).strip()
                if (
                    len(segment_tokens) >= self.min_split_segment_length_words
                    and segment_str
                ):
                    final_segments.append(segment_str)
                    logger.debug(f"Created segment (split): '{segment_str}'")
                elif (
                    final_segments and segment_str
                ):  # Current segment too short, append to previous
                    final_segments[-1] = (
                        final_segments[-1] + " " + segment_str
                    ).strip()
                    logger.debug(
                        f"Appended short segment. Previous segment is now: '{final_segments[-1]}'"
                    )
                elif segment_str:  # First segment and too short
                    final_segments.append(segment_str)  # Keep it for now
                    logger.debug(f"Kept short first segment: '{segment_str}'")

            current_segment_start_token_idx = (
                split_after_token_idx  # Next segment starts after this one ended
            )

        # Add the last remaining segment
        if current_segment_start_token_idx < len(tokens):
            last_segment_tokens = tokens[current_segment_start_token_idx:]
            last_segment_str = " ".join(last_segment_tokens).strip()
            if (
                len(last_segment_tokens) >= self.min_split_segment_length_words
                and last_segment_str
            ):
                final_segments.append(last_segment_str)
                logger.debug(f"Added final remaining segment: '{last_segment_str}'")
            elif (
                final_segments and last_segment_str
            ):  # Last segment too short, append to previous
                final_segments[-1] = (
                    final_segments[-1] + " " + last_segment_str
                ).strip()
                logger.debug(
                    f"Appended short final segment. Previous segment is now: '{final_segments[-1]}'"
                )
            elif last_segment_str:  # Only one segment in total, and it's short
                final_segments.append(last_segment_str)
                logger.debug(f"Kept short (only) final segment: '{last_segment_str}'")

        # Final filter for any empty strings that might have been created
        return [s for s in final_segments if s]


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
