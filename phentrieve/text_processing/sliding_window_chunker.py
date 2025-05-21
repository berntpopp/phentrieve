"""
Implementation of the SlidingWindowSemanticSplitter chunker class.
"""

from typing import List
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from abc import ABC, abstractmethod
from typing import List


# Define a local version of TextChunker to avoid circular imports
class TextChunker(ABC):
    """Abstract base class for text chunking strategies."""

    def __init__(self, language: str = "en", **kwargs):
        """Initialize the chunker."""
        self.language = language

    @abstractmethod
    def chunk(self, text_segments: List[str]) -> List[str]:
        """Process text segments and return a list of chunks."""
        pass


logger = logging.getLogger(__name__)


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
        window_token_spans: List[tuple] = []

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
