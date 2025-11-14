"""
Text chunking utilities for Phentrieve.

This module provides classes for breaking down text into processable chunks
using various strategies, from simple paragraph splitting to semantic grouping.
The primary semantic chunking is handled by SlidingWindowSemanticSplitter in a separate module.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import pysbd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from phentrieve.text_processing.cleaners import (
    clean_internal_newlines_and_extra_spaces,
    normalize_line_endings,
)
from phentrieve.text_processing.resource_loader import load_language_resource
from phentrieve.utils import load_user_config

# Punctuation to be stripped from the ends of segments
TRAILING_PUNCTUATION_CHARS = ",.;:?!\"')}]"
# Punctuation to be stripped from the beginnings of segments
LEADING_PUNCTUATION_CHARS = "\"'([{"

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
    def chunk(self, text_segments: list[str]) -> list[str]:
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


class FinalChunkCleaner(TextChunker):
    """
    Chunker that post-processes text segments to remove common leading/trailing
    non-semantic elements like conjunctions, articles, and punctuation.

    This is typically used as the final step in a chunking pipeline to clean up
    the output of other chunkers.
    """

    def __init__(
        self,
        language: str = "en",
        min_cleaned_chunk_length_chars: int = 1,  # Minimum chars after cleaning
        filter_short_low_value_chunks_max_words: int = 2,  # Max words for low value check
        max_cleanup_passes: int = 3,  # Prevent infinite loops
        custom_leading_words_to_remove: Optional[list[str]] = None,
        custom_trailing_words_to_remove: Optional[list[str]] = None,
        custom_trailing_punctuation: Optional[str] = None,
        custom_leading_punctuation: Optional[str] = None,
        custom_low_value_words: Optional[list[str]] = None,  # Custom low-value words
        **kwargs,
    ):
        """
        Initialize the FinalChunkCleaner.

        Args:
            language: ISO language code for language-specific cleanup rules
            min_cleaned_chunk_length_chars: Minimum character length a chunk must have after cleaning
            filter_short_low_value_chunks_max_words: Maximum word count for chunks to check for low semantic value
            max_cleanup_passes: Maximum number of cleanup passes to perform
            custom_leading_words_to_remove: Custom list of leading words to remove
            custom_trailing_words_to_remove: Custom list of trailing words to remove
            custom_trailing_punctuation: Custom string of trailing punctuation to remove
            custom_leading_punctuation: Custom string of leading punctuation to remove
            custom_low_value_words: Custom list of words considered to have low semantic value
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(language=language, **kwargs)
        self.min_cleaned_chunk_length_chars = max(
            1, min_cleaned_chunk_length_chars
        )  # Minimum 1 character
        self.filter_short_low_value_chunks_max_words = max(
            1, filter_short_low_value_chunks_max_words
        )  # At least 1 word
        self.max_cleanup_passes = max(1, max_cleanup_passes)  # Ensure at least one pass

        # Simple tokenizer for word counting and stop word checks
        self.tokenizer = lambda text: [token for token in text.split() if token.strip()]

        # Load user configuration
        user_config_main = load_user_config()
        language_resources_section = user_config_main.get("language_resources", {})

        # Load language-specific or custom lists
        # Ensure custom lists are all lowercase and have correct spacing if provided
        if custom_leading_words_to_remove is not None:
            self.leading_words_to_strip = [
                w.lower() for w in custom_leading_words_to_remove
            ]
        else:
            # Load from resource files with the new mechanism
            leading_cleanup_resources = load_language_resource(
                default_resource_filename="leading_cleanup_words.json",
                config_key_for_custom_file="leading_cleanup_words_file",
                language_resources_config_section=language_resources_section,
            )
            self.leading_words_to_strip = leading_cleanup_resources.get(
                self.language.lower(), leading_cleanup_resources.get("en", [])
            )

        if custom_trailing_words_to_remove is not None:
            self.trailing_words_to_strip = [
                w.lower() for w in custom_trailing_words_to_remove
            ]
        else:
            # Load from resource files with the new mechanism
            trailing_cleanup_resources = load_language_resource(
                default_resource_filename="trailing_cleanup_words.json",
                config_key_for_custom_file="trailing_cleanup_words_file",
                language_resources_config_section=language_resources_section,
            )
            self.trailing_words_to_strip = trailing_cleanup_resources.get(
                self.language.lower(), trailing_cleanup_resources.get("en", [])
            )

        self.trailing_punctuation_to_strip = (
            custom_trailing_punctuation
            if custom_trailing_punctuation is not None
            else TRAILING_PUNCTUATION_CHARS
        )
        self.leading_punctuation_to_strip = (
            custom_leading_punctuation
            if custom_leading_punctuation is not None
            else LEADING_PUNCTUATION_CHARS
        )

        # Sort by length descending to attempt removal of longer phrases first (e.g., "negative for " before "for ")
        self.leading_words_to_strip.sort(key=len, reverse=True)
        self.trailing_words_to_strip.sort(key=len, reverse=True)

        # Load low_value_words for filtering short chunks
        if custom_low_value_words:
            self.low_value_words = {s.lower() for s in custom_low_value_words}
        else:
            # Load from resource files with the new mechanism
            low_value_resources = load_language_resource(
                default_resource_filename="low_semantic_value_words.json",
                config_key_for_custom_file="low_semantic_value_words_file",
                language_resources_config_section=language_resources_section,
            )
            self.low_value_words = set(
                low_value_resources.get(
                    self.language.lower(), low_value_resources.get("en", [])
                )
            )

        logger.info(
            f"Initialized FinalChunkCleaner for language '{self.language}' with "
            f"min_chars={self.min_cleaned_chunk_length_chars}, "
            f"filter_short_low_value_max_words={self.filter_short_low_value_chunks_max_words}, "
            f"max_passes={self.max_cleanup_passes}."
        )
        logger.debug(f"Leading words for cleanup: {self.leading_words_to_strip}")
        logger.debug(f"Trailing words for cleanup: {self.trailing_words_to_strip}")
        logger.debug(
            f"Leading punctuation for cleanup: '{self.leading_punctuation_to_strip}'"
        )
        logger.debug(
            f"Trailing punctuation for cleanup: '{self.trailing_punctuation_to_strip}'"
        )
        logger.debug(
            f"Low-value words for lang '{self.language}': {sorted(self.low_value_words)[:20]}..."
        )

    def chunk(self, text_segments: list[str]) -> list[str]:
        """
        Process a list of text segments to clean up leading/trailing non-semantic elements.
        Also filters out short segments consisting entirely of low semantic value words.

        Args:
            text_segments: List of input text segments to clean

        Returns:
            List of cleaned text segments, with any segments that are too short after cleaning
            or consist entirely of low semantic value words removed
        """
        cleaned_segments_accumulator: list[str] = []
        for segment_str_input in text_segments:
            if not segment_str_input or not segment_str_input.strip():
                logger.debug(
                    "FinalChunkCleaner: Skipping empty or whitespace-only input segment."
                )
                continue

            # 1. Clean edges (leading/trailing punctuation and words)
            edge_cleaned_segment = self._clean_single_segment(segment_str_input)

            # 2. Basic filter: if empty or below min char length after edge cleaning, discard
            if (
                not edge_cleaned_segment
                or len(edge_cleaned_segment) < self.min_cleaned_chunk_length_chars
            ):
                logger.debug(
                    f"Input: '{segment_str_input[:50]}...' -> Edge-Cleaned: '{edge_cleaned_segment}' "
                    f"(Discarded - empty or below char threshold {self.min_cleaned_chunk_length_chars})"
                )
                continue

            # 3. Tokenize for word count and low-value word check
            # Always use the edge_cleaned_segment for tokenization
            words_in_edge_cleaned_segment = [
                word for word in self.tokenizer(edge_cleaned_segment.lower()) if word
            ]
            num_words = len(words_in_edge_cleaned_segment)

            keep_segment = True
            discard_reason = ""

            # 4. Apply the short, low-value-word-only chunk filter
            if (
                num_words > 0
                and num_words <= self.filter_short_low_value_chunks_max_words
            ):
                if self.low_value_words and all(
                    word in self.low_value_words
                    for word in words_in_edge_cleaned_segment
                ):
                    keep_segment = False
                    discard_reason = (
                        f"short ({num_words} words) and all words are low-value: "
                        f"'{', '.join(words_in_edge_cleaned_segment)}'"
                    )

            if keep_segment:
                # IMPORTANT: Append the segment with original casing, not the lowercased one
                cleaned_segments_accumulator.append(edge_cleaned_segment)
                logger.debug(
                    f"Input: '{segment_str_input[:50]}...' -> Final: '{edge_cleaned_segment}' (Kept - words: {num_words})"
                )
            else:
                logger.debug(
                    f"Input: '{segment_str_input[:50]}...' -> Edge-Cleaned: '{edge_cleaned_segment}' (Discarded - {discard_reason})"
                )

        logger.info(
            f"FinalChunkCleaner processed {len(text_segments)} input segments "
            f"into {len(cleaned_segments_accumulator)} final segments."
        )
        return cleaned_segments_accumulator

    def _clean_single_segment(self, segment: str) -> str:
        """
        Clean a single text segment by removing leading/trailing non-semantic elements.

        Args:
            segment: Input text segment to clean

        Returns:
            Cleaned text segment
        """
        temp_segment = segment.strip()
        if not temp_segment:
            return ""

        for _pass_num in range(self.max_cleanup_passes):
            original_segment_for_pass = temp_segment

            # 1. Strip leading and trailing punctuation
            # Trailing punctuation
            while (
                temp_segment and temp_segment[-1] in self.trailing_punctuation_to_strip
            ):
                temp_segment = temp_segment[:-1].rstrip()
            # Leading punctuation
            while temp_segment and temp_segment[0] in self.leading_punctuation_to_strip:
                temp_segment = temp_segment[1:].lstrip()

            # 2. Strip leading words (case-insensitive match, actual strip preserves case of rest)
            # Iterate because stripping one might reveal another
            made_leading_change_in_iteration = True
            while made_leading_change_in_iteration and temp_segment:
                made_leading_change_in_iteration = False
                for word_to_remove_spaced in (
                    self.leading_words_to_strip
                ):  # Assumes word_to_remove has trailing space
                    # Ensure word_to_remove_spaced ends with a space for proper prefix matching
                    if not word_to_remove_spaced.endswith(" "):
                        word_to_remove_spaced_adjusted = word_to_remove_spaced + " "
                    else:
                        word_to_remove_spaced_adjusted = word_to_remove_spaced

                    if temp_segment.lower().startswith(
                        word_to_remove_spaced_adjusted.lower()
                    ):
                        temp_segment = temp_segment[
                            len(word_to_remove_spaced_adjusted) :
                        ].lstrip()
                        made_leading_change_in_iteration = True
                        break  # Restart checking leading words from the beginning of the modified segment

            # 3. Strip trailing words (case-insensitive match)
            made_trailing_change_in_iteration = True
            while made_trailing_change_in_iteration and temp_segment:
                made_trailing_change_in_iteration = False
                for word_to_remove_spaced in (
                    self.trailing_words_to_strip
                ):  # Assumes word_to_remove has leading space
                    # Ensure word_to_remove_spaced starts with a space
                    if not word_to_remove_spaced.startswith(" "):
                        word_to_remove_spaced_adjusted = " " + word_to_remove_spaced
                    else:
                        word_to_remove_spaced_adjusted = word_to_remove_spaced

                    if temp_segment.lower().endswith(
                        word_to_remove_spaced_adjusted.lower()
                    ):
                        temp_segment = temp_segment[
                            : -len(word_to_remove_spaced_adjusted)
                        ].rstrip()
                        made_trailing_change_in_iteration = True
                        break  # Restart checking trailing words

            if (
                temp_segment == original_segment_for_pass
            ):  # No changes in this entire pass (punct + words)
                break  # Optimization: if nothing changed, further passes won't help

        return temp_segment.strip()  # Final strip after all passes


class NoOpChunker(TextChunker):
    """
    Simple chunker that returns the text segments unchanged.

    This is useful as a baseline or when no chunking is desired.
    """

    def chunk(self, text_segments: list[str]) -> list[str]:
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

    def chunk(self, text_segments: list[str]) -> list[str]:
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

    def chunk(self, text_segments: list[str]) -> list[str]:
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

    def _segment_into_sentences(self, text: str, lang: str = "en") -> list[str]:
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

    def chunk(self, text_segments: list[str]) -> list[str]:
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

            # Split on punctuation followed by whitespace
            # We'll use a simpler approach that doesn't rely on variable-width look-behinds
            # First, let's define common abbreviations to preserve
            abbreviations = [
                r"\bDr\.",
                r"\bMs\.",
                r"\bMr\.",
                r"\bMrs\.",
                r"\bPh\.D\.",
                r"\bed\.",
                r"\bp\.",
                r"\bie\.",
                r"\beg\.",
                r"\bcf\.",
                r"\bvs\.",
                r"\bSt\.",
                r"\bJr\.",
                r"\bSr\.",
                r"[A-Z]\.[A-Z]\.",
            ]

            # Temporarily replace abbreviations with placeholders
            placeholders = {}
            for i, abbr in enumerate(abbreviations):
                placeholder = f"__ABBR{i}__"
                pattern = re.compile(abbr)
                segment_str = pattern.sub(placeholder, segment_str)
                placeholders[placeholder] = abbr

            # Also preserve decimal numbers
            decimal_pattern = re.compile(r"\d+[.,]\d+")
            decimal_matches = decimal_pattern.finditer(segment_str)
            for i, match in enumerate(decimal_matches):
                placeholder = f"__DECIMAL{i}__"
                segment_str = segment_str.replace(match.group(0), placeholder)
                placeholders[placeholder] = match.group(0)

            # Now split on punctuation followed by whitespace
            segments = re.split(r"[.,:;?!]\s+", segment_str)

            # Restore abbreviations and decimal numbers
            for i, _segment in enumerate(segments):
                for placeholder, original in placeholders.items():
                    segments[i] = segments[i].replace(placeholder, original)

            # Remove empty strings and whitespace
            fine_grained_chunks = [s.strip() for s in segments if s.strip()]
            all_fine_grained_chunks.extend(fine_grained_chunks)

        logger.debug(
            f"FineGrainedPunctuationChunker produced {len(all_fine_grained_chunks)} chunks "
            f"from {len(text_segments)} input segments."
        )
        return all_fine_grained_chunks


class ConjunctionChunker(TextChunker):
    """
    Chunker that splits text segments at specified coordinating conjunctions.
    The split occurs before the conjunction, and the conjunction becomes
    part of the following chunk.
    """

    def __init__(self, language: str = "en", **kwargs):
        super().__init__(language=language, **kwargs)

        # Load user configuration
        user_config_main = load_user_config()
        language_resources_section = user_config_main.get("language_resources", {})

        # Load coordinating conjunctions from resource files
        coordinating_conjunctions = load_language_resource(
            default_resource_filename="coordinating_conjunctions.json",
            config_key_for_custom_file="coordinating_conjunctions_file",
            language_resources_config_section=language_resources_section,
        )

        # Get conjunctions for the current language, defaulting to English
        self.conjunctions = coordinating_conjunctions.get(
            self.language.lower(), coordinating_conjunctions.get("en", [])
        )
        # Create a regex pattern for splitting.
        # We want to split *before* the conjunction.
        # The pattern should match space(s) + conjunction + space(s)
        # We'll use a lookahead to include the conjunction in the next split.
        self.split_pattern: re.Pattern[str] | None
        if self.conjunctions:
            # Escape conjunctions in case they contain regex special characters (unlikely for these)
            escaped_conjunctions = [re.escape(c) for c in self.conjunctions]
            # Pattern to split *before* " conjunction " (case-insensitive)
            # Using word boundaries (\b) is important
            self.split_pattern = re.compile(
                r"\s+(?=(\b(?:" + "|".join(escaped_conjunctions) + r")\b\s+))",
                re.IGNORECASE,
            )
            logger.info(
                f"ConjunctionChunker for lang '{self.language}' initialized with conjunctions: {self.conjunctions}"
            )
        else:
            self.split_pattern = None
            logger.warning(
                f"ConjunctionChunker for lang '{self.language}' has no conjunctions defined. Will act as a NoOp."
            )

    def chunk(self, text_segments: list[str]) -> list[str]:
        """
        Split text segments at coordinating conjunctions.

        Args:
            text_segments: List of text segments to split

        Returns:
            List of conjunction-split chunks
        """
        if not self.split_pattern:
            return text_segments  # Act as NoOp if no conjunctions

        all_conjunction_split_chunks = []
        for segment_str in text_segments:
            if not segment_str or not segment_str.strip():
                logger.debug("ConjunctionChunker: Skipping empty segment.")
                continue

            # Perform the split. re.split with a capturing group in lookahead
            # doesn't directly give what we want (keeping delimiter with next part).
            # A simpler approach is to find all matches and reconstruct.

            parts = []
            last_end = 0
            for match in self.split_pattern.finditer(segment_str):
                start_of_split_point = (
                    match.start()
                )  # This is the space *before* the conjunction
                # The part before the split point (and the conjunction)
                parts.append(segment_str[last_end:start_of_split_point].strip())
                last_end = start_of_split_point  # Next part starts from here (including the space and conjunction)

            # Add the final part of the segment
            parts.append(segment_str[last_end:].strip())

            # Filter out any empty strings resulting from multiple spaces or edge cases
            all_conjunction_split_chunks.extend([p for p in parts if p])

        logger.debug(
            f"ConjunctionChunker produced {len(all_conjunction_split_chunks)} chunks "
            f"from {len(text_segments)} input segments."
        )
        return all_conjunction_split_chunks


# Language-specific constants for negation-aware merging
NEGATION_PREFIXES = {
    "en": ["no", "not", "non", "without", "zero"],
    "de": ["kein", "keine", "keinen", "keiner", "keines", "ohne", "nicht"],
    "fr": ["non", "pas", "sans", "aucun", "aucune", "jamais", "nul", "nulle"],
    "es": ["no", "sin", "ningún", "ninguna", "nunca", "jamás", "ni"],
    "nl": ["geen", "niet", "zonder", "nooit", "nee", "niks", "nergens"],
}

# Dictionary of coordinating conjunctions for supported languages
# These are used by ConjunctionChunker to split text at conjunction points
# All conjunctions must be lowercase
COORDINATING_CONJUNCTIONS = {
    "en": ["and", "or", "but"],
    "de": ["und", "oder", "aber", "sondern"],
    "fr": ["et", "ou", "mais"],
    "es": ["y", "e", "o", "u", "pero"],  # 'e' before 'i'/'hi', 'u' before 'o'/'ho'
    "nl": ["en", "of", "maar"],
}

# A small list of words that, if they are the `next_segment`, might prevent merging a negation prefix.
# These should be words that don't typically form a tight semantic unit with a preceding standalone negation.
AVOID_MERGE_AFTER_NEGATION_IF_NEXT_IS = {
    "en": [
        "and",
        "or",
        "but",
        "so",
        "yet",
        "for",
        "nor",
        "is",
        "are",
        "was",
        "were",
        "if",
        "when",
        "then",
        "while",
        "although",
        "though",
        "however",
    ],
    "de": [
        "und",
        "oder",
        "aber",
        "sondern",
        "denn",
        "als",
        "wenn",
        "ist",
        "sind",
        "falls",
        "dann",
        "während",
        "obwohl",
        "jedoch",
    ],
    "fr": [
        "et",
        "ou",
        "mais",
        "donc",
        "car",
        "ni",
        "est",
        "sont",
        "était",
        "étaient",
        "comme",
        "si",
        "quand",
        "alors",
        "bien que",
        "quoique",
        "pendant",
        "cependant",
    ],
    "es": [
        "y",
        "o",
        "pero",
        "así",
        "que",
        "porque",
        "ni",
        "es",
        "son",
        "era",
        "eran",
        "como",
        "si",
        "cuando",
        "entonces",
        "mientras",
        "aunque",
        "sin embargo",
    ],
    "nl": [
        "en",
        "of",
        "maar",
        "dus",
        "want",
        "noch",
        "is",
        "zijn",
        "was",
        "waren",
        "als",
        "wanneer",
        "dan",
        "terwijl",
        "hoewel",
        "echter",
    ],
}


class SlidingWindowSemanticSplitter(TextChunker):
    """
    Chunker that splits text segments using a sliding window of token embeddings to
    identify semantic boundaries. This enables more fine-grained semantic splitting
    at sub-sentence level, with special handling for negation patterns.
    """

    def __init__(
        self,
        language: str = "en",
        model: SentenceTransformer | None = None,
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

    def chunk(self, text_segments: list[str]) -> list[str]:
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
        output_segments: list[str] = []

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
    ) -> list[str]:
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

        window_texts: list[str] = []
        window_token_spans: list[tuple[int, int]] = []

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

        similarities_between_windows: list[float] = []
        for j in range(len(window_embeddings) - 1):
            sim = cosine_similarity(
                window_embeddings[j].reshape(1, -1),
                window_embeddings[j + 1].reshape(1, -1),
            )[0][0]
            similarities_between_windows.append(sim)

        # Identify indices of the *first* window in a pair that has low similarity to the next one
        # These mark potential *ends* of semantic segments.
        potential_split_marker_indices: list[int] = []
        for k, sim_score in enumerate(similarities_between_windows):
            logger.debug(
                f"Similarity between window {k} (tokens {window_token_spans[k]}) and "
                f"window {k + 1} (tokens {window_token_spans[k + 1]}): {sim_score:.4f}"
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

        final_segments: list[str] = []
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
        final_segments = [s for s in final_segments if s]

        # Apply negation-aware merging to fix splits within negation patterns
        final_segments = self._apply_negation_aware_merging(final_segments)

        return final_segments

    def _apply_negation_aware_merging(self, segments: list[str]) -> list[str]:
        """
        Merge segments that were incorrectly split within negation patterns.

        Args:
            segments: List of text segments to process

        Returns:
            List of segments with negation patterns properly merged
        """
        if not segments or len(segments) < 2:
            return segments

        # Load user configuration
        user_config_main = load_user_config()
        language_resources_section = user_config_main.get("language_resources", {})

        lang_key = self.language.lower()

        # Load negation prefixes from resource files
        negation_prefixes = load_language_resource(
            default_resource_filename="negation_prefixes.json",
            config_key_for_custom_file="negation_prefixes_file",
            language_resources_config_section=language_resources_section,
        )

        # Create a clean set of negation prefixes by stripping whitespace
        current_neg_standalone_prefixes = {
            prefix.strip()
            for prefix in negation_prefixes.get(
                lang_key, negation_prefixes.get("en", [])
            )
        }

        # Load words to avoid merging after negation from resource files
        avoid_merge_resources = load_language_resource(
            default_resource_filename="avoid_merge_after_negation_if_next_is.json",
            config_key_for_custom_file="avoid_merge_after_negation_file",
            language_resources_config_section=language_resources_section,
        )

        current_avoid_merge_next_starts_with = avoid_merge_resources.get(
            lang_key, avoid_merge_resources.get("en", [])
        )

        merged_segments: list[str] = []
        i = 0

        while i < len(segments):
            current_segment = segments[i]
            current_segment_lower_stripped = current_segment.lower().strip()

            if (
                not current_segment_lower_stripped
            ):  # Handle empty string after potential splits
                i += 1
                continue

            # Check if current segment is a standalone negation prefix
            is_standalone_neg_prefix = (
                current_segment_lower_stripped in current_neg_standalone_prefixes
                and len(self.tokenizer(current_segment_lower_stripped)) == 1
            )

            # Check if segment ends with a negation word (and is not just the negation word itself)
            ends_with_neg_word = False
            neg_suffix_found_for_log = None
            if not is_standalone_neg_prefix:
                tokenized_current = self.tokenizer(current_segment_lower_stripped)
                if (
                    tokenized_current and len(tokenized_current) > 1
                ):  # More than one word
                    last_word = tokenized_current[-1]
                    if last_word in current_neg_standalone_prefixes:
                        ends_with_neg_word = True
                        neg_suffix_found_for_log = last_word

            # Determine if we should attempt to merge
            attempt_merge = is_standalone_neg_prefix or ends_with_neg_word

            if attempt_merge and (i + 1) < len(segments):
                next_segment = segments[i + 1]
                next_segment_lower_stripped = next_segment.lower().strip()

                if (
                    not next_segment_lower_stripped
                ):  # Next segment is empty, don't merge
                    attempt_merge = False
                else:
                    tokenized_next = self.tokenizer(next_segment_lower_stripped)
                    next_first_word = tokenized_next[0] if tokenized_next else ""

                    # Don't merge if next segment starts with a word to avoid
                    if next_first_word in current_avoid_merge_next_starts_with:
                        attempt_merge = False

                    # Additional heuristic for ends_with_neg_word case to prevent over-merging
                    if (
                        ends_with_neg_word and len(tokenized_next) > 5
                    ):  # Tunable parameter: max 5 words
                        logger.debug(
                            f"Segment '{current_segment}' ends with negation '{neg_suffix_found_for_log}', "
                            f"but next segment '{next_segment[:30]}...' (len {len(tokenized_next)}) is long. No merge."
                        )
                        attempt_merge = False

            if attempt_merge and (i + 1) < len(segments):
                merged_text = (
                    current_segment.rstrip() + " " + segments[i + 1].lstrip()
                ).strip()
                merged_segments.append(merged_text)
                log_reason = (
                    "standalone prefix"
                    if is_standalone_neg_prefix
                    else f"suffix '{neg_suffix_found_for_log}'"
                )
                logger.debug(
                    f"Merged negation pattern ({log_reason}): '{current_segment}' + '{segments[i + 1]}' -> '{merged_text}'"
                )
                i += 2
            else:
                merged_segments.append(current_segment)
                i += 1

        logger.debug(
            f"After negation merging: {len(segments)} -> {len(merged_segments)} segments"
        )
        return merged_segments
