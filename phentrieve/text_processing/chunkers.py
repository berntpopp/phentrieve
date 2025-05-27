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
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)

# Language-specific cleanup words and punctuation
# Ensure all are lowercase
LEADING_CLEANUP_WORDS = {
    "en": [
        "a ",
        "an ",
        "the ",
        "and ",
        "or ",
        "but ",
        "with ",
        "of ",
        "in ",
        "on ",
        "at ",
        "for ",
        "to ",
        "is ",
        "are ",
        "was ",
        "were ",
        "has ",
        "have ",
        "had ",
        "also ",
        "it ",
        "its ",
        "he ",
        "she ",
        "they ",
        "them ",
        "their ",
        "this ",
        "that ",
        "these ",
        "those ",
        "by ",
        "from ",
        "as ",
    ],
    "de": [
        "der ",
        "die ",
        "das ",
        "dem ",
        "den ",
        "des ",
        "ein ",
        "eine ",
        "eines ",
        "einem ",
        "einen ",
        "und ",
        "oder ",
        "aber ",
        "sondern ",
        "mit ",
        "von ",
        "zu ",
        "in ",
        "im ",
        "am ",
        "auf ",
        "aus ",
        "bei ",
        "ist ",
        "sind ",
        "war ",
        "waren ",
        "hat ",
        "haben ",
        "hatte ",
        "hatten ",
        "auch ",
        "als ",
        "wenn ",
        "dass ",
        "daß ",
        "so ",
        "wie ",
        "nur ",
        "durch ",
        "für ",
    ],
    "fr": [
        "le ",
        "la ",
        "l'",
        "les ",
        "un ",
        "une ",
        "des ",
        "et ",
        "ou ",
        "mais ",
        "avec ",
        "de ",
        "d'",
        "du ",
        "en ",
        "dans ",
        "sur ",
        "pour ",
        "à ",
        "au ",
        "aux ",
        "est ",
        "sont ",
        "était ",
        "étaient ",
        "a ",
        "ont ",
        "avait ",
        "avaient ",
        "aussi ",
        "il ",
        "elle ",
        "ils ",
        "elles ",
        "ce ",
        "cet ",
        "cette ",
        "ces ",
        "par ",
        "comme ",
    ],
    "es": [
        "el ",
        "la ",
        "lo ",
        "los ",
        "las ",
        "un ",
        "una ",
        "unos ",
        "unas ",
        "y ",
        "e ",
        "o ",
        "u ",
        "pero ",
        "con ",
        "de ",
        "del ",
        "en ",
        "sobre ",
        "para ",
        "a ",
        "al ",
        "es ",
        "son ",
        "era ",
        "eran ",
        "fue ",
        "fueron ",
        "ha ",
        "han ",
        "había ",
        "habían ",
        "también ",
        "él ",
        "ella ",
        "ellos ",
        "ellas ",
        "esto ",
        "esta ",
        "estos ",
        "estas ",
        "eso ",
        "esa ",
        "esos ",
        "esas ",
        "por ",
        "como ",
    ],
    "nl": [
        "de ",
        "het ",
        "een ",
        "en ",
        "of ",
        "maar ",
        "met ",
        "van ",
        "in ",
        "op ",
        "aan ",
        "voor ",
        "tot ",
        "is ",
        "zijn ",
        "was ",
        "waren ",
        "heeft ",
        "hebben ",
        "had ",
        "hadden ",
        "ook ",
        "hij ",
        "zij ",
        "ze ",
        "dit ",
        "dat ",
        "deze ",
        "die ",
        "door ",
        "als ",
    ],
}

TRAILING_CLEANUP_WORDS = {
    "en": [
        " and",
        " or",
        " but",
        " with",
        " of",
        " for",
        " to",
        " is",
        " are",
        " was",
        " were",
        " by",
        " from",
        " as",
        # Trailing articles like " a", " the" are less common but can be added if observed
    ],
    "de": [
        " und",
        " oder",
        " aber",
        " sondern",
        " als",
        " wenn",
        " ist",
        " sind",
        " mit",
        " von",
        " zu",
        " durch",
        " für",
    ],
    "fr": [
        " et",
        " ou",
        " mais",
        " avec",
        " de",
        " d'",
        " pour",
        " à",
        " par",
        " comme",
    ],
    "es": [
        " y",
        " e",
        " o",
        " u",
        " pero",
        " con",
        " de",
        " para",
        " a",
        " por",
        " como",
    ],
    "nl": [
        " en",
        " of",
        " maar",
        " met",
        " van",
        " voor",
        " tot",
        " door",
        " als",
    ],
}

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
        min_cleaned_chunk_length_chars: int = 2,  # Minimum length in characters after cleaning
        max_cleanup_passes: int = 3,  # To prevent potential infinite loops with complex patterns
        custom_leading_words_to_remove: Optional[List[str]] = None,
        custom_trailing_words_to_remove: Optional[List[str]] = None,
        custom_trailing_punctuation: Optional[str] = None,
        custom_leading_punctuation: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the FinalChunkCleaner.

        Args:
            language: ISO language code for language-specific cleanup rules
            min_cleaned_chunk_length_words: Minimum number of words a chunk must have after cleaning
            max_cleanup_passes: Maximum number of cleanup passes to perform
            custom_leading_words_to_remove: Custom list of leading words to remove
            custom_trailing_words_to_remove: Custom list of trailing words to remove
            custom_trailing_punctuation: Custom string of trailing punctuation to remove
            custom_leading_punctuation: Custom string of leading punctuation to remove
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(language=language, **kwargs)
        self.min_cleaned_chunk_length_chars = max(
            1, min_cleaned_chunk_length_chars
        )  # Minimum 1 character
        self.max_cleanup_passes = max(1, max_cleanup_passes)  # Ensure at least one pass

        # Load language-specific or custom lists
        # Ensure custom lists are all lowercase and have correct spacing if provided
        self.leading_words_to_strip = (
            [w.lower() for w in custom_leading_words_to_remove]
            if custom_leading_words_to_remove is not None
            else LEADING_CLEANUP_WORDS.get(
                self.language.lower(), LEADING_CLEANUP_WORDS.get("en", [])
            )
        )

        self.trailing_words_to_strip = (
            [w.lower() for w in custom_trailing_words_to_remove]
            if custom_trailing_words_to_remove is not None
            else TRAILING_CLEANUP_WORDS.get(
                self.language.lower(), TRAILING_CLEANUP_WORDS.get("en", [])
            )
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

        logger.info(
            f"Initialized FinalChunkCleaner for language '{self.language}' with "
            f"min_length_chars={self.min_cleaned_chunk_length_chars}, max_passes={self.max_cleanup_passes}."
        )
        logger.debug(f"Leading words for cleanup: {self.leading_words_to_strip}")
        logger.debug(f"Trailing words for cleanup: {self.trailing_words_to_strip}")
        logger.debug(
            f"Leading punctuation for cleanup: '{self.leading_punctuation_to_strip}'"
        )
        logger.debug(
            f"Trailing punctuation for cleanup: '{self.trailing_punctuation_to_strip}'"
        )

    def chunk(self, text_segments: List[str]) -> List[str]:
        """
        Process a list of text segments to clean up leading/trailing non-semantic elements.

        Args:
            text_segments: List of input text segments to clean

        Returns:
            List of cleaned text segments, with any segments that are too short after cleaning removed
        """
        cleaned_segments_accumulator: List[str] = []
        for segment_str in text_segments:
            if (
                not segment_str or not segment_str.strip()
            ):  # Skip if segment is already effectively empty
                logger.debug(
                    "FinalChunkCleaner: Skipping empty or whitespace-only input segment."
                )
                continue

            cleaned_segment = self._clean_single_segment(segment_str)

            # Check final length in characters (after cleaning)
            if len(cleaned_segment) >= self.min_cleaned_chunk_length_chars:
                cleaned_segments_accumulator.append(cleaned_segment)
                logger.debug(
                    f"Original: '{segment_str}' -> Cleaned: '{cleaned_segment}' (Kept - {len(cleaned_segment)} chars)"
                )
            else:
                logger.debug(
                    f"Original: '{segment_str}' -> Cleaned: '{cleaned_segment}' (Discarded - {len(cleaned_segment)} chars < {self.min_cleaned_chunk_length_chars} min chars)"
                )

        logger.info(
            f"FinalChunkCleaner processed {len(text_segments)} input segments "
            f"into {len(cleaned_segments_accumulator)} cleaned segments."
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
                for (
                    word_to_remove_spaced
                ) in (
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
                for (
                    word_to_remove_spaced
                ) in (
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
            for i, segment in enumerate(segments):
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


# Language-specific constants for negation-aware merging
NEGATION_PREFIXES = {
    "en": ["no", "not", "non", "without", "zero"],
    "de": ["kein", "keine", "keinen", "keiner", "keines", "ohne", "nicht"],
    "fr": ["non", "pas", "sans", "aucun", "aucune", "jamais", "nul", "nulle"],
    "es": ["no", "sin", "ningún", "ninguna", "nunca", "jamás", "ni"],
    "nl": ["geen", "niet", "zonder", "nooit", "nee", "niks", "nergens"],
}

# A small list of words that, if they are the `next_segment`, might prevent merging a negation prefix.
# These should be words that don't typically form a tight semantic unit with a preceding standalone negation.
AVOID_MERGE_AFTER_NEGATION_IF_NEXT_IS = {
    "en": ["and", "or", "but", "so", "yet", "for", "nor", "is", "are", "was", "were"],
    "de": ["und", "oder", "aber", "sondern", "denn", "als", "wenn", "ist", "sind"],
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
        final_segments = [s for s in final_segments if s]

        # Apply negation-aware merging to fix splits within negation patterns
        final_segments = self._apply_negation_aware_merging(final_segments)

        return final_segments

    def _apply_negation_aware_merging(self, segments: List[str]) -> List[str]:
        """
        Merge segments that were incorrectly split within negation patterns.

        Args:
            segments: List of text segments to process

        Returns:
            List of segments with negation patterns properly merged
        """
        if not segments or len(segments) < 2:
            return segments

        lang_key = self.language.lower()
        # Create a clean set of negation prefixes by stripping whitespace
        current_neg_standalone_prefixes = set(prefix.strip() for prefix in NEGATION_PREFIXES.get(lang_key, NEGATION_PREFIXES["en"]))
        current_avoid_merge_next_starts_with = AVOID_MERGE_AFTER_NEGATION_IF_NEXT_IS.get(
            lang_key, AVOID_MERGE_AFTER_NEGATION_IF_NEXT_IS["en"]
        )

        merged_segments: List[str] = []
        i = 0

        while i < len(segments):
            current_segment = segments[i]
            current_segment_lower_stripped = current_segment.lower().strip()

            if not current_segment_lower_stripped:  # Handle empty string after potential splits
                i += 1
                continue

            # Check if current segment is a standalone negation prefix
            is_standalone_neg_prefix = (
                current_segment_lower_stripped in current_neg_standalone_prefixes and
                len(self.tokenizer(current_segment_lower_stripped)) == 1
            )

            # Check if segment ends with a negation word (and is not just the negation word itself)
            ends_with_neg_word = False
            neg_suffix_found_for_log = None
            if not is_standalone_neg_prefix:
                tokenized_current = self.tokenizer(current_segment_lower_stripped)
                if tokenized_current and len(tokenized_current) > 1:  # More than one word
                    last_word = tokenized_current[-1]
                    if last_word in current_neg_standalone_prefixes:
                        ends_with_neg_word = True
                        neg_suffix_found_for_log = last_word
            
            # Determine if we should attempt to merge
            attempt_merge = is_standalone_neg_prefix or ends_with_neg_word
            
            if attempt_merge and (i + 1) < len(segments):
                next_segment = segments[i + 1]
                next_segment_lower_stripped = next_segment.lower().strip()
                
                if not next_segment_lower_stripped:  # Next segment is empty, don't merge
                    attempt_merge = False
                else:
                    tokenized_next = self.tokenizer(next_segment_lower_stripped)
                    next_first_word = tokenized_next[0] if tokenized_next else ""
                    
                    # Don't merge if next segment starts with a word to avoid
                    if next_first_word in current_avoid_merge_next_starts_with:
                        attempt_merge = False
                    
                    # Additional heuristic for ends_with_neg_word case to prevent over-merging
                    if ends_with_neg_word and len(tokenized_next) > 5:  # Tunable parameter: max 5 words
                        logger.debug(
                            f"Segment '{current_segment}' ends with negation '{neg_suffix_found_for_log}', "
                            f"but next segment '{next_segment[:30]}...' (len {len(tokenized_next)}) is long. No merge."
                        )
                        attempt_merge = False
            
            if attempt_merge and (i + 1) < len(segments):
                merged_text = (current_segment.rstrip() + " " + segments[i+1].lstrip()).strip()
                merged_segments.append(merged_text)
                log_reason = "standalone prefix" if is_standalone_neg_prefix else f"suffix '{neg_suffix_found_for_log}'"
                logger.debug(
                    f"Merged negation pattern ({log_reason}): '{current_segment}' + '{segments[i+1]}' -> '{merged_text}'"
                )
                i += 2
            else:
                merged_segments.append(current_segment)
                i += 1

        logger.debug(
            f"After negation merging: {len(segments)} -> {len(merged_segments)} segments"
        )
        return merged_segments
