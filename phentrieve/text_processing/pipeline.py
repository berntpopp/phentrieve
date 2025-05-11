"""
Text processing pipeline orchestrator for Phentrieve.

This module provides pipeline components that manage the sequence
of text processing operations including chunking and assertion detection.
"""

import logging
from typing import List, Dict, Any, Optional, Type

from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)
from phentrieve.text_processing.chunkers import (
    TextChunker,
    NoOpChunker,
    ParagraphChunker,
    SentenceChunker,
    SemanticChunker,
    FineGrainedPunctuationChunker,
)
from phentrieve.text_processing.assertion_detection import (
    AssertionDetector,
    AssertionStatus,
    KeywordAssertionDetector,
    DependencyAssertionDetector,
    CombinedAssertionDetector,
)

logger = logging.getLogger(__name__)


class TextProcessingPipeline:
    """
    Pipeline orchestrator for text processing.

    This class manages the sequence of text processing steps,
    including text chunking and assertion detection.
    """

    def __init__(
        self,
        language: str,
        chunking_pipeline_config: List[Dict],
        assertion_config: Dict,
        sbert_model_for_semantic_chunking: Optional[SentenceTransformer] = None,
    ):
        """
        Initialize the text processing pipeline.

        Args:
            language: ISO language code ('en', 'de', etc.)
            chunking_pipeline_config: List of dicts configuring the chunking pipeline.
                Example: [
                    {'type': 'paragraph'},
                    {'type': 'semantic', 'similarity_threshold': 0.4,
                     'min_sentences': 1, 'max_sentences': 3},
                    {'type': 'fine_grained_punctuation'}
                ]
            assertion_config: Dict configuring assertion detection.
                Example: {
                    'enable_keyword': True,
                    'enable_dependency': True,
                    'preference': 'dependency'
                }
            sbert_model_for_semantic_chunking: Pre-loaded SentenceTransformer model
                for semantic chunking (required if 'semantic' type is in the pipeline)
        """
        self.language = language
        self.chunking_pipeline_config = chunking_pipeline_config
        self.assertion_config = assertion_config
        self.sbert_model = sbert_model_for_semantic_chunking

        # Create chunkers list
        self.chunkers = self._create_chunkers()

        # Create assertion detector
        self.assertion_detector = self._create_assertion_detector()

        logger.info(
            f"Initialized TextProcessingPipeline with {len(self.chunkers)} chunking stages "
            f"and assertion detection config: {assertion_config}"
        )

    def _create_chunkers(self) -> List[TextChunker]:
        """
        Create chunker instances based on configuration.

        Returns:
            List of initialized chunker instances
        """
        chunkers = []

        for config in self.chunking_pipeline_config:
            chunker_type = config.get("type", "").lower()
            chunker_config = config.get("config", {})

            # Common parameters for all chunkers
            params = {"language": self.language}

            if chunker_type == "paragraph":
                chunkers.append(ParagraphChunker(**params))

            elif chunker_type == "sentence":
                chunkers.append(SentenceChunker(**params))

            elif chunker_type == "semantic":
                if not self.sbert_model:
                    raise ValueError(
                        "SentenceTransformer model required for semantic chunking "
                        "but none was provided"
                    )

                # Get semantic chunker specific parameters
                similarity_threshold = chunker_config.get(
                    "similarity_threshold", config.get("similarity_threshold", 0.4)
                )
                min_chunk_sentences = chunker_config.get(
                    "min_chunk_sentences",
                    config.get("min_chunk_sentences", config.get("min_sentences", 1)),
                )
                max_chunk_sentences = chunker_config.get(
                    "max_chunk_sentences",
                    config.get("max_chunk_sentences", config.get("max_sentences", 5)),
                )

                chunkers.append(
                    SemanticChunker(
                        **params,
                        model=self.sbert_model,
                        similarity_threshold=similarity_threshold,
                        min_chunk_sentences=min_chunk_sentences,
                        max_chunk_sentences=max_chunk_sentences,
                    )
                )

            elif chunker_type == "fine_grained_punctuation":
                chunkers.append(FineGrainedPunctuationChunker(**params))

            elif chunker_type == "noop" or chunker_type == "no_op":
                chunkers.append(NoOpChunker(**params))

            else:
                logger.warning(
                    f"Unknown chunker type '{chunker_type}'. "
                    f"Valid types: paragraph, sentence, semantic, fine_grained_punctuation, noop."
                )

        if not chunkers:
            # Default to NoOpChunker if no valid chunkers specified
            logger.warning(
                "No valid chunkers specified in config, using NoOpChunker as fallback."
            )
            chunkers.append(NoOpChunker(language=self.language))

        return chunkers

    def _create_assertion_detector(self) -> AssertionDetector:
        """
        Create assertion detector based on configuration.

        Returns:
            Initialized assertion detector
        """
        # Check if assertion detection is disabled
        if self.assertion_config.get("disable", False):
            # Return a placeholder detector that always returns AFFIRMED
            class PlaceholderDetector(AssertionDetector):
                def detect(self, text_chunk: str):
                    return AssertionStatus.AFFIRMED, {"detection_disabled": True}

            return PlaceholderDetector(language=self.language)

        # Extract configuration
        enable_keyword = self.assertion_config.get("enable_keyword", True)
        enable_dependency = self.assertion_config.get("enable_dependency", True)
        preference = self.assertion_config.get("preference", "dependency")

        # Create combined detector with the specified configuration
        return CombinedAssertionDetector(
            language=self.language,
            enable_keyword=enable_keyword,
            enable_dependency=enable_dependency,
            preference=preference,
        )

    def process(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Process raw text through the chunking pipeline and assertion detection.

        Args:
            raw_text: The raw text to process

        Returns:
            List of processed chunks with assertion information:
            [
                {
                    'text': str,                    # The chunk text
                    'status': AssertionStatus,      # Assertion status enum
                    'assertion_details': Dict,      # Details about assertion detection
                    'source_indices': Dict          # Tracking information about source
                },
                ...
            ]
        """
        if not raw_text or not raw_text.strip():
            logger.warning(
                "Empty input text provided to TextProcessingPipeline.process"
            )
            return []

        # Apply initial normalization
        normalized_text = normalize_line_endings(raw_text)

        # Start with a single chunk
        current_chunks = [normalized_text]

        # Track source information (optional, mainly for debugging/analysis)
        source_indices = [{"original": True}]

        # Apply chunking pipeline stages iteratively
        for chunker_idx, chunker in enumerate(self.chunkers):
            logger.debug(
                f"Applying chunker {chunker_idx + 1}/{len(self.chunkers)}: "
                f"{chunker.__class__.__name__} to {len(current_chunks)} chunks"
            )

            next_stage_chunks = []
            next_stage_indices = []

            # Process each chunk through the current chunker
            for chunk_idx, chunk_to_process in enumerate(current_chunks):
                # Clean the chunk
                cleaned_chunk = clean_internal_newlines_and_extra_spaces(
                    chunk_to_process
                )
                if not cleaned_chunk:
                    continue

                # Apply the chunker
                newly_chunked_parts = chunker.chunk(cleaned_chunk)

                # Add new chunks with source tracking
                for part_idx, part in enumerate(newly_chunked_parts):
                    next_stage_chunks.append(part)

                    # Track source indices
                    source_info = {
                        f"stage_{chunker_idx}": {
                            "parent_idx": chunk_idx,
                            "part_idx": part_idx,
                            "chunker": chunker.__class__.__name__,
                        }
                    }

                    # Copy previous source info if available
                    if chunk_idx < len(source_indices):
                        source_info.update(source_indices[chunk_idx])

                    next_stage_indices.append(source_info)

            # Update for next stage
            current_chunks = next_stage_chunks
            source_indices = next_stage_indices

            logger.debug(
                f"Chunker {chunker_idx + 1} produced {len(current_chunks)} chunks"
            )

        # Final raw chunks after all chunking stages
        final_raw_chunks = current_chunks
        final_source_indices = source_indices

        # Process final chunks for assertion
        processed_chunks_with_assertion = []

        for chunk_idx, final_raw_chunk in enumerate(final_raw_chunks):
            # Clean the final chunk
            cleaned_final_chunk = clean_internal_newlines_and_extra_spaces(
                final_raw_chunk
            )
            if not cleaned_final_chunk:
                continue

            # Detect assertion status
            assertion_status, assertion_details = self.assertion_detector.detect(
                cleaned_final_chunk
            )

            # Store processed chunk with all information
            source_info = (
                final_source_indices[chunk_idx]
                if chunk_idx < len(final_source_indices)
                else {}
            )

            processed_chunks_with_assertion.append(
                {
                    "text": cleaned_final_chunk,
                    "status": assertion_status,
                    "assertion_details": assertion_details,
                    "source_indices": source_info,
                }
            )

        logger.info(
            f"TextProcessingPipeline processed text into {len(processed_chunks_with_assertion)} "
            f"final chunks with assertion status"
        )

        return processed_chunks_with_assertion
