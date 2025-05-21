"""
Text processing pipeline orchestrator for Phentrieve.

This module provides pipeline components that manage the sequence
of text processing operations including chunking and assertion detection.
"""

import copy
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import spacy
from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.cleaners import (
    normalize_line_endings,
    clean_internal_newlines_and_extra_spaces,
)
from phentrieve.text_processing.chunkers import (
    FineGrainedPunctuationChunker,
    NoOpChunker,
    ParagraphChunker,
    PreChunkSemanticGrouper,
    SentenceChunker,
    TextChunker,
)
from phentrieve.text_processing.sliding_window_chunker import (
    SlidingWindowSemanticSplitter,
)
from phentrieve.text_processing.assertion_detection import (
    AssertionDetector,
    AssertionStatus,
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

        for stage_config in self.chunking_pipeline_config:
            # Get chunker configuration
            if isinstance(stage_config, dict):
                chunker_type = stage_config.get("type", "unknown")
                chunker_config = stage_config.get("config", {})
            else:
                chunker_type = stage_config
                chunker_config = {}

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

                # For backward compatibility: redirect 'semantic' chunker type to use
                # the sliding window semantic splitter, which gives better results
                logger.info(
                    "'semantic' chunker type is deprecated, using sliding_window_semantic instead"
                )

                # Get sliding window parameters (with defaults tuned to match old semantic chunker behavior)
                window_size = chunker_config.get("window_size_tokens", 3)
                step_size = chunker_config.get("step_size_tokens", 1)
                threshold = chunker_config.get("similarity_threshold", 0.4)
                min_segment_length = chunker_config.get(
                    "min_split_segment_length_words", 2
                )

                # Create sliding window semantic splitter
                sliding_window_params = {
                    **params,
                    "model": self.sbert_model,
                    "window_size_tokens": window_size,
                    "step_size_tokens": step_size,
                    "splitting_threshold": threshold,
                    "min_split_segment_length_words": min_segment_length,
                }
                # Use sliding window instead of semantic chunker
                chunkers.append(SlidingWindowSemanticSplitter(**sliding_window_params))

            elif chunker_type == "pre_chunk_semantic_grouper":
                if not self.sbert_model:
                    raise ValueError(
                        "SentenceTransformer model required for PreChunkSemanticGrouper "
                        "but not provided to pipeline."
                    )

                # Get grouper specific parameters
                similarity_threshold = chunker_config.get("similarity_threshold", 0.5)
                min_group_size = chunker_config.get("min_group_size", 1)
                max_group_size = chunker_config.get("max_group_size", 7)

                # Create pre-chunk semantic grouper
                grouper_params = {
                    **params,
                    "model": self.sbert_model,
                    "similarity_threshold": similarity_threshold,
                    "min_group_size": min_group_size,
                    "max_group_size": max_group_size,
                }
                chunkers.append(PreChunkSemanticGrouper(**grouper_params))

            elif chunker_type == "fine_grained_punctuation":
                chunkers.append(FineGrainedPunctuationChunker(**params))

            elif chunker_type == "noop":
                chunkers.append(NoOpChunker(**params))

            elif chunker_type == "sliding_window_semantic":
                if not self.sbert_model:
                    raise ValueError(
                        "SentenceTransformer model required for sliding window semantic splitting "
                        "but none was provided"
                    )

                # Get sliding window specific parameters
                window_size = chunker_config.get("window_size_tokens", 4)
                step_size = chunker_config.get("step_size_tokens", 2)
                threshold = chunker_config.get("splitting_threshold", 0.5)
                min_segment_length = chunker_config.get(
                    "min_split_segment_length_words", 50
                )

                # Create sliding window semantic splitter
                sliding_window_params = {
                    **params,
                    "model": self.sbert_model,
                    "window_size_tokens": window_size,
                    "step_size_tokens": step_size,
                    "splitting_threshold": threshold,
                    "min_split_segment_length_words": min_segment_length,
                }
                chunkers.append(SlidingWindowSemanticSplitter(**sliding_window_params))

            else:
                logger.warning(
                    f"Unknown chunker type '{chunker_type}' in config, skipping"
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
                    'text': str,                # The chunk text
                    'status': AssertionStatus, # Enum value indicating assertion status
                    'assertion_details': Dict,  # Details about the assertion analysis
                    'source_indices': Dict      # Tracking information about source
                },
                ...
            ]
        """
        if not raw_text or not raw_text.strip():
            logger.warning(
                "TextProcessingPipeline.process called with empty input text."
            )
            return []

        normalized_text = normalize_line_endings(raw_text)
        current_segments_for_stage: List[str] = [normalized_text]

        # Simplified source tracking for this refactor:
        # Each element in current_source_info_list corresponds to a segment in current_segments_for_stage
        # It stores a list of strings describing the applied chunkers.
        current_source_info_list: List[List[str]] = [["initial_raw_text"]]

        for chunker_idx, chunker_instance in enumerate(self.chunkers):
            chunker_name = chunker_instance.__class__.__name__
            logger.debug(
                f"Pipeline Stage {chunker_idx + 1}: Applying {chunker_name} "
                f"to {len(current_segments_for_stage)} segment(s)."
            )

            input_segments_to_current_chunker = current_segments_for_stage
            input_source_info_to_current_chunker = current_source_info_list

            output_segments_from_chunker = chunker_instance.chunk(
                input_segments_to_current_chunker
            )

            # Basic source tracking update:
            # If a chunker doesn't change the number of segments, source info maps 1:1.
            # If it splits one segment into many, all new segments inherit and append the current chunker's name.
            # This is a simplification. True lineage is harder.
            # For now, we'll just track that this chunker was applied.
            # A more robust way would be for chunkers to also return source mapping.

            updated_source_info_list = []
            if (
                len(input_segments_to_current_chunker)
                == len(output_segments_from_chunker)
                and len(input_segments_to_current_chunker) > 0
            ):
                # Assume 1:1 mapping if segment count doesn't change
                for i, _ in enumerate(output_segments_from_chunker):
                    updated_source_info_list.append(
                        input_source_info_to_current_chunker[i] + [chunker_name]
                    )
            else:  # Segment count changed, apply new source info more generically
                for _ in output_segments_from_chunker:
                    # This is a simplification; ideally, we'd know which input segment(s) an output segment came from.
                    # For now, just indicate this chunker ran. A more complex approach would be needed for precise lineage.
                    updated_source_info_list.append([f"processed_by_{chunker_name}"])

            current_segments_for_stage = output_segments_from_chunker
            current_source_info_list = updated_source_info_list

            logger.debug(
                f"Stage {chunker_idx + 1} ({chunker_name}) produced {len(current_segments_for_stage)} segment(s)."
            )

        final_raw_chunks_text_only: List[str] = current_segments_for_stage

        processed_chunks_with_assertion: List[Dict[str, Any]] = []
        for idx, final_text_chunk in enumerate(final_raw_chunks_text_only):
            cleaned_final_chunk = clean_internal_newlines_and_extra_spaces(
                final_text_chunk
            )
            if not cleaned_final_chunk:
                continue

            assertion_status, assertion_details = self.assertion_detector.detect(
                cleaned_final_chunk
            )

            source_info_for_chunk = (
                current_source_info_list[idx]
                if idx < len(current_source_info_list)
                else ["unknown_source"]
            )

            processed_chunks_with_assertion.append(
                {
                    "text": cleaned_final_chunk,
                    "status": assertion_status,  # This is the AssertionStatus Enum object
                    "assertion_details": assertion_details,
                    "source_indices": {
                        "processing_stages": source_info_for_chunk
                    },  # Simplified source info
                }
            )

        logger.info(
            f"TextProcessingPipeline processed text into {len(processed_chunks_with_assertion)} final asserted chunks."
        )
        return processed_chunks_with_assertion
