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
    PreChunkSemanticGrouper,
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

                # Get semantic chunker specific parameters
                similarity_threshold = chunker_config.get("similarity_threshold", 0.4)
                min_chunk_sentences = chunker_config.get("min_sentences", 1)
                max_chunk_sentences = chunker_config.get("max_sentences", 5)

                # Create semantic chunker
                semantic_params = {
                    **params,
                    "model": self.sbert_model,
                    "similarity_threshold": similarity_threshold,
                    "min_chunk_sentences": min_chunk_sentences,
                    "max_chunk_sentences": max_chunk_sentences,
                }
                chunkers.append(SemanticChunker(**semantic_params))

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
                    'text': 'chunk text',         # The chunk text
                    'status': AssertionStatus,    # POSITIVE, NEGATIVE, UNCERTAIN
                    'assertion_details': Dict,    # Assertion detection details
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

        # Initialize tracking of chunks and their groups (from paragraphs)
        # Outermost list: stages of pipeline
        # Middle list: groups of chunks (a paragraph's output becomes one group)
        # Innermost list: the chunks themselves (strings)
        list_of_chunk_groups_after_each_stage = [[[normalized_text]]]

        # Track source information with the same structure
        current_chunk_source_indices = [[{"original": True}]]

        # Apply chunking pipeline stages iteratively
        for chunker_idx, chunker in enumerate(self.chunkers):
            logger.debug(
                f"Pipeline Stage {chunker_idx + 1}: Applying {chunker.__class__.__name__}"
            )

            # Get the input groups for this chunker stage
            input_groups_for_this_stage = list_of_chunk_groups_after_each_stage[-1]
            output_groups_for_next_stage = []
            next_stage_source_indices_groups = []

            # Special handling for PreChunkSemanticGrouper
            if isinstance(chunker, PreChunkSemanticGrouper):
                logger.debug(
                    f"Using special processing for PreChunkSemanticGrouper with "
                    f"{len(input_groups_for_this_stage)} input groups"
                )

                # Process each group of pre-chunks (from a single paragraph)
                for group_idx, current_input_group in enumerate(
                    input_groups_for_this_stage
                ):
                    # current_input_group is a List[str] of pre-chunks from one paragraph
                    if not current_input_group:  # Skip empty groups
                        continue

                    # Get source indices for this group
                    group_source_indices_base = (
                        current_chunk_source_indices[-1][group_idx]
                        if group_idx < len(current_chunk_source_indices[-1])
                        else {}
                    )

                    # Log the group processing
                    logger.debug(
                        f"  PreChunkSemanticGrouper processing a group of "
                        f"{len(current_input_group)} pre-chunks."
                    )

                    # Process the group using chunk_pre_chunks method
                    processed_by_grouper = chunker.chunk_pre_chunks(current_input_group)

                    # Each output becomes a separate group of one chunk for next stage
                    for part_idx, final_chunk_str in enumerate(processed_by_grouper):
                        output_groups_for_next_stage.append([final_chunk_str])

                        # Track source information
                        # Initialize an empty dict to avoid issues
                        new_source_info = {}

                        # Ensure we're working with a dictionary for source tracking
                        if isinstance(group_source_indices_base, dict):
                            new_source_info = {**group_source_indices_base}
                        # If we get a list with one dict inside, use that dict
                        elif (
                            isinstance(group_source_indices_base, list)
                            and group_source_indices_base
                        ):
                            if isinstance(group_source_indices_base[0], dict):
                                new_source_info = {**group_source_indices_base[0]}
                            else:
                                logger.debug(
                                    f"Expected dict in list for semantic grouper source indices, got {type(group_source_indices_base[0])}"
                                )
                        else:
                            logger.debug(
                                f"Unexpected semantic grouper source indices type: {type(group_source_indices_base)}"
                            )

                        new_source_info[f"stage_{chunker_idx}_grouper"] = {
                            "input_pre_chunk_count": len(current_input_group),
                            "output_chunk_idx_in_group": part_idx,
                            "chunker": chunker.__class__.__name__,
                        }
                        next_stage_source_indices_groups.append([new_source_info])

            else:  # Standard chunker processing
                # Process each group (usually a paragraph)
                for group_idx, current_input_group in enumerate(
                    input_groups_for_this_stage
                ):
                    output_chunks_from_this_group = []
                    source_indices_for_this_group = []

                    # Get source indices for this group
                    group_source_indices_base = (
                        current_chunk_source_indices[-1][group_idx]
                        if group_idx < len(current_chunk_source_indices[-1])
                        else {}
                    )

                    # Process each item in the group individually
                    for item_idx, text_item in enumerate(current_input_group):
                        # Clean the item
                        cleaned_item = clean_internal_newlines_and_extra_spaces(
                            text_item
                        )
                        if not cleaned_item:
                            continue

                        # Apply the chunker
                        chunked_parts = chunker.chunk(cleaned_item)

                        # Add to output chunks for this group
                        output_chunks_from_this_group.extend(chunked_parts)

                        # Track source indices for each part
                        for part_idx, _ in enumerate(chunked_parts):
                            # Create source info for this part
                            # Initialize an empty dict to avoid issues
                            new_source_info = {}

                            # Ensure we're working with a dictionary for source tracking
                            if isinstance(group_source_indices_base, dict):
                                new_source_info = {**group_source_indices_base}
                            # If we get a list with one dict inside, use that dict
                            elif (
                                isinstance(group_source_indices_base, list)
                                and group_source_indices_base
                            ):
                                if isinstance(group_source_indices_base[0], dict):
                                    new_source_info = {**group_source_indices_base[0]}
                                else:
                                    logger.debug(
                                        f"Expected dict in list for source indices, got {type(group_source_indices_base[0])}"
                                    )
                            else:
                                logger.debug(
                                    f"Unexpected source indices type: {type(group_source_indices_base)}"
                                )

                            new_source_info[f"stage_{chunker_idx}"] = {
                                "item_idx_in_group": item_idx,
                                "output_part_idx": part_idx,
                                "chunker": chunker.__class__.__name__,
                            }
                            source_indices_for_this_group.append(new_source_info)

                    # Only add the group if it has output
                    if output_chunks_from_this_group:
                        # For regular chunkers, we don't need to maintain a
                        # paragraph-to-chunks relationship except for PreChunkSemanticGrouper
                        # For normal chunkers, we flatten the structure and each chunk becomes a group

                        # If next chunker is PreChunkSemanticGrouper, keep the chunks as one group
                        if chunker_idx + 1 < len(self.chunkers) and isinstance(
                            self.chunkers[chunker_idx + 1], PreChunkSemanticGrouper
                        ):
                            output_groups_for_next_stage.append(
                                output_chunks_from_this_group
                            )
                            next_stage_source_indices_groups.append(
                                source_indices_for_this_group
                            )
                        else:
                            # Otherwise, each chunk becomes its own group
                            for i, chunk in enumerate(output_chunks_from_this_group):
                                output_groups_for_next_stage.append([chunk])
                                if i < len(source_indices_for_this_group):
                                    # Ensure we're passing a dictionary, not a list
                                    source_index = source_indices_for_this_group[i]
                                    if isinstance(source_index, dict):
                                        next_stage_source_indices_groups.append(
                                            [source_index]
                                        )
                                    else:
                                        # Convert to dict if needed
                                        logger.debug(
                                            f"Converting source index type {type(source_index)} to dict"
                                        )
                                        next_stage_source_indices_groups.append([{}])
                                else:
                                    # Should not happen, but just in case
                                    next_stage_source_indices_groups.append([{}])

            # Update for next stage
            list_of_chunk_groups_after_each_stage.append(output_groups_for_next_stage)
            current_chunk_source_indices.append(next_stage_source_indices_groups)

            # Count total chunks for logging
            total_chunks = sum(len(group) for group in output_groups_for_next_stage)
            logger.debug(
                f"Chunker {chunker_idx + 1} produced {total_chunks} chunks "
                f"in {len(output_groups_for_next_stage)} groups"
            )

        # Get final chunks (flatten the groups structure)
        final_raw_chunks = []
        final_source_indices = []

        for group_idx, group in enumerate(list_of_chunk_groups_after_each_stage[-1]):
            final_raw_chunks.extend(group)
            if group_idx < len(current_chunk_source_indices[-1]):
                sources_for_group = current_chunk_source_indices[-1][group_idx]
                # Handle cases where source indices might be a list instead of dict
                for source in sources_for_group:
                    if isinstance(source, dict):
                        final_source_indices.append(source)
                    else:
                        logger.warning(
                            f"Expected dict for source indices, got {type(source)}"
                        )
                        final_source_indices.append({})
            else:
                final_source_indices.extend([{}] * len(group))

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
