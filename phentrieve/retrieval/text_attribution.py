"""
Module for text attribution functionality in Phentrieve.

This module provides functions to identify text spans in source chunks that
correspond to HPO terms, supporting attribution of phenotype observations to
specific parts of the text.
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_text_attributions(
    source_chunk_text: str,
    hpo_term_label: str,
    hpo_term_synonyms: Optional[list[str]] = None,
    hpo_term_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Find text spans in a chunk that correspond to an HPO term or its synonyms.

    Args:
        source_chunk_text: The text of a single source chunk to search within.
        hpo_term_label: The primary label of the HPO term.
        hpo_term_synonyms: Optional list of synonyms for the HPO term.
        hpo_term_id: Optional HPO ID for logging/context.

    Returns:
        List of dictionaries, each containing:
        - start_char: Starting character position of the match in the chunk.
        - end_char: Ending character position of the match in the chunk.
        - matched_text_in_chunk: The exact text that matched.
    """
    if not source_chunk_text:
        logger.debug(f"Empty source chunk text for HPO term {hpo_term_id}")
        return []

    attribution_spans = []

    # Create a list of phrases to search for
    search_phrases = [hpo_term_label]
    if hpo_term_synonyms:
        search_phrases.extend(hpo_term_synonyms)

    # Remove duplicates and empty phrases
    search_phrases = [phrase for phrase in set(search_phrases) if phrase]

    # Sort by length (descending) to prioritize longer matches
    search_phrases.sort(key=len, reverse=True)

    # Keep track of matched spans to avoid duplicates
    matched_spans = set()

    for phrase in search_phrases:
        try:
            # Escape special regex characters but allow for flexible whitespace
            # This converts spaces to \s+ to match variations in whitespace
            escaped_phrase = re.escape(phrase).replace(r"\ ", r"\s+")

            # Find all case-insensitive matches
            for match in re.finditer(escaped_phrase, source_chunk_text, re.IGNORECASE):
                span = (match.start(), match.end())

                # Check if this span overlaps with any previously matched span
                overlaps = False
                for existing_span in matched_spans:
                    # Check for overlap
                    if max(span[0], existing_span[0]) < min(span[1], existing_span[1]):
                        overlaps = True
                        break

                if not overlaps:
                    matched_spans.add(span)
                    attribution_spans.append(
                        {
                            "start_char": span[0],
                            "end_char": span[1],
                            "matched_text_in_chunk": match.group(0),
                        }
                    )

                    logger.debug(
                        f"Found attribution for {hpo_term_id} ({phrase}) at "
                        f"positions {span[0]}-{span[1]}: '{match.group(0)}'"
                    )
        except re.error as e:
            logger.warning(f"Regex error when searching for '{phrase}' in chunk: {e}")
            continue

    if attribution_spans:
        logger.debug(
            f"Found {len(attribution_spans)} attribution spans for {hpo_term_id}"
        )
    else:
        logger.debug(f"No attribution spans found for {hpo_term_id}")

    return attribution_spans
