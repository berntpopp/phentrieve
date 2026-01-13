"""
Family history processing module for extracting phenotypes from family history contexts.

This module provides functionality to enhance HPO term extraction from clinical text
by identifying and processing family history mentions separately. This addresses the
challenge where specific clinical phenotypes mentioned within family history contexts
are not properly extracted due to semantic dilution.

Key Features:
    - Detects family history mentions in clinical text using pattern matching
    - Extracts specific phenotypes from family history contexts
    - Matches extracted phenotypes to HPO terms via dense retrieval
    - Annotates results with family relationship metadata
    - Preserves both the family history context AND specific clinical terms

Problem Addressed:
    When clinical text mentions "Family history is significant for epilepsy in the
    maternal uncle", traditional chunking often groups the entire phrase together,
    causing:
    1. High similarity to HP:0032316 (Family history) - correct but generic
    2. Low similarity to HP:0001250 (Seizure) - the specific phenotype is lost

    This module solves this by:
    1. Detecting the family history chunk
    2. Extracting "epilepsy" as a separate phenotype
    3. Querying the retriever specifically for "epilepsy"
    4. Matching it to HP:0001250 (Seizure) with proper text attribution
    5. Marking it with family_history=True and relationship="maternal uncle"

Usage Example:
    >>> from phentrieve.text_processing.family_history_processor import (
    ...     is_family_history_chunk,
    ...     extract_phenotypes_from_family_history
    ... )
    >>>
    >>> text = "Family history is significant for epilepsy in the maternal uncle."
    >>> is_family_history_chunk(text)
    True
    >>>
    >>> extractions = extract_phenotypes_from_family_history(text)
    >>> for ext in extractions:
    ...     print(f"Phenotype: {ext.phenotype_text}, Relationship: {ext.relationship}")
    Phenotype: epilepsy, Relationship: maternal uncle

Integration:
    This module is integrated into the HPO extraction orchestrator via the
    `process_family_history_chunks()` function, which is called after initial
    retrieval to enhance results with family history phenotype extractions.

    Enable via CLI:
        phentrieve text process "..." --enable-family-history-extraction

    Or programmatically:
        orchestrate_hpo_extraction(..., enable_family_history_extraction=True)

Notes:
    - This feature is DISABLED by default to maintain backward compatibility
    - Extraction patterns are optimized for English clinical text
    - Family relationships are identified using regex patterns
    - Extracted phenotypes are filtered to remove stopwords and non-medical terms

See Also:
    - phentrieve/text_processing/hpo_extraction_orchestrator.py
    - phentrieve/cli/text_commands.py (--enable-family-history-extraction flag)
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from phentrieve.retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


# Family history trigger patterns (case-insensitive)
FAMILY_HISTORY_PATTERNS = [
    r"\bfamily\s+history\b",
    r"\bfamilial\b",
    r"\bmother\s+(?:has|had|with)\b",
    r"\bfather\s+(?:has|had|with)\b",
    r"\bsister\s+(?:has|had|with)\b",
    r"\bbrother\s+(?:has|had|with)\b",
    r"\bmaternal\s+(?:uncle|aunt|grandfather|grandmother)\b",
    r"\bpaternal\s+(?:uncle|aunt|grandfather|grandmother)\b",
    r"\bgrandmother\s+(?:has|had|with)\b",
    r"\bgrandfather\s+(?:has|had|with)\b",
    r"\bsibling\s+(?:has|had|with)\b",
    r"\brelative\s+(?:has|had|with)\b",
    r"\bfamily\s+member\b",
    r"\bin\s+the\s+family\b",
]

# Compile patterns for efficiency
FAMILY_HISTORY_REGEX = re.compile("|".join(FAMILY_HISTORY_PATTERNS), re.IGNORECASE)

# Clinical phenotype extraction patterns
# These capture specific medical terms within family history contexts
PHENOTYPE_EXTRACTION_PATTERNS = [
    # "family history (is significant) for <phenotype>"
    r"(?:family\s+history|familial)(?:\s+is\s+significant)?\s+(?:of|for)\s+([a-z][a-z\s]{2,30}?)(?:\s+in\s+|\s+and\s+|\.|\,|$)",
    # "<phenotype> in <relative>"
    r"\b([a-z][a-z\s]{2,30}?)\s+in\s+(?:the\s+)?(?:maternal|paternal|mother|father|sister|brother|uncle|aunt|grandfather|grandmother)",
    # "<relative> has/had <phenotype>"
    r"(?:mother|father|sister|brother|uncle|aunt|grandfather|grandmother|relative|sibling)\s+(?:has|had|with)\s+([a-z][a-z\s]{2,30}?)(?:\s+and\s+|\.|\,|$)",
]


@dataclass
class FamilyHistoryExtraction:
    """Represents a phenotype extracted from family history context.

    Attributes:
        phenotype_text: The extracted phenotype text (e.g., "epilepsy")
        start_char: Start position in original chunk
        end_char: End position in original chunk
        family_context: The broader family history context
        relationship: Specific family relationship if identified (e.g., "maternal uncle")
    """

    phenotype_text: str
    start_char: int
    end_char: int
    family_context: str
    relationship: Optional[str] = None


def is_family_history_chunk(chunk_text: str) -> bool:
    """
    Detect if a text chunk contains family history information.

    Args:
        chunk_text: Text chunk to analyze

    Returns:
        True if chunk contains family history patterns
    """
    return bool(FAMILY_HISTORY_REGEX.search(chunk_text))


def extract_phenotypes_from_family_history(
    chunk_text: str,
) -> list[FamilyHistoryExtraction]:
    """
    Extract specific phenotypes mentioned within family history contexts.

    This function identifies clinical terms (like "epilepsy", "seizures") within
    family history statements and returns them as separate extraction targets.

    Args:
        chunk_text: Text chunk containing family history information

    Returns:
        List of extracted phenotypes with their positions and context
    """
    extractions: list[FamilyHistoryExtraction] = []

    if not is_family_history_chunk(chunk_text):
        return extractions

    # Try each extraction pattern
    for pattern in PHENOTYPE_EXTRACTION_PATTERNS:
        for match in re.finditer(pattern, chunk_text, re.IGNORECASE):
            if match.groups():
                phenotype = match.group(1).strip()

                # Filter out non-medical terms and stopwords
                if _is_valid_phenotype_text(phenotype):
                    # Find the actual position in the original text
                    start = match.start(1)
                    end = match.end(1)

                    # Try to identify the specific family relationship
                    relationship = _extract_relationship(chunk_text)

                    extractions.append(
                        FamilyHistoryExtraction(
                            phenotype_text=phenotype,
                            start_char=start,
                            end_char=end,
                            family_context=chunk_text,
                            relationship=relationship,
                        )
                    )

    # Deduplicate based on phenotype text (keep first occurrence)
    seen = set()
    unique_extractions = []
    for ext in extractions:
        key = ext.phenotype_text.lower()
        if key not in seen:
            seen.add(key)
            unique_extractions.append(ext)

    return unique_extractions


def _is_valid_phenotype_text(text: str) -> bool:
    """
    Validate that extracted text is likely a clinical phenotype.

    Args:
        text: Extracted text to validate

    Returns:
        True if text is likely a valid phenotype mention
    """
    # Filter too short or too long
    if len(text) < 3 or len(text) > 50:
        return False

    # Filter common stopwords and non-medical terms
    stopwords = {
        "the",
        "and",
        "or",
        "but",
        "with",
        "for",
        "is",
        "was",
        "has",
        "had",
        "been",
        "being",
        "have",
        "are",
        "were",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "can",
        "must",
        "shall",
        "significant",
        "history",
        "family",
        "member",
        "relative",
        "maternal",
        "paternal",
    }

    text_lower = text.lower().strip()
    if text_lower in stopwords:
        return False

    # Must contain at least one letter
    if not re.search(r"[a-z]", text, re.IGNORECASE):
        return False

    return True


def _extract_relationship(text: str) -> Optional[str]:
    """
    Extract the specific family relationship from text.

    Args:
        text: Text to search for relationship

    Returns:
        Identified relationship or None
    """
    relationship_patterns = [
        (r"maternal\s+uncle", "maternal uncle"),
        (r"maternal\s+aunt", "maternal aunt"),
        (r"paternal\s+uncle", "paternal uncle"),
        (r"paternal\s+aunt", "paternal aunt"),
        (r"maternal\s+grandfather", "maternal grandfather"),
        (r"maternal\s+grandmother", "maternal grandmother"),
        (r"paternal\s+grandfather", "paternal grandfather"),
        (r"paternal\s+grandmother", "paternal grandmother"),
        (r"\bmother\b", "mother"),
        (r"\bfather\b", "father"),
        (r"\bsister\b", "sister"),
        (r"\bbrother\b", "brother"),
        (r"\bgrandmother\b", "grandmother"),
        (r"\bgrandfather\b", "grandfather"),
        (r"\buncle\b", "uncle"),
        (r"\baunt\b", "aunt"),
        (r"\bsibling\b", "sibling"),
    ]

    for pattern, label in relationship_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return label

    return None


def process_family_history_chunks(
    chunk_results: list[dict[str, Any]],
    retriever: DenseRetriever,
    num_results: int = 10,
    retrieval_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Process family history chunks to extract and annotate specific phenotypes.

    This function enhances the standard chunk processing by:
    1. Identifying family history chunks
    2. Extracting specific phenotypes mentioned within
    3. Retrieving HPO terms for those phenotypes
    4. Annotating them with family_history=True

    Args:
        chunk_results: List of chunk results from standard processing
        retriever: Dense retriever for HPO term matching
        num_results: Number of HPO matches to retrieve per phenotype
        retrieval_threshold: Minimum similarity threshold

    Returns:
        Enhanced chunk results with family history phenotypes added
    """
    enhanced_results = []

    for chunk_result in chunk_results:
        chunk_text = chunk_result.get("chunk_text", "")
        chunk_idx = chunk_result.get("chunk_idx", 0)

        # Keep the original result
        enhanced_results.append(chunk_result)

        # Check if this is a family history chunk
        if is_family_history_chunk(chunk_text):
            logger.debug(
                f"Detected family history in chunk {chunk_idx}: {chunk_text[:50]}..."
            )

            # Extract specific phenotypes
            phenotype_extractions = extract_phenotypes_from_family_history(chunk_text)

            if phenotype_extractions:
                logger.info(
                    f"Extracted {len(phenotype_extractions)} phenotypes from family history chunk {chunk_idx}"
                )

                # Query retriever for each extracted phenotype
                phenotype_texts = [ext.phenotype_text for ext in phenotype_extractions]
                batch_results = retriever.query_batch(
                    texts=phenotype_texts,
                    n_results=num_results,
                    include_similarities=True,
                )

                # Process results for each phenotype
                for ext, query_result in zip(phenotype_extractions, batch_results):
                    if (
                        not query_result.get("metadatas")
                        or not query_result["metadatas"][0]
                    ):
                        logger.debug(
                            f"No query results for phenotype '{ext.phenotype_text}'"
                        )
                        continue

                    # Build matches for this phenotype
                    phenotype_matches = []
                    total_candidates = len(query_result["metadatas"][0])
                    filtered_count = 0

                    for i, metadata in enumerate(query_result["metadatas"][0]):
                        similarity = (
                            query_result.get("distances", [[]])[0][i]
                            if query_result.get("distances")
                            else 0.0
                        )
                        hpo_id = metadata.get("hpo_id", "")
                        hpo_name = metadata.get("hpo_name", "")

                        if similarity < retrieval_threshold:
                            filtered_count += 1
                            if i < 3:  # Log top 3 filtered matches
                                logger.debug(
                                    f"Filtered phenotype '{ext.phenotype_text}' -> {hpo_id} ({hpo_name}): "
                                    f"score {similarity:.4f} < threshold {retrieval_threshold}"
                                )
                            continue

                        phenotype_matches.append(
                            {
                                "id": hpo_id,
                                "name": hpo_name,
                                "score": float(similarity),
                                "assertion_status": "affirmed",  # Family history is typically affirmed
                                "family_history": True,  # Mark as family history
                                "family_relationship": ext.relationship,
                                "text_attributions": [
                                    {
                                        "start_char": ext.start_char,
                                        "end_char": ext.end_char,
                                        "matched_text_in_chunk": ext.phenotype_text,
                                    }
                                ],
                            }
                        )

                    if phenotype_matches:
                        logger.info(
                            f"Found {len(phenotype_matches)} HPO matches for phenotype '{ext.phenotype_text}' "
                            f"(filtered {filtered_count}/{total_candidates} below threshold)"
                        )

                        # Create a synthetic chunk result for this phenotype
                        synthetic_chunk = {
                            "chunk_idx": chunk_idx,  # Same chunk index
                            "chunk_text": ext.phenotype_text,  # Just the phenotype text
                            "start_char": chunk_result.get("start_char", 0)
                            + ext.start_char,
                            "end_char": chunk_result.get("start_char", 0)
                            + ext.end_char,
                            "matches": phenotype_matches,
                            "assertion_status": "affirmed",
                            "is_family_history_extraction": True,  # Flag as synthetic
                            "parent_chunk_idx": chunk_idx,
                            "family_context": chunk_text,
                        }
                        enhanced_results.append(synthetic_chunk)
                    else:
                        logger.debug(
                            f"No matches above threshold for phenotype '{ext.phenotype_text}' "
                            f"({filtered_count}/{total_candidates} candidates filtered)"
                        )

    return enhanced_results
