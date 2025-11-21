"""
Output formatters for query results.

This module provides functions to format query results into different
output formats (text, JSON, JSON Lines) for the query commands.
"""

import json
from typing import Any


def format_results_as_text(
    structured_query_results: list[dict[str, Any]], sentence_mode: bool
) -> str:
    """
    Format structured query results as a human-readable text string.

    Args:
        structured_query_results: List of structured result dictionaries
        sentence_mode: Whether the results are in sentence mode (multiple segments)

    Returns:
        Formatted text string for display
    """
    output_lines = []

    for _i, result_set in enumerate(structured_query_results):
        # Add a header for each segment if in sentence mode with multiple segments
        if sentence_mode and len(structured_query_results) > 1:
            output_lines.append("")
            query_text = result_set["query_text_processed"]
            output_lines.append(f"==== Results for: {query_text} ====")

        # Display original query assertion status if available
        og_query_status_val = result_set.get(
            "original_query_assertion_status"
        ) or result_set.get("original_query_assertion_status_value")
        if og_query_status_val:
            # It's important to clarify this status is for the *original* query,
            # especially if sentence_mode=True and query_text_segment is just one sentence.
            output_lines.append(
                f"Detected Assertion for Original Input Query: {og_query_status_val.upper()}"
            )

        # Add the header info (e.g., "Found X matching HPO terms:")
        output_lines.append(result_set["header_info"])

        # No results case
        if not result_set.get("results") or len(result_set["results"]) == 0:
            continue

        # Format each result
        for result in result_set["results"]:
            rank_display = f"{result['rank']}."
            hpo_id = result["hpo_id"]
            label = result["label"]
            similarity = f"{result['similarity']:.2f}"

            # Add re-ranking information if available
            reranking_info = ""
            if "cross_encoder_score" in result:
                ce_score = f"{result['cross_encoder_score']:.2f}"
                original_rank = result.get("original_rank", "?")
                reranking_info = (
                    f" [re-ranked from #{original_rank}, cross-encoder: {ce_score}]"
                )

            # Format the line
            line = (
                f"{rank_display:3} {hpo_id:11} {label} "
                f"(similarity: {similarity}){reranking_info}"
            )
            output_lines.append(line)

            # Add definition and synonyms if available
            definition = result.get("definition")
            synonyms = result.get("synonyms")

            if definition:
                output_lines.append(f"    Definition: {definition}")

            if synonyms:
                synonyms_str = ", ".join(synonyms)
                output_lines.append(f"    Synonyms: {synonyms_str}")

    return "\n".join(output_lines)


def format_results_as_json(
    structured_query_results: list[dict[str, Any]], sentence_mode: bool
) -> str:
    """
    Format structured query results as a JSON string.

    Args:
        structured_query_results: List of structured result dictionaries
        sentence_mode: Whether the results are in sentence mode (multiple segments)

    Returns:
        Formatted JSON string
    """
    # Transform the structure to match the harmonized design
    transformed_results = []

    for result_set in structured_query_results:
        # Get assertion status with fallback
        assertion_status = result_set.get("original_query_assertion_status")
        if not assertion_status:
            assertion_status = result_set.get("original_query_assertion_status_value")

        transformed_result = {
            "query_text_processed": result_set["query_text_processed"],
            "retrieval_info": {"header": result_set["header_info"]},
            "hpo_terms": [],
            "original_query_assertion_status": assertion_status,
        }

        # Transform each HPO term result
        for result in result_set.get("results", []):
            hpo_term = {
                "rank": result["rank"],
                "hpo_id": result["hpo_id"],
                "name": result["label"],
                "confidence": result["similarity"],
            }

            # Add cross-encoder score if available
            if "cross_encoder_score" in result:
                hpo_term["cross_encoder_score"] = result["cross_encoder_score"]

            # Add original rank if available
            if "original_rank" in result:
                hpo_term["original_rank"] = result["original_rank"]

            # Add definition and synonyms if available
            if "definition" in result:
                hpo_term["definition"] = result["definition"]

            if "synonyms" in result:
                hpo_term["synonyms"] = result["synonyms"]

            transformed_result["hpo_terms"].append(hpo_term)

        transformed_results.append(transformed_result)

    # If not in sentence mode or only one result, return as a JSON object
    if not sentence_mode or len(transformed_results) == 1:
        return json.dumps(transformed_results[0], indent=2, ensure_ascii=False)

    # Otherwise, return the full list for sentence mode with multiple segments
    return json.dumps(transformed_results, indent=2, ensure_ascii=False)


def format_results_as_jsonl(structured_query_results: list[dict[str, Any]]) -> str:
    """
    Format structured query results as a JSON Lines string.

    Args:
        structured_query_results: List of structured result dictionaries

    Returns:
        Formatted JSON Lines string (one JSON object per line)
    """
    # Transform the structure to match the harmonized design
    transformed_results = []

    for result_set in structured_query_results:
        # Get assertion status with fallback
        assertion_status = result_set.get("original_query_assertion_status")
        if not assertion_status:
            assertion_status = result_set.get("original_query_assertion_status_value")

        transformed_result = {
            "query_text_processed": result_set["query_text_processed"],
            "retrieval_info": {"header": result_set["header_info"]},
            "hpo_terms": [],
            "original_query_assertion_status": assertion_status,
        }

        # Transform each HPO term result
        for result in result_set.get("results", []):
            hpo_term = {
                "rank": result["rank"],
                "hpo_id": result["hpo_id"],
                "name": result["label"],
                "confidence": result["similarity"],
            }

            # Add cross-encoder score if available
            if "cross_encoder_score" in result:
                hpo_term["cross_encoder_score"] = result["cross_encoder_score"]

            # Add original rank if available
            if "original_rank" in result:
                hpo_term["original_rank"] = result["original_rank"]

            # Add definition and synonyms if available
            if "definition" in result:
                hpo_term["definition"] = result["definition"]

            if "synonyms" in result:
                hpo_term["synonyms"] = result["synonyms"]

            transformed_result["hpo_terms"].append(hpo_term)

        transformed_results.append(transformed_result)

    # Concatenate each result as a JSON line
    return "\n".join(
        json.dumps(result, ensure_ascii=False) for result in transformed_results
    )
