"""
Document-level aggregation for mention-based HPO extraction.

This module aggregates mention groups to produce document-level
output that is compatible with existing benchmark evaluation
infrastructure.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from phentrieve.text_processing.mention import (
    DocumentMentions,
    MentionGroup,
    map_assertion_to_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """
    Configuration for document-level aggregation.

    Attributes:
        min_confidence: Minimum confidence for inclusion
        max_terms: Maximum terms to return (0 = unlimited)
        resolve_conflicts: How to resolve conflicting assertions
        include_family_history: Include family history findings
        dataset_format: Target dataset for assertion mapping
        include_details: Include detailed mention information
        include_alternatives: Include alternative HPO candidates above threshold
        alternative_threshold: Min score for alternatives (relative to top)
    """

    min_confidence: float = 0.0  # No filtering by default to preserve recall
    max_terms: int = 0
    resolve_conflicts: str = "majority"  # "majority", "max_confidence", "first"
    include_family_history: bool = True
    dataset_format: str = "phenobert"
    include_details: bool = False
    include_alternatives: bool = False  # Disabled - adds too many false positives
    alternative_threshold: float = 0.95  # Include only very close alternatives


class MentionAggregator:
    """
    Aggregate mention groups to document-level output.

    Handles:
    - Deduplication of HPO terms across groups
    - Conflict resolution for assertion status
    - Dataset-specific label mapping
    - Benchmark-compatible output format

    Example:
        >>> aggregator = MentionAggregator(config)
        >>> doc_output = aggregator.aggregate(document_mentions)
        >>> benchmark_tuples = doc_output["benchmark_format"]
    """

    def __init__(self, config: AggregationConfig | None = None):
        """
        Initialize the aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()

    def aggregate(self, doc_mentions: DocumentMentions) -> dict[str, Any]:
        """
        Aggregate document mentions to final output.

        Args:
            doc_mentions: DocumentMentions with groups

        Returns:
            Aggregated document-level output
        """
        logger.debug(
            f"Aggregating {doc_mentions.num_groups} groups for {doc_mentions.doc_id}"
        )

        # Collect terms from all groups
        hpo_terms = self._collect_terms(doc_mentions.groups)

        # Resolve conflicts
        hpo_terms = self._resolve_conflicts(hpo_terms)

        # Filter by confidence
        hpo_terms = self._filter_by_confidence(hpo_terms)

        # Filter family history if needed
        if not self.config.include_family_history:
            hpo_terms = self._filter_family_history(hpo_terms)

        # Apply max terms limit
        if self.config.max_terms > 0:
            hpo_terms = hpo_terms[: self.config.max_terms]

        # Build output
        output = self._build_output(doc_mentions, hpo_terms)

        return output

    def _collect_terms(
        self,
        groups: list[MentionGroup],
    ) -> list[dict[str, Any]]:
        """
        Collect terms from all groups.

        Args:
            groups: List of mention groups

        Returns:
            List of term dictionaries
        """
        # Collect by HPO ID to handle duplicates
        term_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for group in groups:
            if group.final_hpo is None:
                continue

            canonical_assertion = group.get_canonical_assertion()
            dataset_assertion = group.get_dataset_assertion(self.config.dataset_format)
            
            # Collect the final HPO
            hpo_id = group.final_hpo.hpo_id
            term_info = {
                "hpo_id": hpo_id,
                "label": group.final_hpo.label,
                "score": group.final_hpo.effective_score,
                "canonical_assertion": canonical_assertion,
                "dataset_assertion": dataset_assertion,
                "group_id": group.group_id,
                "num_mentions": group.num_mentions,
                "is_family_history": group.final_assertion.family_history
                if group.final_assertion
                else False,
                "confidence": group.confidence,
                "alternatives": [
                    {"hpo_id": c.hpo_id, "label": c.label, "score": c.effective_score}
                    for c in group.ranked_hpo_explanations[:3]
                ],
            }
            term_map[hpo_id].append(term_info)
            
            # Also collect alternatives above threshold for better recall
            if self.config.include_alternatives and group.ranked_hpo_explanations:
                top_score = group.final_hpo.effective_score
                threshold = top_score * self.config.alternative_threshold
                
                for alt in group.ranked_hpo_explanations[1:]:  # Skip the first (already added)
                    if alt.effective_score >= threshold and alt.hpo_id != hpo_id:
                        alt_info = {
                            "hpo_id": alt.hpo_id,
                            "label": alt.label,
                            "score": alt.effective_score,
                            "canonical_assertion": canonical_assertion,
                            "dataset_assertion": dataset_assertion,
                            "group_id": group.group_id,
                            "num_mentions": group.num_mentions,
                            "is_family_history": group.final_assertion.family_history
                            if group.final_assertion
                            else False,
                            "confidence": group.confidence * 0.9,  # Slightly lower confidence
                            "is_alternative": True,
                            "alternatives": [],
                        }
                        term_map[alt.hpo_id].append(alt_info)

        # Merge duplicate HPO IDs
        merged_terms: list[dict[str, Any]] = []
        for hpo_id, term_list in term_map.items():
            merged = self._merge_term_entries(term_list)
            merged_terms.append(merged)

        # Sort by score
        merged_terms.sort(key=lambda t: t["score"], reverse=True)

        return merged_terms

    def _merge_term_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Merge multiple entries for the same HPO ID.

        Args:
            entries: List of term entries for same HPO ID

        Returns:
            Merged term dictionary
        """
        if len(entries) == 1:
            return entries[0]

        # Use best score
        best_entry = max(entries, key=lambda e: e["score"])

        # Aggregate counts
        total_mentions = sum(e["num_mentions"] for e in entries)

        # Collect all group IDs
        group_ids = [e["group_id"] for e in entries]

        # Merge assertions - collect all for conflict resolution
        assertion_entries = [
            {
                "assertion": e["canonical_assertion"],
                "confidence": e.get("confidence", 1.0),
                "count": e["num_mentions"],
            }
            for e in entries
        ]

        merged = {
            **best_entry,
            "num_mentions": total_mentions,
            "group_ids": group_ids,
            "assertion_entries": assertion_entries,
        }

        return merged

    def _resolve_conflicts(
        self,
        terms: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Resolve assertion conflicts for terms with multiple entries.

        Args:
            terms: List of term dictionaries

        Returns:
            Terms with resolved assertions
        """
        for term in terms:
            if "assertion_entries" not in term:
                continue

            entries = term["assertion_entries"]
            if len(entries) <= 1:
                continue

            # Resolve based on strategy
            if self.config.resolve_conflicts == "majority":
                resolved = self._resolve_by_majority(entries)
            elif self.config.resolve_conflicts == "max_confidence":
                resolved = self._resolve_by_confidence(entries)
            else:  # "first"
                resolved = entries[0]["assertion"]

            term["canonical_assertion"] = resolved
            term["dataset_assertion"] = map_assertion_to_dataset(
                resolved, self.config.dataset_format
            )

            # Clean up
            del term["assertion_entries"]

        return terms

    def _resolve_by_majority(
        self,
        entries: list[dict[str, Any]],
    ) -> str:
        """Resolve by majority vote weighted by mention count."""
        assertion_counts: dict[str, int] = defaultdict(int)

        for entry in entries:
            assertion_counts[entry["assertion"]] += entry["count"]

        return max(assertion_counts, key=lambda a: assertion_counts[a])

    def _resolve_by_confidence(
        self,
        entries: list[dict[str, Any]],
    ) -> str:
        """Resolve by highest confidence."""
        best = max(entries, key=lambda e: e["confidence"])
        return str(best["assertion"])

    def _filter_by_confidence(
        self,
        terms: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter terms below minimum confidence."""
        if self.config.min_confidence <= 0:
            return terms

        return [t for t in terms if t["score"] >= self.config.min_confidence]

    def _filter_family_history(
        self,
        terms: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter out family history terms."""
        return [t for t in terms if not t.get("is_family_history", False)]

    def _build_output(
        self,
        doc_mentions: DocumentMentions,
        terms: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build the final output dictionary.

        Args:
            doc_mentions: Original document mentions
            terms: Filtered and resolved terms

        Returns:
            Output dictionary
        """
        # Benchmark format: list of (hpo_id, assertion) tuples
        benchmark_format = [(t["hpo_id"], t["dataset_assertion"]) for t in terms]

        output = {
            "doc_id": doc_mentions.doc_id,
            "num_terms": len(terms),
            "benchmark_format": benchmark_format,
            "terms": [
                {
                    "id": t["hpo_id"],
                    "name": t["label"],
                    "score": t["score"],
                    "assertion": t["dataset_assertion"],
                    "num_mentions": t["num_mentions"],
                }
                for t in terms
            ],
        }

        if self.config.include_details:
            output["detailed_terms"] = terms
            output["num_mentions"] = doc_mentions.num_mentions
            output["num_groups"] = doc_mentions.num_groups
            output["mentions"] = [m.to_dict() for m in doc_mentions.mentions]
            output["groups"] = [g.to_dict() for g in doc_mentions.groups]

        return output


def aggregate_mentions_to_document(
    doc_mentions: DocumentMentions,
    config: AggregationConfig | None = None,
) -> dict[str, Any]:
    """
    Convenience function to aggregate mentions to document level.

    Args:
        doc_mentions: Document mentions with groups
        config: Aggregation configuration

    Returns:
        Aggregated document output
    """
    aggregator = MentionAggregator(config=config)
    return aggregator.aggregate(doc_mentions)


def aggregate_to_benchmark_format(
    doc_mentions: DocumentMentions,
    dataset: str = "phenobert",
) -> list[tuple[str, str]]:
    """
    Aggregate to benchmark evaluation format.

    Args:
        doc_mentions: Document mentions
        dataset: Target dataset for assertion mapping

    Returns:
        List of (hpo_id, assertion) tuples
    """
    config = AggregationConfig(dataset_format=dataset)
    output = aggregate_mentions_to_document(doc_mentions, config)
    benchmark_format: list[tuple[str, str]] = output["benchmark_format"]
    return benchmark_format
