"""
Core mention-level data structures for HPO extraction.

This module provides dataclasses representing clinical finding mentions
in text with their associated HPO candidates, assertions, and groupings.
These structures support mention-level processing while maintaining
compatibility with document-level benchmark evaluation.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from phentrieve.text_processing.assertion_representation import AssertionVector


class CanonicalAssertion(Enum):
    """
    Canonical internal assertion labels.

    These are the internal labels used during mention-level processing.
    They are mapped to dataset-specific labels at output time.
    """

    AFFIRMED = "affirmed"  # Finding is present
    NEGATED = "negated"  # Finding is absent
    UNCERTAIN = "uncertain"  # Epistemic uncertainty
    NORMAL = "normal"  # Within normal limits
    HISTORICAL = "historical"  # Past finding
    FAMILY = "family"  # Family member finding


# Dataset-specific assertion mappings
DATASET_ASSERTION_MAPS: dict[str, dict[str, str]] = {
    "phenobert": {
        "affirmed": "PRESENT",
        "negated": "ABSENT",
        "uncertain": "UNCERTAIN",
        "normal": "PRESENT",  # Normal is still a present finding
        "historical": "PRESENT",
        "family": "PRESENT",  # Tracked separately in metadata
    },
    "gsc_plus": {
        "affirmed": "PRESENT",
        "negated": "ABSENT",
        "uncertain": "UNCERTAIN",
        "normal": "PRESENT",
        "historical": "PRESENT",
        "family": "PRESENT",
    },
    "id_68": {
        "affirmed": "PRESENT",
        "negated": "ABSENT",
        "uncertain": "UNCERTAIN",
        "normal": "PRESENT",
        "historical": "PRESENT",
        "family": "PRESENT",
    },
    "gene_reviews": {
        "affirmed": "PRESENT",
        "negated": "ABSENT",
        "uncertain": "UNCERTAIN",
        "normal": "PRESENT",
        "historical": "PRESENT",
        "family": "PRESENT",
    },
}


def map_assertion_to_dataset(
    canonical_assertion: str, dataset: str = "phenobert"
) -> str:
    """
    Map a canonical assertion label to a dataset-specific label.

    Args:
        canonical_assertion: Canonical assertion string (e.g., "affirmed")
        dataset: Target dataset name (e.g., "phenobert", "gsc_plus")

    Returns:
        Dataset-specific assertion label (e.g., "PRESENT")
    """
    mapping = DATASET_ASSERTION_MAPS.get(dataset, DATASET_ASSERTION_MAPS["phenobert"])
    return mapping.get(canonical_assertion.lower(), "PRESENT")


@dataclass
class HPOCandidate:
    """
    A candidate HPO term for a clinical finding mention.

    Represents a potential HPO term match with various scores used
    for ranking and selection during the extraction pipeline.

    Attributes:
        hpo_id: HPO identifier (e.g., "HP:0001250")
        label: Human-readable label (e.g., "Seizure")
        score: Initial retrieval similarity score (0.0 to 1.0)
        refined_score: Score after cross-encoder re-ranking (optional)
        specificity_score: Ontology depth-based specificity score (0.0 to 1.0)
        depth: Ontology depth (distance from root)
        is_generic: Flag indicating if this is a generic/root-level term
        synonyms: List of HPO term synonyms
        definition: HPO term definition (optional)
        metadata: Additional metadata for extensibility

    Example:
        >>> candidate = HPOCandidate(
        ...     hpo_id="HP:0001250",
        ...     label="Seizure",
        ...     score=0.85,
        ...     depth=4,
        ... )
        >>> candidate.is_generic
        False
    """

    hpo_id: str
    label: str
    score: float
    refined_score: float | None = None
    specificity_score: float = 0.5
    depth: int = 0
    is_generic: bool = False
    synonyms: list[str] = field(default_factory=list)
    definition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        # Mark as generic if depth is shallow (less than 3 levels from root)
        if self.depth < 3 and not self.metadata.get("is_generic_override"):
            self.is_generic = True

    @property
    def effective_score(self) -> float:
        """
        Get the effective score for ranking.

        Uses refined_score if available, otherwise falls back to score.

        Returns:
            The score to use for ranking candidates
        """
        return self.refined_score if self.refined_score is not None else self.score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hpo_id": self.hpo_id,
            "label": self.label,
            "score": self.score,
            "refined_score": self.refined_score,
            "specificity_score": self.specificity_score,
            "depth": self.depth,
            "is_generic": self.is_generic,
            "synonyms": self.synonyms,
            "definition": self.definition,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HPOCandidate:
        """Create HPOCandidate from dictionary."""
        return cls(
            hpo_id=data["hpo_id"],
            label=data["label"],
            score=data.get("score", 0.0),
            refined_score=data.get("refined_score"),
            specificity_score=data.get("specificity_score", 0.5),
            depth=data.get("depth", 0),
            is_generic=data.get("is_generic", False),
            synonyms=data.get("synonyms", []),
            definition=data.get("definition"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Mention:
    """
    A clinical finding mention in text with span and semantic information.

    Represents a text span that potentially describes a clinical finding,
    with associated assertion status and HPO term candidates. This is the
    core unit of processing in the mention-level extraction pipeline.

    Attributes:
        mention_id: Unique identifier for this mention
        text: Surface text of the mention
        start_char: Start character position in document
        end_char: End character position in document
        sentence_idx: Index of the containing sentence
        section_type: Section context (e.g., "family_history", "current_findings")
        embedding: Computed embedding vector (populated during processing)
        assertion: Multi-dimensional assertion vector
        hpo_candidates: List of ranked HPO term candidates
        context_window: Surrounding text for context (optional)
        metadata: Additional metadata for extensibility

    Example:
        >>> from phentrieve.text_processing.assertion_representation import AssertionVector
        >>> mention = Mention(
        ...     text="seizures",
        ...     start_char=45,
        ...     end_char=53,
        ...     sentence_idx=2,
        ... )
        >>> mention.mention_id  # Auto-generated UUID
        'a1b2c3d4-...'
    """

    text: str
    start_char: int
    end_char: int
    sentence_idx: int = 0
    mention_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    section_type: str | None = None
    embedding: np.ndarray | None = None
    assertion: AssertionVector | None = None
    hpo_candidates: list[HPOCandidate] = field(default_factory=list)
    context_window: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by mention_id for use in sets/dicts."""
        return hash(self.mention_id)

    def __eq__(self, other: object) -> bool:
        """Equality by mention_id."""
        if not isinstance(other, Mention):
            return False
        return self.mention_id == other.mention_id

    @property
    def span_length(self) -> int:
        """Length of the mention span in characters."""
        return self.end_char - self.start_char

    @property
    def top_candidate(self) -> HPOCandidate | None:
        """
        Get the top-ranked HPO candidate.

        Returns:
            The highest-scoring HPO candidate, or None if no candidates
        """
        if not self.hpo_candidates:
            return None
        return max(self.hpo_candidates, key=lambda c: c.effective_score)

    @property
    def top_candidates(self) -> list[HPOCandidate]:
        """
        Get candidates sorted by effective score (descending).

        Returns:
            List of candidates sorted by score
        """
        return sorted(
            self.hpo_candidates, key=lambda c: c.effective_score, reverse=True
        )

    def get_canonical_assertion(self) -> str:
        """
        Get the canonical assertion label.

        Converts the AssertionVector to a canonical string label.

        Returns:
            Canonical assertion string (e.g., "affirmed", "negated")
        """
        if self.assertion is None:
            return "affirmed"

        status = self.assertion.to_status()
        return status.value

    def is_in_family_history(self) -> bool:
        """Check if this mention is in a family history context."""
        if self.assertion is not None and self.assertion.family_history:
            return True
        return self.section_type == "family_history"

    def overlaps_with(self, other: Mention) -> bool:
        """
        Check if this mention overlaps with another.

        Args:
            other: Another Mention to check

        Returns:
            True if the spans overlap
        """
        return not (
            self.end_char <= other.start_char or self.start_char >= other.end_char
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "mention_id": self.mention_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "sentence_idx": self.sentence_idx,
            "section_type": self.section_type,
            "hpo_candidates": [c.to_dict() for c in self.hpo_candidates],
            "context_window": self.context_window,
            "metadata": self.metadata,
        }
        if self.assertion is not None:
            result["assertion"] = self.assertion.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mention:
        """Create Mention from dictionary."""
        from phentrieve.text_processing.assertion_representation import AssertionVector

        assertion = None
        if "assertion" in data:
            assertion = AssertionVector.from_dict(data["assertion"])

        return cls(
            mention_id=data.get("mention_id", str(uuid.uuid4())),
            text=data["text"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            sentence_idx=data.get("sentence_idx", 0),
            section_type=data.get("section_type"),
            assertion=assertion,
            hpo_candidates=[
                HPOCandidate.from_dict(c) for c in data.get("hpo_candidates", [])
            ],
            context_window=data.get("context_window"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MentionGroup:
    """
    Group of mentions referring to the same clinical phenomenon.

    Represents a cluster of mentions that are determined to describe
    the same underlying clinical finding, with ranked alternative
    HPO explanations for the group.

    Attributes:
        group_id: Unique identifier for this group
        mentions: List of mentions in this group
        representative_mention: Best exemplar mention for the group
        ranked_hpo_explanations: Merged and ranked HPO candidates
        final_hpo: Selected HPO term for output
        final_assertion: Aggregated assertion for the group
        confidence: Confidence in the grouping (0.0 to 1.0)
        metadata: Additional metadata for extensibility

    Example:
        >>> group = MentionGroup(
        ...     mentions=[mention1, mention2],
        ...     representative_mention=mention1,
        ... )
        >>> group.num_mentions
        2
    """

    mentions: list[Mention]
    representative_mention: Mention | None = None
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ranked_hpo_explanations: list[HPOCandidate] = field(default_factory=list)
    final_hpo: HPOCandidate | None = None
    final_assertion: AssertionVector | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set representative mention if not provided."""
        if self.representative_mention is None and self.mentions:
            # Use mention with highest top candidate score
            scored_mentions = [
                (m, m.top_candidate.effective_score if m.top_candidate else 0.0)
                for m in self.mentions
            ]
            if scored_mentions:
                self.representative_mention = max(scored_mentions, key=lambda x: x[1])[
                    0
                ]

    @property
    def num_mentions(self) -> int:
        """Number of mentions in this group."""
        return len(self.mentions)

    @property
    def all_hpo_ids(self) -> set[str]:
        """Get all unique HPO IDs from all mentions in the group."""
        ids: set[str] = set()
        for mention in self.mentions:
            for candidate in mention.hpo_candidates:
                ids.add(candidate.hpo_id)
        return ids

    @property
    def span_range(self) -> tuple[int, int]:
        """Get the character span range covering all mentions."""
        if not self.mentions:
            return (0, 0)
        start = min(m.start_char for m in self.mentions)
        end = max(m.end_char for m in self.mentions)
        return (start, end)

    def get_canonical_assertion(self) -> str:
        """
        Get the canonical assertion label for the group.

        Uses the final_assertion if available, otherwise aggregates
        from individual mentions.

        Returns:
            Canonical assertion string
        """
        if self.final_assertion is not None:
            return self.final_assertion.to_status().value

        if self.representative_mention is not None:
            return self.representative_mention.get_canonical_assertion()

        return "affirmed"

    def get_dataset_assertion(self, dataset: str = "phenobert") -> str:
        """
        Get the assertion label mapped to a specific dataset format.

        Args:
            dataset: Target dataset name

        Returns:
            Dataset-specific assertion label
        """
        canonical = self.get_canonical_assertion()
        return map_assertion_to_dataset(canonical, dataset)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "group_id": self.group_id,
            "mentions": [m.to_dict() for m in self.mentions],
            "ranked_hpo_explanations": [
                c.to_dict() for c in self.ranked_hpo_explanations
            ],
            "confidence": self.confidence,
            "metadata": self.metadata,
            "span_range": self.span_range,
        }
        if self.representative_mention is not None:
            result["representative_mention_id"] = self.representative_mention.mention_id
        if self.final_hpo is not None:
            result["final_hpo"] = self.final_hpo.to_dict()
        if self.final_assertion is not None:
            result["final_assertion"] = self.final_assertion.to_dict()
        return result


@dataclass
class DocumentMentions:
    """
    Container for all mentions extracted from a document.

    Provides document-level organization of mentions and groups,
    with methods for aggregation to benchmark-compatible output.

    Attributes:
        doc_id: Document identifier
        full_text: Original document text
        mentions: All extracted mentions
        groups: Mention groups (populated after grouping stage)
        sentences: List of sentence boundaries (start, end tuples)
        sections: List of section boundaries with types
        metadata: Document-level metadata

    Example:
        >>> doc_mentions = DocumentMentions(
        ...     doc_id="case_001",
        ...     full_text="Patient presents with seizures...",
        ... )
        >>> doc_mentions.add_mention(mention)
    """

    doc_id: str
    full_text: str
    mentions: list[Mention] = field(default_factory=list)
    groups: list[MentionGroup] = field(default_factory=list)
    sentences: list[tuple[int, int]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_mentions(self) -> int:
        """Total number of mentions."""
        return len(self.mentions)

    @property
    def num_groups(self) -> int:
        """Number of mention groups."""
        return len(self.groups)

    def add_mention(self, mention: Mention) -> None:
        """Add a mention to the document."""
        self.mentions.append(mention)

    def add_group(self, group: MentionGroup) -> None:
        """Add a mention group to the document."""
        self.groups.append(group)

    def get_mentions_in_section(self, section_type: str) -> list[Mention]:
        """Get all mentions in a specific section type."""
        return [m for m in self.mentions if m.section_type == section_type]

    def get_mentions_in_sentence(self, sentence_idx: int) -> list[Mention]:
        """Get all mentions in a specific sentence."""
        return [m for m in self.mentions if m.sentence_idx == sentence_idx]

    def to_document_level_output(
        self,
        dataset: str = "phenobert",
        include_details: bool = False,
    ) -> dict[str, Any]:
        """
        Aggregate to document-level output for benchmark evaluation.

        Args:
            dataset: Target dataset for assertion mapping
            include_details: Whether to include mention-level details

        Returns:
            Dictionary with document-level HPO terms and assertions
        """
        # Collect final HPO terms from groups
        hpo_terms: list[dict[str, Any]] = []
        seen_hpo_ids: set[str] = set()

        for group in self.groups:
            if group.final_hpo is None:
                continue

            hpo_id = group.final_hpo.hpo_id
            if hpo_id in seen_hpo_ids:
                continue
            seen_hpo_ids.add(hpo_id)

            term_output = {
                "id": hpo_id,
                "name": group.final_hpo.label,
                "assertion": group.get_dataset_assertion(dataset),
                "score": group.final_hpo.effective_score,
                "count": group.num_mentions,
            }

            if include_details:
                term_output["group_id"] = group.group_id
                term_output["mention_ids"] = [m.mention_id for m in group.mentions]
                term_output["alternatives"] = [
                    {"hpo_id": c.hpo_id, "label": c.label, "score": c.effective_score}
                    for c in group.ranked_hpo_explanations[:3]
                ]

            hpo_terms.append(term_output)

        # Sort by score
        hpo_terms.sort(key=lambda x: x["score"], reverse=True)

        result = {
            "doc_id": self.doc_id,
            "hpo_terms": hpo_terms,
            "num_terms": len(hpo_terms),
        }

        if include_details:
            result["num_mentions"] = self.num_mentions
            result["num_groups"] = self.num_groups
            result["mentions"] = [m.to_dict() for m in self.mentions]
            result["groups"] = [g.to_dict() for g in self.groups]

        return result

    def to_benchmark_format(self, dataset: str = "phenobert") -> list[tuple[str, str]]:
        """
        Convert to benchmark evaluation format (list of (hpo_id, assertion) tuples).

        This format is compatible with the existing benchmark infrastructure.

        Args:
            dataset: Target dataset for assertion mapping

        Returns:
            List of (hpo_id, assertion) tuples
        """
        result: list[tuple[str, str]] = []
        seen_hpo_ids: set[str] = set()

        for group in self.groups:
            if group.final_hpo is None:
                continue

            hpo_id = group.final_hpo.hpo_id
            if hpo_id in seen_hpo_ids:
                continue
            seen_hpo_ids.add(hpo_id)

            assertion = group.get_dataset_assertion(dataset)
            result.append((hpo_id, assertion))

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "full_text": self.full_text,
            "mentions": [m.to_dict() for m in self.mentions],
            "groups": [g.to_dict() for g in self.groups],
            "sentences": self.sentences,
            "sections": self.sections,
            "metadata": self.metadata,
        }
