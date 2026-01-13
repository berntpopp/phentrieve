"""
Mention grouping for clinical phenomenon clustering.

This module groups mentions that refer to the same underlying
clinical phenomenon, then ranks alternative HPO explanations
for each group.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.assertion_representation import AssertionVector
from phentrieve.text_processing.mention import HPOCandidate, Mention, MentionGroup

logger = logging.getLogger(__name__)

# Default grouping configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_HPO_OVERLAP_THRESHOLD = 0.5
DEFAULT_SENTENCE_DISTANCE_THRESHOLD = 3


@dataclass
class MentionGroupingConfig:
    """
    Configuration for mention grouping.

    Attributes:
        similarity_threshold: Min text similarity for grouping (0-1)
        hpo_overlap_threshold: Min HPO candidate overlap for grouping (0-1)
        sentence_distance_threshold: Max sentence distance for grouping
        same_section_only: Only group mentions in same section
        use_text_similarity: Use text embedding similarity
        use_hpo_overlap: Use HPO candidate overlap
        min_group_size: Minimum mentions per group (smaller become singleton)
        top_n_alternatives: Number of alternative HPO explanations per group
    """

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    hpo_overlap_threshold: float = DEFAULT_HPO_OVERLAP_THRESHOLD
    sentence_distance_threshold: int = DEFAULT_SENTENCE_DISTANCE_THRESHOLD
    same_section_only: bool = True
    use_text_similarity: bool = True
    use_hpo_overlap: bool = True
    min_group_size: int = 1
    top_n_alternatives: int = 3


class MentionGrouper:
    """
    Group mentions referring to the same clinical phenomenon.

    Uses multiple signals (text similarity, HPO overlap, proximity)
    to cluster related mentions, then ranks alternative HPO
    explanations for each cluster.

    Example:
        >>> grouper = MentionGrouper(model=sbert_model)
        >>> mentions = [...]  # Mentions with hpo_candidates
        >>> groups = grouper.group(mentions)
        >>> for g in groups:
        ...     print(f"Group with {g.num_mentions} mentions")
    """

    def __init__(
        self,
        config: MentionGroupingConfig | None = None,
        model: SentenceTransformer | None = None,
    ):
        """
        Initialize the mention grouper.

        Args:
            config: Grouping configuration
            model: Optional embedding model for text similarity
        """
        self.config = config or MentionGroupingConfig()
        self.model = model

    def group(self, mentions: list[Mention]) -> list[MentionGroup]:
        """
        Group mentions into clusters.

        Args:
            mentions: List of mentions to group

        Returns:
            List of MentionGroup objects
        """
        if not mentions:
            return []

        logger.debug(f"Grouping {len(mentions)} mentions")

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(mentions)

        # Cluster using connected components with threshold
        clusters = self._cluster_mentions(mentions, similarity_matrix)

        # Create MentionGroup objects
        groups = self._create_groups(clusters, mentions)

        logger.debug(f"Created {len(groups)} mention groups")
        return groups

    def _compute_similarity_matrix(
        self,
        mentions: list[Mention],
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for mentions.

        Combines multiple signals: text similarity, HPO overlap, proximity.

        Args:
            mentions: List of mentions

        Returns:
            Similarity matrix (n x n)
        """
        n = len(mentions)
        similarity = np.zeros((n, n))

        # Compute text similarity if enabled
        text_sim = None
        if self.config.use_text_similarity:
            text_sim = self._compute_text_similarity(mentions)

        # Compute HPO overlap if enabled
        hpo_overlap = None
        if self.config.use_hpo_overlap:
            hpo_overlap = self._compute_hpo_overlap(mentions)

        # Combine signals
        for i in range(n):
            for j in range(i + 1, n):
                m1, m2 = mentions[i], mentions[j]

                # Check section constraint
                if self.config.same_section_only:
                    if m1.section_type != m2.section_type:
                        continue

                # Check sentence distance
                sentence_dist = abs(m1.sentence_idx - m2.sentence_idx)
                if sentence_dist > self.config.sentence_distance_threshold:
                    continue

                # Combine signals
                scores: list[float] = []

                if text_sim is not None:
                    scores.append(text_sim[i, j])

                if hpo_overlap is not None:
                    scores.append(hpo_overlap[i, j])

                # Proximity bonus
                proximity_score = 1.0 / (1.0 + sentence_dist * 0.2)
                scores.append(proximity_score)

                # Average of all signals
                if scores:
                    sim = sum(scores) / len(scores)
                    similarity[i, j] = sim
                    similarity[j, i] = sim

        return similarity

    def _compute_text_similarity(self, mentions: list[Mention]) -> np.ndarray:
        """Compute text embedding similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity

        n = len(mentions)
        embeddings: list[np.ndarray | None] = []

        # Collect embeddings, computing if needed
        texts_to_embed: list[str] = []
        indices: list[int] = []

        for i, mention in enumerate(mentions):
            if mention.embedding is not None:
                embeddings.append(mention.embedding)
            else:
                embeddings.append(None)
                texts_to_embed.append(mention.text)
                indices.append(i)

        # Compute missing embeddings
        if texts_to_embed and self.model is not None:
            new_embeddings = self.model.encode(texts_to_embed, show_progress_bar=False)
            for idx, i in enumerate(indices):
                embeddings[i] = new_embeddings[idx]
                mentions[i].embedding = new_embeddings[idx]

        # Build similarity matrix
        valid_embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)

        sim_matrix = np.zeros((n, n))

        if len(valid_embeddings) >= 2:
            emb_array = np.array(valid_embeddings)
            pairwise_sim = cosine_similarity(emb_array)

            for a in range(len(valid_indices)):
                for b in range(a + 1, len(valid_indices)):
                    i, j = valid_indices[a], valid_indices[b]
                    sim_matrix[i, j] = pairwise_sim[a, b]
                    sim_matrix[j, i] = pairwise_sim[a, b]

        return sim_matrix

    def _compute_hpo_overlap(self, mentions: list[Mention]) -> np.ndarray:
        """Compute HPO candidate overlap matrix."""
        n = len(mentions)
        overlap = np.zeros((n, n))

        # Get top HPO IDs for each mention
        mention_hpos: list[set[str]] = []
        for mention in mentions:
            top_hpos = {c.hpo_id for c in mention.hpo_candidates[:5]}
            mention_hpos.append(top_hpos)

        # Compute Jaccard similarity
        for i in range(n):
            for j in range(i + 1, n):
                hpos_i = mention_hpos[i]
                hpos_j = mention_hpos[j]

                if not hpos_i or not hpos_j:
                    continue

                intersection = len(hpos_i & hpos_j)
                union = len(hpos_i | hpos_j)

                if union > 0:
                    jaccard = intersection / union
                    overlap[i, j] = jaccard
                    overlap[j, i] = jaccard

        return overlap

    def _cluster_mentions(
        self,
        mentions: list[Mention],
        similarity: np.ndarray,
    ) -> list[list[int]]:
        """
        Cluster mentions using connected components.

        Args:
            mentions: List of mentions
            similarity: Similarity matrix

        Returns:
            List of clusters (each cluster is list of mention indices)
        """
        n = len(mentions)
        threshold = self.config.similarity_threshold

        # Build adjacency from similarity matrix
        adjacency: dict[int, set[int]] = defaultdict(set)

        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] >= threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Find connected components
        visited: set[int] = set()
        clusters: list[list[int]] = []

        for start in range(n):
            if start in visited:
                continue

            # BFS to find component
            component: list[int] = []
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if component:
                clusters.append(sorted(component))

        return clusters

    def _create_groups(
        self,
        clusters: list[list[int]],
        mentions: list[Mention],
    ) -> list[MentionGroup]:
        """
        Create MentionGroup objects from clusters.

        Args:
            clusters: List of mention index clusters
            mentions: Original mentions

        Returns:
            List of MentionGroup objects
        """
        groups: list[MentionGroup] = []

        for cluster_indices in clusters:
            cluster_mentions = [mentions[i] for i in cluster_indices]

            # Skip if below minimum size
            if len(cluster_mentions) < self.config.min_group_size:
                continue

            # Find representative mention (best top candidate score)
            representative = max(
                cluster_mentions,
                key=lambda m: m.top_candidate.effective_score if m.top_candidate else 0,
            )

            # Merge and rank HPO explanations
            ranked_hpos = self._merge_hpo_candidates(cluster_mentions)

            # Select final HPO
            final_hpo = ranked_hpos[0] if ranked_hpos else None

            # Aggregate assertion
            final_assertion = self._aggregate_assertions(cluster_mentions)

            group = MentionGroup(
                mentions=cluster_mentions,
                representative_mention=representative,
                ranked_hpo_explanations=ranked_hpos[: self.config.top_n_alternatives],
                final_hpo=final_hpo,
                final_assertion=final_assertion,
            )
            groups.append(group)

        return groups

    def _merge_hpo_candidates(
        self,
        mentions: list[Mention],
    ) -> list[HPOCandidate]:
        """
        Merge and rank HPO candidates from multiple mentions.

        Args:
            mentions: List of mentions

        Returns:
            Merged and sorted list of HPOCandidate
        """
        # Collect candidates by HPO ID
        hpo_candidates: dict[str, list[HPOCandidate]] = defaultdict(list)

        for mention in mentions:
            for candidate in mention.hpo_candidates:
                hpo_candidates[candidate.hpo_id].append(candidate)

        # Merge candidates for each HPO ID
        merged: list[HPOCandidate] = []

        for hpo_id, candidates in hpo_candidates.items():
            # Take the best version of each candidate
            best = max(candidates, key=lambda c: c.effective_score)

            # Boost score based on number of mentions supporting this HPO
            support_count = len(candidates)
            support_boost = min(0.1 * (support_count - 1), 0.3)  # Max 0.3 boost

            # Create merged candidate
            merged_candidate = HPOCandidate(
                hpo_id=hpo_id,
                label=best.label,
                score=best.score,
                refined_score=(best.refined_score or best.score) + support_boost,
                specificity_score=best.specificity_score,
                depth=best.depth,
                is_generic=best.is_generic,
                synonyms=best.synonyms,
                definition=best.definition,
                metadata={
                    **best.metadata,
                    "support_count": support_count,
                },
            )
            merged.append(merged_candidate)

        # Sort by effective score
        merged.sort(key=lambda c: c.effective_score, reverse=True)
        return merged

    def _aggregate_assertions(
        self,
        mentions: list[Mention],
    ) -> AssertionVector | None:
        """
        Aggregate assertions from multiple mentions.

        Uses weighted averaging based on confidence.

        Args:
            mentions: List of mentions

        Returns:
            Aggregated AssertionVector or None
        """
        assertions = [m.assertion for m in mentions if m.assertion is not None]

        if not assertions:
            return None

        if len(assertions) == 1:
            return assertions[0]

        # Weighted average based on evidence confidence
        total_weight = sum(a.evidence_confidence for a in assertions)
        if total_weight == 0:
            total_weight = len(assertions)
            weights = [1.0 / total_weight] * len(assertions)
        else:
            weights = [a.evidence_confidence / total_weight for a in assertions]

        avg_negation = sum(a.negation_score * w for a, w in zip(assertions, weights))
        avg_uncertainty = sum(
            a.uncertainty_score * w for a, w in zip(assertions, weights)
        )
        avg_normality = sum(a.normality_score * w for a, w in zip(assertions, weights))
        avg_confidence = sum(
            a.evidence_confidence * w for a, w in zip(assertions, weights)
        )

        # Boolean fields use OR
        any_historical = any(a.historical for a in assertions)
        any_hypothetical = any(a.hypothetical for a in assertions)
        any_family = any(a.family_history for a in assertions)

        from phentrieve.text_processing.assertion_representation import AssertionVector

        return AssertionVector(
            negation_score=avg_negation,
            uncertainty_score=avg_uncertainty,
            normality_score=avg_normality,
            historical=any_historical,
            hypothetical=any_hypothetical,
            family_history=any_family,
            evidence_source="aggregated",
            evidence_confidence=avg_confidence,
        )


def group_mentions(
    mentions: list[Mention],
    model: SentenceTransformer | None = None,
    config: MentionGroupingConfig | None = None,
) -> list[MentionGroup]:
    """
    Convenience function to group mentions.

    Args:
        mentions: List of mentions
        model: Optional embedding model
        config: Grouping configuration

    Returns:
        List of MentionGroup objects
    """
    grouper = MentionGrouper(config=config, model=model)
    return grouper.group(mentions)
