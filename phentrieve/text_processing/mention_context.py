"""
Controlled contextual influence for mentions.

This module provides cross-mention context propagation with gating
based on proximity and document region. It integrates with the
existing SemanticDocumentGraph for optional graph-based refinement.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.assertion_representation import AssertionVector
from phentrieve.text_processing.mention import Mention

logger = logging.getLogger(__name__)

# Default context configuration
DEFAULT_CONTEXT_RADIUS = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_MAX_SENTENCE_DISTANCE = 3


@dataclass
class ContextPropagationConfig:
    """
    Configuration for context propagation.

    Attributes:
        context_radius: Graph radius for context propagation
        similarity_threshold: Min similarity for context influence
        max_sentence_distance: Max sentence distance for influence
        same_section_only: Only propagate within same section
        propagation_weight: Weight for propagated information (0-1)
        enable_assertion_propagation: Propagate assertion information
        enable_candidate_influence: Let context influence candidates
    """

    context_radius: int = DEFAULT_CONTEXT_RADIUS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_sentence_distance: int = DEFAULT_MAX_SENTENCE_DISTANCE
    same_section_only: bool = True
    propagation_weight: float = 0.3
    enable_assertion_propagation: bool = True
    enable_candidate_influence: bool = True


@dataclass
class MentionGraphNode:
    """
    Node in the mention graph.

    Attributes:
        mention: The mention
        neighbors: Indices of neighboring mentions
        weights: Edge weights to neighbors
        propagated_assertion: Assertion after propagation
    """

    mention: Mention
    neighbors: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    propagated_assertion: AssertionVector | None = None


class MentionContextGraph:
    """
    Graph structure for mention context propagation.

    Builds a graph of mentions connected by proximity and similarity,
    then propagates context information between connected mentions.

    Example:
        >>> graph = MentionContextGraph(config=ContextPropagationConfig())
        >>> graph.build_from_mentions(mentions, model=sbert_model)
        >>> graph.propagate_context()
    """

    def __init__(
        self,
        config: ContextPropagationConfig | None = None,
    ):
        """
        Initialize the mention context graph.

        Args:
            config: Context propagation configuration
        """
        self.config = config or ContextPropagationConfig()
        self._nodes: list[MentionGraphNode] = []
        self._adjacency: dict[int, list[tuple[int, float]]] = {}

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Total number of edges in the graph."""
        return sum(len(edges) for edges in self._adjacency.values())

    def build_from_mentions(
        self,
        mentions: list[Mention],
        model: SentenceTransformer | None = None,
    ) -> None:
        """
        Build the graph from mentions.

        Args:
            mentions: List of mentions
            model: Optional embedding model for similarity edges
        """
        if not mentions:
            return

        logger.debug(f"Building mention context graph from {len(mentions)} mentions")

        # Clear existing graph
        self._nodes = []
        self._adjacency = {}

        # Create nodes
        for mention in mentions:
            node = MentionGraphNode(mention=mention)
            self._nodes.append(node)
            self._adjacency[len(self._nodes) - 1] = []

        # Compute embeddings if needed and model provided
        if model is not None:
            self._compute_embeddings(mentions, model)

        # Build edges
        self._build_proximity_edges(mentions)

        if model is not None:
            self._build_similarity_edges(mentions)

        logger.debug(
            f"Mention context graph: {self.num_nodes} nodes, {self.num_edges} edges"
        )

    def _compute_embeddings(
        self,
        mentions: list[Mention],
        model: SentenceTransformer,
    ) -> None:
        """Compute embeddings for mentions that don't have them."""
        texts_to_embed: list[str] = []
        indices: list[int] = []

        for i, mention in enumerate(mentions):
            if mention.embedding is None:
                texts_to_embed.append(mention.text)
                indices.append(i)

        if texts_to_embed:
            embeddings = model.encode(texts_to_embed, show_progress_bar=False)
            for idx, i in enumerate(indices):
                mentions[i].embedding = embeddings[idx]

    def _build_proximity_edges(self, mentions: list[Mention]) -> None:
        """Build edges based on sentence/position proximity."""
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                m1, m2 = mentions[i], mentions[j]

                # Check section constraint
                if self.config.same_section_only:
                    if m1.section_type != m2.section_type:
                        continue

                # Check sentence distance
                sentence_dist = abs(m1.sentence_idx - m2.sentence_idx)
                if sentence_dist > self.config.max_sentence_distance:
                    continue

                # Calculate proximity weight
                weight = 1.0 / (1.0 + sentence_dist)

                # Add bidirectional edges
                self._add_edge(i, j, weight)
                self._add_edge(j, i, weight)

    def _build_similarity_edges(self, mentions: list[Mention]) -> None:
        """Build edges based on embedding similarity."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Get embeddings
        embeddings = []
        valid_indices = []
        for i, mention in enumerate(mentions):
            if mention.embedding is not None:
                embeddings.append(mention.embedding)
                valid_indices.append(i)

        if len(embeddings) < 2:
            return

        embeddings_array = np.array(embeddings)
        sim_matrix = cosine_similarity(embeddings_array)

        # Add edges for similar mentions
        for a in range(len(valid_indices)):
            for b in range(a + 1, len(valid_indices)):
                i, j = valid_indices[a], valid_indices[b]
                similarity = sim_matrix[a, b]

                if similarity >= self.config.similarity_threshold:
                    m1, m2 = mentions[i], mentions[j]

                    # Check section constraint
                    if self.config.same_section_only:
                        if m1.section_type != m2.section_type:
                            continue

                    # Add bidirectional edges
                    self._add_edge(i, j, similarity)
                    self._add_edge(j, i, similarity)

    def _add_edge(self, source: int, target: int, weight: float) -> None:
        """Add an edge to the graph."""
        # Check if edge already exists
        existing = self._adjacency.get(source, [])
        for idx, (t, w) in enumerate(existing):
            if t == target:
                # Update weight if new weight is higher
                if weight > w:
                    existing[idx] = (target, weight)
                return

        # Add new edge
        if source not in self._adjacency:
            self._adjacency[source] = []
        self._adjacency[source].append((target, weight))

        # Update node neighbors
        if source < len(self._nodes):
            self._nodes[source].neighbors.append(target)
            self._nodes[source].weights.append(weight)

    def propagate_context(self) -> None:
        """
        Propagate context information between connected mentions.

        Updates assertion information based on neighboring mentions.
        """
        if not self._nodes:
            return

        logger.debug("Propagating context between mentions")

        if self.config.enable_assertion_propagation:
            self._propagate_assertions()

        if self.config.enable_candidate_influence:
            self._propagate_candidate_influence()

    def _propagate_assertions(self) -> None:
        """Propagate assertion information from neighbors."""
        for node in self._nodes:
            if not node.neighbors:
                continue

            mention = node.mention
            if mention.assertion is None:
                continue

            # Collect weighted neighbor assertions
            neighbor_assertions: list[tuple[AssertionVector, float]] = []

            for neighbor_idx, weight in zip(node.neighbors, node.weights):
                if neighbor_idx >= len(self._nodes):
                    continue

                neighbor_mention = self._nodes[neighbor_idx].mention
                if neighbor_mention.assertion is not None:
                    neighbor_assertions.append((neighbor_mention.assertion, weight))

            if not neighbor_assertions:
                continue

            # Combine assertions with propagation weight
            combined = self._combine_assertions(
                mention.assertion,
                neighbor_assertions,
                self.config.propagation_weight,
            )

            node.propagated_assertion = combined

    def _propagate_candidate_influence(self) -> None:
        """Let context influence candidate scoring."""
        for node in self._nodes:
            if not node.neighbors or not node.mention.hpo_candidates:
                continue

            mention = node.mention

            # Collect HPO candidates from neighbors
            neighbor_hpo_ids: dict[str, float] = {}

            for neighbor_idx, weight in zip(node.neighbors, node.weights):
                if neighbor_idx >= len(self._nodes):
                    continue

                neighbor_mention = self._nodes[neighbor_idx].mention
                for candidate in neighbor_mention.hpo_candidates:
                    hpo_id = candidate.hpo_id
                    score = candidate.effective_score * weight

                    if hpo_id in neighbor_hpo_ids:
                        neighbor_hpo_ids[hpo_id] = max(neighbor_hpo_ids[hpo_id], score)
                    else:
                        neighbor_hpo_ids[hpo_id] = score

            # Boost candidates that appear in neighbors
            for candidate in mention.hpo_candidates:
                if candidate.hpo_id in neighbor_hpo_ids:
                    neighbor_support = neighbor_hpo_ids[candidate.hpo_id]
                    boost = neighbor_support * self.config.propagation_weight
                    current = candidate.refined_score or candidate.score
                    candidate.refined_score = min(1.0, current + boost * 0.1)

    def _combine_assertions(
        self,
        base: AssertionVector,
        neighbors: list[tuple[AssertionVector, float]],
        propagation_weight: float,
    ) -> AssertionVector:
        """
        Combine base assertion with neighbor assertions.

        Args:
            base: Base assertion vector
            neighbors: List of (assertion, weight) tuples
            propagation_weight: Overall weight for propagation

        Returns:
            Combined AssertionVector
        """
        if not neighbors:
            return base

        # Calculate weighted average of neighbor assertions
        total_weight = sum(w for _, w in neighbors)
        if total_weight == 0:
            return base

        avg_negation = sum(a.negation_score * w for a, w in neighbors) / total_weight
        avg_uncertainty = (
            sum(a.uncertainty_score * w for a, w in neighbors) / total_weight
        )
        avg_normality = sum(a.normality_score * w for a, w in neighbors) / total_weight

        # Combine with base using propagation weight
        base_weight = 1.0 - propagation_weight

        combined_negation = (
            base.negation_score * base_weight + avg_negation * propagation_weight
        )
        combined_uncertainty = (
            base.uncertainty_score * base_weight + avg_uncertainty * propagation_weight
        )
        combined_normality = (
            base.normality_score * base_weight + avg_normality * propagation_weight
        )

        # Family history propagates with OR logic
        any_family = base.family_history or any(a.family_history for a, _ in neighbors)

        return base.with_updates(
            negation_score=combined_negation,
            uncertainty_score=combined_uncertainty,
            normality_score=combined_normality,
            family_history=any_family,
            evidence_source="propagated",
        )

    def get_neighbors(self, mention_idx: int) -> list[Mention]:
        """Get neighbor mentions for a mention."""
        if mention_idx >= len(self._nodes):
            return []

        node = self._nodes[mention_idx]
        return [
            self._nodes[n_idx].mention
            for n_idx in node.neighbors
            if n_idx < len(self._nodes)
        ]

    def get_propagated_assertion(self, mention_idx: int) -> AssertionVector | None:
        """Get the propagated assertion for a mention."""
        if mention_idx >= len(self._nodes):
            return None
        return self._nodes[mention_idx].propagated_assertion


def propagate_mention_context(
    mentions: list[Mention],
    model: SentenceTransformer | None = None,
    config: ContextPropagationConfig | None = None,
) -> MentionContextGraph:
    """
    Convenience function to propagate context between mentions.

    Args:
        mentions: List of mentions
        model: Optional embedding model
        config: Propagation configuration

    Returns:
        Built and propagated MentionContextGraph
    """
    graph = MentionContextGraph(config=config)
    graph.build_from_mentions(mentions, model=model)
    graph.propagate_context()
    return graph
