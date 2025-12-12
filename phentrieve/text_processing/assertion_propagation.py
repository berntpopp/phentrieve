"""
Graph-based assertion propagation for document-level consistency.

This module implements message-passing algorithms to propagate assertion
evidence through the semantic document graph. This enables:

1. Document-level assertion consistency
2. Conflict detection and resolution
3. Evidence strengthening from multiple sources

The AssertionPropagator is an ADDITIVE extension that works alongside
the existing chunk-local assertion detection, enabling comparison between
methods.

See: plan/00-planning/GRAPH-BASED-EXTENSION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phentrieve.text_processing.assertion_representation import AssertionVector
    from phentrieve.text_processing.semantic_graph import SemanticDocumentGraph

logger = logging.getLogger(__name__)


@dataclass
class PropagationConfig:
    """
    Configuration for assertion propagation.

    Attributes:
        max_iterations: Maximum number of propagation iterations
        damping_factor: Weight for local vs. propagated evidence (like PageRank)
        convergence_threshold: Stop when average change falls below this
        edge_type_weights: Propagation weight for each edge type
        preserve_strong_evidence: Don't override high-confidence local evidence
        strong_evidence_threshold: Threshold for "strong" evidence
    """

    max_iterations: int = 3
    damping_factor: float = 0.85  # Higher = more weight on local evidence
    convergence_threshold: float = 0.01

    # Edge type weights for propagation
    edge_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sequential": 0.9,  # Strong local context
            "semantic": 0.7,  # Similar content
            "hpo_coreference": 1.0,  # Same HPO term - full propagation
        }
    )

    # Preserve strong local evidence
    preserve_strong_evidence: bool = True
    strong_evidence_threshold: float = 0.8


@dataclass
class PropagationResult:
    """
    Result of assertion propagation.

    Attributes:
        assertions: Updated assertion arrays per chunk_idx
        iterations_run: Number of iterations before convergence
        converged: Whether propagation converged
        conflict_chunks: Chunk pairs with conflicting assertions
    """

    assertions: dict[int, np.ndarray]
    iterations_run: int
    converged: bool
    conflict_chunks: list[tuple[int, int]] = field(default_factory=list)


class AssertionPropagator:
    """
    Propagate assertion evidence through semantic document graph.

    Uses a message-passing algorithm:
    1. Initialize node assertions from local detection
    2. Iteratively propagate weighted assertions through edges
    3. Apply conflict resolution rules
    4. Converge to document-consistent assertions

    Example:
        >>> from phentrieve.text_processing.semantic_graph import SemanticDocumentGraph
        >>> graph = SemanticDocumentGraph()
        >>> # ... build graph with chunks ...
        >>> propagator = AssertionPropagator()
        >>> result = propagator.propagate(graph)
        >>> # result.assertions contains updated assertion vectors
    """

    def __init__(self, config: PropagationConfig | None = None):
        """
        Initialize the assertion propagator.

        Args:
            config: Propagation configuration. Defaults to PropagationConfig().
        """
        self.config = config or PropagationConfig()

    def propagate(self, graph: SemanticDocumentGraph) -> PropagationResult:
        """
        Run assertion propagation on the graph.

        Args:
            graph: SemanticDocumentGraph with nodes and edges

        Returns:
            PropagationResult with updated assertions and metadata
        """
        if graph.num_nodes == 0:
            return PropagationResult(
                assertions={},
                iterations_run=0,
                converged=True,
            )

        # Initialize from local assertions
        assertions = self._initialize_assertions(graph)
        original_assertions = {k: v.copy() for k, v in assertions.items()}

        # Iterative propagation
        converged = False
        iterations_run = 0

        for iteration in range(self.config.max_iterations):
            iterations_run = iteration + 1
            new_assertions = self._propagation_step(graph, assertions)

            # Preserve strong local evidence if configured
            if self.config.preserve_strong_evidence:
                new_assertions = self._preserve_strong_evidence(
                    original_assertions, new_assertions
                )

            # Check convergence
            if self._has_converged(assertions, new_assertions):
                converged = True
                assertions = new_assertions
                break

            assertions = new_assertions

        # Detect conflicts
        conflict_chunks = self._detect_conflicts(graph, assertions)

        logger.debug(
            "Propagation completed: %d iterations, converged=%s, %d conflicts",
            iterations_run,
            converged,
            len(conflict_chunks),
        )

        return PropagationResult(
            assertions=assertions,
            iterations_run=iterations_run,
            converged=converged,
            conflict_chunks=conflict_chunks,
        )

    def _initialize_assertions(
        self,
        graph: SemanticDocumentGraph,
    ) -> dict[int, np.ndarray]:
        """
        Extract initial assertion vectors from graph nodes.

        Each assertion is represented as a 3D array:
        [negation_score, uncertainty_score, normality_score]
        """
        assertions = {}

        for node in graph.get_nodes():
            if node.assertion_vector is not None:
                assertions[node.chunk_idx] = self._vector_to_array(
                    node.assertion_vector
                )
            else:
                # Default: neutral assertion (all zeros = affirmed)
                assertions[node.chunk_idx] = np.zeros(3)

        return assertions

    def _propagation_step(
        self,
        graph: SemanticDocumentGraph,
        current_assertions: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Single propagation iteration using message passing."""
        new_assertions = {}

        for node in graph.get_nodes():
            node_idx = node.chunk_idx
            local_assertion = current_assertions.get(node_idx, np.zeros(3))

            # Collect messages from predecessors (incoming edges)
            messages = []
            incoming_edges = graph.get_incoming_edges(node_idx)

            for edge in incoming_edges:
                neighbor_idx = edge.source_idx
                neighbor_assertion = current_assertions.get(
                    neighbor_idx, np.zeros(3)
                )

                # Weight by edge type and edge weight
                edge_type_weight = self.config.edge_type_weights.get(
                    edge.edge_type, 0.5
                )
                combined_weight = edge.weight * edge_type_weight

                message = neighbor_assertion * combined_weight
                messages.append(message)

            # Combine local and propagated evidence
            if messages:
                # Average of incoming messages
                propagated = np.mean(messages, axis=0)

                # Damped update: blend local and propagated
                new_assertion = (
                    self.config.damping_factor * local_assertion
                    + (1 - self.config.damping_factor) * propagated
                )
            else:
                new_assertion = local_assertion

            # Normalize to [0, 1]
            new_assertions[node_idx] = np.clip(new_assertion, 0, 1)

        return new_assertions

    def _preserve_strong_evidence(
        self,
        original: dict[int, np.ndarray],
        updated: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """
        Preserve strong local evidence from being overwritten.

        If any dimension of the original assertion exceeds the threshold,
        preserve that dimension's value.
        """
        preserved = {}
        threshold = self.config.strong_evidence_threshold

        for node_idx, updated_arr in updated.items():
            original_arr = original.get(node_idx, np.zeros(3))
            result = updated_arr.copy()

            # For each dimension, if original was strong, preserve it
            for dim in range(3):
                if original_arr[dim] >= threshold:
                    result[dim] = original_arr[dim]

            preserved[node_idx] = result

        return preserved

    def _has_converged(
        self,
        old: dict[int, np.ndarray],
        new: dict[int, np.ndarray],
    ) -> bool:
        """Check if assertions have converged."""
        if not old:
            return True

        total_diff = 0.0
        for node_idx in old:
            old_arr = old[node_idx]
            new_arr = new.get(node_idx, old_arr)
            diff = np.linalg.norm(old_arr - new_arr)
            total_diff += diff

        avg_diff = total_diff / len(old)
        return avg_diff < self.config.convergence_threshold

    def _detect_conflicts(
        self,
        graph: SemanticDocumentGraph,
        assertions: dict[int, np.ndarray],
    ) -> list[tuple[int, int]]:
        """
        Detect pairs of chunks with conflicting assertions for same HPO.

        A conflict is when one chunk has high negation and another
        has low negation (affirmed) for the same HPO term.
        """
        conflicts = []

        # Group chunks by HPO
        hpo_to_chunks: dict[str, list[int]] = {}
        for node in graph.get_nodes():
            for hpo_id in node.hpo_matches:
                if hpo_id not in hpo_to_chunks:
                    hpo_to_chunks[hpo_id] = []
                hpo_to_chunks[hpo_id].append(node.chunk_idx)

        # Check for conflicts within each HPO group
        for hpo_id, chunk_indices in hpo_to_chunks.items():
            if len(chunk_indices) < 2:
                continue

            for i in range(len(chunk_indices)):
                for j in range(i + 1, len(chunk_indices)):
                    idx_i, idx_j = chunk_indices[i], chunk_indices[j]
                    arr_i = assertions.get(idx_i, np.zeros(3))
                    arr_j = assertions.get(idx_j, np.zeros(3))

                    # Conflict: one negated (>0.5) and one affirmed (<0.3)
                    neg_i, neg_j = arr_i[0], arr_j[0]
                    if (neg_i > 0.5 and neg_j < 0.3) or (neg_j > 0.5 and neg_i < 0.3):
                        conflicts.append((idx_i, idx_j))

        return conflicts

    def _vector_to_array(self, vec: AssertionVector) -> np.ndarray:
        """Convert AssertionVector to numpy array."""
        return np.array(
            [
                vec.negation_score,
                vec.uncertainty_score,
                vec.normality_score,
            ]
        )

    def array_to_vector(
        self,
        arr: np.ndarray,
        source: str = "propagated",
    ) -> AssertionVector:
        """
        Convert numpy array back to AssertionVector.

        Args:
            arr: 3D array [negation, uncertainty, normality]
            source: Evidence source string

        Returns:
            AssertionVector with propagated values
        """
        from phentrieve.text_processing.assertion_representation import AssertionVector

        # Compute confidence based on certainty (inverse of uncertainty)
        uncertainty = float(arr[1])
        confidence = 1.0 / (1.0 + uncertainty)

        return AssertionVector(
            negation_score=float(arr[0]),
            uncertainty_score=uncertainty,
            normality_score=float(arr[2]),
            evidence_source=source,
            evidence_confidence=confidence,
        )

    def resolve_hpo_assertion(
        self,
        graph: SemanticDocumentGraph,
        hpo_id: str,
        propagated_assertions: dict[int, np.ndarray],
    ) -> AssertionVector:
        """
        Resolve conflicting assertions for a specific HPO term.

        Collects assertions from all chunks mentioning this HPO,
        weights by graph connectivity and evidence strength,
        and applies clinical resolution rules.

        Args:
            graph: The semantic document graph
            hpo_id: HPO term to resolve
            propagated_assertions: Propagated assertion arrays

        Returns:
            Resolved AssertionVector for this HPO term
        """
        from phentrieve.text_processing.assertion_representation import (
            AssertionVector,
        )

        # Find chunks mentioning this HPO
        relevant_chunks = []
        for node in graph.get_nodes():
            if hpo_id in node.hpo_matches:
                relevant_chunks.append(node.chunk_idx)

        if not relevant_chunks:
            return AssertionVector(evidence_source="no_evidence")

        # Collect assertions with weights
        weighted_assertions: list[tuple[np.ndarray, float]] = []
        for chunk_idx in relevant_chunks:
            assertion_arr = propagated_assertions.get(chunk_idx)
            if assertion_arr is not None:
                # Weight by graph connectivity (degree centrality proxy)
                out_edges = len(graph.get_edges(chunk_idx))
                in_edges = len(graph.get_incoming_edges(chunk_idx))
                degree = out_edges + in_edges
                weight = 1.0 + (degree / 10.0)  # Scale factor
                weighted_assertions.append((assertion_arr, weight))

        if not weighted_assertions:
            return AssertionVector(evidence_source="no_assertion")

        # Weighted combination with clinical rules
        return self._apply_clinical_resolution_rules(weighted_assertions)

    def _apply_clinical_resolution_rules(
        self,
        weighted_assertions: list[tuple[np.ndarray, float]],
    ) -> AssertionVector:
        """
        Apply clinical rules for assertion resolution.

        Rules:
        1. Negation dominance: Explicit negation often overrides affirmation
        2. Conflict detection: Conflicting evidence increases uncertainty
        3. Weighted averaging for base scores
        """
        from phentrieve.text_processing.assertion_representation import AssertionVector

        total_weight = sum(w for _, w in weighted_assertions)
        if total_weight == 0:
            return AssertionVector(evidence_source="zero_weight")

        # Weighted average of scores
        neg_score = sum(a[0] * w for a, w in weighted_assertions) / total_weight
        unc_score = sum(a[1] * w for a, w in weighted_assertions) / total_weight
        norm_score = sum(a[2] * w for a, w in weighted_assertions) / total_weight

        # Clinical rule: Detect conflict between negated and affirmed
        has_negation = any(a[0] > 0.5 for a, _ in weighted_assertions)
        has_affirmation = any(a[0] < 0.3 and a[1] < 0.3 for a, _ in weighted_assertions)

        if has_negation and has_affirmation:
            # Conflict detected - increase uncertainty
            unc_score = max(unc_score, 0.5)

        # Compute confidence
        confidence = 1.0 / (1.0 + unc_score)

        return AssertionVector(
            negation_score=float(neg_score),
            uncertainty_score=float(unc_score),
            normality_score=float(norm_score),
            evidence_source="resolved",
            evidence_confidence=float(confidence),
        )


def propagate_assertions(
    graph: SemanticDocumentGraph,
    config: PropagationConfig | None = None,
) -> PropagationResult:
    """
    Convenience function to run assertion propagation.

    Args:
        graph: SemanticDocumentGraph with chunks and edges
        config: Optional propagation configuration

    Returns:
        PropagationResult with updated assertions
    """
    propagator = AssertionPropagator(config)
    return propagator.propagate(graph)
