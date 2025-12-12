"""
Hybrid inference engine combining text graph and ontology evidence.

This module coordinates:
1. Semantic text graph assertion propagation
2. HPO ontology consistency checking
3. Weighted combination of evidence sources
4. Final assertion output with full provenance

The HybridInferenceEngine provides the complete graph-based reasoning
pipeline as an ADDITIVE extension to existing extraction methods.

See: plan/00-planning/GRAPH-BASED-EXTENSION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from phentrieve.reasoning.hpo_consistency import (
        ConsistencyCheckResult,
        ConsistencyViolation,
        HPOConsistencyChecker,
    )
    from phentrieve.text_processing.assertion_propagation import (
        AssertionPropagator,
        PropagationResult,
    )
    from phentrieve.text_processing.assertion_representation import AssertionVector
    from phentrieve.text_processing.semantic_graph import SemanticDocumentGraph

logger = logging.getLogger(__name__)


@dataclass
class HybridInferenceConfig:
    """
    Configuration for hybrid graph inference.

    Attributes:
        text_graph_weight: Weight for text graph evidence (0-1)
        ontology_weight: Weight for ontology constraints (0-1)
        consistency_penalty: Penalty applied for consistency violations
        min_confidence_threshold: Minimum confidence to include in output
        resolve_conflicts: Whether to automatically resolve detected conflicts
        include_provenance: Whether to include detailed provenance in output
    """

    text_graph_weight: float = 0.6
    ontology_weight: float = 0.4
    consistency_penalty: float = 0.2
    min_confidence_threshold: float = 0.1
    resolve_conflicts: bool = True
    include_provenance: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        total = self.text_graph_weight + self.ontology_weight
        if not (0.99 <= total <= 1.01):
            # Normalize weights
            self.text_graph_weight /= total
            self.ontology_weight /= total


@dataclass
class EvidenceSource:
    """
    Tracks the source of evidence for an assertion.

    Attributes:
        source_type: Type of evidence (text_graph, ontology, combined)
        chunk_indices: Text chunks that contributed evidence
        propagation_path: Path through graph for propagated evidence
        ontology_inference: Whether ontology inference was applied
        confidence_breakdown: Confidence from each source
    """

    source_type: str
    chunk_indices: list[int] = field(default_factory=list)
    propagation_path: list[int] = field(default_factory=list)
    ontology_inference: bool = False
    confidence_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class FinalAssertion:
    """
    Final assertion with full provenance for an HPO term.

    Attributes:
        hpo_id: The HPO term ID
        assertion_vector: Final combined assertion
        legacy_status: Backward-compatible status string
        text_evidence: Evidence from text graph
        ontology_evidence: Evidence from ontology reasoning
        consistency_violations: Any violations involving this term
        inference_method: Method used ("hybrid", "text_only", "ontology_only")
        provenance: Detailed provenance information
    """

    hpo_id: str
    assertion_vector: AssertionVector
    legacy_status: str
    text_evidence: EvidenceSource | None = None
    ontology_evidence: EvidenceSource | None = None
    consistency_violations: list[ConsistencyViolation] = field(default_factory=list)
    inference_method: str = "hybrid"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_affirmed(self) -> bool:
        """Returns True if assertion is affirmed (not negated/uncertain)."""
        return (
            self.assertion_vector.negation_score < 0.5
            and self.assertion_vector.uncertainty_score < 0.5
        )

    @property
    def confidence(self) -> float:
        """Overall confidence in this assertion."""
        return self.assertion_vector.evidence_confidence


@dataclass
class HybridInferenceResult:
    """
    Complete result from hybrid inference.

    Attributes:
        assertions: Final assertions per HPO term
        propagation_result: Result from text graph propagation
        consistency_result: Result from ontology consistency check
        inference_config: Configuration used
        timing_ms: Processing time in milliseconds
    """

    assertions: dict[str, FinalAssertion]
    propagation_result: PropagationResult | None = None
    consistency_result: ConsistencyCheckResult | None = None
    inference_config: HybridInferenceConfig | None = None
    timing_ms: float = 0.0

    @property
    def affirmed_terms(self) -> list[str]:
        """Get list of affirmed HPO terms."""
        return [
            hpo_id for hpo_id, a in self.assertions.items() if a.is_affirmed
        ]

    @property
    def negated_terms(self) -> list[str]:
        """Get list of negated HPO terms."""
        return [
            hpo_id
            for hpo_id, a in self.assertions.items()
            if a.assertion_vector.negation_score > 0.5
        ]

    @property
    def uncertain_terms(self) -> list[str]:
        """Get list of uncertain HPO terms."""
        return [
            hpo_id
            for hpo_id, a in self.assertions.items()
            if a.assertion_vector.uncertainty_score > 0.5
        ]


class HybridInferenceEngine:
    """
    Combine text graph evidence with ontology constraints.

    Pipeline:
    1. Run text graph assertion propagation
    2. Map propagated assertions to HPO terms
    3. Apply ontology consistency checks
    4. Resolve conflicts using weighted combination
    5. Output final assertions with provenance

    Example:
        >>> engine = HybridInferenceEngine(
        ...     text_graph=graph,
        ...     propagator=propagator,
        ...     consistency_checker=checker,
        ... )
        >>> result = engine.infer(hpo_matches)
        >>> for hpo_id, assertion in result.assertions.items():
        ...     print(f"{hpo_id}: {assertion.legacy_status}")
    """

    def __init__(
        self,
        text_graph: SemanticDocumentGraph,
        propagator: AssertionPropagator,
        consistency_checker: HPOConsistencyChecker | None = None,
        config: HybridInferenceConfig | None = None,
    ):
        """
        Initialize the hybrid inference engine.

        Args:
            text_graph: SemanticDocumentGraph with text chunks
            propagator: AssertionPropagator for graph-based propagation
            consistency_checker: Optional HPOConsistencyChecker for ontology
            config: Inference configuration
        """
        self.text_graph = text_graph
        self.propagator = propagator
        self.consistency_checker = consistency_checker
        self.config = config or HybridInferenceConfig()

    def infer(
        self,
        hpo_matches: dict[str, list[int]],
    ) -> HybridInferenceResult:
        """
        Run hybrid inference pipeline.

        Args:
            hpo_matches: Mapping from HPO IDs to chunk indices where matched

        Returns:
            HybridInferenceResult with final assertions and metadata
        """
        import time

        start_time = time.perf_counter()

        # Step 1: Text graph propagation
        propagation_result = self.propagator.propagate(self.text_graph)

        # Step 2: Aggregate to HPO level
        hpo_assertions = self._aggregate_to_hpo(hpo_matches, propagation_result)

        # Step 3: Ontology consistency check (if checker provided)
        consistency_result = None
        if self.consistency_checker:
            consistency_result = self.consistency_checker.check_consistency(
                hpo_assertions
            )

            # Step 4: Apply consistency constraints if configured
            if self.config.resolve_conflicts and not consistency_result.is_consistent:
                hpo_assertions = self.consistency_checker.resolve_violations(
                    hpo_assertions,
                    consistency_result,
                )

        # Step 5: Format final output with provenance
        final_assertions = self._create_final_assertions(
            hpo_matches,
            hpo_assertions,
            propagation_result,
            consistency_result,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Hybrid inference complete: %d terms, %.1f ms",
            len(final_assertions),
            elapsed_ms,
        )

        return HybridInferenceResult(
            assertions=final_assertions,
            propagation_result=propagation_result,
            consistency_result=consistency_result,
            inference_config=self.config,
            timing_ms=elapsed_ms,
        )

    def _aggregate_to_hpo(
        self,
        hpo_matches: dict[str, list[int]],
        propagation_result: PropagationResult,
    ) -> dict[str, AssertionVector]:
        """Aggregate chunk-level assertions to HPO level."""
        hpo_assertions = {}

        for hpo_id, chunk_indices in hpo_matches.items():
            assertion = self.propagator.resolve_hpo_assertion(
                self.text_graph,
                hpo_id,
                propagation_result.assertions,
            )
            hpo_assertions[hpo_id] = assertion

        return hpo_assertions

    def _create_final_assertions(
        self,
        hpo_matches: dict[str, list[int]],
        hpo_assertions: dict[str, AssertionVector],
        propagation_result: PropagationResult,
        consistency_result: ConsistencyCheckResult | None,
    ) -> dict[str, FinalAssertion]:
        """Create final assertions with provenance."""
        final_assertions = {}

        for hpo_id, assertion in hpo_assertions.items():
            # Skip low-confidence assertions
            if assertion.evidence_confidence < self.config.min_confidence_threshold:
                continue

            # Get violations for this term
            term_violations = []
            if consistency_result:
                term_violations = [
                    v
                    for v in consistency_result.violations
                    if v.hpo_id_primary == hpo_id or v.hpo_id_secondary == hpo_id
                ]

            # Apply consistency penalty if violations exist
            final_assertion = assertion
            if term_violations and self.config.consistency_penalty > 0:
                final_assertion = self._apply_consistency_penalty(
                    assertion, len(term_violations)
                )

            # Build provenance
            provenance = {}
            text_evidence = None
            if self.config.include_provenance:
                chunk_indices = hpo_matches.get(hpo_id, [])
                text_evidence = EvidenceSource(
                    source_type="text_graph",
                    chunk_indices=chunk_indices,
                    confidence_breakdown={
                        "propagation": final_assertion.evidence_confidence,
                    },
                )
                provenance = {
                    "chunks": chunk_indices,
                    "propagation_converged": propagation_result.converged,
                    "propagation_iterations": propagation_result.iterations_run,
                    "has_conflicts": len(term_violations) > 0,
                }

            final_assertions[hpo_id] = FinalAssertion(
                hpo_id=hpo_id,
                assertion_vector=final_assertion,
                legacy_status=final_assertion.to_status().value,
                text_evidence=text_evidence,
                consistency_violations=term_violations,
                inference_method="hybrid" if self.consistency_checker else "text_only",
                provenance=provenance,
            )

        return final_assertions

    def _apply_consistency_penalty(
        self,
        assertion: AssertionVector,
        num_violations: int,
    ) -> AssertionVector:
        """Apply penalty for consistency violations."""
        from phentrieve.text_processing.assertion_representation import AssertionVector

        penalty = self.config.consistency_penalty * min(num_violations, 3)

        return AssertionVector(
            negation_score=assertion.negation_score,
            uncertainty_score=min(1.0, assertion.uncertainty_score + penalty * 0.5),
            normality_score=assertion.normality_score,
            historical=assertion.historical,
            hypothetical=assertion.hypothetical,
            family_history=assertion.family_history,
            evidence_source=assertion.evidence_source,
            evidence_confidence=max(0.0, assertion.evidence_confidence * (1 - penalty)),
        )

    def infer_with_comparison(
        self,
        hpo_matches: dict[str, list[int]],
        legacy_assertions: dict[str, str],
    ) -> tuple[HybridInferenceResult, dict[str, dict[str, Any]]]:
        """
        Run inference with comparison to legacy method.

        Args:
            hpo_matches: HPO to chunk mapping
            legacy_assertions: Legacy assertion status per HPO

        Returns:
            Tuple of (inference_result, comparison_metrics)
        """
        result = self.infer(hpo_matches)

        comparison = {}
        for hpo_id in set(result.assertions.keys()) | set(legacy_assertions.keys()):
            hybrid_status = (
                result.assertions[hpo_id].legacy_status
                if hpo_id in result.assertions
                else "missing"
            )
            legacy_status = legacy_assertions.get(hpo_id, "missing")

            comparison[hpo_id] = {
                "hybrid_status": hybrid_status,
                "legacy_status": legacy_status,
                "agreement": hybrid_status == legacy_status,
                "hybrid_confidence": (
                    result.assertions[hpo_id].confidence
                    if hpo_id in result.assertions
                    else 0.0
                ),
            }

        return result, comparison


def create_inference_engine(
    text_graph: SemanticDocumentGraph,
    ancestors_map: dict[str, set[str]] | None = None,
    depths_map: dict[str, int] | None = None,
    config: HybridInferenceConfig | None = None,
) -> HybridInferenceEngine:
    """
    Factory function to create a complete inference engine.

    Args:
        text_graph: SemanticDocumentGraph with processed chunks
        ancestors_map: Optional HPO ancestors for consistency checking
        depths_map: Optional HPO depths for consistency checking
        config: Inference configuration

    Returns:
        Configured HybridInferenceEngine
    """
    from phentrieve.text_processing.assertion_propagation import AssertionPropagator

    propagator = AssertionPropagator()

    consistency_checker = None
    if ancestors_map and depths_map:
        from phentrieve.reasoning.hpo_consistency import HPOConsistencyChecker

        consistency_checker = HPOConsistencyChecker(ancestors_map, depths_map)

    return HybridInferenceEngine(
        text_graph=text_graph,
        propagator=propagator,
        consistency_checker=consistency_checker,
        config=config,
    )


@dataclass
class MethodComparisonResult:
    """Result of comparing legacy and graph-based methods."""

    total_terms: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Agreement: {self.agreement_count}/{self.total_terms} "
            f"({self.agreement_rate:.1%})"
        )


def compare_methods(
    hybrid_result: HybridInferenceResult,
    legacy_assertions: dict[str, str],
) -> MethodComparisonResult:
    """
    Compare hybrid inference results with legacy method.

    Args:
        hybrid_result: Result from HybridInferenceEngine
        legacy_assertions: {hpo_id: assertion_status_string}

    Returns:
        MethodComparisonResult with agreement metrics
    """
    all_terms = set(hybrid_result.assertions.keys()) | set(legacy_assertions.keys())
    total = len(all_terms)

    agreements = 0
    disagreements = []

    for hpo_id in all_terms:
        hybrid_status = (
            hybrid_result.assertions[hpo_id].legacy_status
            if hpo_id in hybrid_result.assertions
            else "missing"
        )
        legacy_status = legacy_assertions.get(hpo_id, "missing")

        if hybrid_status == legacy_status:
            agreements += 1
        else:
            disagreements.append(
                {
                    "hpo_id": hpo_id,
                    "hybrid": hybrid_status,
                    "legacy": legacy_status,
                }
            )

    return MethodComparisonResult(
        total_terms=total,
        agreement_count=agreements,
        disagreement_count=len(disagreements),
        agreement_rate=agreements / total if total > 0 else 1.0,
        disagreements=disagreements,
    )
