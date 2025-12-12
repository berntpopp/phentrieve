"""
HPO ontology-based consistency checking and inference.

This module enforces logical consistency using HPO hierarchy structure:

1. Ancestor conflict detection: If HP:child is affirmed, HP:ancestor cannot be negated
2. Descendant propagation: Affirming HP:ancestor implies possible HP:descendants
3. Redundancy detection: Affirming HP:child makes HP:ancestor redundant
4. Specificity preference: Prefer most specific matching term

The HPOConsistencyChecker is an ADDITIVE extension that provides ontology-based
validation of extraction results.

See: plan/00-planning/GRAPH-BASED-EXTENSION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phentrieve.text_processing.assertion_representation import AssertionVector

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of consistency violations in HPO assertions."""

    ANCESTOR_CONFLICT = "ancestor_conflict"
    REDUNDANT_ANCESTOR = "redundant_ancestor"
    CONTRADICTORY_SIBLINGS = "contradictory_siblings"
    MISSING_IMPLIED_ANCESTOR = "missing_implied_ancestor"


class ViolationSeverity(Enum):
    """Severity levels for consistency violations."""

    INFO = "info"  # Informational only, no action needed
    WARNING = "warning"  # Potential issue, should review
    ERROR = "error"  # Logical inconsistency, requires resolution


@dataclass
class ConsistencyViolation:
    """
    Represents a logical inconsistency in HPO assertions.

    Attributes:
        violation_type: Type of violation detected
        severity: Severity level
        hpo_id_primary: Primary HPO term involved
        hpo_id_secondary: Secondary HPO term (if applicable)
        description: Human-readable description
        confidence: Confidence in this violation (0.0-1.0)
        suggested_resolution: Suggested way to resolve
    """

    violation_type: ViolationType
    severity: ViolationSeverity
    hpo_id_primary: str
    hpo_id_secondary: str | None
    description: str
    confidence: float = 1.0
    suggested_resolution: str | None = None


@dataclass
class ConsistencyCheckResult:
    """
    Result of consistency checking.

    Attributes:
        violations: List of detected violations
        is_consistent: Whether assertions are fully consistent
        error_count: Number of ERROR-level violations
        warning_count: Number of WARNING-level violations
    """

    violations: list[ConsistencyViolation] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        """Returns True if no ERROR-level violations."""
        return not any(
            v.severity == ViolationSeverity.ERROR for v in self.violations
        )

    @property
    def error_count(self) -> int:
        """Count of ERROR-level violations."""
        return sum(1 for v in self.violations if v.severity == ViolationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING-level violations."""
        return sum(
            1 for v in self.violations if v.severity == ViolationSeverity.WARNING
        )


class HPOConsistencyChecker:
    """
    Enforce logical consistency using HPO ontology structure.

    Uses the HPO hierarchy to detect and resolve:
    - Ancestor-descendant conflicts
    - Redundant terms
    - Missing implied ancestors

    Example:
        >>> checker = HPOConsistencyChecker(ancestors_map, depths_map)
        >>> result = checker.check_consistency(hpo_assertions)
        >>> if not result.is_consistent:
        ...     resolved = checker.resolve_violations(hpo_assertions, result)
    """

    def __init__(
        self,
        ancestors_map: dict[str, set[str]],
        depths_map: dict[str, int],
        term_names: dict[str, str] | None = None,
    ):
        """
        Initialize the consistency checker.

        Args:
            ancestors_map: {hpo_id: set of ancestor HPO IDs}
            depths_map: {hpo_id: depth in ontology (0=root)}
            term_names: Optional {hpo_id: term name} for readable messages
        """
        self.ancestors_map = ancestors_map
        self.depths_map = depths_map
        self.term_names = term_names or {}
        self._descendants_map: dict[str, set[str]] = {}
        self._build_descendants_map()

    def _build_descendants_map(self) -> None:
        """Build reverse mapping from ancestors to descendants."""
        for term, ancestors in self.ancestors_map.items():
            for ancestor in ancestors:
                if ancestor not in self._descendants_map:
                    self._descendants_map[ancestor] = set()
                self._descendants_map[ancestor].add(term)

    def _get_term_name(self, hpo_id: str) -> str:
        """Get readable name for HPO term."""
        return self.term_names.get(hpo_id, hpo_id)

    def check_consistency(
        self,
        hpo_assertions: dict[str, AssertionVector],
    ) -> ConsistencyCheckResult:
        """
        Check for logical inconsistencies in HPO assertions.

        Args:
            hpo_assertions: {hpo_id: AssertionVector}

        Returns:
            ConsistencyCheckResult with all detected violations
        """
        violations: list[ConsistencyViolation] = []

        # Check ancestor conflicts
        violations.extend(self._check_ancestor_conflicts(hpo_assertions))

        # Check redundancy
        violations.extend(self._check_redundancy(hpo_assertions))

        logger.debug(
            "Consistency check complete: %d violations (%d errors, %d warnings)",
            len(violations),
            sum(1 for v in violations if v.severity == ViolationSeverity.ERROR),
            sum(1 for v in violations if v.severity == ViolationSeverity.WARNING),
        )

        return ConsistencyCheckResult(violations=violations)

    def _check_ancestor_conflicts(
        self,
        hpo_assertions: dict[str, AssertionVector],
    ) -> list[ConsistencyViolation]:
        """
        Detect affirmed child with negated ancestor.

        If HP:0001234 (child) is affirmed, HP:0000001 (ancestor) cannot
        be negated - that would be a logical contradiction.
        """
        violations = []

        for hpo_id, assertion in hpo_assertions.items():
            # Skip if this term is negated or uncertain
            if assertion.negation_score > 0.5 or assertion.uncertainty_score > 0.5:
                continue

            # Check if any ancestor is negated
            ancestors = self.ancestors_map.get(hpo_id, set())
            for ancestor_id in ancestors:
                ancestor_assertion = hpo_assertions.get(ancestor_id)
                if ancestor_assertion and ancestor_assertion.negation_score > 0.5:
                    violations.append(
                        ConsistencyViolation(
                            violation_type=ViolationType.ANCESTOR_CONFLICT,
                            severity=ViolationSeverity.ERROR,
                            hpo_id_primary=hpo_id,
                            hpo_id_secondary=ancestor_id,
                            description=(
                                f"Affirmed '{self._get_term_name(hpo_id)}' conflicts with "
                                f"negated ancestor '{self._get_term_name(ancestor_id)}'"
                            ),
                            confidence=min(
                                1.0 - assertion.negation_score,
                                ancestor_assertion.negation_score,
                            ),
                            suggested_resolution=(
                                f"Consider removing negation from {ancestor_id} "
                                f"or marking {hpo_id} as uncertain"
                            ),
                        )
                    )

        return violations

    def _check_redundancy(
        self,
        hpo_assertions: dict[str, AssertionVector],
    ) -> list[ConsistencyViolation]:
        """
        Detect redundant ancestor terms when more specific term is present.

        If HP:0001234 (specific) is affirmed, including HP:0000001 (general)
        is redundant since it's implied by the more specific term.
        """
        violations = []

        # Find affirmed terms
        affirmed_terms = {
            hpo_id
            for hpo_id, a in hpo_assertions.items()
            if a.negation_score < 0.3 and a.uncertainty_score < 0.3
        }

        for hpo_id in affirmed_terms:
            ancestors = self.ancestors_map.get(hpo_id, set())
            redundant_ancestors = ancestors & affirmed_terms

            for ancestor_id in redundant_ancestors:
                # Skip root terms (HP:0000001 Phenotypic abnormality)
                if self.depths_map.get(ancestor_id, 0) <= 1:
                    continue

                violations.append(
                    ConsistencyViolation(
                        violation_type=ViolationType.REDUNDANT_ANCESTOR,
                        severity=ViolationSeverity.INFO,
                        hpo_id_primary=ancestor_id,
                        hpo_id_secondary=hpo_id,
                        description=(
                            f"Ancestor '{self._get_term_name(ancestor_id)}' is redundant "
                            f"with more specific '{self._get_term_name(hpo_id)}'"
                        ),
                        confidence=0.9,
                        suggested_resolution=(
                            f"Consider removing {ancestor_id} in favor of {hpo_id}"
                        ),
                    )
                )

        return violations

    def resolve_violations(
        self,
        hpo_assertions: dict[str, AssertionVector],
        check_result: ConsistencyCheckResult,
        resolution_strategy: str = "conservative",
    ) -> dict[str, AssertionVector]:
        """
        Resolve detected violations by adjusting assertions.

        Args:
            hpo_assertions: Original assertions
            check_result: Result from check_consistency()
            resolution_strategy: "conservative" (add uncertainty) or
                                "aggressive" (modify assertions)

        Returns:
            Updated assertions with violations resolved
        """
        from phentrieve.text_processing.assertion_representation import AssertionVector

        resolved = dict(hpo_assertions)

        for violation in check_result.violations:
            if violation.severity != ViolationSeverity.ERROR:
                continue

            if violation.violation_type == ViolationType.ANCESTOR_CONFLICT:
                # Resolution: Add uncertainty to both conflicting terms
                for hpo_id in [violation.hpo_id_primary, violation.hpo_id_secondary]:
                    if hpo_id and hpo_id in resolved:
                        old = resolved[hpo_id]
                        if resolution_strategy == "conservative":
                            # Just add uncertainty
                            resolved[hpo_id] = AssertionVector(
                                negation_score=old.negation_score,
                                uncertainty_score=max(old.uncertainty_score, 0.4),
                                normality_score=old.normality_score,
                                historical=old.historical,
                                hypothetical=old.hypothetical,
                                family_history=old.family_history,
                                evidence_source="consistency_resolved",
                                evidence_confidence=old.evidence_confidence * 0.8,
                            )
                        else:
                            # Aggressive: favor the more specific term
                            depth_primary = self.depths_map.get(
                                violation.hpo_id_primary, 0
                            )
                            depth_secondary = self.depths_map.get(
                                violation.hpo_id_secondary or "", 0
                            )
                            if hpo_id == violation.hpo_id_secondary:
                                # This is the ancestor - reduce negation
                                if depth_primary > depth_secondary:
                                    resolved[hpo_id] = AssertionVector(
                                        negation_score=old.negation_score * 0.5,
                                        uncertainty_score=max(
                                            old.uncertainty_score, 0.3
                                        ),
                                        normality_score=old.normality_score,
                                        evidence_source="consistency_resolved",
                                        evidence_confidence=old.evidence_confidence
                                        * 0.7,
                                    )

        logger.debug(
            "Resolved %d violations using %s strategy",
            check_result.error_count,
            resolution_strategy,
        )

        return resolved

    def propagate_through_hierarchy(
        self,
        hpo_assertions: dict[str, AssertionVector],
        propagation_mode: str = "conservative",
    ) -> dict[str, AssertionVector]:
        """
        Propagate assertions through HPO hierarchy.

        Args:
            hpo_assertions: Current assertions
            propagation_mode:
                "conservative" - Only propagate high-confidence upward
                "upward" - Affirmed child implies possible ancestors
                "downward" - Negated ancestor implies negated descendants

        Returns:
            Assertions with hierarchy-propagated evidence
        """
        from phentrieve.text_processing.assertion_representation import AssertionVector

        propagated = dict(hpo_assertions)

        if propagation_mode in ("conservative", "upward"):
            # Upward propagation: affirmed child implies possible ancestor
            for hpo_id, assertion in hpo_assertions.items():
                if assertion.negation_score < 0.3:  # Affirmed
                    ancestors = self.ancestors_map.get(hpo_id, set())
                    for ancestor_id in ancestors:
                        if ancestor_id not in propagated:
                            # Add weak evidence for ancestor
                            propagated[ancestor_id] = AssertionVector(
                                evidence_source="hierarchy_propagated",
                                evidence_confidence=0.3,
                            )

        if propagation_mode == "downward":
            # Downward propagation: negated ancestor implies negated descendants
            for hpo_id, assertion in hpo_assertions.items():
                if assertion.negation_score > 0.7:  # Strongly negated
                    descendants = self._descendants_map.get(hpo_id, set())
                    for desc_id in descendants:
                        if desc_id not in propagated:
                            # Propagate negation with attenuation
                            propagated[desc_id] = AssertionVector(
                                negation_score=assertion.negation_score * 0.7,
                                evidence_source="hierarchy_propagated",
                                evidence_confidence=0.5,
                            )

        return propagated

    def get_most_specific_terms(
        self,
        hpo_assertions: dict[str, AssertionVector],
        include_negated: bool = False,
    ) -> set[str]:
        """
        Get the most specific (deepest) affirmed terms.

        Filters out terms that have affirmed descendants (more specific).

        Args:
            hpo_assertions: Current assertions
            include_negated: Whether to include negated terms

        Returns:
            Set of most specific HPO IDs
        """
        # Get affirmed terms
        if include_negated:
            terms = set(hpo_assertions.keys())
        else:
            terms = {
                hpo_id
                for hpo_id, a in hpo_assertions.items()
                if a.negation_score < 0.5 and a.uncertainty_score < 0.5
            }

        most_specific = set()

        for hpo_id in terms:
            # Check if any descendant is also in terms
            descendants = self._descendants_map.get(hpo_id, set())
            has_more_specific = bool(descendants & terms)

            if not has_more_specific:
                most_specific.add(hpo_id)

        return most_specific

    def compute_term_specificity(self, hpo_id: str) -> float:
        """
        Compute specificity score for an HPO term.

        Higher depth = more specific. Normalized to [0, 1].
        """
        depth = self.depths_map.get(hpo_id, 0)
        max_depth = max(self.depths_map.values()) if self.depths_map else 1
        return depth / max_depth if max_depth > 0 else 0.0


@dataclass
class ConsistencyConfig:
    """Configuration for consistency checking."""

    check_ancestor_conflicts: bool = True
    check_redundancy: bool = True
    resolution_strategy: str = "conservative"
    min_confidence_for_check: float = 0.3


def check_hpo_consistency(
    hpo_assertions: dict[str, AssertionVector],
    ancestors_map: dict[str, set[str]],
    depths_map: dict[str, int],
    config: ConsistencyConfig | None = None,
) -> ConsistencyCheckResult:
    """
    Convenience function to check HPO assertion consistency.

    Args:
        hpo_assertions: {hpo_id: AssertionVector}
        ancestors_map: HPO ancestor mapping
        depths_map: HPO depth mapping
        config: Optional configuration

    Returns:
        ConsistencyCheckResult with violations
    """
    config = config or ConsistencyConfig()
    checker = HPOConsistencyChecker(ancestors_map, depths_map)
    return checker.check_consistency(hpo_assertions)
