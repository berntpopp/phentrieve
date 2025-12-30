"""
Multi-dimensional assertion representation with confidence scores.

This module extends the existing AssertionStatus enum with a richer
AssertionVector representation that captures:
- Confidence scores for negation, uncertainty, and normality
- Contextual modifiers (historical, hypothetical, family history)
- Evidence provenance tracking

The AssertionVector is designed to be used alongside (not replace) the
existing AssertionStatus enum, enabling comparison between methods.

See: plan/00-planning/GRAPH-BASED-EXTENSION-PLAN.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phentrieve.text_processing.assertion_detection import AssertionStatus


@dataclass(frozen=True)
class AssertionVector:
    """
    Multi-dimensional assertion representation with confidence scores.

    This dataclass provides a richer representation of assertion status
    compared to the discrete AssertionStatus enum. It captures:

    1. **Core assertion dimensions** (0.0 to 1.0):
       - negation_score: Confidence that the finding is negated
       - uncertainty_score: Epistemic uncertainty (e.g., "possible", "rule out")
       - normality_score: Finding is within normal limits

    2. **Contextual modifiers**:
       - historical: Finding is from the past (not current)
       - hypothetical: Conditional statement (e.g., "if symptoms appear")
       - family_history: Subject is a family member, not the patient

    3. **Evidence tracking**:
       - evidence_source: Origin of the assertion ("keyword", "dependency", "propagated")
       - evidence_confidence: Confidence in the evidence itself

    The class provides backward-compatible conversion to/from AssertionStatus
    via to_status() and from_status() methods.

    Attributes:
        negation_score: Confidence in negation (0.0 = affirmed, 1.0 = negated)
        uncertainty_score: Epistemic uncertainty (0.0 = certain, 1.0 = uncertain)
        normality_score: Within normal limits (0.0 = abnormal, 1.0 = normal)
        historical: True if finding is from the past
        hypothetical: True if conditional/hypothetical statement
        family_history: True if subject is a family member
        evidence_source: Source of assertion evidence
        evidence_confidence: Confidence in the evidence (0.0 to 1.0)

    Example:
        >>> # Create from explicit negation detection
        >>> vec = AssertionVector(negation_score=0.9, evidence_source="keyword")
        >>> vec.to_status()
        <AssertionStatus.NEGATED: 'negated'>

        >>> # Create from legacy status
        >>> from phentrieve.text_processing.assertion_detection import AssertionStatus
        >>> vec = AssertionVector.from_status(AssertionStatus.AFFIRMED, confidence=0.95)
        >>> vec.negation_score
        0.0
    """

    # Core assertion dimensions (0.0 to 1.0)
    negation_score: float = 0.0
    uncertainty_score: float = 0.0
    normality_score: float = 0.0

    # Contextual modifiers
    historical: bool = False
    hypothetical: bool = False
    family_history: bool = False

    # Evidence tracking
    evidence_source: str = "unknown"
    evidence_confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate score ranges after initialization."""
        # Note: frozen=True prevents direct assignment, but we validate here
        # to catch invalid construction. Validation uses object.__setattr__
        # workaround if needed, but for frozen dataclass we just check values.
        if not (0.0 <= self.negation_score <= 1.0):
            raise ValueError(
                f"negation_score must be in [0, 1], got {self.negation_score}"
            )
        if not (0.0 <= self.uncertainty_score <= 1.0):
            raise ValueError(
                f"uncertainty_score must be in [0, 1], got {self.uncertainty_score}"
            )
        if not (0.0 <= self.normality_score <= 1.0):
            raise ValueError(
                f"normality_score must be in [0, 1], got {self.normality_score}"
            )
        if not (0.0 <= self.evidence_confidence <= 1.0):
            raise ValueError(
                f"evidence_confidence must be in [0, 1], got {self.evidence_confidence}"
            )

    def to_status(self) -> AssertionStatus:
        """
        Convert to legacy discrete AssertionStatus for backward compatibility.

        Uses threshold-based conversion with priority ordering:
        1. NEGATED if negation_score > 0.5
        2. UNCERTAIN if uncertainty_score > 0.5
        3. NORMAL if normality_score > 0.5
        4. AFFIRMED otherwise

        Returns:
            AssertionStatus enum value

        Example:
            >>> vec = AssertionVector(negation_score=0.8)
            >>> vec.to_status()
            <AssertionStatus.NEGATED: 'negated'>
        """
        # Import here to avoid circular dependency
        from phentrieve.text_processing.assertion_detection import AssertionStatus

        if self.negation_score > 0.5:
            return AssertionStatus.NEGATED
        if self.uncertainty_score > 0.5:
            return AssertionStatus.UNCERTAIN
        if self.normality_score > 0.5:
            return AssertionStatus.NORMAL
        return AssertionStatus.AFFIRMED

    @classmethod
    def from_status(
        cls,
        status: AssertionStatus,
        confidence: float = 1.0,
        evidence_source: str = "legacy_conversion",
    ) -> AssertionVector:
        """
        Create AssertionVector from legacy AssertionStatus.

        Converts discrete status to vector representation by setting
        the appropriate dimension to the specified confidence.

        Args:
            status: Legacy AssertionStatus enum value
            confidence: Confidence score to assign (0.0 to 1.0)
            evidence_source: Source of the assertion

        Returns:
            New AssertionVector instance

        Example:
            >>> from phentrieve.text_processing.assertion_detection import AssertionStatus
            >>> vec = AssertionVector.from_status(AssertionStatus.NEGATED, confidence=0.9)
            >>> vec.negation_score
            0.9
        """
        # Import here to avoid circular dependency
        from phentrieve.text_processing.assertion_detection import AssertionStatus

        if status == AssertionStatus.NEGATED:
            return cls(
                negation_score=confidence,
                evidence_source=evidence_source,
                evidence_confidence=confidence,
            )
        elif status == AssertionStatus.UNCERTAIN:
            return cls(
                uncertainty_score=confidence,
                evidence_source=evidence_source,
                evidence_confidence=confidence,
            )
        elif status == AssertionStatus.NORMAL:
            return cls(
                normality_score=confidence,
                evidence_source=evidence_source,
                evidence_confidence=confidence,
            )
        # AFFIRMED - no dimensions set, just confidence
        return cls(
            evidence_source=evidence_source,
            evidence_confidence=confidence,
        )

    def with_updates(self, **kwargs: Any) -> AssertionVector:
        """
        Create a new AssertionVector with updated fields.

        Since AssertionVector is frozen (immutable), this creates a new
        instance with the specified fields updated.

        Args:
            **kwargs: Fields to update

        Returns:
            New AssertionVector with updated fields

        Example:
            >>> vec = AssertionVector(negation_score=0.5)
            >>> updated = vec.with_updates(negation_score=0.9)
            >>> updated.negation_score
            0.9
        """
        return AssertionVector(
            negation_score=kwargs.get("negation_score", self.negation_score),
            uncertainty_score=kwargs.get("uncertainty_score", self.uncertainty_score),
            normality_score=kwargs.get("normality_score", self.normality_score),
            historical=kwargs.get("historical", self.historical),
            hypothetical=kwargs.get("hypothetical", self.hypothetical),
            family_history=kwargs.get("family_history", self.family_history),
            evidence_source=kwargs.get("evidence_source", self.evidence_source),
            evidence_confidence=kwargs.get(
                "evidence_confidence", self.evidence_confidence
            ),
        )

    def is_affirmed(self, threshold: float = 0.5) -> bool:
        """
        Check if the assertion is effectively affirmed.

        An assertion is affirmed if none of the negation, uncertainty,
        or normality scores exceed the threshold.

        Args:
            threshold: Score threshold (default 0.5)

        Returns:
            True if effectively affirmed
        """
        return (
            self.negation_score <= threshold
            and self.uncertainty_score <= threshold
            and self.normality_score <= threshold
        )

    def is_negated(self, threshold: float = 0.5) -> bool:
        """Check if negation score exceeds threshold."""
        return self.negation_score > threshold

    def is_uncertain(self, threshold: float = 0.5) -> bool:
        """Check if uncertainty score exceeds threshold."""
        return self.uncertainty_score > threshold

    def is_normal(self, threshold: float = 0.5) -> bool:
        """Check if normality score exceeds threshold."""
        return self.normality_score > threshold

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the vector
        """
        return {
            "negation_score": self.negation_score,
            "uncertainty_score": self.uncertainty_score,
            "normality_score": self.normality_score,
            "historical": self.historical,
            "hypothetical": self.hypothetical,
            "family_history": self.family_history,
            "evidence_source": self.evidence_source,
            "evidence_confidence": self.evidence_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssertionVector:
        """
        Create AssertionVector from dictionary.

        Args:
            data: Dictionary with vector fields

        Returns:
            New AssertionVector instance
        """
        return cls(
            negation_score=data.get("negation_score", 0.0),
            uncertainty_score=data.get("uncertainty_score", 0.0),
            normality_score=data.get("normality_score", 0.0),
            historical=data.get("historical", False),
            hypothetical=data.get("hypothetical", False),
            family_history=data.get("family_history", False),
            evidence_source=data.get("evidence_source", "unknown"),
            evidence_confidence=data.get("evidence_confidence", 1.0),
        )

    def combine_with(
        self,
        other: AssertionVector,
        self_weight: float = 0.5,
    ) -> AssertionVector:
        """
        Combine this vector with another using weighted averaging.

        Useful for aggregating assertions from multiple sources or
        during graph-based propagation.

        Args:
            other: Another AssertionVector to combine with
            self_weight: Weight for this vector (0.0 to 1.0)

        Returns:
            New combined AssertionVector
        """
        other_weight = 1.0 - self_weight

        return AssertionVector(
            negation_score=(
                self.negation_score * self_weight + other.negation_score * other_weight
            ),
            uncertainty_score=(
                self.uncertainty_score * self_weight
                + other.uncertainty_score * other_weight
            ),
            normality_score=(
                self.normality_score * self_weight
                + other.normality_score * other_weight
            ),
            # For boolean fields, use OR logic
            historical=self.historical or other.historical,
            hypothetical=self.hypothetical or other.hypothetical,
            family_history=self.family_history or other.family_history,
            evidence_source="combined",
            evidence_confidence=(
                self.evidence_confidence * self_weight
                + other.evidence_confidence * other_weight
            ),
        )


# Convenience factory functions


def affirmed_vector(confidence: float = 1.0) -> AssertionVector:
    """Create an affirmed assertion vector."""
    return AssertionVector(
        evidence_source="factory",
        evidence_confidence=confidence,
    )


def negated_vector(confidence: float = 1.0) -> AssertionVector:
    """Create a negated assertion vector."""
    return AssertionVector(
        negation_score=confidence,
        evidence_source="factory",
        evidence_confidence=confidence,
    )


def uncertain_vector(confidence: float = 1.0) -> AssertionVector:
    """Create an uncertain assertion vector."""
    return AssertionVector(
        uncertainty_score=confidence,
        evidence_source="factory",
        evidence_confidence=confidence,
    )


def normal_vector(confidence: float = 1.0) -> AssertionVector:
    """Create a normal assertion vector."""
    return AssertionVector(
        normality_score=confidence,
        evidence_source="factory",
        evidence_confidence=confidence,
    )
