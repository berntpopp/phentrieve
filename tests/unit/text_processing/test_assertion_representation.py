"""
Unit tests for AssertionVector multi-dimensional assertion representation.

Tests cover:
- Construction and validation
- Conversion to/from legacy AssertionStatus
- Score threshold behavior
- Serialization/deserialization
- Combination of vectors
"""

import pytest

from phentrieve.text_processing.assertion_detection import AssertionStatus
from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    affirmed_vector,
    negated_vector,
    normal_vector,
    uncertain_vector,
)


class TestAssertionVectorConstruction:
    """Tests for AssertionVector construction and validation."""

    def test_default_construction(self):
        """Default vector should be all zeros (affirmed)."""
        vec = AssertionVector()
        assert vec.negation_score == 0.0
        assert vec.uncertainty_score == 0.0
        assert vec.normality_score == 0.0
        assert vec.historical is False
        assert vec.hypothetical is False
        assert vec.family_history is False
        assert vec.evidence_source == "unknown"
        assert vec.evidence_confidence == 1.0

    def test_construction_with_scores(self):
        """Construction with explicit scores."""
        vec = AssertionVector(
            negation_score=0.8,
            uncertainty_score=0.2,
            normality_score=0.1,
            evidence_source="keyword",
            evidence_confidence=0.9,
        )
        assert vec.negation_score == 0.8
        assert vec.uncertainty_score == 0.2
        assert vec.normality_score == 0.1
        assert vec.evidence_source == "keyword"
        assert vec.evidence_confidence == 0.9

    def test_construction_with_modifiers(self):
        """Construction with contextual modifiers."""
        vec = AssertionVector(
            historical=True,
            hypothetical=True,
            family_history=True,
        )
        assert vec.historical is True
        assert vec.hypothetical is True
        assert vec.family_history is True

    def test_invalid_negation_score_high(self):
        """Negation score > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="negation_score must be in"):
            AssertionVector(negation_score=1.5)

    def test_invalid_negation_score_low(self):
        """Negation score < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="negation_score must be in"):
            AssertionVector(negation_score=-0.1)

    def test_invalid_uncertainty_score(self):
        """Uncertainty score out of range should raise ValueError."""
        with pytest.raises(ValueError, match="uncertainty_score must be in"):
            AssertionVector(uncertainty_score=2.0)

    def test_invalid_normality_score(self):
        """Normality score out of range should raise ValueError."""
        with pytest.raises(ValueError, match="normality_score must be in"):
            AssertionVector(normality_score=-0.5)

    def test_invalid_evidence_confidence(self):
        """Evidence confidence out of range should raise ValueError."""
        with pytest.raises(ValueError, match="evidence_confidence must be in"):
            AssertionVector(evidence_confidence=1.1)

    def test_boundary_values(self):
        """Boundary values (0.0 and 1.0) should be valid."""
        vec = AssertionVector(
            negation_score=0.0,
            uncertainty_score=1.0,
            normality_score=0.0,
            evidence_confidence=1.0,
        )
        assert vec.negation_score == 0.0
        assert vec.uncertainty_score == 1.0

    def test_frozen_immutability(self):
        """AssertionVector should be immutable (frozen dataclass)."""
        vec = AssertionVector(negation_score=0.5)
        with pytest.raises(AttributeError):
            vec.negation_score = 0.9  # type: ignore


class TestAssertionVectorToStatus:
    """Tests for conversion to legacy AssertionStatus."""

    def test_to_status_negated(self):
        """High negation score should convert to NEGATED."""
        vec = AssertionVector(negation_score=0.8)
        assert vec.to_status() == AssertionStatus.NEGATED

    def test_to_status_uncertain(self):
        """High uncertainty score should convert to UNCERTAIN."""
        vec = AssertionVector(uncertainty_score=0.7)
        assert vec.to_status() == AssertionStatus.UNCERTAIN

    def test_to_status_normal(self):
        """High normality score should convert to NORMAL."""
        vec = AssertionVector(normality_score=0.6)
        assert vec.to_status() == AssertionStatus.NORMAL

    def test_to_status_affirmed(self):
        """Low scores should convert to AFFIRMED."""
        vec = AssertionVector(
            negation_score=0.3,
            uncertainty_score=0.2,
            normality_score=0.1,
        )
        assert vec.to_status() == AssertionStatus.AFFIRMED

    def test_to_status_negation_priority(self):
        """Negation should take priority when multiple scores are high."""
        vec = AssertionVector(
            negation_score=0.9,
            uncertainty_score=0.8,
            normality_score=0.7,
        )
        assert vec.to_status() == AssertionStatus.NEGATED

    def test_to_status_uncertainty_over_normal(self):
        """Uncertainty should take priority over normality."""
        vec = AssertionVector(
            negation_score=0.3,
            uncertainty_score=0.8,
            normality_score=0.7,
        )
        assert vec.to_status() == AssertionStatus.UNCERTAIN

    def test_to_status_threshold_boundary(self):
        """Exactly 0.5 should not trigger (needs > 0.5)."""
        vec = AssertionVector(negation_score=0.5)
        assert vec.to_status() == AssertionStatus.AFFIRMED

        vec2 = AssertionVector(negation_score=0.51)
        assert vec2.to_status() == AssertionStatus.NEGATED


class TestAssertionVectorFromStatus:
    """Tests for creation from legacy AssertionStatus."""

    def test_from_status_negated(self):
        """NEGATED status should set negation_score."""
        vec = AssertionVector.from_status(AssertionStatus.NEGATED, confidence=0.9)
        assert vec.negation_score == 0.9
        assert vec.uncertainty_score == 0.0
        assert vec.normality_score == 0.0
        assert vec.evidence_confidence == 0.9

    def test_from_status_uncertain(self):
        """UNCERTAIN status should set uncertainty_score."""
        vec = AssertionVector.from_status(AssertionStatus.UNCERTAIN, confidence=0.8)
        assert vec.uncertainty_score == 0.8
        assert vec.negation_score == 0.0

    def test_from_status_normal(self):
        """NORMAL status should set normality_score."""
        vec = AssertionVector.from_status(AssertionStatus.NORMAL, confidence=0.75)
        assert vec.normality_score == 0.75
        assert vec.negation_score == 0.0

    def test_from_status_affirmed(self):
        """AFFIRMED status should not set any dimension scores."""
        vec = AssertionVector.from_status(AssertionStatus.AFFIRMED, confidence=0.95)
        assert vec.negation_score == 0.0
        assert vec.uncertainty_score == 0.0
        assert vec.normality_score == 0.0
        assert vec.evidence_confidence == 0.95

    def test_from_status_default_confidence(self):
        """Default confidence should be 1.0."""
        vec = AssertionVector.from_status(AssertionStatus.NEGATED)
        assert vec.negation_score == 1.0
        assert vec.evidence_confidence == 1.0

    def test_from_status_custom_source(self):
        """Evidence source should be set."""
        vec = AssertionVector.from_status(
            AssertionStatus.NEGATED,
            confidence=0.9,
            evidence_source="test_source",
        )
        assert vec.evidence_source == "test_source"


class TestAssertionVectorRoundTrip:
    """Tests for round-trip conversion (status -> vector -> status)."""

    def test_roundtrip_negated(self):
        """NEGATED should round-trip correctly."""
        original = AssertionStatus.NEGATED
        vec = AssertionVector.from_status(original)
        result = vec.to_status()
        assert result == original

    def test_roundtrip_uncertain(self):
        """UNCERTAIN should round-trip correctly."""
        original = AssertionStatus.UNCERTAIN
        vec = AssertionVector.from_status(original)
        result = vec.to_status()
        assert result == original

    def test_roundtrip_normal(self):
        """NORMAL should round-trip correctly."""
        original = AssertionStatus.NORMAL
        vec = AssertionVector.from_status(original)
        result = vec.to_status()
        assert result == original

    def test_roundtrip_affirmed(self):
        """AFFIRMED should round-trip correctly."""
        original = AssertionStatus.AFFIRMED
        vec = AssertionVector.from_status(original)
        result = vec.to_status()
        assert result == original


class TestAssertionVectorHelpers:
    """Tests for helper methods."""

    def test_is_affirmed(self):
        """is_affirmed should return True when all scores are low."""
        vec = AssertionVector(negation_score=0.2, uncertainty_score=0.3)
        assert vec.is_affirmed() is True

        vec2 = AssertionVector(negation_score=0.6)
        assert vec2.is_affirmed() is False

    def test_is_negated(self):
        """is_negated should check negation_score."""
        assert AssertionVector(negation_score=0.8).is_negated() is True
        assert AssertionVector(negation_score=0.3).is_negated() is False

    def test_is_uncertain(self):
        """is_uncertain should check uncertainty_score."""
        assert AssertionVector(uncertainty_score=0.7).is_uncertain() is True
        assert AssertionVector(uncertainty_score=0.4).is_uncertain() is False

    def test_is_normal(self):
        """is_normal should check normality_score."""
        assert AssertionVector(normality_score=0.6).is_normal() is True
        assert AssertionVector(normality_score=0.5).is_normal() is False

    def test_custom_threshold(self):
        """Helper methods should respect custom threshold."""
        vec = AssertionVector(negation_score=0.4)
        assert vec.is_negated(threshold=0.3) is True
        assert vec.is_negated(threshold=0.5) is False


class TestAssertionVectorWithUpdates:
    """Tests for with_updates method."""

    def test_with_updates_single_field(self):
        """Update single field."""
        original = AssertionVector(negation_score=0.5)
        updated = original.with_updates(negation_score=0.9)
        assert updated.negation_score == 0.9
        assert original.negation_score == 0.5  # Original unchanged

    def test_with_updates_multiple_fields(self):
        """Update multiple fields."""
        original = AssertionVector(negation_score=0.5, evidence_source="original")
        updated = original.with_updates(
            negation_score=0.9,
            evidence_source="updated",
            historical=True,
        )
        assert updated.negation_score == 0.9
        assert updated.evidence_source == "updated"
        assert updated.historical is True


class TestAssertionVectorSerialization:
    """Tests for to_dict and from_dict."""

    def test_to_dict(self):
        """to_dict should include all fields."""
        vec = AssertionVector(
            negation_score=0.8,
            uncertainty_score=0.2,
            historical=True,
            evidence_source="test",
        )
        data = vec.to_dict()
        assert data["negation_score"] == 0.8
        assert data["uncertainty_score"] == 0.2
        assert data["historical"] is True
        assert data["evidence_source"] == "test"

    def test_from_dict(self):
        """from_dict should reconstruct vector."""
        data = {
            "negation_score": 0.7,
            "uncertainty_score": 0.3,
            "normality_score": 0.1,
            "historical": True,
            "hypothetical": False,
            "family_history": True,
            "evidence_source": "test",
            "evidence_confidence": 0.9,
        }
        vec = AssertionVector.from_dict(data)
        assert vec.negation_score == 0.7
        assert vec.historical is True
        assert vec.family_history is True

    def test_from_dict_missing_fields(self):
        """from_dict should use defaults for missing fields."""
        data = {"negation_score": 0.5}
        vec = AssertionVector.from_dict(data)
        assert vec.negation_score == 0.5
        assert vec.uncertainty_score == 0.0
        assert vec.evidence_source == "unknown"

    def test_roundtrip_serialization(self):
        """to_dict -> from_dict should preserve all data."""
        original = AssertionVector(
            negation_score=0.8,
            uncertainty_score=0.2,
            normality_score=0.1,
            historical=True,
            hypothetical=True,
            family_history=True,
            evidence_source="original",
            evidence_confidence=0.85,
        )
        data = original.to_dict()
        restored = AssertionVector.from_dict(data)
        assert restored == original


class TestAssertionVectorCombination:
    """Tests for combine_with method."""

    def test_combine_equal_weights(self):
        """Equal weights should average scores."""
        vec1 = AssertionVector(negation_score=0.8)
        vec2 = AssertionVector(negation_score=0.4)
        combined = vec1.combine_with(vec2, self_weight=0.5)
        assert combined.negation_score == pytest.approx(0.6)

    def test_combine_unequal_weights(self):
        """Unequal weights should weighted average."""
        vec1 = AssertionVector(negation_score=1.0)
        vec2 = AssertionVector(negation_score=0.0)
        combined = vec1.combine_with(vec2, self_weight=0.8)
        assert combined.negation_score == pytest.approx(0.8)

    def test_combine_boolean_or(self):
        """Boolean fields should use OR logic."""
        vec1 = AssertionVector(historical=True, hypothetical=False)
        vec2 = AssertionVector(historical=False, hypothetical=True)
        combined = vec1.combine_with(vec2)
        assert combined.historical is True
        assert combined.hypothetical is True

    def test_combine_source(self):
        """Combined vectors should have 'combined' source."""
        vec1 = AssertionVector(evidence_source="keyword")
        vec2 = AssertionVector(evidence_source="dependency")
        combined = vec1.combine_with(vec2)
        assert combined.evidence_source == "combined"


class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_affirmed_vector(self):
        """affirmed_vector should create affirmed assertion."""
        vec = affirmed_vector(confidence=0.9)
        assert vec.to_status() == AssertionStatus.AFFIRMED
        assert vec.evidence_confidence == 0.9

    def test_negated_vector(self):
        """negated_vector should create negated assertion."""
        vec = negated_vector(confidence=0.85)
        assert vec.to_status() == AssertionStatus.NEGATED
        assert vec.negation_score == 0.85

    def test_uncertain_vector(self):
        """uncertain_vector should create uncertain assertion."""
        vec = uncertain_vector(confidence=0.75)
        assert vec.to_status() == AssertionStatus.UNCERTAIN
        assert vec.uncertainty_score == 0.75

    def test_normal_vector(self):
        """normal_vector should create normal assertion."""
        vec = normal_vector(confidence=0.8)
        assert vec.to_status() == AssertionStatus.NORMAL
        assert vec.normality_score == 0.8

    def test_default_confidence(self):
        """Factory functions should default to confidence=1.0."""
        assert negated_vector().negation_score == 1.0
        assert uncertain_vector().uncertainty_score == 1.0
        assert normal_vector().normality_score == 1.0
