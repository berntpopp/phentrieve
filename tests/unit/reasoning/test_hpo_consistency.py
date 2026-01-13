"""
Unit tests for HPO consistency checking module.

Tests cover:
- ConsistencyViolation dataclass
- HPOConsistencyChecker initialization
- Ancestor conflict detection
- Redundancy detection
- Violation resolution
- Hierarchy propagation
"""

import pytest

from phentrieve.reasoning.hpo_consistency import (
    ConsistencyCheckResult,
    ConsistencyConfig,
    ConsistencyViolation,
    HPOConsistencyChecker,
    ViolationSeverity,
    ViolationType,
    check_hpo_consistency,
)
from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    negated_vector,
)


class TestConsistencyViolation:
    """Tests for ConsistencyViolation dataclass."""

    def test_construction(self):
        """Violation should be constructable."""
        violation = ConsistencyViolation(
            violation_type=ViolationType.ANCESTOR_CONFLICT,
            severity=ViolationSeverity.ERROR,
            hpo_id_primary="HP:0001234",
            hpo_id_secondary="HP:0000001",
            description="Test conflict",
        )
        assert violation.violation_type == ViolationType.ANCESTOR_CONFLICT
        assert violation.severity == ViolationSeverity.ERROR
        assert violation.hpo_id_primary == "HP:0001234"
        assert violation.hpo_id_secondary == "HP:0000001"

    def test_default_values(self):
        """Default values should be set."""
        violation = ConsistencyViolation(
            violation_type=ViolationType.REDUNDANT_ANCESTOR,
            severity=ViolationSeverity.INFO,
            hpo_id_primary="HP:0001234",
            hpo_id_secondary=None,
            description="Test",
        )
        assert violation.confidence == 1.0
        assert violation.suggested_resolution is None


class TestConsistencyCheckResult:
    """Tests for ConsistencyCheckResult dataclass."""

    def test_empty_result_is_consistent(self):
        """Empty violations list should be consistent."""
        result = ConsistencyCheckResult(violations=[])
        assert result.is_consistent is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_warning_still_consistent(self):
        """Warnings should not affect is_consistent."""
        result = ConsistencyCheckResult(
            violations=[
                ConsistencyViolation(
                    violation_type=ViolationType.REDUNDANT_ANCESTOR,
                    severity=ViolationSeverity.WARNING,
                    hpo_id_primary="HP:0001234",
                    hpo_id_secondary=None,
                    description="Warning",
                )
            ]
        )
        assert result.is_consistent is True
        assert result.warning_count == 1

    def test_error_not_consistent(self):
        """Errors should make result inconsistent."""
        result = ConsistencyCheckResult(
            violations=[
                ConsistencyViolation(
                    violation_type=ViolationType.ANCESTOR_CONFLICT,
                    severity=ViolationSeverity.ERROR,
                    hpo_id_primary="HP:0001234",
                    hpo_id_secondary="HP:0000001",
                    description="Error",
                )
            ]
        )
        assert result.is_consistent is False
        assert result.error_count == 1


class TestHPOConsistencyCheckerInit:
    """Tests for HPOConsistencyChecker initialization."""

    def test_basic_init(self):
        """Basic initialization should work."""
        ancestors = {"HP:0001234": {"HP:0000001"}}
        depths = {"HP:0001234": 3, "HP:0000001": 1}

        checker = HPOConsistencyChecker(ancestors, depths)

        assert checker.ancestors_map == ancestors
        assert checker.depths_map == depths

    def test_descendants_map_built(self):
        """Descendants map should be built from ancestors."""
        ancestors = {
            "HP:0001234": {"HP:0000001", "HP:0000002"},
            "HP:0005678": {"HP:0000001"},
        }
        depths = {}

        checker = HPOConsistencyChecker(ancestors, depths)

        # HP:0000001 should have both as descendants
        assert "HP:0001234" in checker._descendants_map.get("HP:0000001", set())
        assert "HP:0005678" in checker._descendants_map.get("HP:0000001", set())


class TestAncestorConflictDetection:
    """Tests for ancestor conflict detection."""

    @pytest.fixture
    def simple_hierarchy(self):
        """Create a simple HPO hierarchy."""
        # HP:0001945 (Fever) is a descendant of HP:0000001 (Phenotypic abnormality)
        ancestors = {"HP:0001945": {"HP:0000001", "HP:0001939"}}
        depths = {"HP:0001945": 5, "HP:0001939": 3, "HP:0000001": 1}
        return ancestors, depths

    def test_no_conflict_both_affirmed(self, simple_hierarchy):
        """No conflict when both terms affirmed."""
        ancestors, depths = simple_hierarchy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(),  # Affirmed
            "HP:0001939": AssertionVector(),  # Affirmed
        }

        result = checker.check_consistency(assertions)
        conflicts = [
            v
            for v in result.violations
            if v.violation_type == ViolationType.ANCESTOR_CONFLICT
        ]
        assert len(conflicts) == 0

    def test_conflict_child_affirmed_ancestor_negated(self, simple_hierarchy):
        """Conflict when child affirmed but ancestor negated."""
        ancestors, depths = simple_hierarchy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(),  # Affirmed (Fever)
            "HP:0001939": negated_vector(0.9),  # Negated (ancestor)
        }

        result = checker.check_consistency(assertions)
        conflicts = [
            v
            for v in result.violations
            if v.violation_type == ViolationType.ANCESTOR_CONFLICT
        ]
        assert len(conflicts) == 1
        assert conflicts[0].severity == ViolationSeverity.ERROR

    def test_no_conflict_both_negated(self, simple_hierarchy):
        """No conflict when both terms negated."""
        ancestors, depths = simple_hierarchy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": negated_vector(0.9),  # Negated
            "HP:0001939": negated_vector(0.8),  # Negated
        }

        result = checker.check_consistency(assertions)
        conflicts = [
            v
            for v in result.violations
            if v.violation_type == ViolationType.ANCESTOR_CONFLICT
        ]
        assert len(conflicts) == 0

    def test_no_conflict_uncertain_child(self, simple_hierarchy):
        """No conflict when child is uncertain."""
        ancestors, depths = simple_hierarchy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(uncertainty_score=0.8),  # Uncertain
            "HP:0001939": negated_vector(0.9),  # Negated
        }

        result = checker.check_consistency(assertions)
        conflicts = [
            v
            for v in result.violations
            if v.violation_type == ViolationType.ANCESTOR_CONFLICT
        ]
        assert len(conflicts) == 0


class TestRedundancyDetection:
    """Tests for redundancy detection."""

    @pytest.fixture
    def hierarchy_with_redundancy(self):
        """Create hierarchy where child makes ancestor redundant."""
        ancestors = {"HP:0001945": {"HP:0001939", "HP:0000001"}}
        depths = {"HP:0001945": 5, "HP:0001939": 3, "HP:0000001": 1}
        return ancestors, depths

    def test_detect_redundant_ancestor(self, hierarchy_with_redundancy):
        """Should detect when ancestor is redundant."""
        ancestors, depths = hierarchy_with_redundancy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(),  # Specific term
            "HP:0001939": AssertionVector(),  # More general ancestor
        }

        result = checker.check_consistency(assertions)
        redundancies = [
            v
            for v in result.violations
            if v.violation_type == ViolationType.REDUNDANT_ANCESTOR
        ]
        assert len(redundancies) == 1
        assert redundancies[0].hpo_id_primary == "HP:0001939"  # Redundant one
        assert redundancies[0].hpo_id_secondary == "HP:0001945"  # More specific

    def test_skip_root_terms(self, hierarchy_with_redundancy):
        """Should not flag root terms as redundant."""
        ancestors, depths = hierarchy_with_redundancy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(),
            "HP:0000001": AssertionVector(),  # Root term (depth 1)
        }

        result = checker.check_consistency(assertions)
        redundancies = [
            v
            for v in result.violations
            if v.violation_type == ViolationType.REDUNDANT_ANCESTOR
            and v.hpo_id_primary == "HP:0000001"
        ]
        assert len(redundancies) == 0


class TestViolationResolution:
    """Tests for violation resolution."""

    @pytest.fixture
    def conflict_setup(self):
        """Create a conflict scenario."""
        ancestors = {"HP:0001945": {"HP:0001939"}}
        depths = {"HP:0001945": 5, "HP:0001939": 3}
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(),  # Affirmed
            "HP:0001939": negated_vector(0.9),  # Negated ancestor
        }

        result = checker.check_consistency(assertions)
        return checker, assertions, result

    def test_conservative_resolution_adds_uncertainty(self, conflict_setup):
        """Conservative resolution should add uncertainty."""
        checker, assertions, result = conflict_setup

        resolved = checker.resolve_violations(
            assertions, result, resolution_strategy="conservative"
        )

        # Both terms should have increased uncertainty
        assert resolved["HP:0001945"].uncertainty_score >= 0.4
        assert resolved["HP:0001939"].uncertainty_score >= 0.4

    def test_resolution_reduces_confidence(self, conflict_setup):
        """Resolution should reduce evidence confidence."""
        checker, assertions, result = conflict_setup

        resolved = checker.resolve_violations(assertions, result)

        original_conf = assertions["HP:0001939"].evidence_confidence
        resolved_conf = resolved["HP:0001939"].evidence_confidence
        assert resolved_conf < original_conf


class TestHierarchyPropagation:
    """Tests for hierarchy propagation."""

    @pytest.fixture
    def propagation_hierarchy(self):
        """Create hierarchy for propagation testing."""
        ancestors = {"HP:0001945": {"HP:0001939", "HP:0000001"}}
        depths = {"HP:0001945": 5, "HP:0001939": 3, "HP:0000001": 1}
        return ancestors, depths

    def test_upward_propagation(self, propagation_hierarchy):
        """Affirmed child should imply possible ancestors."""
        ancestors, depths = propagation_hierarchy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {"HP:0001945": AssertionVector()}  # Only child affirmed

        propagated = checker.propagate_through_hierarchy(
            assertions, propagation_mode="upward"
        )

        # Ancestors should now have weak evidence
        assert "HP:0001939" in propagated
        assert "HP:0000001" in propagated
        assert propagated["HP:0001939"].evidence_confidence < 0.5

    def test_downward_propagation(self, propagation_hierarchy):
        """Negated ancestor should propagate to descendants."""
        ancestors, depths = propagation_hierarchy
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {"HP:0001939": negated_vector(0.9)}  # Ancestor negated

        propagated = checker.propagate_through_hierarchy(
            assertions, propagation_mode="downward"
        )

        # Descendant should inherit negation
        assert "HP:0001945" in propagated
        assert propagated["HP:0001945"].negation_score > 0.5


class TestMostSpecificTerms:
    """Tests for most specific term selection."""

    def test_get_most_specific(self):
        """Should return only most specific terms."""
        ancestors = {
            "HP:0001945": {"HP:0001939", "HP:0000001"},
            "HP:0001939": {"HP:0000001"},
        }
        depths = {"HP:0001945": 5, "HP:0001939": 3, "HP:0000001": 1}
        checker = HPOConsistencyChecker(ancestors, depths)

        assertions = {
            "HP:0001945": AssertionVector(),  # Most specific
            "HP:0001939": AssertionVector(),  # Less specific
            "HP:0000001": AssertionVector(),  # Root
        }

        most_specific = checker.get_most_specific_terms(assertions)

        assert "HP:0001945" in most_specific
        assert "HP:0001939" not in most_specific
        assert "HP:0000001" not in most_specific


class TestTermSpecificity:
    """Tests for term specificity computation."""

    def test_compute_specificity(self):
        """Deeper terms should have higher specificity."""
        ancestors = {}
        depths = {"HP:0001945": 8, "HP:0001939": 3, "HP:0000001": 1}
        checker = HPOConsistencyChecker(ancestors, depths)

        spec_deep = checker.compute_term_specificity("HP:0001945")
        spec_mid = checker.compute_term_specificity("HP:0001939")
        spec_shallow = checker.compute_term_specificity("HP:0000001")

        assert spec_deep > spec_mid > spec_shallow
        assert 0.0 <= spec_deep <= 1.0


class TestConvenienceFunction:
    """Tests for check_hpo_consistency convenience function."""

    def test_convenience_function(self):
        """Convenience function should work."""
        ancestors = {"HP:0001234": {"HP:0000001"}}
        depths = {"HP:0001234": 3, "HP:0000001": 1}
        assertions = {"HP:0001234": AssertionVector()}

        result = check_hpo_consistency(assertions, ancestors, depths)

        assert isinstance(result, ConsistencyCheckResult)

    def test_convenience_with_config(self):
        """Convenience function should accept config."""
        ancestors = {"HP:0001234": {"HP:0000001"}}
        depths = {"HP:0001234": 3, "HP:0000001": 1}
        assertions = {"HP:0001234": AssertionVector()}
        config = ConsistencyConfig(check_ancestor_conflicts=True)

        result = check_hpo_consistency(assertions, ancestors, depths, config)

        assert isinstance(result, ConsistencyCheckResult)
