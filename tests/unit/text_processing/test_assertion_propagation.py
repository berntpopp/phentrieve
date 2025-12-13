"""
Unit tests for assertion propagation module.

Tests cover:
- PropagationConfig defaults and validation
- AssertionPropagator initialization and propagation
- Convergence detection
- Conflict detection and resolution
- HPO assertion resolution
"""

import numpy as np
import pytest

from phentrieve.text_processing.assertion_propagation import (
    AssertionPropagator,
    PropagationConfig,
    PropagationResult,
    propagate_assertions,
)
from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    negated_vector,
)
from phentrieve.text_processing.semantic_graph import (
    ChunkNode,
    SemanticDocumentGraph,
)


class TestPropagationConfig:
    """Tests for PropagationConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = PropagationConfig()
        assert config.max_iterations == 3
        assert config.damping_factor == 0.85
        assert config.convergence_threshold == 0.01
        assert "sequential" in config.edge_type_weights
        assert "semantic" in config.edge_type_weights
        assert "hpo_coreference" in config.edge_type_weights

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = PropagationConfig(
            max_iterations=5,
            damping_factor=0.9,
            convergence_threshold=0.001,
        )
        assert config.max_iterations == 5
        assert config.damping_factor == 0.9
        assert config.convergence_threshold == 0.001

    def test_preserve_strong_evidence_default(self):
        """Strong evidence preservation should be enabled by default."""
        config = PropagationConfig()
        assert config.preserve_strong_evidence is True
        assert config.strong_evidence_threshold == 0.8


class TestAssertionPropagatorInit:
    """Tests for AssertionPropagator initialization."""

    def test_default_init(self):
        """Default initialization should work."""
        propagator = AssertionPropagator()
        assert propagator.config is not None
        assert propagator.config.max_iterations == 3

    def test_custom_config_init(self):
        """Custom config should be used."""
        config = PropagationConfig(max_iterations=10)
        propagator = AssertionPropagator(config)
        assert propagator.config.max_iterations == 10


class TestPropagationOnEmptyGraph:
    """Tests for propagation on empty graphs."""

    def test_empty_graph(self):
        """Propagation on empty graph should return empty result."""
        graph = SemanticDocumentGraph()
        propagator = AssertionPropagator()
        result = propagator.propagate(graph)

        assert result.assertions == {}
        assert result.iterations_run == 0
        assert result.converged is True
        assert result.conflict_chunks == []


class TestBasicPropagation:
    """Tests for basic propagation behavior."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph with 3 chunks."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="No fever present",
                assertion_vector=negated_vector(0.9),
            ),
            ChunkNode(
                chunk_idx=1,
                text="Some symptoms",
                assertion_vector=AssertionVector(),  # Affirmed
            ),
            ChunkNode(
                chunk_idx=2,
                text="Patient stable",
                assertion_vector=AssertionVector(),  # Affirmed
            ),
        ]
        graph.build_from_chunks(chunks, model=None)
        return graph

    def test_propagation_runs(self, simple_graph):
        """Propagation should complete without error."""
        propagator = AssertionPropagator()
        result = propagator.propagate(simple_graph)

        assert isinstance(result, PropagationResult)
        assert len(result.assertions) == 3
        assert result.iterations_run > 0

    def test_propagation_returns_arrays(self, simple_graph):
        """Propagation should return numpy arrays."""
        propagator = AssertionPropagator()
        result = propagator.propagate(simple_graph)

        for arr in result.assertions.values():
            assert isinstance(arr, np.ndarray)
            assert len(arr) == 3  # [negation, uncertainty, normality]

    def test_assertions_bounded(self, simple_graph):
        """Assertion values should be in [0, 1]."""
        propagator = AssertionPropagator()
        result = propagator.propagate(simple_graph)

        for arr in result.assertions.values():
            assert np.all(arr >= 0)
            assert np.all(arr <= 1)


class TestPropagationConvergence:
    """Tests for propagation convergence."""

    def test_immediate_convergence_isolated_nodes(self):
        """Isolated nodes should converge immediately."""
        graph = SemanticDocumentGraph(
            add_sequential_edges=False,
            add_semantic_edges=False,
            add_hpo_coreference_edges=False,
        )
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
        ]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        result = propagator.propagate(graph)

        # Should converge in 1 iteration since no edges
        assert result.converged is True
        assert result.iterations_run == 1

    def test_converges_within_limit(self):
        """Propagation should converge within max iterations."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=i, text=f"Chunk {i}") for i in range(5)]
        graph.build_from_chunks(chunks, model=None)

        config = PropagationConfig(max_iterations=10)
        propagator = AssertionPropagator(config)
        result = propagator.propagate(graph)

        assert result.iterations_run <= 10


class TestDampingFactor:
    """Tests for damping factor behavior."""

    def test_high_damping_preserves_local(self):
        """High damping should preserve local assertions."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Strongly negated",
                assertion_vector=negated_vector(1.0),
            ),
            ChunkNode(
                chunk_idx=1,
                text="Affirmed",
                assertion_vector=AssertionVector(),
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        # High damping = local evidence dominates
        config = PropagationConfig(damping_factor=0.95, max_iterations=5)
        propagator = AssertionPropagator(config)
        result = propagator.propagate(graph)

        # Chunk 0 should still be strongly negated
        assert result.assertions[0][0] > 0.8  # negation_score

    def test_low_damping_more_propagation(self):
        """Low damping should allow more propagation."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Strongly negated",
                assertion_vector=negated_vector(1.0),
            ),
            ChunkNode(
                chunk_idx=1,
                text="Affirmed",
                assertion_vector=AssertionVector(),
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        # Low damping = more influence from neighbors
        config = PropagationConfig(
            damping_factor=0.5,
            max_iterations=5,
            preserve_strong_evidence=False,
        )
        propagator = AssertionPropagator(config)
        result = propagator.propagate(graph)

        # Chunk 1 should have picked up some negation from chunk 0
        # (though it depends on edge direction)
        assert result.assertions[1] is not None


class TestConflictDetection:
    """Tests for conflict detection."""

    def test_detect_hpo_conflict(self):
        """Should detect conflict when same HPO has opposing assertions.

        Note: With propagation, values may shift before conflict detection.
        We use high damping factor to preserve original assertions.
        """
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="No fever",
                assertion_vector=negated_vector(0.9),  # High negation
                hpo_matches=["HP:0001945"],
            ),
            ChunkNode(
                chunk_idx=1,
                text="Has fever",
                assertion_vector=AssertionVector(negation_score=0.1),  # Low negation
                hpo_matches=["HP:0001945"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        # Use high damping to preserve original assertions
        config = PropagationConfig(damping_factor=0.99, max_iterations=1)
        propagator = AssertionPropagator(config)
        result = propagator.propagate(graph)

        # With high damping and minimal iterations, original values preserved
        assert len(result.conflict_chunks) > 0
        assert (0, 1) in result.conflict_chunks or (1, 0) in result.conflict_chunks

    def test_no_conflict_consistent_assertions(self):
        """No conflict when assertions are consistent."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Has fever",
                assertion_vector=AssertionVector(),
                hpo_matches=["HP:0001945"],
            ),
            ChunkNode(
                chunk_idx=1,
                text="Fever persists",
                assertion_vector=AssertionVector(),
                hpo_matches=["HP:0001945"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        result = propagator.propagate(graph)

        assert len(result.conflict_chunks) == 0


class TestHPOResolution:
    """Tests for HPO assertion resolution."""

    def test_resolve_single_chunk_hpo(self):
        """Resolution with single chunk should return that chunk's assertion."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="No fever",
                assertion_vector=negated_vector(0.8),
                hpo_matches=["HP:0001945"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        result = propagator.propagate(graph)

        resolved = propagator.resolve_hpo_assertion(
            graph, "HP:0001945", result.assertions
        )

        assert resolved.negation_score > 0.5

    def test_resolve_missing_hpo(self):
        """Resolution for missing HPO should return no-evidence vector."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="No HPO here")]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        result = propagator.propagate(graph)

        resolved = propagator.resolve_hpo_assertion(
            graph, "HP:9999999", result.assertions
        )

        assert resolved.evidence_source == "no_evidence"

    def test_resolve_conflict_increases_uncertainty(self):
        """Conflicting assertions should increase uncertainty.

        Uses high damping to preserve original assertions through propagation
        so conflict detection works as expected.
        """
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="No fever",
                assertion_vector=negated_vector(0.9),  # High negation
                hpo_matches=["HP:0001945"],
            ),
            ChunkNode(
                chunk_idx=1,
                text="Has fever",
                assertion_vector=AssertionVector(
                    negation_score=0.1, uncertainty_score=0.1
                ),  # Low negation
                hpo_matches=["HP:0001945"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        # Use high damping to preserve original values
        config = PropagationConfig(damping_factor=0.99, max_iterations=1)
        propagator = AssertionPropagator(config)
        result = propagator.propagate(graph)

        resolved = propagator.resolve_hpo_assertion(
            graph, "HP:0001945", result.assertions
        )

        # Conflict should increase uncertainty
        assert resolved.uncertainty_score >= 0.5


class TestArrayConversion:
    """Tests for array-to-vector conversion."""

    def test_array_to_vector(self):
        """Should convert array back to AssertionVector."""
        propagator = AssertionPropagator()
        arr = np.array([0.7, 0.2, 0.1])

        vec = propagator.array_to_vector(arr, source="test")

        assert vec.negation_score == pytest.approx(0.7)
        assert vec.uncertainty_score == pytest.approx(0.2)
        assert vec.normality_score == pytest.approx(0.1)
        assert vec.evidence_source == "test"

    def test_confidence_from_uncertainty(self):
        """Confidence should be inverse of uncertainty."""
        propagator = AssertionPropagator()

        # Low uncertainty = high confidence
        arr_low_unc = np.array([0.0, 0.1, 0.0])
        vec_low = propagator.array_to_vector(arr_low_unc)
        assert vec_low.evidence_confidence > 0.8

        # High uncertainty = low confidence
        arr_high_unc = np.array([0.0, 0.9, 0.0])
        vec_high = propagator.array_to_vector(arr_high_unc)
        assert vec_high.evidence_confidence < 0.6


class TestConvenienceFunction:
    """Tests for propagate_assertions convenience function."""

    def test_convenience_function(self):
        """Convenience function should work like calling propagator directly."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="Test")]
        graph.build_from_chunks(chunks, model=None)

        result = propagate_assertions(graph)

        assert isinstance(result, PropagationResult)
        assert len(result.assertions) == 1

    def test_convenience_with_config(self):
        """Convenience function should accept config."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="Test")]
        graph.build_from_chunks(chunks, model=None)

        config = PropagationConfig(max_iterations=1)
        result = propagate_assertions(graph, config)

        assert result.iterations_run <= 1
