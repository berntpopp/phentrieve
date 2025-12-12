"""
Unit tests for hybrid inference engine.

Tests cover:
- HybridInferenceConfig validation
- HybridInferenceEngine initialization
- Complete inference pipeline
- Provenance tracking
- Method comparison
"""

import pytest
import numpy as np

from phentrieve.reasoning.hybrid_inference import (
    EvidenceSource,
    FinalAssertion,
    HybridInferenceConfig,
    HybridInferenceEngine,
    HybridInferenceResult,
    MethodComparisonResult,
    compare_methods,
    create_inference_engine,
)
from phentrieve.reasoning.hpo_consistency import HPOConsistencyChecker
from phentrieve.text_processing.assertion_propagation import AssertionPropagator
from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    negated_vector,
)
from phentrieve.text_processing.semantic_graph import (
    ChunkNode,
    SemanticDocumentGraph,
)


class TestHybridInferenceConfig:
    """Tests for HybridInferenceConfig."""

    def test_default_config(self):
        """Default config should have valid weights."""
        config = HybridInferenceConfig()
        assert config.text_graph_weight == 0.6
        assert config.ontology_weight == 0.4
        total = config.text_graph_weight + config.ontology_weight
        assert abs(total - 1.0) < 0.01

    def test_weight_normalization(self):
        """Weights should be normalized if they don't sum to 1."""
        config = HybridInferenceConfig(
            text_graph_weight=0.3,
            ontology_weight=0.3,
        )
        total = config.text_graph_weight + config.ontology_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_config(self):
        """Custom config should be accepted."""
        config = HybridInferenceConfig(
            text_graph_weight=0.8,
            ontology_weight=0.2,
            consistency_penalty=0.3,
            min_confidence_threshold=0.2,
        )
        assert config.consistency_penalty == 0.3
        assert config.min_confidence_threshold == 0.2


class TestEvidenceSource:
    """Tests for EvidenceSource dataclass."""

    def test_construction(self):
        """EvidenceSource should be constructable."""
        evidence = EvidenceSource(
            source_type="text_graph",
            chunk_indices=[0, 1, 2],
        )
        assert evidence.source_type == "text_graph"
        assert evidence.chunk_indices == [0, 1, 2]
        assert evidence.ontology_inference is False


class TestFinalAssertion:
    """Tests for FinalAssertion dataclass."""

    def test_is_affirmed(self):
        """is_affirmed should correctly identify affirmed assertions."""
        affirmed = FinalAssertion(
            hpo_id="HP:0001945",
            assertion_vector=AssertionVector(),
            legacy_status="affirmed",
        )
        assert affirmed.is_affirmed is True

        negated = FinalAssertion(
            hpo_id="HP:0001945",
            assertion_vector=negated_vector(0.9),
            legacy_status="negated",
        )
        assert negated.is_affirmed is False

        uncertain = FinalAssertion(
            hpo_id="HP:0001945",
            assertion_vector=AssertionVector(uncertainty_score=0.8),
            legacy_status="uncertain",
        )
        assert uncertain.is_affirmed is False

    def test_confidence(self):
        """confidence property should return evidence_confidence."""
        assertion = FinalAssertion(
            hpo_id="HP:0001945",
            assertion_vector=AssertionVector(evidence_confidence=0.85),
            legacy_status="affirmed",
        )
        assert assertion.confidence == 0.85


class TestHybridInferenceResult:
    """Tests for HybridInferenceResult."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample inference result."""
        assertions = {
            "HP:0001945": FinalAssertion(
                hpo_id="HP:0001945",
                assertion_vector=AssertionVector(),
                legacy_status="affirmed",
            ),
            "HP:0002315": FinalAssertion(
                hpo_id="HP:0002315",
                assertion_vector=negated_vector(0.9),
                legacy_status="negated",
            ),
            "HP:0001251": FinalAssertion(
                hpo_id="HP:0001251",
                assertion_vector=AssertionVector(uncertainty_score=0.8),
                legacy_status="uncertain",
            ),
        }
        return HybridInferenceResult(assertions=assertions)

    def test_affirmed_terms(self, sample_result):
        """Should return list of affirmed terms."""
        affirmed = sample_result.affirmed_terms
        assert "HP:0001945" in affirmed
        assert "HP:0002315" not in affirmed
        assert "HP:0001251" not in affirmed

    def test_negated_terms(self, sample_result):
        """Should return list of negated terms."""
        negated = sample_result.negated_terms
        assert "HP:0002315" in negated
        assert "HP:0001945" not in negated

    def test_uncertain_terms(self, sample_result):
        """Should return list of uncertain terms."""
        uncertain = sample_result.uncertain_terms
        assert "HP:0001251" in uncertain
        assert "HP:0001945" not in uncertain


class TestHybridInferenceEngineInit:
    """Tests for HybridInferenceEngine initialization."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple graph and propagator."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Patient has fever",
                assertion_vector=AssertionVector(),
                hpo_matches=["HP:0001945"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)
        propagator = AssertionPropagator()
        return graph, propagator

    def test_basic_init(self, simple_setup):
        """Basic initialization should work."""
        graph, propagator = simple_setup
        engine = HybridInferenceEngine(
            text_graph=graph,
            propagator=propagator,
        )
        assert engine.text_graph is graph
        assert engine.propagator is propagator
        assert engine.consistency_checker is None

    def test_init_with_consistency_checker(self, simple_setup):
        """Init with consistency checker should work."""
        graph, propagator = simple_setup
        checker = HPOConsistencyChecker(
            ancestors_map={"HP:0001945": {"HP:0000001"}},
            depths_map={"HP:0001945": 3, "HP:0000001": 1},
        )

        engine = HybridInferenceEngine(
            text_graph=graph,
            propagator=propagator,
            consistency_checker=checker,
        )

        assert engine.consistency_checker is checker


class TestHybridInference:
    """Tests for hybrid inference pipeline."""

    @pytest.fixture
    def inference_setup(self):
        """Create complete inference setup."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Patient has fever",
                assertion_vector=AssertionVector(),
                hpo_matches=["HP:0001945"],
            ),
            ChunkNode(
                chunk_idx=1,
                text="No headache noted",
                assertion_vector=negated_vector(0.8),
                hpo_matches=["HP:0002315"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        engine = HybridInferenceEngine(
            text_graph=graph,
            propagator=propagator,
        )

        hpo_matches = {
            "HP:0001945": [0],
            "HP:0002315": [1],
        }

        return engine, hpo_matches

    def test_inference_returns_result(self, inference_setup):
        """Inference should return HybridInferenceResult."""
        engine, hpo_matches = inference_setup
        result = engine.infer(hpo_matches)

        assert isinstance(result, HybridInferenceResult)
        assert len(result.assertions) == 2

    def test_inference_preserves_assertions(self, inference_setup):
        """Inference should preserve basic assertion status."""
        engine, hpo_matches = inference_setup
        result = engine.infer(hpo_matches)

        assert "HP:0001945" in result.assertions
        assert "HP:0002315" in result.assertions

        # Fever was affirmed
        fever = result.assertions["HP:0001945"]
        assert fever.legacy_status == "affirmed"

        # Headache was negated
        headache = result.assertions["HP:0002315"]
        assert headache.legacy_status == "negated"

    def test_inference_includes_timing(self, inference_setup):
        """Inference should track timing."""
        engine, hpo_matches = inference_setup
        result = engine.infer(hpo_matches)

        assert result.timing_ms > 0

    def test_inference_with_provenance(self, inference_setup):
        """Inference should include provenance when configured."""
        engine, hpo_matches = inference_setup
        engine.config.include_provenance = True
        result = engine.infer(hpo_matches)

        fever = result.assertions["HP:0001945"]
        assert fever.provenance is not None
        assert "chunks" in fever.provenance


class TestInferenceWithOntology:
    """Tests for inference with ontology consistency checking."""

    @pytest.fixture
    def ontology_setup(self):
        """Create setup with ontology checker."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Patient has fever",
                assertion_vector=AssertionVector(),
                hpo_matches=["HP:0001945"],  # Fever (child)
            ),
            ChunkNode(
                chunk_idx=1,
                text="No symptoms",
                assertion_vector=negated_vector(0.9),
                hpo_matches=["HP:0001939"],  # Body temp (ancestor)
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        checker = HPOConsistencyChecker(
            ancestors_map={"HP:0001945": {"HP:0001939", "HP:0000001"}},
            depths_map={"HP:0001945": 5, "HP:0001939": 3, "HP:0000001": 1},
        )

        engine = HybridInferenceEngine(
            text_graph=graph,
            propagator=propagator,
            consistency_checker=checker,
        )

        hpo_matches = {
            "HP:0001945": [0],
            "HP:0001939": [1],
        }

        return engine, hpo_matches

    def test_detects_consistency_violations(self, ontology_setup):
        """Should detect ancestor-child conflict."""
        engine, hpo_matches = ontology_setup
        result = engine.infer(hpo_matches)

        assert result.consistency_result is not None
        # Should have detected conflict
        assert not result.consistency_result.is_consistent

    def test_resolves_conflicts_when_configured(self, ontology_setup):
        """Should resolve conflicts when configured."""
        engine, hpo_matches = ontology_setup
        engine.config.resolve_conflicts = True
        result = engine.infer(hpo_matches)

        # Assertions should have increased uncertainty
        fever = result.assertions["HP:0001945"]
        assert fever.assertion_vector.uncertainty_score > 0


class TestInferenceComparison:
    """Tests for comparing hybrid with legacy methods."""

    @pytest.fixture
    def comparison_setup(self):
        """Create setup for comparison testing."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Has fever",
                assertion_vector=AssertionVector(),
                hpo_matches=["HP:0001945"],
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        propagator = AssertionPropagator()
        engine = HybridInferenceEngine(
            text_graph=graph,
            propagator=propagator,
        )

        return engine

    def test_infer_with_comparison(self, comparison_setup):
        """Should compare with legacy assertions."""
        engine = comparison_setup
        hpo_matches = {"HP:0001945": [0]}
        legacy = {"HP:0001945": "affirmed"}

        result, comparison = engine.infer_with_comparison(hpo_matches, legacy)

        assert isinstance(result, HybridInferenceResult)
        assert "HP:0001945" in comparison
        assert comparison["HP:0001945"]["agreement"] is True


class TestMethodComparison:
    """Tests for compare_methods function."""

    def test_full_agreement(self):
        """Should report 100% agreement when identical."""
        assertions = {
            "HP:0001": FinalAssertion(
                hpo_id="HP:0001",
                assertion_vector=AssertionVector(),
                legacy_status="affirmed",
            ),
        }
        result = HybridInferenceResult(assertions=assertions)
        legacy = {"HP:0001": "affirmed"}

        comparison = compare_methods(result, legacy)

        assert comparison.agreement_rate == 1.0
        assert comparison.disagreement_count == 0

    def test_disagreement(self):
        """Should detect disagreements."""
        assertions = {
            "HP:0001": FinalAssertion(
                hpo_id="HP:0001",
                assertion_vector=negated_vector(0.9),
                legacy_status="negated",
            ),
        }
        result = HybridInferenceResult(assertions=assertions)
        legacy = {"HP:0001": "affirmed"}

        comparison = compare_methods(result, legacy)

        assert comparison.disagreement_count == 1
        assert len(comparison.disagreements) == 1
        assert comparison.disagreements[0]["hpo_id"] == "HP:0001"

    def test_missing_terms_counted(self):
        """Missing terms should be counted as disagreements."""
        assertions = {
            "HP:0001": FinalAssertion(
                hpo_id="HP:0001",
                assertion_vector=AssertionVector(),
                legacy_status="affirmed",
            ),
        }
        result = HybridInferenceResult(assertions=assertions)
        legacy = {"HP:0002": "affirmed"}  # Different term

        comparison = compare_methods(result, legacy)

        assert comparison.total_terms == 2
        assert comparison.disagreement_count == 2

    def test_summary_format(self):
        """Summary should be human-readable."""
        result = MethodComparisonResult(
            total_terms=10,
            agreement_count=8,
            disagreement_count=2,
            agreement_rate=0.8,
        )
        assert "8/10" in result.summary
        assert "80" in result.summary


class TestCreateInferenceEngine:
    """Tests for create_inference_engine factory function."""

    def test_create_without_ontology(self):
        """Should create engine without ontology checker."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="Test")]
        graph.build_from_chunks(chunks, model=None)

        engine = create_inference_engine(graph)

        assert isinstance(engine, HybridInferenceEngine)
        assert engine.consistency_checker is None

    def test_create_with_ontology(self):
        """Should create engine with ontology checker."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="Test")]
        graph.build_from_chunks(chunks, model=None)

        ancestors = {"HP:0001": {"HP:0000"}}
        depths = {"HP:0001": 2, "HP:0000": 1}

        engine = create_inference_engine(
            graph,
            ancestors_map=ancestors,
            depths_map=depths,
        )

        assert engine.consistency_checker is not None

    def test_create_with_config(self):
        """Should accept custom config."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="Test")]
        graph.build_from_chunks(chunks, model=None)

        config = HybridInferenceConfig(min_confidence_threshold=0.5)
        engine = create_inference_engine(graph, config=config)

        assert engine.config.min_confidence_threshold == 0.5
