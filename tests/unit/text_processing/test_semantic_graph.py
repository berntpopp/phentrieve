"""
Unit tests for SemanticDocumentGraph.

Tests cover:
- ChunkNode and GraphEdge dataclasses
- Graph construction and node/edge management
- Edge creation (sequential, semantic, HPO coreference)
- Graph traversal and neighborhood queries
- Serialization
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from phentrieve.text_processing.semantic_graph import (
    ChunkNode,
    GraphEdge,
    SemanticDocumentGraph,
)


class TestChunkNode:
    """Tests for ChunkNode dataclass."""

    def test_construction(self):
        """ChunkNode should be constructable with all fields."""
        embedding = np.array([0.1, 0.2, 0.3])
        node = ChunkNode(
            chunk_idx=0,
            text="The patient has no fever.",
            embedding=embedding,
            legacy_assertion_status="negated",
            hpo_matches=["HP:0001945"],
            start_char=0,
            end_char=25,
        )
        assert node.chunk_idx == 0
        assert node.text == "The patient has no fever."
        assert node.start_char == 0
        assert node.end_char == 25
        assert node.legacy_assertion_status == "negated"
        assert node.hpo_matches == ["HP:0001945"]
        assert node.embedding is not None
        assert len(node.embedding) == 3

    def test_construction_minimal(self):
        """ChunkNode should work with minimal required fields."""
        node = ChunkNode(
            chunk_idx=0,
            text="test text",
        )
        assert node.chunk_idx == 0
        assert node.legacy_assertion_status is None
        assert node.hpo_matches == []
        assert node.embedding is None
        assert node.start_char == -1
        assert node.end_char == -1

    def test_default_factory_isolation(self):
        """Default list factory should not share between instances."""
        node1 = ChunkNode(chunk_idx=0, text="t1")
        node2 = ChunkNode(chunk_idx=1, text="t2")
        node1.hpo_matches.append("HP:0001234")
        assert node2.hpo_matches == []

    def test_hash_by_chunk_idx(self):
        """Nodes with same chunk_idx should have same hash."""
        node1 = ChunkNode(chunk_idx=5, text="text1")
        node2 = ChunkNode(chunk_idx=5, text="different text")
        assert hash(node1) == hash(node2)

    def test_equality_by_chunk_idx(self):
        """Nodes with same chunk_idx should be equal."""
        node1 = ChunkNode(chunk_idx=5, text="text1")
        node2 = ChunkNode(chunk_idx=5, text="different text")
        assert node1 == node2

        node3 = ChunkNode(chunk_idx=6, text="text1")
        assert node1 != node3


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_construction(self):
        """GraphEdge should be constructable with all fields."""
        edge = GraphEdge(
            source_idx=0,
            target_idx=1,
            edge_type="sequential",
            weight=1.0,
            metadata={"key": "value"},
        )
        assert edge.source_idx == 0
        assert edge.target_idx == 1
        assert edge.edge_type == "sequential"
        assert edge.weight == 1.0
        assert edge.metadata == {"key": "value"}

    def test_construction_defaults(self):
        """GraphEdge should have sensible defaults."""
        edge = GraphEdge(
            source_idx=0,
            target_idx=1,
            edge_type="semantic",
        )
        assert edge.weight == 1.0
        assert edge.metadata == {}


class TestSemanticDocumentGraphConstruction:
    """Tests for graph construction."""

    def test_empty_graph(self):
        """Empty graph should be constructable."""
        graph = SemanticDocumentGraph()
        assert graph.num_nodes == 0
        assert graph.num_edges == 0

    def test_custom_threshold(self):
        """Graph should accept custom similarity threshold."""
        graph = SemanticDocumentGraph(similarity_threshold=0.8)
        assert graph.similarity_threshold == 0.8

    def test_add_node(self):
        """Adding a single node should increase node count."""
        graph = SemanticDocumentGraph()
        chunk = ChunkNode(chunk_idx=0, text="Test chunk")
        graph.add_node(chunk)
        assert graph.num_nodes == 1

    def test_add_multiple_nodes(self):
        """Adding multiple nodes should work correctly."""
        graph = SemanticDocumentGraph()
        for i in range(5):
            chunk = ChunkNode(chunk_idx=i, text=f"Chunk {i}")
            graph.add_node(chunk)
        assert graph.num_nodes == 5

    def test_get_node(self):
        """get_node should return correct node."""
        graph = SemanticDocumentGraph()
        chunk = ChunkNode(chunk_idx=0, text="Test chunk")
        graph.add_node(chunk)
        retrieved = graph.get_node(0)
        assert retrieved is not None
        assert retrieved.text == "Test chunk"

    def test_get_nonexistent_node(self):
        """get_node for nonexistent index should return None."""
        graph = SemanticDocumentGraph()
        assert graph.get_node(999) is None

    def test_get_nodes_iterator(self):
        """get_nodes should iterate over all nodes."""
        graph = SemanticDocumentGraph()
        for i in range(3):
            graph.add_node(ChunkNode(chunk_idx=i, text=f"Chunk {i}"))
        nodes = list(graph.get_nodes())
        assert len(nodes) == 3


class TestEdgeManagement:
    """Tests for edge management."""

    def test_add_edge(self):
        """Adding an edge should increase edge count."""
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="A"))
        graph.add_node(ChunkNode(chunk_idx=1, text="B"))

        edge = GraphEdge(source_idx=0, target_idx=1, edge_type="sequential")
        graph.add_edge(edge)

        assert graph.num_edges == 1

    def test_get_edges(self):
        """get_edges should return outgoing edges."""
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="A"))
        graph.add_node(ChunkNode(chunk_idx=1, text="B"))

        edge = GraphEdge(source_idx=0, target_idx=1, edge_type="sequential")
        graph.add_edge(edge)

        edges = graph.get_edges(0)
        assert len(edges) == 1
        assert edges[0].target_idx == 1

    def test_get_incoming_edges(self):
        """get_incoming_edges should return incoming edges."""
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="A"))
        graph.add_node(ChunkNode(chunk_idx=1, text="B"))

        edge = GraphEdge(source_idx=0, target_idx=1, edge_type="sequential")
        graph.add_edge(edge)

        incoming = graph.get_incoming_edges(1)
        assert len(incoming) == 1
        assert incoming[0].source_idx == 0

    def test_get_neighbors(self):
        """get_neighbors should yield neighbor indices."""
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="A"))
        graph.add_node(ChunkNode(chunk_idx=1, text="B"))
        graph.add_node(ChunkNode(chunk_idx=2, text="C"))

        graph.add_edge(GraphEdge(source_idx=0, target_idx=1, edge_type="seq"))
        graph.add_edge(GraphEdge(source_idx=0, target_idx=2, edge_type="seq"))

        neighbors = list(graph.get_neighbors(0))
        assert set(neighbors) == {1, 2}

    def test_get_predecessors(self):
        """get_predecessors should yield predecessor indices."""
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="A"))
        graph.add_node(ChunkNode(chunk_idx=1, text="B"))
        graph.add_node(ChunkNode(chunk_idx=2, text="C"))

        graph.add_edge(GraphEdge(source_idx=0, target_idx=2, edge_type="seq"))
        graph.add_edge(GraphEdge(source_idx=1, target_idx=2, edge_type="seq"))

        predecessors = list(graph.get_predecessors(2))
        assert set(predecessors) == {0, 1}


class TestBuildFromChunks:
    """Tests for build_from_chunks method."""

    def test_build_empty_chunks(self):
        """Building from empty list should not raise."""
        graph = SemanticDocumentGraph()
        graph.build_from_chunks([], model=None)
        assert graph.num_nodes == 0

    def test_build_sequential_edges_only(self):
        """Building with sequential edges only should work."""
        graph = SemanticDocumentGraph(
            add_sequential_edges=True,
            add_semantic_edges=False,
            add_hpo_coreference_edges=False,
        )
        chunks = [
            ChunkNode(chunk_idx=0, text="First"),
            ChunkNode(chunk_idx=1, text="Second"),
            ChunkNode(chunk_idx=2, text="Third"),
        ]
        graph.build_from_chunks(chunks, model=None)

        assert graph.num_nodes == 3
        assert graph.num_edges == 2  # 0->1, 1->2

    def test_build_hpo_coreference_edges(self):
        """Building with HPO coreference edges should connect same-HPO chunks."""
        graph = SemanticDocumentGraph(
            add_sequential_edges=False,
            add_semantic_edges=False,
            add_hpo_coreference_edges=True,
        )
        chunks = [
            ChunkNode(chunk_idx=0, text="Fever present", hpo_matches=["HP:0001945"]),
            ChunkNode(chunk_idx=1, text="Headache", hpo_matches=["HP:0002315"]),
            ChunkNode(chunk_idx=2, text="Still has fever", hpo_matches=["HP:0001945"]),
        ]
        graph.build_from_chunks(chunks, model=None)

        assert graph.num_nodes == 3
        # Chunks 0 and 2 share HP:0001945, bidirectional edges
        assert graph.num_edges == 2  # 0->2 and 2->0

    def test_build_requires_model_for_semantic(self):
        """Building with semantic edges should require model."""
        graph = SemanticDocumentGraph(
            add_sequential_edges=False,
            add_semantic_edges=True,
            add_hpo_coreference_edges=False,
        )
        chunks = [ChunkNode(chunk_idx=0, text="Test")]

        with pytest.raises(ValueError, match="SentenceTransformer model required"):
            graph.build_from_chunks(chunks, model=None)

    def test_build_with_mock_model(self):
        """Building with semantic edges using mock model."""
        graph = SemanticDocumentGraph(
            similarity_threshold=0.9,
            add_sequential_edges=False,
            add_semantic_edges=True,
            add_hpo_coreference_edges=False,
        )
        chunks = [
            ChunkNode(chunk_idx=0, text="Similar text A"),
            ChunkNode(chunk_idx=1, text="Similar text B"),
        ]

        # Mock model that returns identical embeddings
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # Identical = similarity 1.0
            ]
        )

        graph.build_from_chunks(chunks, model=mock_model)

        assert graph.num_nodes == 2
        assert graph.num_edges == 2  # Bidirectional semantic edges

    def test_build_clears_existing_graph(self):
        """build_from_chunks should clear existing graph."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)

        # First build
        chunks1 = [ChunkNode(chunk_idx=0, text="First")]
        graph.build_from_chunks(chunks1, model=None)
        assert graph.num_nodes == 1

        # Second build should replace
        chunks2 = [
            ChunkNode(chunk_idx=0, text="New first"),
            ChunkNode(chunk_idx=1, text="New second"),
        ]
        graph.build_from_chunks(chunks2, model=None)
        assert graph.num_nodes == 2


class TestNeighborhood:
    """Tests for neighborhood/BFS methods."""

    def test_get_neighborhood_radius_0(self):
        """Radius 0 with include_self should return only starting node."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
        ]
        graph.build_from_chunks(chunks, model=None)

        neighborhood = graph.get_neighborhood(0, radius=0, include_self=True)
        assert neighborhood == {0}

    def test_get_neighborhood_radius_1(self):
        """Radius 1 should include direct neighbors."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
            ChunkNode(chunk_idx=2, text="C"),
        ]
        graph.build_from_chunks(chunks, model=None)

        neighborhood = graph.get_neighborhood(0, radius=1, include_self=True)
        assert neighborhood == {0, 1}

    def test_get_neighborhood_radius_2(self):
        """Radius 2 should include 2-hop neighbors."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
            ChunkNode(chunk_idx=2, text="C"),
            ChunkNode(chunk_idx=3, text="D"),
        ]
        graph.build_from_chunks(chunks, model=None)

        neighborhood = graph.get_neighborhood(0, radius=2, include_self=True)
        assert neighborhood == {0, 1, 2}

    def test_get_neighborhood_nonexistent_node(self):
        """get_neighborhood for nonexistent node should return empty set."""
        graph = SemanticDocumentGraph()
        neighborhood = graph.get_neighborhood(999, radius=1)
        assert neighborhood == set()

    def test_get_neighborhood_exclude_self(self):
        """include_self=False should exclude starting node."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
        ]
        graph.build_from_chunks(chunks, model=None)

        neighborhood = graph.get_neighborhood(0, radius=1, include_self=False)
        assert 0 not in neighborhood
        assert 1 in neighborhood


class TestHPOQueries:
    """Tests for HPO-related queries."""

    def test_get_chunks_for_hpo(self):
        """get_chunks_for_hpo should return chunks mentioning that HPO."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="Fever", hpo_matches=["HP:0001945"]),
            ChunkNode(chunk_idx=1, text="Headache", hpo_matches=["HP:0002315"]),
            ChunkNode(chunk_idx=2, text="More fever", hpo_matches=["HP:0001945"]),
        ]
        graph.build_from_chunks(chunks, model=None)

        fever_chunks = graph.get_chunks_for_hpo("HP:0001945")
        assert set(fever_chunks) == {0, 2}

    def test_get_chunks_for_hpo_not_found(self):
        """get_chunks_for_hpo should return empty list if HPO not found."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="No HPO")]
        graph.build_from_chunks(chunks, model=None)

        result = graph.get_chunks_for_hpo("HP:9999999")
        assert result == []


class TestEdgeTypeSummary:
    """Tests for edge type summary."""

    def test_get_edge_types_summary_empty(self):
        """Empty graph should have empty summary."""
        graph = SemanticDocumentGraph()
        summary = graph.get_edge_types_summary()
        assert summary == {}

    def test_get_edge_types_summary_sequential(self):
        """Summary should count sequential edges."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
            ChunkNode(chunk_idx=2, text="C"),
        ]
        graph.build_from_chunks(chunks, model=None)

        summary = graph.get_edge_types_summary()
        assert summary.get("sequential", 0) == 2

    def test_get_edge_types_summary_mixed(self):
        """Summary should count all edge types."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="Fever", hpo_matches=["HP:0001945"]),
            ChunkNode(chunk_idx=1, text="More fever", hpo_matches=["HP:0001945"]),
        ]
        graph.build_from_chunks(chunks, model=None)

        summary = graph.get_edge_types_summary()
        assert summary.get("sequential", 0) == 1
        assert summary.get("hpo_coreference", 0) == 2  # Bidirectional


class TestSerialization:
    """Tests for graph serialization."""

    def test_to_dict_empty(self):
        """to_dict on empty graph should work."""
        graph = SemanticDocumentGraph()
        data = graph.to_dict()
        assert data["num_nodes"] == 0
        assert data["num_edges"] == 0
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_to_dict_with_nodes(self):
        """to_dict should include node data."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(
                chunk_idx=0,
                text="Test",
                hpo_matches=["HP:0001234"],
                legacy_assertion_status="affirmed",
                start_char=0,
                end_char=4,
            ),
        ]
        graph.build_from_chunks(chunks, model=None)

        data = graph.to_dict()
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["chunk_idx"] == 0
        assert data["nodes"][0]["text"] == "Test"
        assert data["nodes"][0]["hpo_matches"] == ["HP:0001234"]
        assert data["nodes"][0]["legacy_status"] == "affirmed"

    def test_to_dict_with_edges(self):
        """to_dict should include edge data."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
        ]
        graph.build_from_chunks(chunks, model=None)

        data = graph.to_dict()
        assert len(data["edges"]) == 1
        assert data["edges"][0]["source_idx"] == 0
        assert data["edges"][0]["target_idx"] == 1
        assert data["edges"][0]["edge_type"] == "sequential"


class TestNetworkXConversion:
    """Tests for NetworkX conversion (optional dependency)."""

    def test_to_networkx_requires_networkx(self):
        """to_networkx should work if networkx available."""
        pytest.importorskip("networkx")

        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [
            ChunkNode(chunk_idx=0, text="A"),
            ChunkNode(chunk_idx=1, text="B"),
        ]
        graph.build_from_chunks(chunks, model=None)

        nx_graph = graph.to_networkx()
        assert nx_graph.number_of_nodes() == 2
        assert nx_graph.number_of_edges() == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_text(self):
        """Graph should handle chunks with very long text."""
        long_text = "word " * 10000
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text=long_text))
        assert graph.num_nodes == 1

    def test_unicode_text(self):
        """Graph should handle unicode text properly."""
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="患者有发热症状 🔥"))
        node = graph.get_node(0)
        assert "发热" in node.text

    def test_many_hpo_ids(self):
        """Graph should handle chunks with many HPO IDs."""
        many_hpos = [f"HP:{str(i).zfill(7)}" for i in range(100)]
        graph = SemanticDocumentGraph()
        graph.add_node(ChunkNode(chunk_idx=0, text="Complex", hpo_matches=many_hpos))
        node = graph.get_node(0)
        assert len(node.hpo_matches) == 100

    def test_single_chunk_no_sequential_edges(self):
        """Single chunk should have no sequential edges."""
        graph = SemanticDocumentGraph(add_semantic_edges=False)
        chunks = [ChunkNode(chunk_idx=0, text="Only chunk")]
        graph.build_from_chunks(chunks, model=None)
        assert graph.num_edges == 0

    def test_disconnected_hpo_chunks(self):
        """Chunks with different HPOs should not be connected via coreference."""
        graph = SemanticDocumentGraph(
            add_sequential_edges=False,
            add_semantic_edges=False,
            add_hpo_coreference_edges=True,
        )
        chunks = [
            ChunkNode(chunk_idx=0, text="A", hpo_matches=["HP:0001"]),
            ChunkNode(chunk_idx=1, text="B", hpo_matches=["HP:0002"]),
        ]
        graph.build_from_chunks(chunks, model=None)
        assert graph.num_edges == 0
