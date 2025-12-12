"""
Semantic document graph construction for document-level assertion reasoning.

This module provides infrastructure for building sentence/chunk-level graphs
that capture semantic relationships within clinical documents. The graph
structure enables:

1. Document-level assertion consistency checking
2. Assertion propagation between related chunks
3. Conflict detection and resolution for HPO term assertions

The SemanticDocumentGraph is an ADDITIVE extension that works alongside
the existing chunk-local assertion detection, enabling comparison between
methods.

See: plan/00-planning/GRAPH-BASED-EXTENSION-PLAN.md
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from phentrieve.text_processing.assertion_representation import AssertionVector

logger = logging.getLogger(__name__)


@dataclass
class ChunkNode:
    """
    Node representing a text chunk in the semantic document graph.

    Stores chunk content, computed embeddings, assertion information,
    and HPO term matches for graph-based reasoning.

    Attributes:
        chunk_idx: Zero-based index of the chunk in the document
        text: The text content of the chunk
        embedding: Computed sentence embedding (populated during graph construction)
        assertion_vector: Multi-dimensional assertion (if graph inference enabled)
        legacy_assertion_status: Original AssertionStatus (always preserved)
        hpo_matches: List of HPO IDs matched to this chunk
        start_char: Start character position in original document
        end_char: End character position in original document
        metadata: Additional metadata for extensibility
    """

    chunk_idx: int
    text: str
    embedding: np.ndarray | None = None
    assertion_vector: AssertionVector | None = None
    legacy_assertion_status: str | None = None  # Preserved original status
    hpo_matches: list[str] = field(default_factory=list)
    start_char: int = -1
    end_char: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by chunk index for use in sets/dicts."""
        return hash(self.chunk_idx)

    def __eq__(self, other: object) -> bool:
        """Equality by chunk index."""
        if not isinstance(other, ChunkNode):
            return False
        return self.chunk_idx == other.chunk_idx


@dataclass
class GraphEdge:
    """
    Edge in the semantic document graph.

    Represents a relationship between two chunks with type and weight.

    Attributes:
        source_idx: Source chunk index
        target_idx: Target chunk index
        edge_type: Type of relationship (sequential, semantic, hpo_coreference)
        weight: Edge weight (e.g., similarity score)
        metadata: Additional edge metadata (e.g., shared HPO ID)
    """

    source_idx: int
    target_idx: int
    edge_type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticDocumentGraph:
    """
    Document-level graph of text chunks with semantic edges.

    This graph structure captures relationships between chunks in a clinical
    document to enable document-level assertion reasoning. Edge types include:

    1. **Sequential edges**: Between adjacent chunks (chunk_i → chunk_{i+1})
    2. **Semantic edges**: Between semantically similar chunks (cosine similarity)
    3. **HPO coreference edges**: Between chunks mentioning the same HPO term

    The graph is implemented using adjacency lists for efficient traversal
    without requiring heavy dependencies like NetworkX (which is optional).

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer("all-MiniLM-L6-v2")
        >>> graph = SemanticDocumentGraph(similarity_threshold=0.5)
        >>> chunks = [
        ...     ChunkNode(chunk_idx=0, text="Patient has fever"),
        ...     ChunkNode(chunk_idx=1, text="No cough observed"),
        ...     ChunkNode(chunk_idx=2, text="High temperature noted"),
        ... ]
        >>> graph.build_from_chunks(chunks, model)
        >>> list(graph.get_neighbors(0))
        [1, 2]  # Adjacent and semantically similar
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        max_neighbor_distance: int = 3,
        add_sequential_edges: bool = True,
        add_semantic_edges: bool = True,
        add_hpo_coreference_edges: bool = True,
    ):
        """
        Initialize the semantic document graph.

        Args:
            similarity_threshold: Minimum cosine similarity for semantic edges (0-1)
            max_neighbor_distance: Maximum chunk index distance for semantic edges
            add_sequential_edges: Whether to add edges between adjacent chunks
            add_semantic_edges: Whether to add similarity-based edges
            add_hpo_coreference_edges: Whether to add HPO coreference edges
        """
        self.similarity_threshold = similarity_threshold
        self.max_neighbor_distance = max_neighbor_distance
        self.add_sequential_edges_flag = add_sequential_edges
        self.add_semantic_edges_flag = add_semantic_edges
        self.add_hpo_coreference_edges_flag = add_hpo_coreference_edges

        # Graph storage
        self._nodes: dict[int, ChunkNode] = {}
        self._adjacency: dict[int, list[GraphEdge]] = {}  # outgoing edges
        self._reverse_adjacency: dict[int, list[GraphEdge]] = {}  # incoming edges
        self._embeddings_cache: dict[int, np.ndarray] = {}

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Total number of edges in the graph."""
        return sum(len(edges) for edges in self._adjacency.values())

    def get_node(self, chunk_idx: int) -> ChunkNode | None:
        """Get node by chunk index."""
        return self._nodes.get(chunk_idx)

    def get_nodes(self) -> Iterator[ChunkNode]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())

    def get_edges(self, source_idx: int) -> list[GraphEdge]:
        """Get outgoing edges from a node."""
        return self._adjacency.get(source_idx, [])

    def get_incoming_edges(self, target_idx: int) -> list[GraphEdge]:
        """Get incoming edges to a node."""
        return self._reverse_adjacency.get(target_idx, [])

    def get_neighbors(self, chunk_idx: int) -> Iterator[int]:
        """Get indices of neighboring chunks (outgoing edges)."""
        for edge in self._adjacency.get(chunk_idx, []):
            yield edge.target_idx

    def get_predecessors(self, chunk_idx: int) -> Iterator[int]:
        """Get indices of predecessor chunks (incoming edges)."""
        for edge in self._reverse_adjacency.get(chunk_idx, []):
            yield edge.source_idx

    def add_node(self, chunk: ChunkNode) -> None:
        """Add a chunk node to the graph."""
        self._nodes[chunk.chunk_idx] = chunk
        if chunk.chunk_idx not in self._adjacency:
            self._adjacency[chunk.chunk_idx] = []
        if chunk.chunk_idx not in self._reverse_adjacency:
            self._reverse_adjacency[chunk.chunk_idx] = []

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if edge.source_idx not in self._adjacency:
            self._adjacency[edge.source_idx] = []
        if edge.target_idx not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_idx] = []

        self._adjacency[edge.source_idx].append(edge)
        self._reverse_adjacency[edge.target_idx].append(edge)

    def build_from_chunks(
        self,
        chunks: list[ChunkNode],
        model: SentenceTransformer | None = None,
    ) -> None:
        """
        Construct graph from processed chunks.

        This method builds the graph by:
        1. Adding all chunks as nodes
        2. Computing embeddings for semantic edge detection
        3. Adding edges based on configured edge types

        Args:
            chunks: List of ChunkNode instances to add to the graph
            model: SentenceTransformer model for computing embeddings
                  (required if add_semantic_edges is True)

        Raises:
            ValueError: If semantic edges requested but no model provided
        """
        if not chunks:
            logger.warning("No chunks provided to build_from_chunks")
            return

        logger.info(
            "Building semantic document graph from %d chunks "
            "(threshold=%.2f, max_distance=%d)",
            len(chunks),
            self.similarity_threshold,
            self.max_neighbor_distance,
        )

        # Clear existing graph
        self._nodes.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        self._embeddings_cache.clear()

        # Add nodes
        for chunk in chunks:
            self.add_node(chunk)

        # Compute embeddings if needed for semantic edges
        embeddings: np.ndarray | None = None
        if self.add_semantic_edges_flag:
            if model is None:
                raise ValueError(
                    "SentenceTransformer model required for semantic edges"
                )
            embeddings = self._compute_embeddings(chunks, model)

        # Add edges
        if self.add_sequential_edges_flag:
            self._build_sequential_edges(chunks)

        if self.add_semantic_edges_flag and embeddings is not None:
            self._build_semantic_edges(chunks, embeddings)

        if self.add_hpo_coreference_edges_flag:
            self._build_hpo_coreference_edges(chunks)

        logger.info(
            "Graph construction complete: %d nodes, %d edges",
            self.num_nodes,
            self.num_edges,
        )

    def _compute_embeddings(
        self,
        chunks: list[ChunkNode],
        model: SentenceTransformer,
    ) -> np.ndarray:
        """Compute embeddings for all chunks in batch."""
        texts = [chunk.text for chunk in chunks]

        logger.debug("Computing embeddings for %d chunks", len(texts))
        embeddings = model.encode(texts, show_progress_bar=False)

        # Store embeddings in nodes and cache
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            self._embeddings_cache[chunk.chunk_idx] = embeddings[i]

        return embeddings

    def _build_sequential_edges(self, chunks: list[ChunkNode]) -> None:
        """Add edges between sequential chunks."""
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_idx)

        for i in range(len(sorted_chunks) - 1):
            edge = GraphEdge(
                source_idx=sorted_chunks[i].chunk_idx,
                target_idx=sorted_chunks[i + 1].chunk_idx,
                edge_type="sequential",
                weight=1.0,
            )
            self.add_edge(edge)

        logger.debug("Added %d sequential edges", len(sorted_chunks) - 1)

    def _build_semantic_edges(
        self,
        chunks: list[ChunkNode],
        embeddings: np.ndarray,
    ) -> None:
        """Add edges between semantically similar chunks."""
        # Lazy import to avoid dependency if not needed
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)
        edge_count = 0

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # Skip if too far apart (index distance)
                idx_distance = abs(chunks[i].chunk_idx - chunks[j].chunk_idx)
                if idx_distance > self.max_neighbor_distance:
                    continue

                similarity = float(sim_matrix[i, j])
                if similarity >= self.similarity_threshold:
                    # Add bidirectional edges for semantic similarity
                    edge_forward = GraphEdge(
                        source_idx=chunks[i].chunk_idx,
                        target_idx=chunks[j].chunk_idx,
                        edge_type="semantic",
                        weight=similarity,
                    )
                    edge_backward = GraphEdge(
                        source_idx=chunks[j].chunk_idx,
                        target_idx=chunks[i].chunk_idx,
                        edge_type="semantic",
                        weight=similarity,
                    )
                    self.add_edge(edge_forward)
                    self.add_edge(edge_backward)
                    edge_count += 2

        logger.debug(
            "Added %d semantic edges (threshold=%.2f)",
            edge_count,
            self.similarity_threshold,
        )

    def _build_hpo_coreference_edges(self, chunks: list[ChunkNode]) -> None:
        """Add edges between chunks mentioning the same HPO term."""
        # Build HPO to chunks mapping
        hpo_to_chunks: dict[str, list[int]] = {}

        for chunk in chunks:
            for hpo_id in chunk.hpo_matches:
                if hpo_id not in hpo_to_chunks:
                    hpo_to_chunks[hpo_id] = []
                hpo_to_chunks[hpo_id].append(chunk.chunk_idx)

        edge_count = 0

        for hpo_id, chunk_indices in hpo_to_chunks.items():
            if len(chunk_indices) > 1:
                # Add edges between all pairs mentioning this HPO term
                for i in range(len(chunk_indices)):
                    for j in range(i + 1, len(chunk_indices)):
                        edge_forward = GraphEdge(
                            source_idx=chunk_indices[i],
                            target_idx=chunk_indices[j],
                            edge_type="hpo_coreference",
                            weight=1.0,
                            metadata={"hpo_id": hpo_id},
                        )
                        edge_backward = GraphEdge(
                            source_idx=chunk_indices[j],
                            target_idx=chunk_indices[i],
                            edge_type="hpo_coreference",
                            weight=1.0,
                            metadata={"hpo_id": hpo_id},
                        )
                        self.add_edge(edge_forward)
                        self.add_edge(edge_backward)
                        edge_count += 2

        logger.debug(
            "Added %d HPO coreference edges for %d shared HPO terms",
            edge_count,
            sum(1 for c in hpo_to_chunks.values() if len(c) > 1),
        )

    def get_neighborhood(
        self,
        chunk_idx: int,
        radius: int = 2,
        include_self: bool = True,
    ) -> set[int]:
        """
        Get chunks within graph distance radius using BFS.

        Args:
            chunk_idx: Starting chunk index
            radius: Maximum graph distance (number of hops)
            include_self: Whether to include the starting chunk

        Returns:
            Set of chunk indices within radius
        """
        if chunk_idx not in self._nodes:
            return set()

        visited: set[int] = set()
        if include_self:
            visited.add(chunk_idx)

        frontier = [chunk_idx]

        for _ in range(radius):
            next_frontier: list[int] = []
            for node in frontier:
                for neighbor_idx in self.get_neighbors(node):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        next_frontier.append(neighbor_idx)
            frontier = next_frontier

        return visited

    def get_chunks_for_hpo(self, hpo_id: str) -> list[int]:
        """Get indices of all chunks that mention a specific HPO term."""
        return [
            chunk.chunk_idx
            for chunk in self._nodes.values()
            if hpo_id in chunk.hpo_matches
        ]

    def get_edge_types_summary(self) -> dict[str, int]:
        """Get count of edges by type."""
        counts: dict[str, int] = {}
        for edges in self._adjacency.values():
            for edge in edges:
                counts[edge.edge_type] = counts.get(edge.edge_type, 0) + 1
        return counts

    def to_networkx(self) -> Any:
        """
        Convert to NetworkX DiGraph for advanced analysis.

        Returns:
            networkx.DiGraph instance

        Raises:
            ImportError: If networkx is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for to_networkx(). "
                "Install with: pip install networkx"
            )

        G = nx.DiGraph()

        # Add nodes with data
        for chunk in self._nodes.values():
            G.add_node(
                chunk.chunk_idx,
                text=chunk.text,
                hpo_matches=chunk.hpo_matches,
                legacy_status=chunk.legacy_assertion_status,
            )

        # Add edges
        for edges in self._adjacency.values():
            for edge in edges:
                G.add_edge(
                    edge.source_idx,
                    edge.target_idx,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    **edge.metadata,
                )

        return G

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary for JSON export."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "similarity_threshold": self.similarity_threshold,
            "max_neighbor_distance": self.max_neighbor_distance,
            "nodes": [
                {
                    "chunk_idx": node.chunk_idx,
                    "text": node.text,
                    "hpo_matches": node.hpo_matches,
                    "legacy_status": node.legacy_assertion_status,
                    "start_char": node.start_char,
                    "end_char": node.end_char,
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "source_idx": edge.source_idx,
                    "target_idx": edge.target_idx,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                }
                for edges in self._adjacency.values()
                for edge in edges
            ],
            "edge_type_summary": self.get_edge_types_summary(),
        }
