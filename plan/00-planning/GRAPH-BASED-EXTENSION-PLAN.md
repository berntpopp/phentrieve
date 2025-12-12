# Graph-Based Extension Plan for Phentrieve

**Status:** Planning  
**Created:** 2025-12-12  
**Author:** Engineering Analysis  
**Scope:** Semantic text graphs, HPO ontology reasoning, hybrid architectures

---

## 1. Current System Overview

### 1.1 Pipeline Stages and Data Flow

The Phentrieve extraction pipeline processes clinical text through the following stages:

```
Raw Text
    │
    ▼
┌──────────────────────────────┐
│ TextProcessingPipeline       │ (pipeline.py)
│  ├─ Text normalization       │
│  ├─ Chunking pipeline        │ (chunkers.py)
│  │   ├─ ParagraphChunker     │
│  │   ├─ SentenceChunker      │
│  │   ├─ FineGrainedPunctuation│
│  │   ├─ ConjunctionChunker   │
│  │   └─ SlidingWindowSemantic│
│  │       Splitter            │
│  └─ Assertion detection      │ (assertion_detection.py)
│      ├─ KeywordAssertionDetector
│      ├─ DependencyAssertionDetector
│      └─ CombinedAssertionDetector
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ HPO Extraction Orchestrator  │ (hpo_extraction_orchestrator.py)
│  ├─ Batch embedding query    │
│  ├─ Dense retrieval          │ (dense_retriever.py)
│  ├─ Optional cross-encoder   │
│  │   reranking               │
│  ├─ Text attribution         │ (text_attribution.py)
│  └─ Evidence aggregation     │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ Output                       │
│  ├─ Aggregated HPO terms     │
│  │   with confidence scores  │
│  ├─ Assertion status per term│
│  └─ Chunk-level matches      │
└──────────────────────────────┘
```

### 1.2 Key Modules and Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **TextProcessingPipeline** | `text_processing/pipeline.py` | Orchestrates chunking and assertion detection |
| **TextChunker (ABC)** | `text_processing/chunkers.py` | Base class for all chunking strategies |
| **SlidingWindowSemanticSplitter** | `text_processing/chunkers.py` | Embedding-based semantic boundary detection |
| **AssertionDetector (ABC)** | `text_processing/assertion_detection.py` | Base class for assertion detection |
| **CombinedAssertionDetector** | `text_processing/assertion_detection.py` | Combines keyword + dependency parsing |
| **ConTextRule** | `text_processing/assertion_detection.py` | Rule representation for ConText algorithm |
| **DenseRetriever** | `retrieval/dense_retriever.py` | ChromaDB-based semantic search |
| **orchestrate_hpo_extraction** | `text_processing/hpo_extraction_orchestrator.py` | End-to-end extraction coordination |
| **HPODatabase** | `data_processing/hpo_database.py` | SQLite storage for HPO terms + graph metadata |
| **HierarchicalMatcher** | `evaluation/hierarchical_matching.py` | NetworkX-based HPO graph operations |
| **DocumentAggregator** | `evaluation/document_aggregation.py` | Chunk-to-document aggregation strategies |
| **CorpusExtractionMetrics** | `evaluation/extraction_metrics.py` | Document-level evaluation metrics |

### 1.3 Existing Graph Infrastructure

**HPO Ontology Data (Already Present):**
- `hpo_database.py`: SQLite with `hpo_graph_metadata` table storing:
  - `term_id`, `depth`, `ancestors` (JSON array)
- `hpo_parser.py`: Parses OBO Graphs JSON, builds parent-child maps
- `metrics.py`: `load_hpo_graph_data()` with LRU cache, `find_lowest_common_ancestor()`, `calculate_resnik_similarity()`
- `hierarchical_matching.py`: `HierarchicalMatcher` using NetworkX DiGraph

**Assertion Detection (Relevant for Text Graphs):**
- `AssertionStatus` enum: `AFFIRMED`, `NEGATED`, `NORMAL`, `UNCERTAIN`
- `ConTextRule` with `Direction` (FORWARD, BACKWARD, BIDIRECTIONAL, TERMINATE, PSEUDO)
- `TriggerCategory`: NEGATED_EXISTENCE, POSSIBLE_EXISTENCE, HISTORICAL, HYPOTHETICAL, FAMILY

### 1.4 Extension Points Identified

1. **After chunking, before assertion detection** (`pipeline.py:process()`)
   - Chunks are available with source positions
   - Can build sentence/chunk-level graphs here

2. **Within assertion detection** (`assertion_detection.py`)
   - spaCy Doc available with dependency parse
   - Can extract syntactic graph structure

3. **After HPO extraction, before aggregation** (`hpo_extraction_orchestrator.py`)
   - Chunk-level matches with scores available
   - Can apply graph-based consistency checks

4. **HPODatabase** (`hpo_database.py`)
   - Can extend schema for graph reasoning cache
   - Already has ancestors/depths infrastructure

5. **DocumentAggregator** (`evaluation/document_aggregation.py`)
   - Natural extension point for graph-based aggregation
   - Currently implements union/intersection/weighted/threshold

### 1.5 Current Limitations

- **Assertion scope is chunk-local**: No document-level consistency
- **No conflict resolution**: Same HPO term with NEGATED in one chunk, AFFIRMED in another
- **No assertion propagation**: Related concepts don't share assertion evidence
- **HPO hierarchy unused at inference**: Only used for evaluation metrics
- **Confidence is simple averaging**: No uncertainty quantification

---

## 2. Problem Decomposition into Extension Components

### 2.1 Component Overview

| # | Component | Graph Type | Priority | Complexity |
|---|-----------|------------|----------|------------|
| 1 | Assertion Representation Upgrade | N/A (data model) | P0 | Low |
| 2 | Semantic Text Graph Construction | Semantic | P1 | Medium |
| 3 | Graph-Based Assertion Propagation | Semantic | P1 | Medium |
| 4 | HPO DAG Consistency Layer | Ontology | P2 | Medium |
| 5 | Hybrid Inference Engine | Hybrid | P3 | High |
| 6 | Temporal/Longitudinal Module | Semantic | P4 | High |

### 2.2 Dependency Graph

```
[P0] Assertion Representation Upgrade
           │
           ▼
[P1] Semantic Text Graph ◄───────────────┐
           │                              │
           ▼                              │
[P1] Assertion Propagation ──────────────┤
           │                              │
           ▼                              │
[P2] HPO DAG Consistency ────────────────┤
           │                              │
           ▼                              │
[P3] Hybrid Inference Engine ────────────┘
           │
           ▼
[P4] Temporal/Longitudinal (Optional)
```

---

## 3. Component-Level Design

### 3.1 Assertion Representation Upgrade (P0)

**Purpose:** Extend assertion representation from discrete categories to multi-dimensional with confidence scores.

**Current State:**
```python
class AssertionStatus(Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
```

**Proposed Enhancement:**

**File:** `phentrieve/text_processing/assertion_representation.py` (new)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class AssertionVector:
    """Multi-dimensional assertion representation with confidence."""
    
    # Core assertion dimensions (0.0 to 1.0)
    negation_score: float = 0.0      # Confidence in negation
    uncertainty_score: float = 0.0   # Epistemic uncertainty
    normality_score: float = 0.0     # "Within normal limits"
    
    # Contextual modifiers
    historical: bool = False         # Past vs. present
    hypothetical: bool = False       # Conditional statements
    family_history: bool = False     # Subject is family member
    
    # Evidence tracking
    evidence_source: str = "unknown"  # "keyword", "dependency", "propagated"
    evidence_confidence: float = 1.0  # Confidence in the evidence itself
    
    def to_status(self) -> "AssertionStatus":
        """Convert to legacy discrete status for backward compatibility."""
        if self.negation_score > 0.5:
            return AssertionStatus.NEGATED
        if self.uncertainty_score > 0.5:
            return AssertionStatus.UNCERTAIN
        if self.normality_score > 0.5:
            return AssertionStatus.NORMAL
        return AssertionStatus.AFFIRMED
    
    @classmethod
    def from_status(cls, status: "AssertionStatus", confidence: float = 1.0) -> "AssertionVector":
        """Create from legacy discrete status."""
        if status == AssertionStatus.NEGATED:
            return cls(negation_score=confidence, evidence_confidence=confidence)
        elif status == AssertionStatus.UNCERTAIN:
            return cls(uncertainty_score=confidence, evidence_confidence=confidence)
        elif status == AssertionStatus.NORMAL:
            return cls(normality_score=confidence, evidence_confidence=confidence)
        return cls(evidence_confidence=confidence)
```

**Integration Points:**
- Update `AssertionDetector.detect()` return type to include `AssertionVector`
- Modify `hpo_extraction_orchestrator.py` to propagate vectors
- API schemas remain backward-compatible via `to_status()`

**Graph Type:** N/A (foundational data model)

**Computational Profile:**
- Memory: +48 bytes per chunk (dataclass with 8 fields)
- CPU: Negligible (simple attribute access)

**Failure Modes:**
- Confidence calibration drift: Mitigate with periodic validation on annotated data
- Backward compatibility breaks: Maintain `to_status()` for all API surfaces

**Tests to Add:**
- Unit: `AssertionVector` serialization/deserialization
- Unit: `to_status()` threshold behavior
- Integration: Pipeline produces consistent vector-to-status conversions

---

### 3.2 Semantic Text Graph Construction (P1)

**Purpose:** Build sentence/chunk-level graphs capturing semantic relationships within documents for assertion consistency.

**Proposed Design:**

**File:** `phentrieve/text_processing/semantic_graph.py` (new)

```python
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class ChunkNode:
    """Node representing a text chunk in the semantic graph."""
    chunk_idx: int
    text: str
    embedding: Optional[np.ndarray] = None
    assertion_vector: Optional["AssertionVector"] = None
    hpo_matches: list[str] = field(default_factory=list)
    start_char: int = -1
    end_char: int = -1

class SemanticDocumentGraph:
    """
    Document-level graph of text chunks with semantic edges.
    
    Node: ChunkNode (text chunk with assertion and HPO matches)
    Edge: Semantic similarity weight, coreference links, discourse relations
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        max_neighbor_distance: int = 3,  # Max chunk index distance for edges
    ):
        self.graph = nx.DiGraph()
        self.similarity_threshold = similarity_threshold
        self.max_neighbor_distance = max_neighbor_distance
        self._embeddings_cache: dict[int, np.ndarray] = {}
    
    def build_from_chunks(
        self,
        chunks: list[ChunkNode],
        model: SentenceTransformer,
    ) -> None:
        """
        Construct graph from processed chunks.
        
        Edges created based on:
        1. Sequential proximity (chunk_i → chunk_{i+1})
        2. Semantic similarity above threshold
        3. Shared HPO term references
        """
        # Add nodes
        for chunk in chunks:
            self.graph.add_node(chunk.chunk_idx, data=chunk)
        
        # Batch encode all chunks
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            self._embeddings_cache[chunk.chunk_idx] = embeddings[i]
        
        # Create edges
        self._add_sequential_edges(chunks)
        self._add_semantic_edges(chunks, embeddings)
        self._add_hpo_coreference_edges(chunks)
    
    def _add_sequential_edges(self, chunks: list[ChunkNode]) -> None:
        """Add edges between sequential chunks."""
        for i in range(len(chunks) - 1):
            self.graph.add_edge(
                chunks[i].chunk_idx,
                chunks[i + 1].chunk_idx,
                edge_type="sequential",
                weight=1.0,
            )
    
    def _add_semantic_edges(
        self,
        chunks: list[ChunkNode],
        embeddings: np.ndarray,
    ) -> None:
        """Add edges between semantically similar chunks."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        sim_matrix = cosine_similarity(embeddings)
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # Skip if too far apart
                if abs(i - j) > self.max_neighbor_distance:
                    continue
                
                similarity = sim_matrix[i, j]
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(
                        chunks[i].chunk_idx,
                        chunks[j].chunk_idx,
                        edge_type="semantic",
                        weight=float(similarity),
                    )
    
    def _add_hpo_coreference_edges(self, chunks: list[ChunkNode]) -> None:
        """Add edges between chunks mentioning the same HPO term."""
        hpo_to_chunks: dict[str, list[int]] = {}
        
        for chunk in chunks:
            for hpo_id in chunk.hpo_matches:
                if hpo_id not in hpo_to_chunks:
                    hpo_to_chunks[hpo_id] = []
                hpo_to_chunks[hpo_id].append(chunk.chunk_idx)
        
        for hpo_id, chunk_indices in hpo_to_chunks.items():
            if len(chunk_indices) > 1:
                for i in range(len(chunk_indices)):
                    for j in range(i + 1, len(chunk_indices)):
                        self.graph.add_edge(
                            chunk_indices[i],
                            chunk_indices[j],
                            edge_type="hpo_coreference",
                            weight=1.0,
                            hpo_id=hpo_id,
                        )
    
    def get_neighborhood(
        self,
        chunk_idx: int,
        radius: int = 2,
    ) -> list[int]:
        """Get chunks within graph distance radius."""
        if chunk_idx not in self.graph:
            return []
        
        # BFS up to radius
        visited = {chunk_idx}
        frontier = [chunk_idx]
        
        for _ in range(radius):
            next_frontier = []
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        
        return list(visited)
```

**Integration Point:** After `TextProcessingPipeline.process()`, before orchestrator aggregation.

**Computational Profile:**
- Memory: O(n²) for similarity matrix where n = number of chunks
- CPU: O(n) for embedding + O(n²) for similarity computation
- For typical clinical notes (10-50 chunks): <100ms on CPU

**Performance Mitigation:**
- Limit `max_neighbor_distance` to avoid O(n²) edge creation
- Use sparse representation for large documents
- Cache embeddings if chunks are reprocessed

**Failure Modes:**
- Over-connected graphs: Limit edges per node, tune similarity threshold
- Missing edges for related content: Lower threshold or add discourse markers

**Tests to Add:**
- Unit: Graph construction from mock chunks
- Unit: Edge type correctness
- Integration: Graph properties on clinical text examples
- Benchmark: Construction time vs. chunk count

---

### 3.3 Graph-Based Assertion Propagation (P1)

**Purpose:** Propagate assertion evidence through the semantic text graph to resolve conflicts and strengthen confidence.

**Proposed Design:**

**File:** `phentrieve/text_processing/assertion_propagation.py` (new)

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PropagationConfig:
    """Configuration for assertion propagation."""
    max_iterations: int = 3
    damping_factor: float = 0.85  # Similar to PageRank
    convergence_threshold: float = 0.01
    negation_propagation_weight: float = 0.7
    uncertainty_propagation_weight: float = 0.5

class AssertionPropagator:
    """
    Propagate assertion evidence through semantic document graph.
    
    Uses message-passing algorithm:
    1. Initialize node assertions from local detection
    2. Iteratively propagate weighted assertions through edges
    3. Apply conflict resolution rules
    4. Converge to document-consistent assertions
    """
    
    def __init__(self, config: Optional[PropagationConfig] = None):
        self.config = config or PropagationConfig()
    
    def propagate(
        self,
        graph: "SemanticDocumentGraph",
    ) -> dict[int, "AssertionVector"]:
        """
        Run assertion propagation on the graph.
        
        Returns:
            Updated assertion vectors per chunk_idx
        """
        # Initialize from local assertions
        assertions = self._initialize_assertions(graph)
        
        # Iterative propagation
        for iteration in range(self.config.max_iterations):
            new_assertions = self._propagation_step(graph, assertions)
            
            # Check convergence
            if self._has_converged(assertions, new_assertions):
                break
            
            assertions = new_assertions
        
        return assertions
    
    def _initialize_assertions(
        self,
        graph: "SemanticDocumentGraph",
    ) -> dict[int, np.ndarray]:
        """Extract initial assertion vectors from graph nodes."""
        assertions = {}
        
        for node_idx in graph.graph.nodes():
            node_data: ChunkNode = graph.graph.nodes[node_idx]["data"]
            if node_data.assertion_vector:
                assertions[node_idx] = self._vector_to_array(
                    node_data.assertion_vector
                )
            else:
                # Default: neutral assertion
                assertions[node_idx] = np.zeros(3)  # [neg, unc, norm]
        
        return assertions
    
    def _propagation_step(
        self,
        graph: "SemanticDocumentGraph",
        current_assertions: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Single propagation iteration."""
        new_assertions = {}
        
        for node_idx in graph.graph.nodes():
            # Get neighbors with edge weights
            neighbor_contributions = []
            
            for neighbor_idx in graph.graph.predecessors(node_idx):
                edge_data = graph.graph.edges[neighbor_idx, node_idx]
                edge_weight = edge_data.get("weight", 1.0)
                edge_type = edge_data.get("edge_type", "unknown")
                
                # Weight contribution by edge type
                type_weight = self._get_edge_type_weight(edge_type)
                
                neighbor_assertion = current_assertions.get(
                    neighbor_idx, np.zeros(3)
                )
                
                contribution = neighbor_assertion * edge_weight * type_weight
                neighbor_contributions.append(contribution)
            
            # Combine local and propagated assertions
            local_assertion = current_assertions.get(node_idx, np.zeros(3))
            
            if neighbor_contributions:
                propagated = np.mean(neighbor_contributions, axis=0)
                # Damped update: local + propagated influence
                new_assertion = (
                    self.config.damping_factor * local_assertion +
                    (1 - self.config.damping_factor) * propagated
                )
            else:
                new_assertion = local_assertion
            
            # Normalize to [0, 1]
            new_assertions[node_idx] = np.clip(new_assertion, 0, 1)
        
        return new_assertions
    
    def _get_edge_type_weight(self, edge_type: str) -> float:
        """Weight for different edge types in propagation."""
        weights = {
            "sequential": 0.9,      # Strong local context
            "semantic": 0.7,        # Similar content
            "hpo_coreference": 1.0, # Same HPO term - full propagation
        }
        return weights.get(edge_type, 0.5)
    
    def _has_converged(
        self,
        old: dict[int, np.ndarray],
        new: dict[int, np.ndarray],
    ) -> bool:
        """Check if assertions have converged."""
        total_diff = 0.0
        for node_idx in old:
            diff = np.linalg.norm(old[node_idx] - new.get(node_idx, old[node_idx]))
            total_diff += diff
        
        avg_diff = total_diff / len(old) if old else 0
        return avg_diff < self.config.convergence_threshold
    
    def _vector_to_array(self, vec: "AssertionVector") -> np.ndarray:
        """Convert AssertionVector to numpy array."""
        return np.array([
            vec.negation_score,
            vec.uncertainty_score,
            vec.normality_score,
        ])
    
    def resolve_hpo_assertion_conflicts(
        self,
        graph: "SemanticDocumentGraph",
        hpo_id: str,
        propagated_assertions: dict[int, "AssertionVector"],
    ) -> "AssertionVector":
        """
        Resolve conflicting assertions for a specific HPO term.
        
        Strategy:
        1. Collect assertions from all chunks mentioning this HPO
        2. Weight by chunk confidence and graph centrality
        3. Apply clinical rules (negation dominance, recency bias)
        """
        # Find chunks mentioning this HPO
        relevant_chunks = []
        for node_idx in graph.graph.nodes():
            node_data: ChunkNode = graph.graph.nodes[node_idx]["data"]
            if hpo_id in node_data.hpo_matches:
                relevant_chunks.append(node_idx)
        
        if not relevant_chunks:
            return AssertionVector()
        
        # Collect assertions with weights
        weighted_assertions = []
        for chunk_idx in relevant_chunks:
            assertion = propagated_assertions.get(chunk_idx)
            if assertion:
                # Weight by graph degree (centrality proxy)
                degree = graph.graph.degree(chunk_idx)
                weight = 1.0 + (degree / 10.0)  # Scale factor
                weighted_assertions.append((assertion, weight))
        
        if not weighted_assertions:
            return AssertionVector()
        
        # Weighted combination with clinical rules
        return self._apply_clinical_resolution_rules(weighted_assertions)
    
    def _apply_clinical_resolution_rules(
        self,
        weighted_assertions: list[tuple["AssertionVector", float]],
    ) -> "AssertionVector":
        """
        Apply clinical rules for assertion resolution.
        
        Rules:
        1. Negation dominance: Explicit negation often overrides affirmation
        2. Recency: Later mentions may update earlier ones
        3. Specificity: More specific assertions take precedence
        """
        total_weight = sum(w for _, w in weighted_assertions)
        
        # Weighted average of scores
        neg_score = sum(a.negation_score * w for a, w in weighted_assertions) / total_weight
        unc_score = sum(a.uncertainty_score * w for a, w in weighted_assertions) / total_weight
        norm_score = sum(a.normality_score * w for a, w in weighted_assertions) / total_weight
        
        # Clinical rule: Negation requires high confidence
        # If negation and affirmation both present, preserve uncertainty
        has_negation = any(a.negation_score > 0.5 for a, _ in weighted_assertions)
        has_affirmation = any(
            a.negation_score < 0.3 and a.uncertainty_score < 0.3
            for a, _ in weighted_assertions
        )
        
        if has_negation and has_affirmation:
            # Conflict detected - increase uncertainty
            unc_score = max(unc_score, 0.5)
        
        return AssertionVector(
            negation_score=neg_score,
            uncertainty_score=unc_score,
            normality_score=norm_score,
            evidence_source="propagated",
            evidence_confidence=1.0 / (1.0 + unc_score),  # Lower if uncertain
        )
```

**Integration Point:** After `SemanticDocumentGraph.build_from_chunks()`, before final aggregation.

**Computational Profile:**
- Memory: O(n) for assertion arrays
- CPU: O(iterations × edges) for propagation
- Typical: <50ms for 50-chunk document with 3 iterations

**Failure Modes:**
- Oscillation: Damping factor prevents; add iteration limit
- Over-smoothing: Limit iterations; preserve strong local evidence
- Clinical rule errors: Validate rules on annotated corpus

**Tests to Add:**
- Unit: Propagation converges on simple graphs
- Unit: Conflict resolution produces expected output
- Integration: Negated term in one chunk, affirmed in another → resolution
- Benchmark: Propagation time vs. graph density

---

### 3.4 HPO DAG Consistency Layer (P2)

**Purpose:** Enforce ontology-based constraints and perform inference using HPO hierarchy.

**Proposed Design:**

**File:** `phentrieve/reasoning/hpo_consistency.py` (new)

```python
from dataclasses import dataclass
from typing import Optional
import networkx as nx

@dataclass
class ConsistencyViolation:
    """Represents a logical inconsistency in HPO assertions."""
    violation_type: str  # "ancestor_conflict", "redundant_specific", "missing_ancestor"
    hpo_id_1: str
    hpo_id_2: Optional[str]
    description: str
    severity: float  # 0.0 to 1.0

class HPOConsistencyChecker:
    """
    Enforce logical consistency using HPO ontology structure.
    
    Rules:
    1. Ancestor conflict: If HP:child is affirmed, HP:ancestor cannot be negated
    2. Descendant propagation: Affirming HP:ancestor implies possible HP:descendants
    3. Redundancy detection: Affirming HP:child makes HP:ancestor redundant
    4. Specificity preference: Prefer most specific matching term
    """
    
    def __init__(
        self,
        ancestors_map: dict[str, set[str]],
        depths_map: dict[str, int],
    ):
        self.ancestors_map = ancestors_map
        self.depths_map = depths_map
        self._build_descendant_map()
    
    def _build_descendant_map(self) -> None:
        """Build reverse mapping from ancestors to descendants."""
        self.descendants_map: dict[str, set[str]] = {}
        
        for term, ancestors in self.ancestors_map.items():
            for ancestor in ancestors:
                if ancestor not in self.descendants_map:
                    self.descendants_map[ancestor] = set()
                self.descendants_map[ancestor].add(term)
    
    def check_consistency(
        self,
        hpo_assertions: dict[str, "AssertionVector"],
    ) -> list[ConsistencyViolation]:
        """
        Check for logical inconsistencies in HPO assertions.
        
        Args:
            hpo_assertions: {hpo_id: AssertionVector}
        
        Returns:
            List of detected violations
        """
        violations = []
        
        # Check ancestor conflicts
        violations.extend(self._check_ancestor_conflicts(hpo_assertions))
        
        # Check redundancy
        violations.extend(self._check_redundancy(hpo_assertions))
        
        return violations
    
    def _check_ancestor_conflicts(
        self,
        hpo_assertions: dict[str, "AssertionVector"],
    ) -> list[ConsistencyViolation]:
        """Detect affirmed child with negated ancestor."""
        violations = []
        
        for hpo_id, assertion in hpo_assertions.items():
            # Skip if not affirmed
            if assertion.negation_score > 0.5:
                continue
            
            # Check ancestors
            ancestors = self.ancestors_map.get(hpo_id, set())
            for ancestor_id in ancestors:
                ancestor_assertion = hpo_assertions.get(ancestor_id)
                if ancestor_assertion and ancestor_assertion.negation_score > 0.5:
                    violations.append(ConsistencyViolation(
                        violation_type="ancestor_conflict",
                        hpo_id_1=hpo_id,
                        hpo_id_2=ancestor_id,
                        description=f"Affirmed {hpo_id} conflicts with negated ancestor {ancestor_id}",
                        severity=ancestor_assertion.negation_score,
                    ))
        
        return violations
    
    def _check_redundancy(
        self,
        hpo_assertions: dict[str, "AssertionVector"],
    ) -> list[ConsistencyViolation]:
        """Detect redundant ancestor terms when more specific term is present."""
        violations = []
        
        affirmed_terms = {
            hpo_id for hpo_id, a in hpo_assertions.items()
            if a.negation_score < 0.3 and a.uncertainty_score < 0.3
        }
        
        for hpo_id in affirmed_terms:
            ancestors = self.ancestors_map.get(hpo_id, set())
            redundant_ancestors = ancestors & affirmed_terms
            
            for ancestor_id in redundant_ancestors:
                violations.append(ConsistencyViolation(
                    violation_type="redundant_specific",
                    hpo_id_1=ancestor_id,
                    hpo_id_2=hpo_id,
                    description=f"Ancestor {ancestor_id} redundant with more specific {hpo_id}",
                    severity=0.3,  # Low severity - informational
                ))
        
        return violations
    
    def resolve_violations(
        self,
        hpo_assertions: dict[str, "AssertionVector"],
        violations: list[ConsistencyViolation],
    ) -> dict[str, "AssertionVector"]:
        """
        Resolve detected violations by adjusting assertions.
        
        Resolution strategies:
        1. Ancestor conflict: Increase uncertainty for conflicting pair
        2. Redundancy: Mark ancestor as subsumption-redundant
        """
        resolved = dict(hpo_assertions)
        
        for violation in violations:
            if violation.violation_type == "ancestor_conflict":
                # Add uncertainty to the conflict
                for hpo_id in [violation.hpo_id_1, violation.hpo_id_2]:
                    if hpo_id and hpo_id in resolved:
                        old = resolved[hpo_id]
                        resolved[hpo_id] = AssertionVector(
                            negation_score=old.negation_score,
                            uncertainty_score=max(old.uncertainty_score, 0.4),
                            normality_score=old.normality_score,
                            evidence_source="consistency_resolved",
                            evidence_confidence=old.evidence_confidence * 0.8,
                        )
        
        return resolved
    
    def propagate_through_hierarchy(
        self,
        hpo_assertions: dict[str, "AssertionVector"],
        propagation_mode: str = "conservative",
    ) -> dict[str, "AssertionVector"]:
        """
        Propagate assertions through HPO hierarchy.
        
        Modes:
        - conservative: Only propagate high-confidence assertions upward
        - aggressive: Propagate in both directions with attenuation
        """
        propagated = dict(hpo_assertions)
        
        if propagation_mode == "conservative":
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
        
        return propagated
```

**Integration Point:** After assertion propagation, before final output formatting.

**Computational Profile:**
- Memory: Uses existing ancestors_map from HPODatabase
- CPU: O(n × avg_ancestors) for consistency check
- Typical: <20ms for 50 HPO terms

**Failure Modes:**
- Over-aggressive resolution: Configurable resolution strictness
- Missing ontology data: Graceful fallback to skip check

**Tests to Add:**
- Unit: Detect ancestor conflict
- Unit: Detect redundancy
- Integration: Resolution changes output assertions
- Benchmark: Check time vs. HPO term count

---

### 3.5 Hybrid Inference Engine (P3)

**Purpose:** Coordinate semantic text graph evidence with ontology constraints for final inference.

**Proposed Design:**

**File:** `phentrieve/reasoning/hybrid_inference.py` (new)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HybridInferenceConfig:
    """Configuration for hybrid graph inference."""
    text_graph_weight: float = 0.6  # Weight for text graph evidence
    ontology_weight: float = 0.4    # Weight for ontology constraints
    consistency_penalty: float = 0.2  # Penalty for consistency violations
    
class HybridInferenceEngine:
    """
    Combine text graph evidence with ontology constraints.
    
    Pipeline:
    1. Run text graph assertion propagation
    2. Map propagated assertions to HPO terms
    3. Apply ontology consistency checks
    4. Resolve conflicts using weighted combination
    5. Output final assertions with provenance
    """
    
    def __init__(
        self,
        text_graph: "SemanticDocumentGraph",
        propagator: "AssertionPropagator",
        consistency_checker: "HPOConsistencyChecker",
        config: Optional[HybridInferenceConfig] = None,
    ):
        self.text_graph = text_graph
        self.propagator = propagator
        self.consistency_checker = consistency_checker
        self.config = config or HybridInferenceConfig()
    
    def infer(
        self,
        hpo_matches: dict[str, list[int]],  # {hpo_id: [chunk_indices]}
    ) -> dict[str, "FinalAssertion"]:
        """
        Run hybrid inference pipeline.
        
        Args:
            hpo_matches: Mapping from HPO IDs to chunk indices where matched
        
        Returns:
            Final assertion for each HPO term with provenance
        """
        # Step 1: Text graph propagation
        propagated_assertions = self.propagator.propagate(self.text_graph)
        
        # Step 2: Aggregate to HPO level
        hpo_assertions = {}
        for hpo_id, chunk_indices in hpo_matches.items():
            hpo_assertions[hpo_id] = self.propagator.resolve_hpo_assertion_conflicts(
                self.text_graph,
                hpo_id,
                propagated_assertions,
            )
        
        # Step 3: Ontology consistency check
        violations = self.consistency_checker.check_consistency(hpo_assertions)
        
        # Step 4: Apply consistency constraints
        if violations:
            hpo_assertions = self.consistency_checker.resolve_violations(
                hpo_assertions,
                violations,
            )
        
        # Step 5: Format final output with provenance
        final_assertions = {}
        for hpo_id, assertion in hpo_assertions.items():
            final_assertions[hpo_id] = FinalAssertion(
                hpo_id=hpo_id,
                assertion_vector=assertion,
                text_evidence_chunks=hpo_matches.get(hpo_id, []),
                consistency_violations=[
                    v for v in violations
                    if v.hpo_id_1 == hpo_id or v.hpo_id_2 == hpo_id
                ],
                inference_method="hybrid",
            )
        
        return final_assertions

@dataclass
class FinalAssertion:
    """Final assertion with full provenance."""
    hpo_id: str
    assertion_vector: "AssertionVector"
    text_evidence_chunks: list[int]
    consistency_violations: list["ConsistencyViolation"]
    inference_method: str
```

**Integration Point:** Replace current aggregation logic in `hpo_extraction_orchestrator.py`.

**Computational Profile:**
- CPU: Sum of propagation + consistency check
- Typical: <100ms for complete inference on clinical note

**Failure Modes:**
- Configuration sensitivity: Default weights validated on clinical data
- Performance regression: Feature flag for gradual rollout

**Tests to Add:**
- Integration: Full pipeline from text to final assertions
- Unit: Weight combination behavior
- Benchmark: End-to-end latency

---

### 3.6 Temporal/Longitudinal Module (P4 - Optional)

**Purpose:** Handle temporal relationships and longitudinal document processing.

**Scope:** Deferred to future phase. Key considerations noted:

1. **Temporal assertions**: HISTORICAL, onset/resolution dates
2. **Cross-document linking**: Patient timeline construction
3. **Assertion evolution**: How assertions change over time

**Not detailed here** due to complexity and dependency on core graph infrastructure.

---

## 4. Prioritized Roadmap

### Phase 1: Foundation (Weeks 1-2)

| Task | Component | Effort | Dependencies |
|------|-----------|--------|--------------|
| 1.1 | Implement `AssertionVector` dataclass | 2 days | None |
| 1.2 | Update `AssertionDetector` to return vectors | 2 days | 1.1 |
| 1.3 | Backward compatibility in API schemas | 1 day | 1.2 |
| 1.4 | Unit tests for assertion representation | 1 day | 1.2 |
| 1.5 | Implement `SemanticDocumentGraph` | 3 days | None |
| 1.6 | Unit tests for graph construction | 1 day | 1.5 |

**Milestone:** Assertion vectors and text graphs available in pipeline.

### Phase 2: Propagation (Weeks 3-4)

| Task | Component | Effort | Dependencies |
|------|-----------|--------|--------------|
| 2.1 | Implement `AssertionPropagator` | 3 days | 1.5, 1.2 |
| 2.2 | Conflict resolution rules | 2 days | 2.1 |
| 2.3 | Integration with orchestrator | 2 days | 2.2 |
| 2.4 | Unit + integration tests | 2 days | 2.3 |
| 2.5 | Benchmark on clinical texts | 1 day | 2.4 |

**Milestone:** Document-level assertion consistency operational.

### Phase 3: Ontology Integration (Weeks 5-6)

| Task | Component | Effort | Dependencies |
|------|-----------|--------|--------------|
| 3.1 | Implement `HPOConsistencyChecker` | 3 days | Existing HPO graph data |
| 3.2 | Hierarchy propagation logic | 2 days | 3.1 |
| 3.3 | Integration tests | 2 days | 3.2 |
| 3.4 | Benchmark consistency checks | 1 day | 3.3 |

**Milestone:** Ontology-based validation integrated.

### Phase 4: Hybrid Inference (Weeks 7-8)

| Task | Component | Effort | Dependencies |
|------|-----------|--------|--------------|
| 4.1 | Implement `HybridInferenceEngine` | 3 days | 2.3, 3.2 |
| 4.2 | End-to-end integration | 2 days | 4.1 |
| 4.3 | API updates for provenance | 2 days | 4.2 |
| 4.4 | Evaluation on extraction benchmarks | 2 days | 4.3 |
| 4.5 | Documentation and examples | 1 day | 4.4 |

**Milestone:** Complete hybrid graph-based inference deployed.

---

## 5. Trade-off Analysis

### 5.1 Why This Sequencing?

**Assertion Representation First (P0):**
- All downstream components depend on rich assertion data
- Backward compatibility ensures no breaking changes
- Low risk, high foundational value

**Text Graph Before Ontology (P1 before P2):**
- Text evidence is the primary signal; ontology provides constraints
- Text graphs are document-specific; ontology is global
- Easier to validate text graph improvements empirically

**Propagation Before Hybrid (P1 before P3):**
- Propagation is a prerequisite for meaningful hybrid inference
- Can evaluate text-only improvements before adding complexity

### 5.2 Alternative Approaches Considered

**Alternative A: Ontology-First Design**
- Pro: Ontology provides strong logical constraints
- Con: Clinical text often violates strict ontology rules
- Con: Requires assertion evidence to be useful for inference
- Decision: Defer ontology to P2

**Alternative B: GNN-Based Propagation**
- Pro: Learnable propagation weights
- Con: Requires labeled training data for assertion propagation
- Con: Adds training infrastructure complexity
- Con: Violates no-LLM constraint if using learned embeddings
- Decision: Use interpretable message-passing instead

**Alternative C: Full Document Embedding + Clustering**
- Pro: Single embedding per document
- Con: Loses chunk-level granularity
- Con: Cannot attribute assertions to text spans
- Decision: Maintain chunk-level representation

**Alternative D: Skip Semantic Graph, Use Sequential Only**
- Pro: Simpler implementation
- Con: Misses semantic similarity between non-adjacent chunks
- Con: Cannot leverage HPO coreference edges
- Decision: Include semantic edges with tunable threshold

### 5.3 Performance vs. Accuracy Trade-offs

| Decision | Performance Impact | Accuracy Impact | Chosen Trade-off |
|----------|-------------------|-----------------|------------------|
| Limit semantic edges | Faster graph construction | May miss distant relationships | max_neighbor_distance=3 |
| Fixed propagation iterations | Predictable latency | May not converge | max_iterations=3 with damping |
| Conservative ontology propagation | Less false positives | May miss implied terms | conservative mode default |
| Cached HPO ancestors | O(1) lookup | Stale if ontology updates | LRU cache with version check |

---

## 6. File Structure Summary

```
phentrieve/
├── text_processing/
│   ├── assertion_representation.py  # NEW: AssertionVector
│   ├── semantic_graph.py            # NEW: SemanticDocumentGraph
│   ├── assertion_propagation.py     # NEW: AssertionPropagator
│   └── ... (existing files)
├── reasoning/                        # NEW DIRECTORY
│   ├── __init__.py
│   ├── hpo_consistency.py           # NEW: HPOConsistencyChecker
│   └── hybrid_inference.py          # NEW: HybridInferenceEngine
└── ... (existing directories)

tests/
├── unit/
│   ├── text_processing/
│   │   ├── test_assertion_representation.py
│   │   ├── test_semantic_graph.py
│   │   └── test_assertion_propagation.py
│   └── reasoning/
│       ├── test_hpo_consistency.py
│       └── test_hybrid_inference.py
└── integration/
    └── test_graph_inference_pipeline.py
```

---

## 7. Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Assertion conflict rate | N/A (no detection) | <5% unresolved | Integration tests |
| Negation F1 | Current detector | +5% improvement | Assertion benchmark |
| Inference latency | N/A | <200ms for typical note | Performance benchmark |
| Memory overhead | Current | <20% increase | Profiling |
| API backward compatibility | 100% | 100% | E2E tests |

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Graph construction too slow | Medium | High | Profile early; add caching |
| Propagation doesn't converge | Low | Medium | Damping + iteration limit |
| Clinical rules incorrect | Medium | High | Validate on annotated corpus |
| Breaking API changes | Low | High | Feature flags; gradual rollout |
| Over-engineering | Medium | Medium | Defer P4; validate P1-P3 first |

---

## 9. References

- `phentrieve/text_processing/pipeline.py` - Current pipeline orchestration
- `phentrieve/text_processing/assertion_detection.py` - Existing assertion infrastructure
- `phentrieve/evaluation/hierarchical_matching.py` - Existing HPO graph usage
- `phentrieve/data_processing/hpo_database.py` - HPO data access patterns
- `plan/01-active/HPO-EXTRACTION-IMPLEMENTATION-PLAN.md` - Related benchmarking work
