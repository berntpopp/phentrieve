# Mention-Level HPO Extraction Implementation Plan

**Status:** ✅ Core Implementation Complete
**Created:** 2025-12-13
**Updated:** 2025-12-13
**Branch:** `feat/graph-based-146`
**Priority:** High
**Estimated Effort:** 2-3 weeks implementation

---

## Progress Summary

### ✅ Completed (Phase 1-8)

| Component | File | Status |
|-----------|------|--------|
| Core dataclasses | `mention.py` | ✅ Complete |
| Document structure | `document_structure.py` | ✅ Complete |
| Mention extraction | `mention_extractor.py` | ✅ Complete |
| Assertion detection | `mention_assertion.py` | ✅ Complete |
| HPO retrieval | `mention_hpo_retriever.py` | ✅ Complete |
| Candidate refinement | `mention_candidate_refiner.py` | ✅ Complete |
| Context propagation | `mention_context.py` | ✅ Complete |
| Mention grouping | `mention_grouper.py` | ✅ Complete |
| Document aggregation | `mention_aggregator.py` | ✅ Complete |
| Orchestrator | `mention_extraction_orchestrator.py` | ✅ Complete |
| Unit tests | 3 test files | ✅ Complete |
| Type checking | mypy | ✅ 0 errors |
| Linting | ruff | ✅ All checks passed |

### 🔄 Remaining Work

- [ ] Integration tests with real DenseRetriever
- [ ] CLI integration 
- [ ] Benchmark evaluation against ID-68, GSC+, GeneReviews
- [ ] Performance optimization (batch processing tuning)
- [ ] Documentation updates

---

## Executive Summary

This plan implements a mention-level HPO extraction system that:
- Extracts clinically relevant HPO terms with dataset-specific assertion labels at the document level
- Uses mention-level representations internally for improved accuracy across chunk boundaries
- Reduces overly generic HPO mappings through specificity control
- Supports grouping of alternative explanations for the same clinical finding
- Maintains full compatibility with existing benchmarks (ID-68, GSC+, GeneReviews)

---

## Architecture Overview

### Core Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Document Input                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage A: Structural Scaffolding                                             │
│  ├── Sentence segmentation                                                   │
│  ├── Lightweight section detection                                           │
│  └── Context boundary markers (e.g., family history regions)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage B: Mention Discovery                                                  │
│  ├── Identify candidate clinical finding spans                               │
│  ├── NP/VP extraction with semantic filtering                                │
│  └── Create Mention objects with span information                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage C: Assertion Interpretation                                           │
│  ├── Per-mention assertion using canonical labels                            │
│  ├── Scope-aware detection (ConText-based)                                   │
│  └── Confidence scores for soft decisions                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage D: HPO Candidate Generation                                           │
│  ├── Dense retrieval per mention                                             │
│  ├── High-recall candidate set (10-20 candidates)                            │
│  └── Preserve local semantic context                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage E: Candidate Refinement & Specificity Control                         │
│  ├── Cross-encoder re-ranking with mention context                           │
│  ├── Ontology-aware specificity scoring                                      │
│  └── Disfavor generic terms when specific alternatives exist                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage F: Controlled Contextual Influence                                    │
│  ├── Limited cross-mention context propagation                               │
│  ├── Gated by proximity and document region                                  │
│  └── Optional graph-based refinement layer                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage G: Content-Based Grouping                                             │
│  ├── Cluster mentions referring to same phenomenon                           │
│  ├── Rank alternative HPO explanations per group                             │
│  └── Soft signals: textual similarity, proximity, ontology structure         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage H: Document-Level Aggregation                                         │
│  ├── Aggregate groups to document-level HPO set                              │
│  ├── Apply dataset-specific assertion label mapping                          │
│  └── Handle conflicts transparently                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Benchmark-Compatible Output                          │
│  ├── Document-level HPO set with assertions                                  │
│  ├── Optional: mention-level details for analysis                            │
│  └── Evaluable against ID-68, GSC+, GeneReviews                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Data Structures (Days 1-2)

**Goal:** Define mention-level representations and grouping structures.

#### 1.1 Mention Dataclass

**File:** `phentrieve/text_processing/mention.py`

```python
@dataclass
class Mention:
    """A clinical finding mention in text with span and semantic info."""
    mention_id: str                      # Unique identifier
    text: str                            # Surface text
    start_char: int                      # Start position in document
    end_char: int                        # End position in document
    sentence_idx: int                    # Sentence index
    section_type: str | None             # Section context (e.g., "family_history")
    embedding: np.ndarray | None         # Computed embedding
    assertion: AssertionVector           # Multi-dimensional assertion
    hpo_candidates: list[HPOCandidate]   # Ranked HPO candidates
    metadata: dict[str, Any]             # Extensibility (future: time, anatomy)
```

#### 1.2 HPO Candidate Dataclass

```python
@dataclass  
class HPOCandidate:
    """A candidate HPO term for a mention."""
    hpo_id: str
    label: str
    score: float                         # Initial retrieval score
    refined_score: float | None          # After re-ranking
    specificity_score: float             # Ontology depth-based
    is_generic: bool                     # Flag for generic terms
```

#### 1.3 Mention Group Dataclass

```python
@dataclass
class MentionGroup:
    """Group of mentions referring to the same clinical phenomenon."""
    group_id: str
    mentions: list[Mention]
    representative_mention: Mention      # Best exemplar
    ranked_hpo_explanations: list[HPOCandidate]  # Merged and ranked
    final_hpo: HPOCandidate | None       # Selected for output
    final_assertion: AssertionVector     # Aggregated assertion
```

### Phase 2: Structural Scaffolding (Day 3)

**Goal:** Lightweight sentence/section detection for context gating.

#### 2.1 Document Structure Detector

**File:** `phentrieve/text_processing/document_structure.py`

- Sentence segmentation using spaCy
- Section header detection (regex + keyword-based)
- Family history region detection (reuse existing `family_history_processor.py`)
- Output: `DocumentStructure` with sentences and section boundaries

### Phase 3: Mention Discovery (Days 4-5)

**Goal:** Identify candidate clinical finding spans.

#### 3.1 Mention Extractor

**File:** `phentrieve/text_processing/mention_extractor.py`

- Use spaCy noun phrase extraction as base
- Apply clinical finding filters (exclude pronouns, stopwords)
- Optionally use dependency patterns for finding descriptions
- Generate `Mention` objects with spans

### Phase 4: Assertion Interpretation (Day 6)

**Goal:** Per-mention assertion detection with canonical labels.

#### 4.1 Mention Assertion Detector

**File:** `phentrieve/text_processing/mention_assertion.py`

- Reuse existing `AssertionDetector` but at mention-level
- Create `AssertionVector` for each mention
- Scope-aware: use mention span, not full chunk
- Map to canonical internal labels

### Phase 5: HPO Candidate Generation (Days 7-8)

**Goal:** Dense retrieval per mention for high-recall candidate sets.

#### 5.1 Mention HPO Retriever

**File:** `phentrieve/text_processing/mention_hpo_retriever.py`

- Batch embed all mentions
- Query retriever with mention text + local context
- Return top-K candidates per mention (K=10-20)
- Preserve similarity scores

### Phase 6: Candidate Refinement (Days 9-10)

**Goal:** Re-rank and apply specificity control.

#### 6.1 Mention Candidate Refiner

**File:** `phentrieve/text_processing/mention_candidate_refiner.py`

- Cross-encoder re-ranking with mention context
- Ontology depth-based specificity scoring
- Soft penalty for generic terms when specific alternatives exist
- Output refined scores

### Phase 7: Controlled Context & Grouping (Days 11-12)

**Goal:** Cross-mention context and phenomenon grouping.

#### 7.1 Context Propagator

**File:** `phentrieve/text_processing/mention_context.py`

- Build mention graph (adjacency, similarity-based edges)
- Gated context influence (same section, proximity constraints)
- Integrate with existing `SemanticDocumentGraph`

#### 7.2 Mention Grouper

**File:** `phentrieve/text_processing/mention_grouper.py`

- Cluster mentions by textual similarity + HPO overlap
- Create `MentionGroup` objects
- Rank alternative HPO explanations per group

### Phase 8: Document-Level Aggregation (Days 13-14)

**Goal:** Produce benchmark-compatible output.

#### 8.1 Document Aggregator

**File:** `phentrieve/text_processing/mention_aggregator.py`

- Aggregate groups to document-level HPO set
- Apply dataset-specific assertion mapping
- Handle conflicts (multiple assertions for same HPO)
- Produce output compatible with existing benchmark format

### Phase 9: Orchestrator Integration (Day 15)

**Goal:** Integrate with existing pipeline.

#### 9.1 Mention-Based Orchestrator

**File:** `phentrieve/text_processing/mention_extraction_orchestrator.py`

- New orchestrator that uses mention-level processing
- Drop-in replacement for `orchestrate_hpo_extraction` with same interface
- Configuration flag to switch between chunk-based and mention-based

### Phase 10: Benchmark Integration & Validation (Days 16-17)

**Goal:** Validate against existing benchmarks.

- Run on ID-68, GSC+, GeneReviews with unchanged scoring
- Compare to chunk-based baseline
- Ablation studies (with/without mention-level, context, grouping)

---

## File Structure

```
phentrieve/text_processing/
├── mention.py                       # Core mention dataclasses (NEW)
├── mention_extractor.py             # Mention discovery (NEW)
├── mention_assertion.py             # Per-mention assertion (NEW)
├── mention_hpo_retriever.py         # Mention-level HPO retrieval (NEW)
├── mention_candidate_refiner.py     # Refinement & specificity (NEW)
├── mention_context.py               # Context propagation (NEW)
├── mention_grouper.py               # Phenomenon grouping (NEW)
├── mention_aggregator.py            # Document-level output (NEW)
├── mention_extraction_orchestrator.py # Main orchestrator (NEW)
├── document_structure.py            # Sentence/section detection (NEW)
├── assertion_detection.py           # (EXISTING - reused)
├── assertion_representation.py      # (EXISTING - reused)
├── semantic_graph.py                # (EXISTING - integrated)
├── hpo_extraction_orchestrator.py   # (EXISTING - preserved)
└── pipeline.py                      # (EXISTING - add config option)
```

---

## Configuration

Add to `phentrieve.yaml.template`:

```yaml
mention_extraction:
  enabled: true
  # Mention discovery
  min_mention_length: 2
  max_mention_length: 50
  # HPO candidate generation
  candidates_per_mention: 15
  retrieval_threshold: 0.25
  # Specificity control
  enable_specificity_scoring: true
  generic_term_penalty: 0.1
  min_specificity_depth: 3
  # Context propagation
  enable_context_propagation: true
  context_radius: 2
  same_section_only: true
  # Grouping
  enable_grouping: true
  grouping_similarity_threshold: 0.7
  grouping_hpo_overlap_threshold: 0.5
  # Output
  output_top_n_per_group: 3
  include_mention_details: false
```

---

## Assertion Label Mapping

### Internal Canonical Labels

```python
class CanonicalAssertion(Enum):
    AFFIRMED = "affirmed"      # Finding is present
    NEGATED = "negated"        # Finding is absent
    UNCERTAIN = "uncertain"    # Epistemic uncertainty
    NORMAL = "normal"          # Within normal limits
    HISTORICAL = "historical"  # Past finding
    FAMILY = "family"          # Family member finding
```

### Dataset-Specific Mappings

```python
DATASET_ASSERTION_MAPS = {
    "phenobert": {
        "AFFIRMED": "PRESENT",
        "NEGATED": "ABSENT",
        "UNCERTAIN": "UNCERTAIN",
        "NORMAL": "PRESENT",  # Normal is still a present finding
        "HISTORICAL": "PRESENT",
        "FAMILY": "PRESENT",  # Tracked separately
    },
    "gsc_plus": {
        # Same as phenobert for now
    },
    # Extensible for future datasets
}
```

---

## Integration with Existing Graph Approach

The existing `SemanticDocumentGraph` integrates naturally:

1. **Nodes**: `ChunkNode` → `MentionNode` (or wrap mentions in chunks)
2. **Edges**: 
   - Sequential edges → mention adjacency
   - Semantic edges → mention similarity
   - HPO coreference edges → same HPO candidate edges
3. **Propagation**: `AssertionPropagator` works at mention level
4. **Consistency**: Ontology checks at aggregation stage

---

## Validation Strategy

### Primary Validation
- Run unchanged benchmark scoring on ID-68, GSC+, GeneReviews
- Compare document-level F1, precision, recall to chunk-based baseline

### Ablation Studies
- With/without mention-level mapping
- With/without context propagation
- With/without grouping
- With/without specificity control

### Proxy Metrics
- Redundancy rate (duplicate HPO assignments)
- Generic term prevalence (proportion of shallow HPO terms)
- Context leakage rate (family history terms on patient)

---

## Testing Strategy

### Unit Tests

```
tests/unit/text_processing/
├── test_mention.py                  # Mention dataclass tests
├── test_mention_extractor.py        # Extraction tests
├── test_mention_assertion.py        # Assertion tests
├── test_mention_hpo_retriever.py    # Retrieval tests
├── test_mention_candidate_refiner.py
├── test_mention_context.py
├── test_mention_grouper.py
├── test_mention_aggregator.py
└── test_mention_extraction_orchestrator.py
```

### Integration Tests

- Full pipeline test with sample documents
- Comparison with chunk-based orchestrator
- Benchmark evaluation tests

---

## Next Steps

1. **Immediate**: Create `mention.py` with core dataclasses
2. **Day 1-2**: Implement `Mention`, `HPOCandidate`, `MentionGroup`
3. **Day 3**: Implement `DocumentStructure` detector
4. **Day 4-5**: Implement `MentionExtractor`
5. **Continue**: Follow phase sequence above

---

## Related Issues

- **Primary**: Graph-based assertion extension (#146)
- **Related**: Full-text HPO extraction benchmark (#17)
- **Related**: Assertion detection improvements (#126)
