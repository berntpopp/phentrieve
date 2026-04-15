# Unified Output Format Architecture with Phenopackets Integration

**Status**: Active Planning
**Priority**: High
**Complexity**: High
**Author**: Senior Data Scientist & Developer Review
**Date**: 2025-01-21

---

## Executive Summary

This document outlines a comprehensive architectural design for unifying Phentrieve's output formats using GA4GH Phenopackets as the internal representation standard, with flexible conversion to multiple export formats (JSON, CSV, TSV, TXT). The design addresses current fragmentation between `query` and `text process` outputs, handles uncertainty quantification, supports interactive workflows, and ensures clinical interoperability.

**Key Goals**:
1. **Internal Standardization**: Use Phenopackets v2.0 as canonical data structure
2. **Format Flexibility**: Support JSON, CSV, TSV, TXT exports with full metadata preservation
3. **Uncertainty Handling**: Extend Phenopackets with confidence scores and rankings
4. **Interactive Workflows**: Enable chunk selection, manual curation, and analysis object persistence
5. **Clinical Interoperability**: Ensure compatibility with GA4GH ecosystem and clinical tools

---

## 1. Current State Analysis

### 1.1 Output Format Fragmentation

**Query Command Output** (`phentrieve/cli/query_commands.py`):
- **Formats**: `text`, `json`, `jsonl`
- **Structure**: Simple list of ranked HPO terms with scores
- **Fields**: `hpo_id`, `label`, `similarity`, `cross_encoder_score`, `original_rank`, `definition`, `synonyms`
- **Limitations**: No chunk provenance, minimal metadata, single query focus

**Text Process Command Output** (`phentrieve/cli/text_commands.py`):
- **Formats**: `json_lines`, `rich_json_summary`, `csv_hpo_list`
- **Structure**: Chunk-level results + aggregated HPO terms
- **Fields**: Per chunk: `chunk_idx`, `chunk_text`, `matches[]`; Aggregated: `hpo_id`, `name`, `confidence`, `evidence_count`, `chunks[]`, `text_attributions[]`
- **Limitations**: Non-standard schema, no clinical context, limited interoperability

**API Schemas** (`api/schemas/`):
- **QueryResponseSegment**: Simple segment-based results
- **AggregatedHPOTermAPI**: Rich metadata with attribution spans
- **ProcessedChunkAPI**: Chunk-level assertions and matches
- **Inconsistency**: Different field names (`id` vs `hpo_id`, `name` vs `label`)

### 1.2 Key Issues

1. **No Unified Data Model**: Query and text outputs use incompatible schemas
2. **Loss of Provenance**: Cannot trace HPO terms back to source text in query mode
3. **Metadata Fragmentation**: Chunking strategy, models used, timestamps scattered or missing
4. **No Clinical Context**: Missing patient ID, encounter date, clinician, document type
5. **Uncertainty Opacity**: Confidence scores present but not standardized
6. **No Persistence Format**: Cannot save full analysis state for later editing
7. **Limited Export Options**: CSV exports lose nested structures (chunks, attributions)
8. **Interactive Mode Gap**: No mechanism to review/edit results before saving

---

## 2. Phenopackets Format: Capabilities & Limitations

### 2.1 What is a Phenopacket?

The **GA4GH Phenopacket** (v2.0) is an open standard for sharing disease and phenotype information, designed for precision medicine and rare disease diagnostics.

**Core Philosophy**:
- Machine-readable, human-friendly (JSON/YAML)
- Ontology-driven (HPO, LOINC, SNOMED CT)
- Interoperable with HL7 FHIR
- Protobuf-based with language bindings (Python, Java, C++)

### 2.2 Phenopacket Schema Structure (v2.0)

```python
from phenopackets.schema.v2 import Phenopacket

phenopacket = Phenopacket(
    id="unique-phenopacket-id",
    subject=Individual(
        id="patient-123",
        time_at_last_encounter=TimeElement(...)
    ),
    phenotypic_features=[
        PhenotypicFeature(
            type=OntologyClass(id="HP:0001234", label="Seizures"),
            excluded=False,              # Observed (True = excluded/absent)
            severity=OntologyClass(...),
            onset=TimeElement(...),
            resolution=TimeElement(...),
            modifiers=[OntologyClass(...)],
            evidence=[Evidence(
                evidence_code=OntologyClass(...),
                reference=ExternalReference(...)
            )]
        )
    ],
    measurements=[...],          # Quantitative data (LOINC)
    medical_actions=[...],       # Treatments, procedures
    diseases=[...],              # Diagnoses
    interpretations=[...],       # Genomic interpretations
    files=[...],                 # References to external files
    meta_data=MetaData(
        created=Timestamp(...),
        created_by="phentrieve-v0.3.0",
        resources=[Resource(...)]  # Ontology versions
    )
)
```

### 2.3 PhenotypicFeature: The Core Element

```python
PhenotypicFeature(
    type=OntologyClass(id="HP:0001250", label="Seizures"),
    description="Brief free-text description (NOT structured)",
    excluded=False,  # False = observed, True = excluded/negated
    severity=OntologyClass(id="HP:0012825", label="Mild"),
    modifiers=[
        OntologyClass(id="HP:0031796", label="Recurrent"),
        OntologyClass(id="HP:0025303", label="Nocturnal")
    ],
    onset=TimeElement(
        age=Age(iso8601duration="P6M")  # 6 months old
    ),
    resolution=TimeElement(...),  # When it resolved (if applicable)
    evidence=[
        Evidence(
            evidence_code=OntologyClass(id="ECO:0000033", label="Author statement"),
            reference=ExternalReference(
                id="chunk-5",
                description="Patient reports nocturnal seizures since 6 months"
            )
        )
    ]
)
```

### 2.4 Capabilities ✅

| Feature | Support | Implementation |
|---------|---------|----------------|
| **Multiple Phenotypes** | ✅ | `phenotypic_features` list |
| **Negation/Exclusion** | ✅ | `excluded` boolean field |
| **Temporal Data** | ✅ | `onset`, `resolution` with flexible TimeElement |
| **Severity** | ✅ | `severity` field (HP:0012824 hierarchy) |
| **Modifiers** | ✅ | `modifiers` list (HP:0012823 hierarchy) |
| **Evidence Tracking** | ✅ | `evidence` list with ECO codes + references |
| **Provenance** | ✅ | `MetaData` with creation time, software version |
| **Ontology Versions** | ✅ | `resources` field (HPO version, date) |
| **JSON Export** | ✅ | `MessageToJson()` from protobuf |
| **Validation** | ✅ | phenopacket-tools library |

### 2.5 Limitations & Workarounds ⚠️

| Limitation | Impact | Workaround Strategy |
|-----------|--------|---------------------|
| **No Confidence Scores** | Cannot represent retrieval uncertainty (0.85 vs 0.42) | 🔧 **Extension Field**: Add `confidence` to Evidence.reference.description or use structured annotations in MetaData.external_references |
| **No Ranking/Ordering** | Cannot indicate "top 3 most likely" | 🔧 **Implicit Ordering**: Array order = rank; store rank in Evidence metadata |
| **No Multi-Result Sets** | Single phenopacket = single patient | 🔧 **Wrapper Object**: Create `PhentrieveAnalysis` container with phenopacket + extra metadata |
| **Binary Excluded Field** | Only True/False, no "uncertain" | 🔧 **Use Modifiers**: HP:0031915 "Uncertain significance" or store in Evidence |
| **Description is Free Text** | Cannot structure chunk provenance | 🔧 **Use Evidence.reference**: Map chunks to ExternalReference with IDs |
| **No Chunk Representation** | No native "text chunk" concept | 🔧 **ExternalReference**: Treat chunks as external references with IDs |

### 2.6 Key Design Decision: Phenopacket as Core, Not Sole Format

**Recommendation**: Use Phenopackets as the **internal canonical representation** but wrap it in a `PhentrieveAnalysis` container to preserve full system metadata.

**Rationale**:
1. Phenopackets standardize clinical phenotype representation ✅
2. Phenopackets lack ML-specific metadata (chunking strategy, model versions, retrieval parameters) ❌
3. Extension approach preserves GA4GH compatibility while adding system-specific context ✅
4. Enables bidirectional conversion: PhentrieveAnalysis ↔ Phenopacket ✅

---

## 3. Unified Architecture Design

### 3.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Analysis Object (Internal Canonical Format)        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  PhentrieveAnalysis                                     │  │
│  │  ├─ metadata: AnalysisMetadata                          │  │
│  │  │   ├─ analysis_id: UUID                               │  │
│  │  │   ├─ created_at: timestamp                           │  │
│  │  │   ├─ phentrieve_version: "0.3.0"                     │  │
│  │  │   ├─ models_used: {embedding, reranker, chunking}    │  │
│  │  │   ├─ chunking_config: {strategy, params}             │  │
│  │  │   ├─ retrieval_config: {threshold, top_k}            │  │
│  │  │   └─ processing_stats: {total_time, chunk_count}     │  │
│  │  ├─ phenopacket: Phenopacket (GA4GH v2.0)              │  │
│  │  ├─ chunks: List[ProcessedChunk]                        │  │
│  │  │   └─ ProcessedChunk:                                 │  │
│  │  │       ├─ chunk_id: int                               │  │
│  │  │       ├─ text: str                                   │  │
│  │  │       ├─ source_indices: Tuple[int, int]             │  │
│  │  │       ├─ assertion_status: AssertionStatus           │  │
│  │  │       ├─ assertion_details: dict                     │  │
│  │  │       └─ hpo_matches: List[HPOMatch]                 │  │
│  │  │           └─ HPOMatch: {hpo_id, score, rank}         │  │
│  │  └─ aggregated_results: List[AggregatedHPOTerm]         │  │
│  │      └─ AggregatedHPOTerm:                              │  │
│  │          ├─ hpo_id, name, definition, synonyms          │  │
│  │          ├─ confidence: float (avg from chunks)         │  │
│  │          ├─ rank: int (overall ranking)                 │  │
│  │          ├─ evidence_count: int                         │  │
│  │          ├─ chunk_ids: List[int]                        │  │
│  │          ├─ assertion_status: "affirmed"|"negated"      │  │
│  │          └─ text_attributions: List[AttributionSpan]    │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Format Converters (Bidirectional Transformers)     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  to_phenopacket() → Pure GA4GH Phenopacket v2.0        │  │
│  │  from_phenopacket() → Reconstruct PhentrieveAnalysis   │  │
│  │  to_json() → Nested JSON with full metadata            │  │
│  │  to_jsonl() → Line-delimited JSON (streaming)          │  │
│  │  to_csv() → Flat CSV (aggregated terms only)           │  │
│  │  to_tsv() → Tab-separated (with evidence columns)      │  │
│  │  to_txt() → Human-readable report                      │  │
│  │  to_legacy_query_format() → Backwards compatibility    │  │
│  │  to_legacy_text_format() → Backwards compatibility     │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Export Formats (User-Facing Outputs)               │
│  ├─ JSON: Full nested structure with all metadata           │
│  ├─ Phenopacket JSON: GA4GH-compliant JSON for sharing      │
│  ├─ JSONL: Streaming format for large analyses              │
│  ├─ CSV: Aggregated terms (flat, Excel-compatible)          │
│  ├─ TSV: Detailed results with evidence (tab-separated)     │
│  └─ TXT: Human-readable clinical report                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Core Data Structures (Python)

#### 3.2.1 PhentrieveAnalysis (Canonical Format)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from phenopackets.schema.v2 import Phenopacket

@dataclass
class AnalysisMetadata:
    """Metadata about the Phentrieve analysis run."""
    analysis_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    phentrieve_version: str = "0.3.0"

    # Model Configuration
    embedding_model: str = "FremyCompany/BioLORD-2023-M"
    reranker_model: Optional[str] = None
    semantic_chunking_model: Optional[str] = None

    # Processing Configuration
    chunking_strategy: str = "sliding_window_punct_conj_cleaned"
    chunking_params: Dict[str, Any] = field(default_factory=dict)  # window_size, step_size, etc.

    # Retrieval Configuration
    chunk_retrieval_threshold: float = 0.3
    aggregated_term_confidence: float = 0.35
    num_results_per_chunk: int = 10
    enable_reranker: bool = False
    reranker_mode: Optional[str] = None

    # Assertion Detection
    assertion_detection_enabled: bool = True
    assertion_preference: str = "dependency"
    language: str = "en"

    # Processing Statistics
    total_processing_time: float = 0.0  # seconds
    total_chunks: int = 0
    total_unique_hpo_terms: int = 0

    # HPO Ontology Version
    hpo_version: str = "2024-12-12"  # From HPO database
    hpo_source: str = "https://github.com/obophenotype/human-phenotype-ontology"


@dataclass
class ProcessedChunk:
    """A single text chunk with its HPO matches."""
    chunk_id: int  # 1-based indexing
    text: str
    source_indices: tuple[int, int]  # Character offsets in original text

    # Assertion Detection Results
    assertion_status: str  # "affirmed", "negated", "uncertain"
    assertion_details: Optional[Dict[str, Any]] = None  # Full ConText output

    # HPO Matches for this chunk
    hpo_matches: List["HPOMatch"] = field(default_factory=list)


@dataclass
class HPOMatch:
    """A single HPO term match within a chunk."""
    hpo_id: str
    name: str
    score: float  # Similarity or cross-encoder score
    rank: int  # Rank within this chunk (1-based)
    original_rank: Optional[int] = None  # Pre-reranking rank
    cross_encoder_score: Optional[float] = None


@dataclass
class AggregatedHPOTerm:
    """An HPO term aggregated across all chunks."""
    hpo_id: str
    name: str
    definition: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)

    # Aggregation Statistics
    confidence: float  # Average score across all evidence chunks
    rank: int  # Overall rank in final results (1-based)
    evidence_count: int  # Number of chunks where this term appears
    chunk_ids: List[int] = field(default_factory=list)  # Which chunks

    # Assertion Status (most common across chunks)
    assertion_status: str = "affirmed"  # "affirmed", "negated", "uncertain"

    # Text Attribution (provenance)
    text_attributions: List["AttributionSpan"] = field(default_factory=list)


@dataclass
class AttributionSpan:
    """A span of text that matches this HPO term."""
    chunk_id: int
    start_char: int  # Within the chunk
    end_char: int    # Within the chunk
    matched_text: str
    match_type: str  # "exact_label", "synonym", "partial"


@dataclass
class PhentrieveAnalysis:
    """Complete analysis object with Phenopacket + system metadata."""

    # System Metadata
    metadata: AnalysisMetadata

    # Core Phenopacket (GA4GH v2.0)
    phenopacket: Phenopacket

    # Extended Phentrieve Data (not in Phenopacket standard)
    chunks: List[ProcessedChunk] = field(default_factory=list)
    aggregated_results: List[AggregatedHPOTerm] = field(default_factory=list)

    # Original Input (for reproducibility)
    original_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": dataclasses.asdict(self.metadata),
            "phenopacket": MessageToDict(self.phenopacket),  # Protobuf → dict
            "chunks": [dataclasses.asdict(c) for c in self.chunks],
            "aggregated_results": [dataclasses.asdict(r) for r in self.aggregated_results],
            "original_text": self.original_text
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhentrieveAnalysis":
        """Deserialize from dictionary."""
        # Implementation details...
        pass

    def save(self, path: Path, format: str = "json") -> None:
        """Save analysis to file in specified format."""
        if format == "json":
            path.write_text(self.to_json())
        elif format == "phenopacket":
            # Export pure GA4GH Phenopacket
            phenopacket_json = MessageToJson(self.phenopacket)
            path.write_text(phenopacket_json)
        elif format == "csv":
            # Export aggregated results as CSV
            self.to_csv(path)
        # ... other formats

    @classmethod
    def load(cls, path: Path) -> "PhentrieveAnalysis":
        """Load analysis from JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)
```

#### 3.2.2 Mapping to Phenopacket Schema

```python
from phenopackets.schema.v2 import (
    Phenopacket, PhenotypicFeature, OntologyClass,
    Evidence, ExternalReference, MetaData, Resource,
    TimeElement, Timestamp, Individual
)
from google.protobuf.json_format import MessageToDict, ParseDict


def phentrieve_to_phenopacket(analysis: PhentrieveAnalysis) -> Phenopacket:
    """
    Convert PhentrieveAnalysis to pure GA4GH Phenopacket v2.0.

    Key Mappings:
    - aggregated_results → phenotypic_features
    - chunks → Evidence.reference (ExternalReference per chunk)
    - confidence scores → Evidence.reference.description (JSON string)
    - text_attributions → Evidence.reference.description
    """
    phenotypic_features = []

    for term in analysis.aggregated_results:
        # Create evidence entries for each chunk
        evidence_list = []
        for chunk_id in term.chunk_ids:
            chunk = next(c for c in analysis.chunks if c.chunk_id == chunk_id)

            # Find the score for this term in this chunk
            chunk_match = next(
                (m for m in chunk.hpo_matches if m.hpo_id == term.hpo_id),
                None
            )
            score = chunk_match.score if chunk_match else 0.0

            # Create external reference for this chunk
            # WORKAROUND: Store structured data in description (JSON string)
            evidence_metadata = {
                "confidence": score,
                "rank": chunk_match.rank if chunk_match else None,
                "attributions": [
                    {
                        "start": attr.start_char,
                        "end": attr.end_char,
                        "text": attr.matched_text,
                        "type": attr.match_type
                    }
                    for attr in term.text_attributions
                    if attr.chunk_id == chunk_id
                ]
            }

            evidence = Evidence(
                evidence_code=OntologyClass(
                    id="ECO:0000501",  # Evidence from automated machine learning
                    label="evidence from machine learning"
                ),
                reference=ExternalReference(
                    id=f"chunk-{chunk_id}",
                    description=f"{chunk.text}\n\n[Metadata: {json.dumps(evidence_metadata)}]"
                )
            )
            evidence_list.append(evidence)

        # Create PhenotypicFeature
        feature = PhenotypicFeature(
            type=OntologyClass(id=term.hpo_id, label=term.name),
            excluded=(term.assertion_status == "negated"),
            evidence=evidence_list
        )

        # Add modifiers if available (e.g., uncertainty)
        if term.assertion_status == "uncertain":
            feature.modifiers.extend([
                OntologyClass(id="HP:0031915", label="Uncertain significance")
            ])

        phenotypic_features.append(feature)

    # Create metadata
    from google.protobuf.timestamp_pb2 import Timestamp
    created_timestamp = Timestamp()
    created_timestamp.GetCurrentTime()

    metadata = MetaData(
        created=created_timestamp,
        created_by=f"phentrieve-{analysis.metadata.phentrieve_version}",
        submitted_by="phentrieve-user",
        resources=[
            Resource(
                id="hp",
                name="Human Phenotype Ontology",
                namespace_prefix="HP",
                url="http://purl.obolibrary.org/obo/hp.owl",
                version=analysis.metadata.hpo_version,
                iri_prefix="http://purl.obolibrary.org/obo/HP_"
            ),
            Resource(
                id="eco",
                name="Evidence and Conclusion Ontology",
                namespace_prefix="ECO",
                url="http://purl.obolibrary.org/obo/eco.owl",
                version="2023-12-15",
                iri_prefix="http://purl.obolibrary.org/obo/ECO_"
            )
        ]
    )

    # Create subject (anonymous patient if not specified)
    subject = Individual(
        id="phentrieve-subject-" + str(analysis.metadata.analysis_id)[:8],
    )

    # Assemble phenopacket
    phenopacket = Phenopacket(
        id=str(analysis.metadata.analysis_id),
        subject=subject,
        phenotypic_features=phenotypic_features,
        meta_data=metadata
    )

    return phenopacket


def phenopacket_to_phentrieve(phenopacket: Phenopacket) -> PhentrieveAnalysis:
    """
    Convert GA4GH Phenopacket back to PhentrieveAnalysis.

    Limitations:
    - Loss of chunking strategy details (not stored in Phenopacket)
    - Loss of model versions (unless stored in MetaData.external_references)
    - Evidence.reference.description must be parsed for confidence scores
    """
    # Extract metadata from phenopacket
    metadata = AnalysisMetadata(
        analysis_id=UUID(phenopacket.id) if phenopacket.id else uuid4(),
        phentrieve_version=phenopacket.meta_data.created_by.split("-")[-1],
        hpo_version=next(
            (r.version for r in phenopacket.meta_data.resources if r.id == "hp"),
            "unknown"
        )
    )

    # Reconstruct aggregated results from phenotypic features
    aggregated_results = []
    chunks_dict = {}  # chunk_id → ProcessedChunk

    for idx, feature in enumerate(phenopacket.phenotypic_features):
        # Extract HPO term
        hpo_id = feature.type.id
        name = feature.type.label

        # Extract evidence and reconstruct chunks
        chunk_ids = []
        attributions = []
        scores = []

        for evidence in feature.evidence:
            if not evidence.reference.id.startswith("chunk-"):
                continue

            chunk_id = int(evidence.reference.id.split("-")[1])
            chunk_ids.append(chunk_id)

            # Parse metadata from description
            description = evidence.reference.description
            metadata_start = description.find("[Metadata: ")
            if metadata_start != -1:
                chunk_text = description[:metadata_start].strip()
                metadata_json = description[metadata_start + 11:-1]  # Remove "[Metadata: " and "]"
                evidence_metadata = json.loads(metadata_json)

                scores.append(evidence_metadata["confidence"])

                # Reconstruct chunk if not already exists
                if chunk_id not in chunks_dict:
                    chunks_dict[chunk_id] = ProcessedChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_indices=(0, 0),  # Unknown
                        assertion_status="affirmed" if not feature.excluded else "negated",
                        hpo_matches=[]
                    )

                # Extract attributions
                for attr_data in evidence_metadata.get("attributions", []):
                    attributions.append(AttributionSpan(
                        chunk_id=chunk_id,
                        start_char=attr_data["start"],
                        end_char=attr_data["end"],
                        matched_text=attr_data["text"],
                        match_type=attr_data["type"]
                    ))

        # Create aggregated term
        aggregated_term = AggregatedHPOTerm(
            hpo_id=hpo_id,
            name=name,
            confidence=sum(scores) / len(scores) if scores else 0.0,
            rank=idx + 1,
            evidence_count=len(chunk_ids),
            chunk_ids=chunk_ids,
            assertion_status="negated" if feature.excluded else "affirmed",
            text_attributions=attributions
        )
        aggregated_results.append(aggregated_term)

    # Create analysis object
    analysis = PhentrieveAnalysis(
        metadata=metadata,
        phenopacket=phenopacket,
        chunks=list(chunks_dict.values()),
        aggregated_results=aggregated_results
    )

    return analysis
```

---

## 4. Format Converters: Export Implementations

### 4.1 JSON (Full Nested Format)

**Use Case**: Save complete analysis state for later loading/editing

```python
def to_json(analysis: PhentrieveAnalysis, indent: int = 2) -> str:
    """Export full analysis as JSON."""
    return analysis.to_json(indent=indent)

# Example output structure:
{
  "metadata": {
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-01-21T10:30:00Z",
    "phentrieve_version": "0.3.0",
    "embedding_model": "FremyCompany/BioLORD-2023-M",
    "chunking_strategy": "sliding_window_punct_conj_cleaned",
    "chunking_params": {"window_size": 3, "step_size": 1},
    "total_chunks": 15,
    "total_unique_hpo_terms": 8,
    "hpo_version": "2024-12-12"
  },
  "phenopacket": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "subject": {"id": "phentrieve-subject-550e8400"},
    "phenotypicFeatures": [...],
    "metaData": {...}
  },
  "chunks": [
    {
      "chunk_id": 1,
      "text": "Patient presents with recurrent seizures",
      "source_indices": [0, 42],
      "assertion_status": "affirmed",
      "hpo_matches": [
        {"hpo_id": "HP:0001250", "name": "Seizures", "score": 0.89, "rank": 1}
      ]
    }
  ],
  "aggregated_results": [
    {
      "hpo_id": "HP:0001250",
      "name": "Seizures",
      "definition": "A seizure is an intermittent abnormality...",
      "synonyms": ["Epileptic seizure", "Convulsions"],
      "confidence": 0.87,
      "rank": 1,
      "evidence_count": 3,
      "chunk_ids": [1, 5, 8],
      "assertion_status": "affirmed",
      "text_attributions": [
        {
          "chunk_id": 1,
          "start_char": 23,
          "end_char": 42,
          "matched_text": "recurrent seizures",
          "match_type": "synonym"
        }
      ]
    }
  ],
  "original_text": "Patient presents with recurrent seizures..."
}
```

### 4.2 Phenopacket JSON (GA4GH Standard)

**Use Case**: Share results with clinical systems, FHIR converters, phenotype databases

```python
def to_phenopacket_json(analysis: PhentrieveAnalysis) -> str:
    """Export as pure GA4GH Phenopacket v2.0 JSON."""
    phenopacket = phentrieve_to_phenopacket(analysis)
    return MessageToJson(phenopacket, indent=2)
```

### 4.3 CSV (Aggregated Terms Only)

**Use Case**: Excel analysis, simple reporting, stakeholder summaries

```python
import csv
from io import StringIO

def to_csv(analysis: PhentrieveAnalysis) -> str:
    """Export aggregated HPO terms as CSV."""
    output = StringIO()
    fieldnames = [
        "rank", "hpo_id", "name", "confidence", "evidence_count",
        "assertion_status", "chunks", "definition"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for term in analysis.aggregated_results:
        writer.writerow({
            "rank": term.rank,
            "hpo_id": term.hpo_id,
            "name": term.name,
            "confidence": f"{term.confidence:.3f}",
            "evidence_count": term.evidence_count,
            "assertion_status": term.assertion_status,
            "chunks": ";".join(str(c) for c in term.chunk_ids),
            "definition": term.definition or ""
        })

    return output.getvalue()
```

**Example CSV Output**:
```csv
rank,hpo_id,name,confidence,evidence_count,assertion_status,chunks,definition
1,HP:0001250,Seizures,0.870,3,affirmed,"1;5;8","A seizure is an intermittent abnormality..."
2,HP:0001298,Encephalopathy,0.821,2,affirmed,"3;7","Encephalopathy is a term for any diffuse disease..."
3,HP:0001263,Developmental delay,0.785,4,affirmed,"2;4;6;9","A delay in the achievement of motor or mental milestones..."
```

### 4.4 TSV (Detailed with Evidence)

**Use Case**: Data science analysis, evidence tracking, chunk-level review

```python
def to_tsv(analysis: PhentrieveAnalysis) -> str:
    """Export with chunk-level evidence as TSV."""
    output = StringIO()
    fieldnames = [
        "rank", "hpo_id", "name", "confidence", "evidence_count",
        "chunk_id", "chunk_text", "chunk_score", "assertion_status",
        "attribution_text"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for term in analysis.aggregated_results:
        for chunk_id in term.chunk_ids:
            chunk = next(c for c in analysis.chunks if c.chunk_id == chunk_id)
            chunk_match = next(
                (m for m in chunk.hpo_matches if m.hpo_id == term.hpo_id),
                None
            )

            # Find attributions for this chunk
            chunk_attributions = [
                attr for attr in term.text_attributions
                if attr.chunk_id == chunk_id
            ]
            attribution_texts = " | ".join(
                attr.matched_text for attr in chunk_attributions
            )

            writer.writerow({
                "rank": term.rank,
                "hpo_id": term.hpo_id,
                "name": term.name,
                "confidence": f"{term.confidence:.3f}",
                "evidence_count": term.evidence_count,
                "chunk_id": chunk_id,
                "chunk_text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                "chunk_score": f"{chunk_match.score:.3f}" if chunk_match else "N/A",
                "assertion_status": chunk.assertion_status,
                "attribution_text": attribution_texts
            })

    return output.getvalue()
```

### 4.5 TXT (Human-Readable Report)

**Use Case**: Clinical documentation, patient reports, non-technical stakeholders

```python
def to_txt(analysis: PhentrieveAnalysis) -> str:
    """Generate human-readable clinical report."""
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("PHENTRIEVE PHENOTYPE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Analysis ID: {analysis.metadata.analysis_id}")
    report_lines.append(f"Generated: {analysis.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Software Version: Phentrieve {analysis.metadata.phentrieve_version}")
    report_lines.append(f"HPO Version: {analysis.metadata.hpo_version}")
    report_lines.append("")

    # Configuration Summary
    report_lines.append("CONFIGURATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Embedding Model: {analysis.metadata.embedding_model}")
    report_lines.append(f"Chunking Strategy: {analysis.metadata.chunking_strategy}")
    report_lines.append(f"Language: {analysis.metadata.language}")
    report_lines.append(f"Total Chunks Processed: {analysis.metadata.total_chunks}")
    report_lines.append("")

    # Aggregated Results
    report_lines.append("IDENTIFIED PHENOTYPES")
    report_lines.append("=" * 80)
    report_lines.append(f"Found {len(analysis.aggregated_results)} unique HPO terms")
    report_lines.append("")

    for term in analysis.aggregated_results:
        report_lines.append(f"{term.rank}. {term.name} ({term.hpo_id})")
        report_lines.append(f"   Status: {term.assertion_status.upper()}")
        report_lines.append(f"   Confidence: {term.confidence:.2f}")
        report_lines.append(f"   Evidence Count: {term.evidence_count} chunk(s)")

        if term.definition:
            report_lines.append(f"   Definition: {term.definition}")

        if term.synonyms:
            report_lines.append(f"   Synonyms: {', '.join(term.synonyms)}")

        # Evidence chunks
        report_lines.append("   Evidence:")
        for chunk_id in term.chunk_ids[:3]:  # Show top 3 chunks
            chunk = next(c for c in analysis.chunks if c.chunk_id == chunk_id)
            report_lines.append(f"      • [Chunk {chunk_id}] {chunk.text[:100]}...")

        if len(term.chunk_ids) > 3:
            report_lines.append(f"      • ... and {len(term.chunk_ids) - 3} more chunks")

        report_lines.append("")

    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)
```

---

## 5. Interactive Mode Design

### 5.1 Interactive Workflow for Text Processing

**Goal**: Allow users to review chunks, select/deselect HPO matches, edit assertion statuses, and save curated analysis.

```python
# CLI Enhancement: phentrieve text process --interactive
@app.command("process")
def process_text_for_hpo_command(
    # ... existing parameters ...
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            "-I",
            help="Enable interactive mode for chunk review and manual curation"
        ),
    ] = False,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output-file",
            "-O",
            help="Save analysis to file after interactive editing (JSON format)"
        ),
    ] = None,
) -> None:
    """Process clinical text with optional interactive curation."""

    # ... existing processing code ...

    # Create PhentrieveAnalysis object
    analysis = PhentrieveAnalysis(
        metadata=AnalysisMetadata(...),
        phenopacket=phentrieve_to_phenopacket(...),
        chunks=processed_chunks,
        aggregated_results=aggregated_results
    )

    if interactive:
        # Launch interactive session
        analysis = interactive_review_session(analysis)

    # Output results in requested format
    if output_file:
        analysis.save(output_file, format="json")
        typer.echo(f"Analysis saved to {output_file}")
    else:
        # Print to stdout in requested format
        if output_format == "json":
            typer.echo(analysis.to_json())
        elif output_format == "csv":
            typer.echo(to_csv(analysis))
        # ... other formats


def interactive_review_session(analysis: PhentrieveAnalysis) -> PhentrieveAnalysis:
    """
    Interactive TUI for reviewing and editing analysis results.

    Features:
    - Browse chunks with navigation (PgUp/PgDn, ↑/↓)
    - View HPO matches per chunk
    - Toggle assertion status (affirmed ↔ negated ↔ uncertain)
    - Select/deselect individual HPO matches
    - View aggregated results with live updates
    - Save and exit
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm

    console = Console()
    console.clear()

    current_chunk_idx = 0
    modified = False

    while True:
        # Display current chunk
        chunk = analysis.chunks[current_chunk_idx]

        # Create chunk display
        chunk_panel = Panel(
            chunk.text,
            title=f"Chunk {chunk.chunk_id}/{len(analysis.chunks)} "
                  f"[Status: {chunk.assertion_status}]",
            border_style="blue"
        )
        console.print(chunk_panel)

        # Display HPO matches table
        table = Table(title="HPO Matches", show_header=True)
        table.add_column("Sel", justify="center", style="cyan", width=3)
        table.add_column("#", justify="right", style="cyan", width=3)
        table.add_column("HPO ID", style="magenta", width=12)
        table.add_column("Name", style="green")
        table.add_column("Score", justify="right", style="yellow", width=6)

        for idx, match in enumerate(chunk.hpo_matches):
            selected = "✓" if match.rank is not None else " "
            table.add_row(
                selected,
                str(idx + 1),
                match.hpo_id,
                match.name,
                f"{match.score:.3f}"
            )

        console.print(table)
        console.print("")

        # Show menu
        console.print("[bold cyan]Commands:[/bold cyan]")
        console.print("  [n]ext chunk  [p]rev chunk  [a]ssertion  [t]oggle match")
        console.print("  [r]eview aggregated  [s]ave & exit  [q]uit without saving")
        console.print("")

        # Get user input
        command = Prompt.ask("Command", choices=["n", "p", "a", "t", "r", "s", "q"])

        if command == "n":
            current_chunk_idx = min(current_chunk_idx + 1, len(analysis.chunks) - 1)
        elif command == "p":
            current_chunk_idx = max(current_chunk_idx - 1, 0)
        elif command == "a":
            # Toggle assertion status
            statuses = ["affirmed", "negated", "uncertain"]
            current = chunk.assertion_status
            next_status = statuses[(statuses.index(current) + 1) % len(statuses)]
            chunk.assertion_status = next_status
            modified = True
            console.print(f"[green]Status changed to: {next_status}[/green]")
        elif command == "t":
            # Toggle match selection
            match_num = int(Prompt.ask("Match number to toggle", default="1"))
            if 0 < match_num <= len(chunk.hpo_matches):
                match = chunk.hpo_matches[match_num - 1]
                # Toggle by setting rank to None (deselected)
                match.rank = None if match.rank is not None else match_num
                modified = True
                console.print(f"[green]Match toggled[/green]")
        elif command == "r":
            # Show aggregated results
            _display_aggregated_results(console, analysis)
            Prompt.ask("Press Enter to continue")
        elif command == "s":
            if modified:
                # Recompute aggregated results
                analysis = _recompute_aggregated(analysis)
            return analysis
        elif command == "q":
            if modified and not Confirm.ask("Discard changes?"):
                continue
            return analysis

        console.clear()

    return analysis


def _display_aggregated_results(console: Console, analysis: PhentrieveAnalysis) -> None:
    """Display aggregated results in interactive mode."""
    table = Table(title="Aggregated HPO Terms", show_header=True)
    table.add_column("Rank", justify="right", style="cyan", width=4)
    table.add_column("HPO ID", style="magenta", width=12)
    table.add_column("Name", style="green")
    table.add_column("Conf", justify="right", style="yellow", width=5)
    table.add_column("Ev", justify="right", style="cyan", width=3)
    table.add_column("Status", style="blue", width=10)

    for term in analysis.aggregated_results:
        table.add_row(
            str(term.rank),
            term.hpo_id,
            term.name,
            f"{term.confidence:.2f}",
            str(term.evidence_count),
            term.assertion_status
        )

    console.print(table)


def _recompute_aggregated(analysis: PhentrieveAnalysis) -> PhentrieveAnalysis:
    """Recompute aggregated results after manual edits."""
    # Re-run orchestrate_hpo_extraction with modified chunks
    # ... implementation details ...
    return analysis
```

### 5.2 Interactive Mode for Query

**Enhancement**: Add `--interactive` flag to query command for multi-iteration refinement:

```python
@app.command("query")
def query_hpo(
    # ... existing parameters ...
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-I", help="Interactive query refinement mode"),
    ] = False,
) -> None:
    """Query HPO terms interactively."""

    if interactive:
        console = Console()
        console.print("[bold cyan]Interactive Query Mode[/bold cyan]")
        console.print("Type your queries (or 'exit' to quit)\n")

        while True:
            query_text = Prompt.ask("Query")
            if query_text.lower() in ["exit", "quit"]:
                break

            # Run query
            results = _run_query(query_text, ...)

            # Display results
            _display_query_results(console, results)

            # Ask for actions
            action = Prompt.ask(
                "Action",
                choices=["r", "d", "s", "n"],
                default="n",
                show_choices=True
            )

            if action == "r":  # Refine query
                continue
            elif action == "d":  # Show details for a result
                result_num = int(Prompt.ask("Result number"))
                _show_result_details(console, results[result_num - 1])
            elif action == "s":  # Save results
                save_path = Prompt.ask("Save path", default="query_results.json")
                _save_results(results, Path(save_path))
                console.print(f"[green]Saved to {save_path}[/green]")
            # elif action == "n": new query (continue loop)
```

---

## 6. Uncertainty & Confidence Handling

### 6.1 Problem: Phenopackets Lack Native Confidence Scores

**Phenopacket PhenotypicFeature**:
- ✅ `excluded: bool` → Observed (False) or Absent (True)
- ❌ No `confidence` field
- ❌ No ranking mechanism

**Phentrieve Needs**:
- Represent retrieval uncertainty (e.g., term A: 0.89 vs term B: 0.42)
- Maintain top-K rankings
- Preserve per-chunk scores vs aggregated scores

### 6.2 Solution: Multi-Strategy Approach

#### Strategy 1: Evidence Metadata (Recommended)

Store confidence in `Evidence.reference.description` as structured JSON:

```python
evidence = Evidence(
    evidence_code=OntologyClass(
        id="ECO:0000501",
        label="evidence from machine learning"
    ),
    reference=ExternalReference(
        id="chunk-5",
        description=json.dumps({
            "text": "Patient presents with recurrent seizures",
            "confidence": 0.89,
            "rank": 1,
            "retrieval_method": "dense_retrieval",
            "model": "FremyCompany/BioLORD-2023-M"
        })
    )
)
```

**Pros**:
- Uses existing Phenopacket structure
- Parseable by systems that understand Evidence
- Maintains GA4GH compliance

**Cons**:
- Requires custom parsing logic
- Not automatically validated by phenopacket-tools

#### Strategy 2: Modifiers for Uncertainty Categories

Use HPO modifiers to indicate broad confidence levels:

```python
feature = PhenotypicFeature(
    type=OntologyClass(id="HP:0001250", label="Seizures"),
    modifiers=[
        OntologyClass(id="HP:0031915", label="Uncertain significance"),  # For low confidence
        # OR
        OntologyClass(id="HP:0032443", label="Past medical history"),  # For context
    ]
)
```

**Pros**:
- Standard HPO vocabulary
- Automatically validated

**Cons**:
- Coarse-grained (no numerical scores)
- Limited modifier options for ML confidence

#### Strategy 3: External Annotations File

Ship Phenopacket + separate JSON sidecar with extended metadata:

```
phenopacket-123.json          # Pure GA4GH Phenopacket
phenopacket-123.metadata.json # Phentrieve-specific extensions
```

**`phenopacket-123.metadata.json`**:
```json
{
  "phenopacket_id": "550e8400-e29b-41d4-a716-446655440000",
  "phentrieve_version": "0.3.0",
  "extensions": {
    "phenotypic_features": {
      "HP:0001250": {
        "confidence": 0.89,
        "rank": 1,
        "evidence_chunks": [
          {"chunk_id": 1, "score": 0.87},
          {"chunk_id": 5, "score": 0.91}
        ]
      }
    }
  }
}
```

**Pros**:
- Keeps Phenopacket pure and compliant
- Full flexibility for extensions

**Cons**:
- Requires managing two files
- Not a standard pattern

#### Strategy 4: PhentrieveAnalysis Wrapper (Recommended ✅)

Use `PhentrieveAnalysis` as the primary format, with Phenopacket embedded inside:

```python
@dataclass
class PhentrieveAnalysis:
    metadata: AnalysisMetadata
    phenopacket: Phenopacket  # GA4GH-compliant embedded object
    chunks: List[ProcessedChunk]
    aggregated_results: List[AggregatedHPOTerm]  # With confidence, rank, etc.
```

**Export Options**:
1. `analysis.to_json()` → Full Phentrieve format with confidence scores
2. `analysis.to_phenopacket_json()` → Pure GA4GH Phenopacket (confidence in Evidence)
3. `analysis.to_csv()` → Flat format with confidence column

**Pros**:
- Best of both worlds
- Full flexibility internally
- GA4GH compliance for sharing

**Cons**:
- More complex data model

### 6.3 Recommendation

**Use Strategy 4 (PhentrieveAnalysis wrapper) + Strategy 1 (Evidence metadata)**:

1. **Internal format**: `PhentrieveAnalysis` with full metadata
2. **Phenopacket export**: Store confidence in `Evidence.reference.description`
3. **CSV/TSV export**: Dedicated confidence columns
4. **Interactive mode**: Display confidence prominently in UI

---

## 7. Implementation Roadmap

### Phase 1: Core Data Structures (Week 1-2)

**Tasks**:
1. ✅ Define `PhentrieveAnalysis`, `AnalysisMetadata`, `ProcessedChunk`, `AggregatedHPOTerm` dataclasses
2. ✅ Implement `to_dict()`, `from_dict()`, `to_json()`, `from_json()` methods
3. ✅ Add `phenopackets` library dependency to `pyproject.toml`
4. ✅ Implement `phentrieve_to_phenopacket()` converter
5. ✅ Implement `phenopacket_to_phentrieve()` converter (with limitations documented)
6. ✅ Add unit tests for serialization/deserialization

**Files to Create/Modify**:
- `phentrieve/data_structures.py` (new) → Core dataclasses
- `phentrieve/converters/phenopacket_converter.py` (new)
- `phentrieve/converters/__init__.py` (new)
- `tests/unit/test_data_structures.py` (new)
- `tests/unit/converters/test_phenopacket_converter.py` (new)

### Phase 2: Format Converters (Week 3)

**Tasks**:
1. ✅ Implement `to_csv()`, `to_tsv()`, `to_txt()` exporters
2. ✅ Implement `to_jsonl()` for streaming
3. ✅ Add `to_legacy_query_format()`, `to_legacy_text_format()` for backwards compatibility
4. ✅ Create `phentrieve/converters/export_formatters.py`
5. ✅ Add tests for all export formats

**Files to Create/Modify**:
- `phentrieve/converters/export_formatters.py` (new)
- `tests/unit/converters/test_export_formatters.py` (new)

### Phase 3: CLI Integration (Week 4)

**Tasks**:
1. ✅ Modify `text_commands.py` to create `PhentrieveAnalysis` objects
2. ✅ Modify `query_commands.py` to create `PhentrieveAnalysis` objects
3. ✅ Add `--output-format` options: `json`, `phenopacket`, `csv`, `tsv`, `txt`, `jsonl`
4. ✅ Add `--output-file` option to save analysis objects
5. ✅ Add `--load-analysis` option to load saved analysis and re-export
6. ✅ Update help text and examples

**CLI Examples**:
```bash
# Text processing with new format
phentrieve text process "clinical text" --output-format json --output-file analysis.json

# Export to Phenopacket
phentrieve text process "clinical text" --output-format phenopacket --output-file phenopacket.json

# Export to CSV for Excel
phentrieve text process "clinical text" --output-format csv > results.csv

# Load and convert formats
phentrieve convert analysis.json --to csv --output results.csv
phentrieve convert analysis.json --to phenopacket --output phenopacket.json
```

### Phase 4: Interactive Mode (Week 5-6)

**Tasks**:
1. ✅ Add `rich` library for TUI (already in dependencies?)
2. ✅ Implement `interactive_review_session()` function
3. ✅ Add chunk navigation (←/→, PgUp/PgDn)
4. ✅ Add assertion status toggling
5. ✅ Add HPO match selection/deselection
6. ✅ Add aggregated results view
7. ✅ Implement live recomputation after edits
8. ✅ Add save/discard workflow

**CLI Examples**:
```bash
phentrieve text process "clinical text" --interactive --output-file curated.json
```

### Phase 5: API Integration (Week 7)

**Tasks**:
1. ✅ Update API schemas to support new format
2. ✅ Add `/text/process` endpoint with `output_format` parameter
3. ✅ Add `/convert` endpoint for format conversion
4. ✅ Add `/analysis/{id}` endpoints for CRUD operations on saved analyses
5. ✅ Update API documentation

**API Examples**:
```bash
# Process text and get Phenopacket
POST /api/text/process
{
  "text_content": "...",
  "output_format": "phenopacket"
}

# Convert existing analysis
POST /api/convert
{
  "analysis_json": "...",
  "target_format": "csv"
}
```

### Phase 6: Documentation & Examples (Week 8)

**Tasks**:
1. ✅ Write `docs/output-formats.md` explaining all formats
2. ✅ Write `docs/phenopackets-integration.md`
3. ✅ Create Jupyter notebook examples
4. ✅ Update README with new CLI options
5. ✅ Add example files to `examples/outputs/`

### Phase 7: Validation & Testing (Week 9-10)

**Tasks**:
1. ✅ Add phenopacket-tools validation
2. ✅ Add round-trip tests (Phentrieve → Phenopacket → Phentrieve)
3. ✅ Add E2E tests for all output formats
4. ✅ Add performance benchmarks (large text handling)
5. ✅ User acceptance testing with clinical stakeholders

---

## 8. Backwards Compatibility Strategy

### 8.1 Deprecation Plan

**Phase 1 (v0.4.0 - Soft Deprecation)**:
- New unified format introduced
- Old formats still supported with deprecation warnings
- `--output-format legacy-query` and `--output-format legacy-text` for explicit opt-in

**Phase 2 (v0.5.0 - Default Switch)**:
- New format becomes default
- Old formats require explicit `--legacy` flag
- Documentation emphasizes new format

**Phase 3 (v1.0.0 - Full Removal)**:
- Old formats removed
- Migration guide provided

### 8.2 Migration Guide for Users

```bash
# Old: Query with JSON output
phentrieve query "clinical text" --output-format json

# New: Query with unified JSON output
phentrieve query "clinical text" --output-format json  # Now returns PhentrieveAnalysis

# Migration: Use legacy format temporarily
phentrieve query "clinical text" --output-format legacy-query

# Recommended: Switch to new format and use converters
phentrieve query "clinical text" --output-format json --output-file analysis.json
phentrieve convert analysis.json --to csv  # Convert as needed
```

---

## 9. Benefits Summary

### 9.1 For Users

1. **Unified Experience**: Same data structure for query and text processing
2. **Flexibility**: Export to any format (JSON, CSV, TSV, TXT, Phenopacket)
3. **Clinical Interoperability**: Phenopacket export for sharing with clinical systems
4. **Reproducibility**: Full metadata preservation (models, configs, timestamps)
5. **Editability**: Interactive mode for manual curation before export
6. **Excel-Friendly**: CSV/TSV exports for non-technical stakeholders

### 9.2 For Developers

1. **Maintainability**: Single data structure instead of fragmented formats
2. **Extensibility**: Easy to add new fields without breaking compatibility
3. **Testability**: Clear serialization contracts
4. **Standards Compliance**: GA4GH Phenopackets integration
5. **Type Safety**: Dataclasses with full type hints

### 9.3 For Researchers

1. **Data Science Ready**: JSON/JSONL for Python/R analysis
2. **Evidence Tracking**: Full provenance from text → chunks → HPO terms
3. **Uncertainty Quantification**: Confidence scores at chunk and aggregate levels
4. **Reanalysis**: Load saved analyses and reprocess with different parameters

### 9.4 For Clinical Users

1. **FHIR Integration**: Phenopackets can be converted to FHIR Observations
2. **EHR Compatibility**: Standard format for clinical systems
3. **Human-Readable Reports**: TXT format for documentation
4. **Validation**: phenopacket-tools ensures schema compliance

---

## 10. Open Questions & Decisions Needed

### 10.1 Questions for Stakeholders

1. **Patient Identification**:
   - Should Phenopackets include actual patient IDs or use anonymous placeholders?
   - GDPR/HIPAA implications for saving patient data in analysis files?

2. **Clinical Context**:
   - Should we capture document type (progress note, discharge summary, etc.)?
   - Should we capture clinician/author information?
   - Should we capture encounter date/time?

3. **Ontology Versioning**:
   - How to handle HPO version updates over time?
   - Should analyses be re-runnable with newer HPO versions?

4. **Batch Processing**:
   - Should we support batch exports (e.g., 100 analyses → single JSONL file)?
   - How to handle multi-patient exports?

5. **Storage**:
   - Should analyses be stored in a database or as files?
   - If database, what schema (PostgreSQL JSONB, MongoDB, etc.)?

### 10.2 Technical Decisions Needed

1. **Default Output Format**:
   - Should default be `json` (full Phentrieve) or `phenopacket` (GA4GH)?
   - Recommendation: `json` for power users, add `--simple-json` for lightweight output

2. **Performance Optimization**:
   - Should we lazy-load Phenopacket generation (only when needed)?
   - Should we cache converted formats?

3. **Validation Level**:
   - Should we validate Phenopackets on every export (slow) or on-demand?
   - Should we fail hard on validation errors or warn?

4. **Field Naming Convention**:
   - `hpo_id` vs `id` (currently inconsistent)
   - `name` vs `label` (currently inconsistent)
   - **Recommendation**: Standardize on `hpo_id` and `name` internally

---

## 11. Success Metrics

### 11.1 Technical Metrics

- [ ] 100% round-trip fidelity (Phentrieve → Phenopacket → Phentrieve)
- [ ] All Phenopacket exports pass phenopacket-tools validation
- [ ] Zero regressions in existing test suite
- [ ] < 10% performance overhead from new data structures
- [ ] Full test coverage (>90%) for all converters

### 11.2 User Metrics

- [ ] 3+ clinical users successfully export to Phenopackets
- [ ] 10+ researchers use CSV/TSV exports in analyses
- [ ] 5+ users complete interactive curation workflows
- [ ] Zero critical bugs reported in first month
- [ ] Positive feedback from usability testing

---

## 12. Conclusion & Recommendations

### 12.1 Summary

This architectural design proposes a **three-layer approach**:
1. **Internal Representation**: `PhentrieveAnalysis` wrapper with embedded Phenopacket
2. **Converters**: Flexible transformers to multiple export formats
3. **Export Formats**: JSON, Phenopacket JSON, CSV, TSV, TXT, JSONL

**Key Design Decisions**:
- ✅ Use Phenopackets v2.0 as embedded standard format
- ✅ Extend with PhentrieveAnalysis wrapper for ML metadata
- ✅ Store confidence scores in Evidence.reference.description
- ✅ Support interactive curation workflows
- ✅ Maintain backwards compatibility during transition

### 12.2 Recommended Next Steps

1. **Week 1**: Review this document with stakeholders, finalize decisions
2. **Week 2**: Implement Phase 1 (Core Data Structures)
3. **Week 3**: Implement Phase 2 (Format Converters)
4. **Week 4**: Implement Phase 3 (CLI Integration)
5. **Week 5-6**: Implement Phase 4 (Interactive Mode)
6. **Week 7**: API Integration
7. **Week 8-10**: Documentation, validation, user testing

### 12.3 Risk Mitigation

**Risk**: Breaking changes disrupt existing users
**Mitigation**: 3-phase deprecation plan, legacy format support, migration guide

**Risk**: Phenopacket limitations constrain functionality
**Mitigation**: PhentrieveAnalysis wrapper preserves full metadata, pure Phenopacket export optional

**Risk**: Performance degradation from complex data structures
**Mitigation**: Lazy loading, caching, profiling benchmarks

**Risk**: Low adoption of new formats
**Mitigation**: Clear documentation, Jupyter examples, interactive mode usability

---

## 13. Appendix

### A. Python Package Requirements

```toml
# Add to pyproject.toml
[project]
dependencies = [
    # ... existing dependencies ...
    "phenopackets>=2.0.2",  # GA4GH Phenopackets Python library
    "protobuf>=4.25.0",     # Required for Phenopackets
]

[project.optional-dependencies]
interactive = [
    "rich>=13.0.0",         # Terminal UI for interactive mode
]
```

### B. Example Files

**Example 1: Full PhentrieveAnalysis JSON**
- See Section 4.1 for detailed structure

**Example 2: Pure Phenopacket JSON**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "subject": {
    "id": "phentrieve-subject-550e8400"
  },
  "phenotypicFeatures": [
    {
      "type": {
        "id": "HP:0001250",
        "label": "Seizures"
      },
      "excluded": false,
      "evidence": [
        {
          "evidenceCode": {
            "id": "ECO:0000501",
            "label": "evidence from machine learning"
          },
          "reference": {
            "id": "chunk-1",
            "description": "Patient presents with recurrent seizures\n\n[Metadata: {\"confidence\": 0.89, \"rank\": 1}]"
          }
        }
      ]
    }
  ],
  "metaData": {
    "created": "2025-01-21T10:30:00Z",
    "createdBy": "phentrieve-0.3.0",
    "resources": [
      {
        "id": "hp",
        "name": "Human Phenotype Ontology",
        "namespacePrefix": "HP",
        "url": "http://purl.obolibrary.org/obo/hp.owl",
        "version": "2024-12-12",
        "iriPrefix": "http://purl.obolibrary.org/obo/HP_"
      }
    ]
  }
}
```

### C. References

1. **GA4GH Phenopackets v2.0**: https://phenopacket-schema.readthedocs.io/
2. **Phenopackets Python Library**: https://github.com/phenopackets/phenopacket-schema
3. **pyphetools**: https://github.com/monarch-initiative/pyphetools
4. **HPO Browser**: https://hpo.jax.org/
5. **Evidence & Conclusion Ontology (ECO)**: http://www.evidenceontology.org/
6. **GA4GH Standards**: https://www.ga4gh.org/

### D. Contributors

- **Senior Data Scientist Review**: Deep dive into Phenopackets capabilities
- **Senior Developer Review**: Architecture design and implementation strategy
- **Clinical Informatics Input**: (Pending stakeholder review)
- **User Experience Design**: Interactive mode workflow

---

**Document Status**: Draft for Review
**Next Review Date**: 2025-01-28
**Implementation Start**: Upon approval
