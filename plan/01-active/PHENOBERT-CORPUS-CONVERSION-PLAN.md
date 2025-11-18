# PhenoBERT Corpus Conversion Plan

**Status:** Active
**Date:** 2025-01-18
**Priority:** High
**Estimated Effort:** 2-3 days

## Executive Summary

Convert **three PhenoBERT corpus datasets** (306 documents, 2,564 annotations) to Phentrieve JSON format with a **simple, focused conversion script**. This is alpha software - prioritize working code over perfect architecture.

**Datasets:**
- **GSC+:** 228 PubMed abstracts (1,933 annotations, 497 unique HPO terms)
- **ID-68:** 68 clinical notes (intellectual disability)
- **GeneReviews:** 10 clinical cases

**Key Deliverables:**
- ✅ Simple conversion script (`scripts/convert_phenobert_data.py`)
- ✅ 306 validated JSON files in Phentrieve format
- ✅ Single README with usage instructions

---

## Table of Contents

1. [Data Source Overview](#1-data-source-overview)
2. [Target Format](#2-target-format)
3. [Simple Architecture](#3-simple-architecture)
4. [Implementation](#4-implementation)
5. [Configuration](#5-configuration)
6. [Testing](#6-testing)
7. [Usage](#7-usage)

---

## 1. Data Source Overview

### Source Repository
- **Repository:** https://github.com/EclipseCN/PhenoBERT
- **Data Path:** `phenobert/data/`
- **License:** Verify from repository

### Directory Structure
```
phenobert/data/
├── GSC+/
│   ├── corpus/          # 228 text files
│   └── ann/             # 228 annotation files
├── ID-68/
│   ├── corpus/          # 68 text files
│   └── ann/             # 68 annotation files
└── GeneReviews/
    ├── corpus/          # 10 text files
    └── ann/             # 10 annotation files
```

### Annotation Formats

**Format 1: Raw** (`[start::end] HPO_ID | text`)
```
[27::42] HP_0000110 | renal dysplasia
[56::78] HP_0001263 | Global developmental delay
```

**Format 2: Processed** (TSV: `start\tend\ttext\tHPO:ID\tconfidence`)
```
9	17	headache	HP:0002315	1.0
45	68	developmental delay	HP:0001263	1.0
```

---

## 2. Target Format

Phentrieve JSON structure:

```json
{
  "doc_id": "gsc_plus_001",
  "language": "en",
  "source": "phenobert",
  "full_text": "Clinical text here...",
  "metadata": {
    "dataset": "GSC+",
    "text_length_chars": 500,
    "num_annotations": 8,
    "conversion_date": "2025-01-18T10:00:00Z"
  },
  "annotations": [
    {
      "hpo_id": "HP:0001263",
      "label": "Global developmental delay",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 26,
          "end_char": 51,
          "text_snippet": "global developmental delay",
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

**Default Values:**
- `language`: `"en"`
- `assertion_status`: `"affirmed"` (no negation detection in source)
- `confidence`: From source, or omit if not present

---

## 3. Simple Architecture

### Component Design (4 Components - Not 10!)

```
┌─────────────────────────────────────────────────┐
│           PhenoBERT Converter                    │
│  Single script: convert_phenobert_data.py       │
└─────────────────────────────────────────────────┘

Component 1: DatasetLoader
  - Discovers corpus and annotation files
  - Matches file pairs
  - Reads text with encoding detection

Component 2: AnnotationParser
  - Auto-detects format (raw vs processed)
  - Single parser with pluggable format handlers
  - Normalizes HPO IDs (HP_NNNN → HP:NNNN)

Component 3: Converter
  - Builds JSON structure
  - Validates spans against text
  - Enriches with HPO labels
  - Uses dependency injection

Component 4: OutputWriter
  - Writes JSON files
  - Generates conversion report
```

### Why Only 4 Components?

**DON'T over-engineer:**
- ❌ Separate Configuration Manager (just load YAML)
- ❌ Separate Dataset Scanner (loader does this)
- ❌ Separate File Pair Matcher (loader does this)
- ❌ Separate Text Loader (loader does this)
- ❌ Separate Span Validator (converter does this)
- ❌ Separate HPO Enricher (converter does this)
- ❌ Separate JSON Constructor (converter does this)
- ❌ Separate Schema Validator (output writer does this)
- ❌ Separate Report Generator (output writer does this)

This is **alpha software** converting 306 files - keep it simple!

---

## 4. Implementation

### File Structure

```
phentrieve/
├── data_processing/
│   ├── __init__.py
│   └── phenobert_converter.py      # All conversion logic
│
scripts/
└── convert_phenobert_data.py       # CLI script

data/
└── test_texts/
    └── phenobert/                  # Output directory
        ├── GSC_plus/               # 228 JSON files
        ├── ID68/                   # 68 JSON files
        ├── GeneReviews/            # 10 JSON files
        ├── conversion_report.json  # Summary
        └── README.md               # Usage docs
```

**Simple structure - no complex nested directories, no unused translation folders!**

### Core Classes

#### 1. DatasetLoader

```python
class DatasetLoader:
    """Loads corpus and annotation file pairs."""

    def __init__(self, phenobert_data_dir: Path):
        self.data_dir = phenobert_data_dir

    def discover_datasets(self) -> List[str]:
        """Discover available datasets (GSC+, ID-68, GeneReviews)."""
        pass

    def load_file_pairs(self, dataset: str) -> Iterator[Tuple[Path, Path]]:
        """
        Yield (corpus_file, annotation_file) pairs.

        Auto-matches by filename: corpus/doc.txt <-> ann/doc.ann
        """
        pass

    def read_text(self, file_path: Path) -> str:
        """Read text with encoding detection (UTF-8, then Latin-1)."""
        pass
```

#### 2. AnnotationParser (DRY - Single Parser!)

```python
class AnnotationParser:
    """
    Unified parser with auto-format detection.

    No separate RawFormatParser and ProcessedFormatParser classes!
    Use format handlers instead (DRY principle).
    """

    def parse(self, ann_file: Path) -> List[Annotation]:
        """Parse annotation file, auto-detecting format."""
        format_type = self._detect_format(ann_file)

        if format_type == "raw":
            return self._parse_raw(ann_file)
        else:
            return self._parse_processed(ann_file)

    def _detect_format(self, ann_file: Path) -> str:
        """Detect format from first line."""
        with open(ann_file) as f:
            first_line = f.readline().strip()

        return "raw" if first_line.startswith("[") else "processed"

    def _parse_raw(self, ann_file: Path) -> List[Annotation]:
        """Parse: [27::42] HP_0000110 | renal dysplasia"""
        pattern = r"\[(\d+)::(\d+)\]\s+(HP_\d+)\s+\|\s+(.+)"
        return self._parse_with_pattern(ann_file, pattern, is_raw=True)

    def _parse_processed(self, ann_file: Path) -> List[Annotation]:
        """Parse: 9\t17\theadache\tHP:0002315\t1.0"""
        pattern = r"^(\d+)\t(\d+)\t([^\t]+)\t(HP:\d+)(?:\t([\d.]+))?$"
        return self._parse_with_pattern(ann_file, pattern, is_raw=False)

    def _parse_with_pattern(
        self,
        ann_file: Path,
        pattern: str,
        is_raw: bool
    ) -> List[Annotation]:
        """
        Shared parsing logic (DRY - no duplication!).

        File reading, error handling, HPO normalization done once.
        """
        annotations = []

        with open(ann_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                match = re.match(pattern, line)
                if not match:
                    logger.warning(f"Invalid format at {ann_file}:{line_num}")
                    continue

                if is_raw:
                    start, end, hpo_id, text = match.groups()
                    confidence = None
                else:
                    start, end, text, hpo_id, confidence = match.groups()
                    confidence = float(confidence) if confidence else None

                # Normalize HPO ID (HP_NNNN → HP:NNNN)
                hpo_id = hpo_id.replace("_", ":")

                annotations.append(Annotation(
                    start=int(start),
                    end=int(end),
                    text=text,
                    hpo_id=hpo_id,
                    confidence=confidence,
                ))

        return annotations
```

#### 3. Converter (With Dependency Injection)

```python
class PhenoBERTConverter:
    """
    Main converter with dependency injection.

    FOCUSED RESPONSIBILITIES (not a God class):
    - Convert documents to JSON
    - Validate spans
    - Enrich with HPO labels
    """

    def __init__(
        self,
        hpo_lookup: HPOLookup,
        dataset_loader: DatasetLoader,
        annotation_parser: AnnotationParser,
    ):
        # Dependency injection for testability
        self.hpo_lookup = hpo_lookup
        self.dataset_loader = dataset_loader
        self.annotation_parser = annotation_parser

    def convert_document(
        self,
        corpus_file: Path,
        ann_file: Path,
        dataset_name: str,
    ) -> dict:
        """Convert single document to Phentrieve JSON."""

        # 1. Load text
        full_text = self.dataset_loader.read_text(corpus_file)

        # 2. Parse annotations
        annotations = self.annotation_parser.parse(ann_file)

        # 3. Validate spans (single validation pass)
        valid_annotations = self._validate_spans(full_text, annotations)

        # 4. Build JSON
        return self._build_json(
            doc_id=f"{dataset_name}_{corpus_file.stem}",
            full_text=full_text,
            annotations=valid_annotations,
            dataset_name=dataset_name,
        )

    def _validate_spans(
        self,
        text: str,
        annotations: List[Annotation]
    ) -> List[Annotation]:
        """
        Single validation pass (not 4 levels!).

        Checks:
        - Offsets within bounds
        - Text snippets match
        """
        valid = []

        for ann in annotations:
            # Check bounds
            if ann.start < 0 or ann.end > len(text):
                logger.warning(
                    f"Out of bounds: [{ann.start}:{ann.end}] "
                    f"for {ann.hpo_id}"
                )
                continue

            # Check text match
            actual = text[ann.start:ann.end]
            if actual != ann.text:
                logger.warning(
                    f"Text mismatch: expected '{ann.text}', "
                    f"got '{actual}'"
                )
                continue

            valid.append(ann)

        return valid

    def _build_json(
        self,
        doc_id: str,
        full_text: str,
        annotations: List[Annotation],
        dataset_name: str,
    ) -> dict:
        """Build Phentrieve JSON structure."""

        # Group annotations by HPO ID
        grouped = defaultdict(list)
        for ann in annotations:
            grouped[ann.hpo_id].append(ann)

        # Build annotations array
        json_annotations = []
        for hpo_id, spans in grouped.items():
            json_annotations.append({
                "hpo_id": hpo_id,
                "label": self.hpo_lookup.get_label(hpo_id) or "UNKNOWN",
                "assertion_status": "affirmed",
                "evidence_spans": [
                    {
                        "start_char": span.start,
                        "end_char": span.end,
                        "text_snippet": span.text,
                        **({"confidence": span.confidence} if span.confidence else {}),
                    }
                    for span in spans
                ],
            })

        return {
            "doc_id": doc_id,
            "language": "en",
            "source": "phenobert",
            "full_text": full_text,
            "metadata": {
                "dataset": dataset_name,
                "text_length_chars": len(full_text),
                "num_annotations": len(json_annotations),
                "conversion_date": datetime.now().isoformat(),
            },
            "annotations": json_annotations,
        }
```

#### 4. HPOLookup (Simple)

```python
class HPOLookup:
    """Simple HPO label lookup from Phentrieve's data."""

    def __init__(self, hpo_data_path: Path):
        self.cache = {}
        self._load_hpo_data(hpo_data_path / "hpo_terms.tsv")

    def _load_hpo_data(self, hpo_file: Path):
        """Load HPO terms into memory."""
        with open(hpo_file) as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.cache[parts[0]] = parts[1]

        logger.info(f"Loaded {len(self.cache)} HPO terms")

    def get_label(self, hpo_id: str) -> Optional[str]:
        """Get label for HPO ID."""
        return self.cache.get(hpo_id)
```

#### 5. OutputWriter

```python
class OutputWriter:
    """Writes converted documents and generates report."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stats = {"datasets": {}, "total_docs": 0, "total_annotations": 0}

    def write_document(self, doc: dict, dataset: str):
        """Write single JSON document."""
        dataset_dir = self.output_dir / dataset / "annotations"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        output_file = dataset_dir / f"{doc['doc_id']}.json"
        with open(output_file, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        # Update stats
        self.stats["total_docs"] += 1
        self.stats["total_annotations"] += len(doc["annotations"])

    def write_report(self):
        """Write conversion report."""
        report_file = self.output_dir / "conversion_report.json"
        with open(report_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Converted {self.stats['total_docs']} documents")
```

### Main Script (Simple!)

```python
#!/usr/bin/env python3
"""
Convert PhenoBERT corpus datasets to Phentrieve JSON format.

Usage:
    python scripts/convert_phenobert_data.py \\
        --phenobert-data path/to/PhenoBERT/phenobert/data \\
        --output data/test_texts/phenobert \\
        --hpo-data data/hpo_core_data

    python scripts/convert_phenobert_data.py --help
"""

import argparse
import logging
from pathlib import Path

from phentrieve.data_processing.phenobert_converter import (
    DatasetLoader,
    AnnotationParser,
    PhenoBERTConverter,
    HPOLookup,
    OutputWriter,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenobert-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hpo-data", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["GSC+", "ID-68", "GeneReviews", "all"],
        default="all",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Initialize components (dependency injection)
    dataset_loader = DatasetLoader(args.phenobert_data)
    annotation_parser = AnnotationParser()
    hpo_lookup = HPOLookup(args.hpo_data)
    converter = PhenoBERTConverter(hpo_lookup, dataset_loader, annotation_parser)
    writer = OutputWriter(args.output)

    # Determine datasets
    datasets = (
        ["GSC+", "ID-68", "GeneReviews"]
        if args.dataset == "all"
        else [args.dataset]
    )

    # Convert each dataset
    for dataset_name in datasets:
        logger.info(f"Converting {dataset_name}...")

        for corpus_file, ann_file in dataset_loader.load_file_pairs(dataset_name):
            try:
                doc = converter.convert_document(corpus_file, ann_file, dataset_name)
                writer.write_document(doc, dataset_name)
            except Exception as e:
                logger.error(f"Error converting {corpus_file.name}: {e}")

    # Write report
    writer.write_report()
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
```

---

## 5. Configuration

### Simple Config (8 Parameters - Not 24!)

```yaml
# config/phenobert_conversion.yaml

# Required paths
phenobert_data_dir: "path/to/PhenoBERT/phenobert/data"
output_dir: "data/test_texts/phenobert"
hpo_data_path: "data/hpo_core_data"

# Dataset selection
datasets:
  - "GSC+"
  - "ID-68"
  - "GeneReviews"

# Logging
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

**That's it! No configuration explosion.**

**Removed unnecessary config:**
- ❌ Validation options (5 params) - use sensible defaults
- ❌ Processing options (4 params) - auto-detect encoding
- ❌ Output options (4 params) - always pretty-print JSON
- ❌ Metadata options (3 params) - auto-generate

---

## 6. Testing

### Focus on Critical Paths (Not Perfect Coverage)

**Unit Tests:**
```python
# tests/unit/data_processing/test_phenobert_converter.py

def test_annotation_parser_raw_format():
    """Test parsing raw GSC+ format."""
    parser = AnnotationParser()
    # Test with sample data

def test_annotation_parser_processed_format():
    """Test parsing processed format."""
    parser = AnnotationParser()
    # Test with sample data

def test_span_validation():
    """Test span validation against text."""
    # Test bounds checking, text matching

def test_hpo_lookup():
    """Test HPO label lookup."""
    lookup = HPOLookup(Path("data/hpo_core_data"))
    assert lookup.get_label("HP:0001263") == "Global developmental delay"
```

**Integration Test:**
```python
def test_convert_single_document(tmp_path):
    """Test end-to-end conversion."""
    # Create test files
    # Run conversion
    # Validate output JSON
```

**Target: >60% coverage** (not >80% - this is alpha!)

---

## 7. Usage

### Quick Start

```bash
# 1. Clone PhenoBERT repository
git clone https://github.com/EclipseCN/PhenoBERT.git

# 2. Run conversion
python scripts/convert_phenobert_data.py \
    --phenobert-data PhenoBERT/phenobert/data \
    --output data/test_texts/phenobert \
    --hpo-data data/hpo_core_data

# 3. Check output
ls data/test_texts/phenobert/
# GSC_plus/    ID68/    GeneReviews/    conversion_report.json
```

### Output Structure

```
data/test_texts/phenobert/
├── GSC_plus/
│   └── annotations/
│       ├── gsc_plus_001.json
│       ├── gsc_plus_002.json
│       └── ... (228 files)
├── ID68/
│   └── annotations/
│       ├── id68_001.json
│       └── ... (68 files)
├── GeneReviews/
│   └── annotations/
│       ├── genereview_001.json
│       └── ... (10 files)
├── conversion_report.json
└── README.md
```

**Simple! No complex nested directories, no unused folders.**

### Loading Converted Data

```python
import json
from pathlib import Path

# Load single document
with open("data/test_texts/phenobert/GSC_plus/annotations/gsc_plus_001.json") as f:
    doc = json.load(f)

print(f"Document: {doc['doc_id']}")
print(f"Annotations: {len(doc['annotations'])}")

# Load all documents from dataset
dataset_dir = Path("data/test_texts/phenobert/GSC_plus/annotations")
for json_file in dataset_dir.glob("*.json"):
    with open(json_file) as f:
        doc = json.load(f)
    # Process document
```

---

## Timeline

### Alpha Software Timeline (2-3 Days)

**Day 1: Implementation**
- [ ] Create module structure
- [ ] Implement DatasetLoader
- [ ] Implement AnnotationParser (unified, DRY)
- [ ] Implement Converter with dependency injection
- [ ] Implement HPOLookup
- [ ] Implement OutputWriter

**Day 2: Conversion & Testing**
- [ ] Write conversion script
- [ ] Run conversion on all datasets
- [ ] Write unit tests (aim for >60% coverage)
- [ ] Fix any errors
- [ ] Generate conversion report

**Day 3: Documentation & Polish**
- [ ] Write single README with usage
- [ ] Add inline docstrings
- [ ] Test end-to-end workflow
- [ ] Commit to repository

**That's it! No excessive documentation, no premature optimization, no over-engineering.**

---

## Success Criteria

**Must Have:**
- ✅ All 306 documents converted
- ✅ Valid JSON format
- ✅ Spans validated against text
- ✅ HPO labels enriched
- ✅ Single README with usage

**Don't Need (Alpha Software):**
- ❌ Perfect test coverage (>80%)
- ❌ Multiple documentation files
- ❌ Benchmarking scripts
- ❌ Comparison tools
- ❌ Complex directory structures
- ❌ Unused translation folders
- ❌ Excessive metadata files

---

## Appendix: Example Output

```json
{
  "doc_id": "gsc_plus_001",
  "language": "en",
  "source": "phenobert",
  "full_text": "Patient presented with global developmental delay, hypotonia, and seizures.",
  "metadata": {
    "dataset": "GSC+",
    "text_length_chars": 76,
    "num_annotations": 3,
    "conversion_date": "2025-01-18T10:30:45Z"
  },
  "annotations": [
    {
      "hpo_id": "HP:0001263",
      "label": "Global developmental delay",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 23,
          "end_char": 48,
          "text_snippet": "global developmental delay",
          "confidence": 1.0
        }
      ]
    },
    {
      "hpo_id": "HP:0001252",
      "label": "Hypotonia",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 50,
          "end_char": 59,
          "text_snippet": "hypotonia"
        }
      ]
    },
    {
      "hpo_id": "HP:0001250",
      "label": "Seizure",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 65,
          "end_char": 73,
          "text_snippet": "seizures",
          "confidence": 0.98
        }
      ]
    }
  ]
}
```

---

**Plan Version:** 2.0 (Simplified)
**Last Updated:** 2025-01-18
**Review Status:** Alpha-focused, ready for implementation

**Key Simplifications from v1.0:**
- ✅ Reduced 10 components → 4 components
- ✅ Eliminated 4 unnecessary scripts (kept only conversion script)
- ✅ Reduced 24 config params → 8 params
- ✅ Simplified directory structure (removed unused translation folders)
- ✅ Unified parser (DRY - no code duplication)
- ✅ Added dependency injection (SOLID compliance)
- ✅ Single validation pass (not 4 levels)
- ✅ Single README (not 5 documentation files)
- ✅ Realistic testing goals (>60%, not >80%)
- ✅ **Alpha software principles throughout**
