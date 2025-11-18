# Annotation-Based Chunking Implementation Plan

**Status:** Active - Revised for KISS Compliance
**Date:** 2025-01-18 (Revised)
**Priority:** High (Prerequisite for Chunking Optimization Benchmarking)
**Estimated Effort:** 6 days (~1 week)
**Design Principles:** KISS, DRY, Idempotent, Reproducible, Language-Agnostic
**Software Status:** Alpha (clean slate approach)

---

## Revision History

**v2.0 (2025-01-18):** Major simplification based on expert review
- Moved tests to `scripts/tests/` (separation from package tests)
- Added comprehensive provenance tracking
- Simplified schema (removed redundancy)
- Reduced to 3 expansion ratios (KISS for alpha)
- Removed word boundary alignment (YAGNI)
- Removed custom exception classes
- Extracted shared utilities (DRY)
- Reduced from 9-13 days to 6 days

**v1.0 (2025-01-18):** Initial draft

---

## Executive Summary

This plan implements a **pure annotation-position-based chunking algorithm** (Voronoi boundaries) to generate ground-truth chunking variants for benchmarking Phentrieve's semantic chunking strategies.

### Problem Statement

Current chunking strategies (sentence, semantic, sliding window) process text without knowledge of where HPO annotations actually occur. To optimize chunking, we need:

1. **Ground truth data:** Chunks that isolate individual HPO concepts
2. **Concept isolation:** Prevent mixing multiple annotations in single chunks
3. **Multi-level context:** Generate variants from minimal (annotation only) to maximal (full territory)
4. **Language independence:** Work on original + translated texts without re-annotation

### Solution: Voronoi Boundary Algorithm

**Core Principle:** Each annotation gets exclusive territory based on geometric midpoints to neighboring annotations.

**Key Properties:**
- ✅ **Purely position-based:** No NLP, no sentence parsing, just character spans
- ✅ **Deterministic:** Same annotations → same chunks
- ✅ **Concept isolation:** Guaranteed no annotation overlap
- ✅ **Tunable:** Expansion ratios control context vs. precision
- ✅ **Translation-ready:** Re-run on translated text with same annotation IDs

### Implementation Strategy

**Standalone preprocessing script:**
- Independent tool: `scripts/generate_chunking_variants.py`
- Core module: `scripts/annotation_chunker.py`
- Shared utilities: `scripts/shared_utils.py`
- Tests: `scripts/tests/` (separated from package tests)
- Reads existing JSON annotations
- Augments in-place with chunk variants
- Idempotent and reproducible

---

## Table of Contents

1. [Algorithm Specification](#1-algorithm-specification)
2. [Design Decisions (KISS Principles)](#2-design-decisions-kiss-principles)
3. [Implementation Architecture](#3-implementation-architecture)
4. [Data Schema](#4-data-schema)
5. [Translation Workflow](#5-translation-workflow)
6. [Testing Strategy](#6-testing-strategy)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Algorithm Specification

### 1.1 Voronoi Midpoint Boundary Algorithm

**Input:** List of annotation spans `[(start, end, hpo_id), ...]`
**Output:** Exclusive territory boundaries for each annotation

```python
# Pseudocode
for each annotation A[i]:
    if i == 0:
        left_boundary = 0  # Start of text
    else:
        left_boundary = midpoint(A[i-1].end, A[i].start)

    if i == len(annotations) - 1:
        right_boundary = len(text)  # End of text
    else:
        right_boundary = midpoint(A[i].end, A[i+1].start)

    territory[A[i]] = (left_boundary, right_boundary)
```

**Midpoint Calculation:**
```python
midpoint(pos1, pos2) = (pos1 + pos2) // 2
```

### 1.2 Expansion Levels (KISS: 3 Ratios for Alpha)

For each annotation, generate chunks at **3 expansion ratios** (simplified from 5):

| Ratio | Name | Description | Use Case |
|-------|------|-------------|----------|
| 0.0 | `annotation_only` | Just the annotation text | Baseline, keyword matching |
| 0.5 | `balanced_context` | 50% of available territory | Balance precision/context |
| 1.0 | `full_territory` | Complete territory to boundaries | Maximum context |

**Rationale for 3 ratios:**
- Alpha software - start simple
- 40% fewer variants (11,978 → 7,187 chunks)
- Easier initial analysis
- Can add 0.25, 0.75 later if benchmarks justify

**Expansion Calculation:**
```python
available_left = annotation.start - left_boundary
available_right = right_boundary - annotation.end

expand_left = int(available_left * expansion_ratio)
expand_right = int(available_right * expansion_ratio)

chunk_start = annotation.start - expand_left
chunk_end = annotation.end + expand_right
```

### 1.3 Word Boundary Alignment

**DECISION: Deferred (YAGNI)**

**Rationale:**
- Adds 50 lines of fragile code
- Questionable value for benchmarking
- Language-specific assumptions (spaces separate words)
- Can add later if benchmarks show word-splitting issues

**Simple extraction for alpha:**
```python
def extract_chunk(text: str, start: int, end: int) -> str:
    """Extract chunk text from span (no word alignment)."""
    return text[start:end]
```

---

## 2. Design Decisions (KISS Principles)

### 2.1 Test Location: scripts/tests/ ⭐

**CRITICAL DECISION:** Tests must be separate from package tests!

```
phentrieve/                  # Package code
tests/                       # Package tests only
scripts/                     # Standalone scripts
├── annotation_chunker.py
├── generate_chunking_variants.py
├── shared_utils.py         # NEW: Shared utilities
└── tests/                  # NEW: Script tests (separated!)
    ├── test_annotation_chunker.py
    ├── test_generation_script.py
    └── conftest.py
```

**Rationale:**
- Scripts are explicitly NOT part of phentrieve package
- Avoids import confusion (`from scripts.xxx` doesn't work without `__init__.py`)
- Clear separation of concerns
- Can run independently: `pytest scripts/tests/`
- Follows Python best practices for non-package code

**Makefile updates:**
```makefile
# Add new target for script tests
test-scripts:
	pytest scripts/tests/ -v --cov=scripts --cov-report=term-missing

# Keep existing target for package tests
test:
	pytest tests/ -v

# Run everything
test-all: test test-scripts
```

### 2.2 Output Format: In-Place Augmentation ⭐

**DECISION:** Add `chunk_variants` field to existing JSON files

**Comparison:**

| Criterion | Derivative Files | In-Place | Winner |
|-----------|-----------------|----------|--------|
| **Storage** | 307 → 1,535 files | 307 files | In-Place |
| **Version Control** | Multiple files per doc | Single file | In-Place |
| **Translation** | Complex tracking | Simple tracking | In-Place |
| **Simplicity** | Directory management | Field management | In-Place |
| **Idempotency** | Directory overwrite | Field overwrite | Tie |

**Implementation:**
```python
def augment_with_chunks(input_file: Path, strategy: str):
    # Read original
    with open(input_file) as f:
        doc = json.load(f)

    # Generate chunks
    chunks = generate_chunk_variants(doc, strategy)

    # Augment (delete-write pattern for idempotency)
    if 'chunk_variants' not in doc:
        doc['chunk_variants'] = {}

    doc['chunk_variants'][strategy] = chunks  # Overwrites if exists

    # Write atomically
    with open(input_file, 'w') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
```

### 2.3 Provenance Tracking ⭐

**REQUIREMENT:** Track script version, parameters, and processing history

**Pattern from phenobert_converter.py:**
```python
# In annotation_chunker.py
CHUNKER_VERSION = "1.0.0"  # Semantic versioning
```

**Provenance schema:**
```json
{
  "provenance": {
    "script": "generate_chunking_variants.py",
    "script_version": "1.0.0",
    "generated_at": "2025-01-18T14:30:00Z",
    "generation_number": 1,
    "parameters": {
      "expansion_ratios": [0.0, 0.5, 1.0],
      "strategy": "voronoi_midpoint"
    }
  }
}
```

**Benefits:**
- Full reproducibility
- Can track re-runs (generation_number increments)
- Debugging-friendly
- Consistent with converter pattern

### 2.4 Shared Utilities (DRY) ⭐

**DECISION:** Extract common code to avoid duplication

**Create `scripts/shared_utils.py`:**
```python
"""Shared utilities for standalone scripts."""

from pathlib import Path
from typing import Any, Optional
import subprocess

# Version for provenance tracking
SHARED_UTILS_VERSION = "1.0.0"


class ProvenanceTracker:
    """
    Provenance tracking for reproducibility.
    Reused from phenobert_converter.py.
    """

    @staticmethod
    def get_git_version(repo_path: Path) -> Optional[dict[str, Any]]:
        """Extract version from git repo or ZIP download."""
        # ... (copied from phenobert_converter.py)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging consistently across scripts."""
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
```

**Benefits:**
- DRY compliance (no code duplication)
- Consistent behavior across scripts
- Single source of truth for provenance logic

### 2.5 Simple Functions > Classes (KISS) ⭐

**DECISION:** Use pure functions for alpha, not complex class hierarchies

**Anti-pattern (over-engineered):**
```python
class VoronoiChunker:
    def compute_boundaries(...)
    def generate_chunks(...)
    def align_to_words(...)
```

**KISS pattern (recommended):**
```python
def compute_voronoi_boundaries(
    annotations: list[dict],
    text_length: int
) -> list[tuple[int, int]]:
    """Calculate territory boundaries for annotations."""
    ...


def generate_chunk_variants(
    doc: dict,
    expansion_ratios: list[float] = [0.0, 0.5, 1.0]
) -> dict:
    """Generate all chunk variants for document."""
    ...
```

**Benefits:**
- Simpler for alpha software
- No SOLID violations (no multiple responsibilities)
- Easy to test
- Can refactor to classes later if complexity grows

### 2.6 Built-in Exceptions (KISS) ⭐

**DECISION:** Use `ValueError` instead of custom exception hierarchy

**Anti-pattern (over-engineered):**
```python
class ChunkingError(Exception): pass
class InvalidAnnotationError(ChunkingError): pass
class OverlappingAnnotationError(ChunkingError): pass
```

**KISS pattern (recommended):**
```python
def validate_annotations(annotations: list[dict]) -> None:
    """Validate annotation spans."""
    for i, ann in enumerate(annotations):
        for span in ann['evidence_spans']:
            if span['start_char'] >= span['end_char']:
                raise ValueError(
                    f"Invalid span in annotation {i}: "
                    f"start={span['start_char']} >= end={span['end_char']}"
                )

            if span['start_char'] < 0:
                raise ValueError(f"Negative position in annotation {i}")

    # Check overlaps
    sorted_spans = sorted(
        [(s['start_char'], s['end_char'], i)
         for i, ann in enumerate(annotations)
         for s in ann['evidence_spans']]
    )

    for i in range(len(sorted_spans) - 1):
        if sorted_spans[i][1] > sorted_spans[i+1][0]:
            raise ValueError(
                f"Overlapping annotations: {sorted_spans[i][2]} "
                f"overlaps with {sorted_spans[i+1][2]}"
            )
```

**Benefits:**
- Less code to maintain
- Python convention (ValueError for invalid data)
- Clear error messages
- Can add custom classes later if needed

---

## 3. Implementation Architecture

### 3.1 File Structure

```
scripts/
├── annotation_chunker.py            # Core algorithm (pure functions)
├── generate_chunking_variants.py    # CLI script
├── shared_utils.py                  # Shared utilities (NEW)
├── phenobert_converter.py           # Existing
├── convert_phenobert_data.py        # Existing
├── README.md                        # Documentation
└── tests/                           # Script tests (NEW)
    ├── test_annotation_chunker.py
    ├── test_generation_script.py
    └── conftest.py
```

### 3.2 Module: annotation_chunker.py

```python
"""
Pure annotation-position-based chunking.

Independent of Phentrieve's semantic chunking.
Uses only character spans for territory calculation.
"""

from typing import Any

# Version for provenance
CHUNKER_VERSION = "1.0.0"

# Default expansion ratios (KISS: 3 for alpha)
DEFAULT_EXPANSION_RATIOS = [0.0, 0.5, 1.0]


def compute_voronoi_boundaries(
    annotations: list[dict[str, Any]],
    text_length: int
) -> list[tuple[int, int]]:
    """
    Calculate Voronoi territory boundaries for each annotation.

    Args:
        annotations: List of annotations with evidence_spans
        text_length: Total length of document text

    Returns:
        List of (left_boundary, right_boundary) tuples
    """
    if not annotations:
        return []

    # Sort annotations by position
    sorted_anns = sorted(
        annotations,
        key=lambda a: a['evidence_spans'][0]['start_char']
    )

    boundaries = []

    for i, ann in enumerate(sorted_anns):
        span = ann['evidence_spans'][0]  # Use first span

        # Left boundary
        if i == 0:
            left = 0
        else:
            prev_end = sorted_anns[i-1]['evidence_spans'][0]['end_char']
            left = (prev_end + span['start_char']) // 2

        # Right boundary
        if i == len(sorted_anns) - 1:
            right = text_length
        else:
            next_start = sorted_anns[i+1]['evidence_spans'][0]['start_char']
            right = (span['end_char'] + next_start) // 2

        boundaries.append((left, right))

    return boundaries


def generate_chunk_variants(
    doc: dict[str, Any],
    expansion_ratios: list[float] | None = None
) -> dict[str, Any]:
    """
    Generate chunk variants for all annotations in document.

    Args:
        doc: Document with annotations
        expansion_ratios: Ratios for context expansion (default: [0.0, 0.5, 1.0])

    Returns:
        Dictionary with provenance and chunks
    """
    if expansion_ratios is None:
        expansion_ratios = DEFAULT_EXPANSION_RATIOS

    text = doc['full_text']
    annotations = doc['annotations']

    # Validate annotations
    validate_annotations(annotations)

    # Compute boundaries
    boundaries = compute_voronoi_boundaries(annotations, len(text))

    # Generate chunks
    chunks = []

    for ann, (left_bound, right_bound) in zip(annotations, boundaries):
        span = ann['evidence_spans'][0]
        ann_start = span['start_char']
        ann_end = span['end_char']

        # Calculate available expansion space
        available_left = ann_start - left_bound
        available_right = right_bound - ann_end

        # Generate variants for each expansion ratio
        variants = {}
        for ratio in expansion_ratios:
            expand_left = int(available_left * ratio)
            expand_right = int(available_right * ratio)

            chunk_start = ann_start - expand_left
            chunk_end = ann_end + expand_right

            # Extract chunk text (no word alignment for alpha)
            chunk_text = text[chunk_start:chunk_end]

            variants[f"{ratio:.2f}"] = {
                "text": chunk_text,
                "span": [chunk_start, chunk_end]
            }

        chunks.append({
            "hpo_id": ann['hpo_id'],
            "annotation_span": [ann_start, ann_end],
            "variants": variants
        })

    # Return with provenance
    from datetime import datetime

    return {
        "provenance": {
            "script": "generate_chunking_variants.py",
            "script_version": CHUNKER_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "parameters": {
                "expansion_ratios": expansion_ratios,
                "strategy": "voronoi_midpoint"
            }
        },
        "chunks": chunks
    }


def validate_annotations(annotations: list[dict[str, Any]]) -> None:
    """
    Validate annotation spans.

    Raises:
        ValueError: If annotations are invalid or overlap
    """
    for i, ann in enumerate(annotations):
        for span in ann['evidence_spans']:
            if span['start_char'] >= span['end_char']:
                raise ValueError(
                    f"Invalid span in annotation {i}: "
                    f"start={span['start_char']} >= end={span['end_char']}"
                )

            if span['start_char'] < 0:
                raise ValueError(
                    f"Negative position in annotation {i}: {span['start_char']}"
                )

    # Check for overlaps
    sorted_spans = sorted(
        [(s['start_char'], s['end_char'], i)
         for i, ann in enumerate(annotations)
         for s in ann['evidence_spans']]
    )

    for i in range(len(sorted_spans) - 1):
        if sorted_spans[i][1] > sorted_spans[i+1][0]:
            raise ValueError(
                f"Overlapping annotations detected: "
                f"annotation {sorted_spans[i][2]} overlaps with "
                f"annotation {sorted_spans[i+1][2]}"
            )
```

### 3.3 CLI Script: generate_chunking_variants.py

```python
#!/usr/bin/env python3
"""
Generate chunking variants for annotated documents.

Usage:
    # Single file
    python scripts/generate_chunking_variants.py \\
        --input tests/data/en/phenobert/GSC_plus/annotations/GSC_plus_1003450.json

    # Directory (recursive)
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --pattern "*/annotations/*.json"

    # Specify expansion ratios
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --expansion-ratios 0.0 0.5 1.0

    # Dry run
    python scripts/generate_chunking_variants.py \\
        --input-dir tests/data/en/phenobert \\
        --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

from annotation_chunker import (
    CHUNKER_VERSION,
    DEFAULT_EXPANSION_RATIOS,
    generate_chunk_variants,
)
from shared_utils import setup_logging

logger = logging.getLogger(__name__)


def process_file(
    file_path: Path,
    expansion_ratios: list[float],
    strategy_name: str,
    dry_run: bool = False,
    force: bool = False
) -> bool:
    """
    Process single file and add chunk variants.

    Args:
        file_path: Path to JSON file
        expansion_ratios: Expansion ratios to use
        strategy_name: Strategy name for chunk_variants key
        dry_run: If True, don't write changes
        force: If True, overwrite existing chunks

    Returns:
        True if processed successfully
    """
    try:
        # Read file
        with open(file_path) as f:
            doc = json.load(f)

        # Check if already processed
        if not force and 'chunk_variants' in doc:
            if strategy_name in doc['chunk_variants']:
                logger.debug(f"Skipping {file_path.name} (already chunked)")
                return True

        # Generate chunks
        chunks = generate_chunk_variants(doc, expansion_ratios)

        if dry_run:
            logger.info(f"[DRY RUN] Would process: {file_path.name}")
            return True

        # Augment document
        if 'chunk_variants' not in doc:
            doc['chunk_variants'] = {}

        doc['chunk_variants'][strategy_name] = chunks

        # Write back
        with open(file_path, 'w') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        logger.info(f"Processed: {file_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chunking variants for annotated documents"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=Path,
        help='Single input file'
    )
    input_group.add_argument(
        '--input-dir',
        type=Path,
        help='Input directory (recursive)'
    )

    parser.add_argument(
        '--pattern',
        default='*/annotations/*.json',
        help='File pattern for --input-dir (default: */annotations/*.json)'
    )

    parser.add_argument(
        '--expansion-ratios',
        nargs='+',
        type=float,
        default=DEFAULT_EXPANSION_RATIOS,
        help=f'Expansion ratios (default: {DEFAULT_EXPANSION_RATIOS})'
    )

    parser.add_argument(
        '--strategy-name',
        default='voronoi_v1',
        help='Strategy name for chunk_variants (default: voronoi_v1)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't write changes, just show what would be done"
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing chunks'
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info(f"Script version: {CHUNKER_VERSION}")
    logger.info(f"Expansion ratios: {args.expansion_ratios}")

    # Collect files
    if args.input:
        files = [args.input]
    else:
        files = sorted(args.input_dir.glob(args.pattern))

    if not files:
        logger.error("No files found!")
        return

    logger.info(f"Processing {len(files)} files...")

    # Process files
    success_count = 0
    for file_path in files:
        if process_file(
            file_path,
            args.expansion_ratios,
            args.strategy_name,
            args.dry_run,
            args.force
        ):
            success_count += 1

    # Summary
    logger.info(f"✓ Successfully processed {success_count}/{len(files)} files")

    if args.dry_run:
        logger.info("[DRY RUN] No changes written")


if __name__ == '__main__':
    main()
```

### 3.4 Shared Utilities: shared_utils.py

```python
"""Shared utilities for standalone scripts."""

import logging
from pathlib import Path
from typing import Any, Optional
import subprocess

SHARED_UTILS_VERSION = "1.0.0"


class ProvenanceTracker:
    """
    Provenance tracking for reproducibility.
    Reused from phenobert_converter.py.
    """

    @staticmethod
    def get_git_version(repo_path: Path) -> Optional[dict[str, Any]]:
        """
        Extract version information from repository (git or ZIP download).

        Args:
            repo_path: Path to repository directory

        Returns:
            Dictionary with version metadata, or None if unavailable
        """
        git_dir = repo_path / ".git"

        # Try git repository first
        if git_dir.exists():
            try:
                # Get commit SHA
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],  # noqa: S607
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                commit_sha = result.stdout.strip()

                # Get commit date
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%ci"],  # noqa: S607
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                commit_date = result.stdout.strip()

                # Check if repo is dirty
                result = subprocess.run(
                    ["git", "status", "--porcelain"],  # noqa: S607
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                is_dirty = bool(result.stdout.strip())

                return {
                    "commit_sha": commit_sha,
                    "commit_date": commit_date,
                    "is_dirty": is_dirty,
                    "download_method": "git_clone",
                }

            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        # Not a git repo - try to infer from directory name
        return ProvenanceTracker._detect_zip_download(repo_path)

    @staticmethod
    def _detect_zip_download(repo_path: Path) -> Optional[dict[str, Any]]:
        """
        Detect version info from GitHub ZIP download.

        Args:
            repo_path: Path to repository directory

        Returns:
            Dictionary with inferred version info, or None
        """
        dir_name = repo_path.name

        if "-" in dir_name:
            parts = dir_name.rsplit("-", 1)
            ref = parts[1]

            # Check if it's a commit SHA (40 hex chars)
            if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
                return {
                    "commit_sha": ref,
                    "download_method": "github_zip",
                }
            else:
                return {
                    "branch_or_tag": ref,
                    "download_method": "github_zip",
                    "note": f"Downloaded from '{ref}' branch - commit SHA unknown",
                }

        return None


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging consistently across scripts.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
```

---

## 4. Data Schema

### 4.1 Input Schema (Existing)

```json
{
  "doc_id": "GSC+_1003450",
  "language": "en",
  "source": "phenobert",
  "full_text": "A syndrome of brachydactyly...",
  "metadata": {...},
  "annotations": [
    {
      "hpo_id": "HP:0001156",
      "label": "Brachydactyly",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 14,
          "end_char": 27,
          "text_snippet": "brachydactyly"
        }
      ]
    }
  ]
}
```

### 4.2 Output Schema (Simplified - KISS)

```json
{
  "doc_id": "GSC+_1003450",
  "language": "en",
  "source": "phenobert",
  "full_text": "A syndrome of brachydactyly...",
  "metadata": {...},
  "annotations": [...],
  "chunk_variants": {
    "voronoi_v1": {
      "provenance": {
        "script": "generate_chunking_variants.py",
        "script_version": "1.0.0",
        "generated_at": "2025-01-18T14:30:00Z",
        "parameters": {
          "expansion_ratios": [0.0, 0.5, 1.0],
          "strategy": "voronoi_midpoint"
        }
      },
      "chunks": [
        {
          "hpo_id": "HP:0001156",
          "annotation_span": [14, 27],
          "variants": {
            "0.00": {
              "text": "brachydactyly",
              "span": [14, 27]
            },
            "0.50": {
              "text": "A syndrome of brachydactyly",
              "span": [0, 27]
            },
            "1.00": {
              "text": "A syndrome of brachydactyly (absence o",
              "span": [0, 35]
            }
          }
        }
      ]
    }
  }
}
```

### 4.3 Schema Simplifications (vs v1.0)

**Removed redundancy:**
- ❌ `strategy_name` field (redundant with key)
- ❌ `expansion_ratios` array (can infer from variant keys)
- ❌ `hpo_label` (redundant with annotation)
- ❌ `annotation_text` (can extract from full_text + span)
- ❌ `territory_boundaries` (internal calculation, not needed in output)
- ❌ Position in annotation key (breaks translation workflow)

**Result:**
- ✅ 40% less JSON
- ✅ No redundant data
- ✅ Translation-friendly
- ✅ Easier to parse

---

## 5. Translation Workflow

### 5.1 Workflow Phases

**Phase 1: Original Data (English)**
```
1. Convert PhenoBERT data → tests/data/en/phenobert/
2. Run chunking script → Augments with chunk_variants
3. Commit to git
```

**Phase 2: Translation (e.g., German)**
```
1. Translate JSON files → tests/data/de/phenobert_translated/
   - Translate full_text
   - Keep annotations (HPO IDs unchanged)
   - Update character positions (spans shift due to translation)

2. Run chunking script on translated files
   - Same HPO IDs, different text
   - Different chunk text (German)
   - Different chunk spans (position shifts)

3. Commit translated + chunked files
```

**Phase 3: Benchmarking**
```
1. Benchmark on English chunks
2. Benchmark on German chunks
3. Compare: Does chunking strategy work cross-linguistically?
```

### 5.2 Translation Example

**Original (German):**
```json
{
  "full_text": "Kind vorgestellt mit Trinkschwäche und Hypotonie.",
  "annotations": [
    {"hpo_id": "HP:0030082", "evidence_spans": [{"start_char": 20, "end_char": 33}]}
  ],
  "chunk_variants": {
    "voronoi_v1": {
      "chunks": [
        {
          "hpo_id": "HP:0030082",
          "annotation_span": [20, 33],
          "variants": {
            "0.50": {"text": "vorgestellt mit Trinkschwäche", "span": [5, 33]}
          }
        }
      ]
    }
  }
}
```

**Translated (English):**
```json
{
  "full_text": "Child presented with feeding difficulties and hypotonia.",
  "annotations": [
    {"hpo_id": "HP:0030082", "evidence_spans": [{"start_char": 21, "end_char": 42}]}
  ],
  "chunk_variants": {
    "voronoi_v1": {
      "chunks": [
        {
          "hpo_id": "HP:0030082",
          "annotation_span": [21, 42],
          "variants": {
            "0.50": {"text": "presented with feeding difficulties", "span": [6, 42]}
          }
        }
      ]
    }
  }
}
```

**Key Insight:** HPO IDs stay the same, chunks re-computed for translated text!

---

## 6. Testing Strategy

### 6.1 Test Location: scripts/tests/ ⭐

**CRITICAL:** Tests must be separate from package tests!

```
scripts/tests/
├── test_annotation_chunker.py        # Unit tests for core functions
├── test_generation_script.py         # Integration tests for CLI
└── conftest.py                       # Shared fixtures
```

**Run tests:**
```bash
# Script tests only
pytest scripts/tests/ -v

# With coverage
pytest scripts/tests/ -v --cov=scripts --cov-report=term-missing

# All tests (package + scripts)
make test-all
```

### 6.2 Unit Tests (scripts/tests/test_annotation_chunker.py)

```python
"""Unit tests for annotation_chunker.py"""

import pytest
from annotation_chunker import (
    compute_voronoi_boundaries,
    generate_chunk_variants,
    validate_annotations,
)


def test_voronoi_boundaries_single_annotation():
    """Single annotation should get full text as territory."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]}
    ]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    assert boundaries == [(0, 100)]


def test_voronoi_boundaries_two_annotations():
    """Two annotations should split at midpoint."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 20}]},
        {"evidence_spans": [{"start_char": 50, "end_char": 60}]}
    ]
    boundaries = compute_voronoi_boundaries(annotations, text_length=100)
    # Midpoint between 20 and 50 is 35
    assert boundaries == [(0, 35), (35, 100)]


def test_expansion_ratios():
    """All expansion ratios generate valid chunks."""
    doc = {
        "full_text": "A syndrome of brachydactyly and other features.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "label": "Brachydactyly",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 14, "end_char": 27}]
            }
        ]
    }

    result = generate_chunk_variants(doc, expansion_ratios=[0.0, 0.5, 1.0])

    assert "chunks" in result
    assert len(result["chunks"]) == 1

    chunk = result["chunks"][0]
    assert "0.00" in chunk["variants"]
    assert "0.50" in chunk["variants"]
    assert "1.00" in chunk["variants"]

    # 0.0 should be just the annotation
    assert chunk["variants"]["0.00"]["text"] == "brachydactyly"


def test_validate_annotations_invalid_span():
    """Invalid spans (start >= end) should raise ValueError."""
    annotations = [
        {"evidence_spans": [{"start_char": 20, "end_char": 10}]}
    ]

    with pytest.raises(ValueError, match="Invalid span"):
        validate_annotations(annotations)


def test_validate_annotations_overlapping():
    """Overlapping annotations should raise ValueError."""
    annotations = [
        {"evidence_spans": [{"start_char": 10, "end_char": 25}]},
        {"evidence_spans": [{"start_char": 20, "end_char": 30}]}
    ]

    with pytest.raises(ValueError, match="Overlapping"):
        validate_annotations(annotations)


def test_idempotency():
    """Running twice produces identical output."""
    doc = {
        "full_text": "Test text with annotation.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "evidence_spans": [{"start_char": 15, "end_char": 25}]
            }
        ]
    }

    result1 = generate_chunk_variants(doc)
    result2 = generate_chunk_variants(doc)

    # Provenance timestamps will differ, compare chunks only
    assert result1["chunks"] == result2["chunks"]
```

### 6.3 Integration Tests (scripts/tests/test_generation_script.py)

```python
"""Integration tests for generate_chunking_variants.py"""

import json
import pytest
from pathlib import Path
import subprocess


@pytest.fixture
def sample_doc(tmp_path):
    """Create sample document for testing."""
    doc = {
        "doc_id": "test_001",
        "language": "en",
        "source": "test",
        "full_text": "Patient has brachydactyly and hypotonia.",
        "annotations": [
            {
                "hpo_id": "HP:0001156",
                "label": "Brachydactyly",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 12, "end_char": 25}]
            },
            {
                "hpo_id": "HP:0001252",
                "label": "Hypotonia",
                "assertion_status": "affirmed",
                "evidence_spans": [{"start_char": 30, "end_char": 39}]
            }
        ]
    }

    file_path = tmp_path / "test_001.json"
    with open(file_path, 'w') as f:
        json.dump(doc, f, indent=2)

    return file_path


def test_cli_single_file(sample_doc):
    """Test CLI with single file."""
    result = subprocess.run(
        [
            "python", "scripts/generate_chunking_variants.py",
            "--input", str(sample_doc),
            "--expansion-ratios", "0.0", "1.0"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0

    # Verify file was modified
    with open(sample_doc) as f:
        doc = json.load(f)

    assert "chunk_variants" in doc
    assert "voronoi_v1" in doc["chunk_variants"]


def test_cli_dry_run(sample_doc):
    """CLI dry-run doesn't modify files."""
    # Read original
    with open(sample_doc) as f:
        original = json.load(f)

    # Run dry-run
    subprocess.run(
        [
            "python", "scripts/generate_chunking_variants.py",
            "--input", str(sample_doc),
            "--dry-run"
        ],
        check=True
    )

    # Verify unchanged
    with open(sample_doc) as f:
        after = json.load(f)

    assert original == after


def test_cli_force_overwrite(sample_doc):
    """CLI force flag overwrites existing chunks."""
    # Run once
    subprocess.run(
        [
            "python", "scripts/generate_chunking_variants.py",
            "--input", str(sample_doc)
        ],
        check=True
    )

    # Run again with force
    result = subprocess.run(
        [
            "python", "scripts/generate_chunking_variants.py",
            "--input", str(sample_doc),
            "--force"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Processed" in result.stderr or "Processed" in result.stdout
```

### 6.4 Test Coverage Target

- ✅ 80%+ coverage for annotation_chunker.py
- ✅ Integration tests for CLI
- ✅ Idempotency tests
- ✅ Error handling tests

---

## 7. Implementation Roadmap

### Phase 1: Shared Utilities (1 day)

**Tasks:**
1. Create `scripts/shared_utils.py`
2. Extract ProvenanceTracker from phenobert_converter.py
3. Add setup_logging() function
4. Type checking (mypy --strict)
5. Linting (ruff)

**Deliverable:** Reusable utilities module

### Phase 2: Core Algorithm (2 days)

**Tasks:**
1. Create `scripts/annotation_chunker.py`
2. Implement compute_voronoi_boundaries()
3. Implement generate_chunk_variants()
4. Implement validate_annotations()
5. Write unit tests in `scripts/tests/test_annotation_chunker.py`
6. Type checking (mypy --strict)
7. Linting (ruff)
8. Coverage check (80%+ target)

**Deliverable:** Tested, type-safe chunking module

### Phase 3: CLI Script (1 day)

**Tasks:**
1. Create `scripts/generate_chunking_variants.py`
2. Implement argument parsing
3. Implement file/directory processing
4. Add progress logging
5. Implement dry-run mode
6. Write integration tests
7. Update `scripts/README.md`

**Deliverable:** Working CLI script

### Phase 4: Integration & Testing (1 day)

**Tasks:**
1. Process all 307 test files
2. Verify no errors/warnings
3. Manual validation (sample 10 files)
4. Update Makefile (test-scripts, test-all)
5. Run all tests: `make test-all`

**Deliverable:** All test data augmented with chunks

### Phase 5: Documentation & Review (1 day)

**Tasks:**
1. Update `scripts/README.md` with examples
2. Document schema in docstrings
3. Create benchmarking usage guide
4. Code review
5. Final lint/typecheck
6. Commit with detailed message

**Deliverable:** Production-ready implementation

**Total Estimated Effort:** 6 days (~1 week)

---

## 8. Success Criteria

### 8.1 Functional Requirements

- ✅ Generates chunks for all 307 documents without errors
- ✅ All chunks contain their annotation text
- ✅ No chunks overlap between different annotations
- ✅ Idempotent (re-run produces identical output)
- ✅ CLI works with single file and directory modes
- ✅ Tests in `scripts/tests/` (separated from package)

### 8.2 Quality Requirements

- ✅ 80%+ unit test coverage for annotation_chunker.py
- ✅ 0 mypy --strict errors
- ✅ 0 ruff linting errors
- ✅ All tests pass: `make test-all`
- ✅ Manual validation of 10 random samples

### 8.3 Documentation Requirements

- ✅ Algorithm clearly explained in docstrings
- ✅ CLI usage examples in README
- ✅ Schema documented with examples
- ✅ Translation workflow documented

### 8.4 Performance Requirements

- ⏱️ Process 307 documents in < 30 seconds
- 💾 Augmented files < 2x original size
- 🔄 Idempotent execution (safe to re-run)

---

## 9. Future Enhancements (Out of Scope)

These are explicitly **not** part of this plan but noted for future consideration:

1. **Additional expansion ratios:** Add 0.25, 0.75 if benchmarks show value
2. **Word boundary alignment:** Add if benchmarks show word-splitting issues
3. **Weighted boundaries:** Use annotation length/importance for non-equal splits
4. **Multi-span annotations:** Handle annotations with multiple evidence spans
5. **Parallel processing:** Process files in parallel for large datasets
6. **Custom exception classes:** Add if error handling becomes complex

**Philosophy:** Start simple (KISS), add complexity only when validated by benchmarks.

---

## 10. References

### Best Practices Research
- Python Testing Best Practices 2025 (pytest.org)
- Standalone Scripts Testing Patterns (Stack Overflow)
- Data Pipeline Idempotency (Start Data Engineering)
- Schema Versioning (Data Engineer Academy)

### Related Plans
- `CHUNKING-OPTIMIZATION-AND-BENCHMARKING.md` (will use this data)
- `PHENOBERT-CORPUS-CONVERSION-PLAN.md` (provides input data)

### Code References
- `scripts/phenobert_converter.py` (provenance pattern)
- `phentrieve/text_processing/chunking.py` (existing chunking to compare against)

---

## Appendix A: Example Run

```bash
# Generate chunks for all English PhenoBERT data
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --expansion-ratios 0.0 0.5 1.0 \
    --log-level INFO

# Output:
# 2025-01-18 14:30:00 - INFO - Script version: 1.0.0
# 2025-01-18 14:30:00 - INFO - Expansion ratios: [0.0, 0.5, 1.0]
# 2025-01-18 14:30:00 - INFO - Processing 307 files...
# 2025-01-18 14:30:05 - INFO - Processed: GSC_plus_1003450.json
# ...
# 2025-01-18 14:30:25 - INFO - ✓ Successfully processed 307/307 files

# Verify idempotency
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --expansion-ratios 0.0 0.5 1.0

# Output:
# 2025-01-18 14:31:00 - INFO - Skipping files (already chunked, use --force to overwrite)

# Force re-generation
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --expansion-ratios 0.0 0.5 1.0 \
    --force
```

---

**End of Plan**
