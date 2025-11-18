# PhenoBERT Corpus Conversion Plan

**Status:** Active
**Date:** 2025-01-18
**Priority:** High
**Related Issues:** [#17](https://github.com/berntpopp/phentrieve/issues/17), [#25](https://github.com/berntpopp/phentrieve/issues/25)
**Related Plans:** [CHUNKING-OPTIMIZATION-AND-BENCHMARKING.md](./CHUNKING-OPTIMIZATION-AND-BENCHMARKING.md)
**Estimated Effort:** 3-5 days

## Executive Summary

This plan outlines the conversion of **three corpus datasets from PhenoBERT** to Phentrieve's standardized JSON annotation format. The conversion will be **reproducible, configurable, and well-documented**, enabling immediate integration into our benchmarking infrastructure.

**Datasets to Convert:**
1. **GSC+ (BiolarkGSC+):** 228 manually annotated PubMed abstracts (1,933 annotations, 497 unique HPO concepts)
2. **ID-68:** 68 de-identified clinical notes from intellectual disability families
3. **GeneReviews:** 10 clinical cases from the GeneReviews database

**Key Deliverables:**
- ✅ Reproducible conversion scripts with comprehensive configuration
- ✅ Validated JSON files in Phentrieve format (306 documents total)
- ✅ Data quality reports with statistics and validation results
- ✅ Integration with existing benchmark infrastructure
- ✅ Complete documentation and usage examples

---

## Table of Contents

1. [Data Source Analysis](#1-data-source-analysis)
2. [Target Format Specification](#2-target-format-specification)
3. [Conversion Architecture](#3-conversion-architecture)
4. [Implementation Specifications](#4-implementation-specifications)
5. [Data Quality Assurance](#5-data-quality-assurance)
6. [Integration Plan](#6-integration-plan)
7. [Configuration & Reproducibility](#7-configuration--reproducibility)
8. [Testing Strategy](#8-testing-strategy)
9. [Documentation Requirements](#9-documentation-requirements)
10. [Timeline & Milestones](#10-timeline--milestones)

---

## 1. Data Source Analysis

### 1.1 Source Repository

**Repository:** https://github.com/EclipseCN/PhenoBERT
**Data Location:** `phenobert/data/`
**License:** To be verified (check repository)

### 1.2 Directory Structure

```
phenobert/data/
├── GSC+/
│   ├── corpus/          # Text files (228 files)
│   ├── ann/             # Processed annotations (228 files)
│   └── raw_ann/         # Raw annotations (228 files)
│
├── ID-68/
│   ├── corpus/          # Text files (68 files)
│   └── ann/             # Annotations (68 files)
│
├── GeneReviews/
│   ├── corpus/          # Text files (10 files)
│   └── ann/             # Annotations (10 files)
│
├── val/                 # Validation set (30 files, format TBD)
├── hpo.json             # HPO ontology (11.1 MB)
├── hpo.obo              # HPO ontology (OBO format)
├── gene_reviews.idx     # GeneReviews index
├── NUM.txt              # Numeric identifiers
└── stopwords.txt        # Stopwords list
```

**Key Observations:**
- **Separate files:** Text (`corpus/`) and annotations (`ann/`) stored separately
- **Filename matching:** Annotations match corpus filenames (e.g., `doc001.txt` ↔ `doc001.ann`)
- **Raw vs processed:** GSC+ includes both raw and processed annotations

### 1.3 Dataset Characteristics

#### GSC+ (BiolarkGSC+)
- **Size:** 228 PubMed abstracts
- **Source:** Manually annotated by Groza et al. (2015), refined by Lobo et al. (2017)
- **Content:** Scientific publication abstracts covering 44 complex dysmorphology syndromes
- **Annotations:** 1,933 annotations covering 497 unique HPO concepts (2,773 mentions)
- **Document length:** Min 138 chars, Max 2,417 chars, Avg ~500 chars
- **Language:** English
- **Annotation level:** Mention-level with character offsets

#### ID-68
- **Size:** 68 de-identified clinical notes
- **Source:** Real-world clinical notes from families with intellectual disabilities
- **Content:** Clinical narratives describing phenotypes in ID patients
- **Annotations:** HPO terms annotated in same format as GSC+
- **Language:** English (presumed)
- **Annotation level:** Mention-level with character offsets

#### GeneReviews
- **Size:** 10 clinical cases
- **Source:** GeneReviews database (NCBI)
- **Content:** Clinical case descriptions from genetic disease reviews
- **Annotations:** HPO terms with character offsets
- **Language:** English
- **Annotation level:** Mention-level with character offsets

### 1.4 Source Annotation Formats

#### Format 1: GSC/GSC+ Raw Format
**File:** `raw_ann/*.ann`
**Format:** `[start::end] HPO_ID | annotation_text`

**Example:**
```
[27::42] HP_0000110 | renal dysplasia
[56::78] HP_0001263 | Global developmental delay
[102::115] HP_0002119 | Ventriculomegaly
```

**Characteristics:**
- Character offsets with `::` separator
- HPO ID with underscore: `HP_NNNNNNN`
- Pipe `|` separates ID from text
- One annotation per line
- Offsets are inclusive ranges `[start, end]`

#### Format 2: PhenoBERT Processed Format
**File:** `ann/*.ann`
**Format:** Tab-separated with 5 columns

**Example:**
```
9	17	headache	HP:0002315	1.0
45	68	developmental delay	HP:0001263	1.0
102	115	seizures	HP:0001250	0.95
```

**Columns:**
1. `start` - Start character offset (0-indexed)
2. `end` - End character offset (exclusive)
3. `text_span` - Extracted text snippet
4. `hpo_id` - HPO identifier (format: `HP:NNNNNNN`)
5. `confidence` - Confidence score (0.0-1.0)

**Characteristics:**
- Tab-separated values (TSV)
- HPO ID with colon: `HP:NNNNNNN`
- Confidence scores included
- 0-indexed offsets (end is exclusive)

#### Format Comparison

| Feature | Raw Format | Processed Format |
|---------|------------|------------------|
| **Separator** | `[start::end]` | Tab-separated |
| **HPO ID** | `HP_NNNNNNN` | `HP:NNNNNNN` |
| **Offsets** | Inclusive `[start, end]` | Exclusive `[start, end)` |
| **Confidence** | None | 0.0-1.0 |
| **Text snippet** | After `\|` | Column 3 |

### 1.5 Data Quality Considerations

**Known Issues:**
1. **Character encoding:** May contain UTF-8 special characters
2. **Offset accuracy:** Need to verify against source text
3. **HPO ID format variation:** Underscore vs colon
4. **Missing metadata:** No assertion status, onset, frequency, etc.
5. **Language detection:** Assumed English, needs verification
6. **Overlapping spans:** Multiple annotations may overlap
7. **Text normalization:** Original text may have whitespace variations

**Validation Required:**
- ✅ Verify all offsets point to correct text spans
- ✅ Check HPO IDs exist in ontology
- ✅ Validate character encodings
- ✅ Ensure corpus-annotation filename matching
- ✅ Detect any missing files

---

## 2. Target Format Specification

### 2.1 Phentrieve JSON Schema

**Format:** Based on `data/test_texts/full_text_hpo_annotations.json`

```json
{
  "doc_id": "string",
  "language": "string",
  "source": "string",
  "source_id": "string (optional)",
  "full_text": "string",
  "metadata": {
    "dataset": "string",
    "original_filename": "string",
    "text_length_chars": "integer",
    "text_length_words": "integer",
    "num_annotations": "integer",
    "num_unique_hpo_terms": "integer",
    "conversion_date": "ISO-8601 datetime",
    "converter_version": "string"
  },
  "annotations": [
    {
      "hpo_id": "string",
      "label": "string",
      "assertion_status": "affirmed | negated | uncertain",
      "evidence_spans": [
        {
          "start_char": "integer",
          "end_char": "integer",
          "text_snippet": "string",
          "confidence": "float (optional)"
        }
      ],
      "frequency": {
        "code": "string (optional)",
        "label": "string (optional)"
      },
      "onset": {
        "code": "string (optional)",
        "label": "string (optional)"
      }
    }
  ]
}
```

### 2.2 Field Mappings

#### Document Level

| Target Field | Source | Transformation |
|--------------|--------|----------------|
| `doc_id` | Derived | `{dataset}_{filename_without_ext}` |
| `language` | Constant | `"en"` (verified during conversion) |
| `source` | Constant | `"phenobert"` or dataset-specific |
| `source_id` | Derived | Original filename |
| `full_text` | `corpus/*.txt` | Read entire file, preserve formatting |

#### Metadata

| Target Field | Source | Transformation |
|--------------|--------|----------------|
| `dataset` | Constant | `"GSC+"`, `"ID-68"`, or `"GeneReviews"` |
| `original_filename` | File path | Basename of corpus file |
| `text_length_chars` | Calculated | `len(full_text)` |
| `text_length_words` | Calculated | `len(full_text.split())` |
| `num_annotations` | Calculated | Count of annotations |
| `num_unique_hpo_terms` | Calculated | Count of unique HPO IDs |
| `conversion_date` | Generated | ISO-8601 timestamp |
| `converter_version` | Constant | Script version string |

#### Annotation Level

| Target Field | Source | Transformation |
|--------------|--------|----------------|
| `hpo_id` | `ann/*.ann` col 4 | Normalize to `HP:NNNNNNN` format |
| `label` | HPO lookup | Fetch from `hpo.json` or API |
| `assertion_status` | Default | `"affirmed"` (no negation in source) |
| `evidence_spans` | `ann/*.ann` cols 1-3 | Array of span objects |
| `start_char` | `ann/*.ann` col 1 | Integer (0-indexed) |
| `end_char` | `ann/*.ann` col 2 | Integer (exclusive end) |
| `text_snippet` | `ann/*.ann` col 3 | String (validate against text) |
| `confidence` | `ann/*.ann` col 5 | Float (optional, if present) |

### 2.3 Default Values

**When source data is missing:**
- `assertion_status`: `"affirmed"` (default, no negation detection in PhenoBERT data)
- `confidence`: Omitted if not in source (GSC+ raw format)
- `frequency`: Omitted (not in source data)
- `onset`: Omitted (not in source data)
- `language`: `"en"` (verified by checking text)

### 2.4 HPO Label Lookup Strategy

**Options:**

1. **Use included `hpo.json`** (11.1 MB file in PhenoBERT repo)
   - ✅ Self-contained, no external dependencies
   - ✅ Fast lookup
   - ❌ May be outdated (need to check version)

2. **Use Phentrieve's HPO data** (`data/hpo_core_data/`)
   - ✅ Already in Phentrieve's data directory
   - ✅ Consistent with other components
   - ❌ Need to ensure it's up-to-date

3. **Use HPO API** (https://hpo.jax.org/api/)
   - ✅ Always up-to-date
   - ❌ Requires internet connection
   - ❌ Slow (rate limits)

**Recommended:** Use Phentrieve's HPO data with fallback to API for missing terms.

---

## 3. Conversion Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Conversion Pipeline                      │
└─────────────────────────────────────────────────────────────┘

┌───────────────┐
│ Configuration │
│   (YAML)      │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│              Dataset Scanner & Discovery                   │
│  - Scan phenobert/data/ for datasets                      │
│  - Discover corpus and annotation files                   │
│  - Validate directory structure                           │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  For Each Dataset     │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              File Pair Matcher                            │
│  - Match corpus files with annotation files              │
│  - Validate filename correspondence                      │
│  - Report any orphaned files                             │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  For Each File Pair   │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              Text Loader                                   │
│  - Read corpus text file                                  │
│  - Detect encoding (UTF-8, Latin-1, etc.)                │
│  - Normalize line endings                                 │
│  - Calculate text statistics                              │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              Annotation Parser                             │
│  - Auto-detect format (raw vs processed)                 │
│  - Parse annotation file                                  │
│  - Normalize HPO IDs (HP_NNNN → HP:NNNN)                │
│  - Group spans by HPO term                                │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              Span Validator                                │
│  - Verify offsets point to correct text                  │
│  - Check text_snippet matches extracted text             │
│  - Flag invalid/out-of-bounds offsets                    │
│  - Report discrepancies                                   │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              HPO Label Enricher                            │
│  - Load HPO ontology data                                 │
│  - Lookup HPO labels for each ID                         │
│  - Cache lookups for performance                         │
│  - Flag unknown HPO IDs                                   │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              JSON Constructor                              │
│  - Build Phentrieve JSON structure                        │
│  - Add metadata fields                                     │
│  - Format annotations array                               │
│  - Add evidence_spans                                      │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              Validator                                     │
│  - Validate JSON schema                                   │
│  - Check required fields                                  │
│  - Verify data types                                      │
│  - Run quality checks                                     │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              JSON Writer                                   │
│  - Write formatted JSON to output directory              │
│  - Organize by dataset                                    │
│  - Generate summary statistics                            │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  End of File Pair     │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              Report Generator                              │
│  - Generate conversion report                             │
│  - Create dataset statistics                              │
│  - Log validation issues                                  │
│  - Output quality metrics                                 │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Component Descriptions

#### 3.2.1 Configuration Manager
**Responsibility:** Load and validate configuration from YAML file.

**Key Settings:**
- Input data directory path
- Output directory path
- HPO data source
- Datasets to convert (all, or specific subsets)
- Validation strictness level
- Logging configuration

#### 3.2.2 Dataset Scanner
**Responsibility:** Discover available datasets and files.

**Tasks:**
- Scan `phenobert/data/` for dataset directories
- Identify corpus and annotation subdirectories
- List all files in each directory
- Validate directory structure matches expected format

#### 3.2.3 File Pair Matcher
**Responsibility:** Match corpus files with annotation files.

**Algorithm:**
```python
for corpus_file in corpus_files:
    basename = corpus_file.stem  # filename without extension
    ann_file = ann_dir / f"{basename}.ann"

    if ann_file.exists():
        yield (corpus_file, ann_file)
    else:
        log_warning(f"No annotation file for {corpus_file}")
```

#### 3.2.4 Text Loader
**Responsibility:** Read and normalize corpus text.

**Features:**
- Auto-detect encoding (try UTF-8, then Latin-1, then chardet)
- Normalize line endings (CRLF → LF)
- Strip leading/trailing whitespace (optional, configurable)
- Calculate text statistics (length, word count)

#### 3.2.5 Annotation Parser
**Responsibility:** Parse annotation files into structured data.

**Format Detection:**
```python
def detect_format(line: str) -> str:
    if line.startswith("[") and "::" in line:
        return "raw"  # GSC+ raw format
    elif "\t" in line:
        return "processed"  # PhenoBERT processed format
    else:
        raise ValueError("Unknown format")
```

**Parsing:**
```python
# Raw format: [27::42] HP_0000110 | renal dysplasia
pattern_raw = r"\[(\d+)::(\d+)\]\s+(HP_\d+)\s+\|\s+(.+)"

# Processed format: 9\t17\theadache\tHP:0002315\t1.0
pattern_processed = r"^(\d+)\t(\d+)\t([^\t]+)\t(HP:\d+)(?:\t([\d.]+))?$"
```

#### 3.2.6 Span Validator
**Responsibility:** Verify annotation offsets match text.

**Validation:**
```python
def validate_span(text: str, start: int, end: int, expected_snippet: str) -> bool:
    actual_snippet = text[start:end]

    # Exact match
    if actual_snippet == expected_snippet:
        return True

    # Fuzzy match (allow minor whitespace differences)
    if actual_snippet.strip() == expected_snippet.strip():
        log_warning(f"Whitespace mismatch at [{start}:{end}]")
        return True

    # Character-by-character comparison for detailed error
    log_error(f"Span mismatch: expected '{expected_snippet}', got '{actual_snippet}'")
    return False
```

#### 3.2.7 HPO Label Enricher
**Responsibility:** Add HPO term labels to annotations.

**Strategy:**
1. Load HPO data from Phentrieve's `data/hpo_core_data/hpo_terms.tsv`
2. Build in-memory lookup dictionary: `{hpo_id: label}`
3. Cache lookups for repeated terms
4. Fallback to HPO API for missing terms (optional)

**Format:**
```python
hpo_lookup = {
    "HP:0001263": "Global developmental delay",
    "HP:0002119": "Ventriculomegaly",
    "HP:0000110": "Renal dysplasia",
    # ...
}
```

#### 3.2.8 JSON Constructor
**Responsibility:** Build Phentrieve JSON structure.

**Implementation:**
```python
def construct_json(
    doc_id: str,
    full_text: str,
    annotations: List[Annotation],
    metadata: dict,
) -> dict:
    # Group annotations by HPO ID
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann.hpo_id].append(ann)

    # Build annotations array
    json_annotations = []
    for hpo_id, spans in grouped.items():
        json_annotations.append({
            "hpo_id": hpo_id,
            "label": hpo_lookup.get(hpo_id, "UNKNOWN"),
            "assertion_status": "affirmed",
            "evidence_spans": [
                {
                    "start_char": span.start,
                    "end_char": span.end,
                    "text_snippet": span.text,
                    "confidence": span.confidence,
                }
                for span in spans
            ],
        })

    return {
        "doc_id": doc_id,
        "language": "en",
        "source": "phenobert",
        "full_text": full_text,
        "metadata": metadata,
        "annotations": json_annotations,
    }
```

#### 3.2.9 Validator
**Responsibility:** Validate JSON output against schema.

**Checks:**
- Required fields present
- Data types correct
- HPO IDs valid format
- Offsets within text bounds
- No duplicate evidence spans
- Confidence scores in [0, 1] range

#### 3.2.10 Report Generator
**Responsibility:** Generate conversion summary report.

**Outputs:**
- Number of documents converted per dataset
- Number of annotations per dataset
- Number of unique HPO terms
- Validation errors/warnings
- Processing time statistics
- Data quality metrics

---

## 4. Implementation Specifications

### 4.1 Module Structure

#### 4.1.1 Architecture Philosophy

**Separation of Concerns:**

1. **`phentrieve/` package** - Core library code
   - Reusable components
   - Well-tested, stable APIs
   - Integrated into main CLI

2. **`scripts/` folder** - Standalone utility scripts
   - Data conversion and preprocessing
   - Validation and quality checks
   - Benchmarking and analysis
   - One-off tasks and experiments
   - May evolve into CLI commands later

**Why `scripts/` for Conversion:**
- ✅ **Not user-facing** - Data conversion is typically one-time setup
- ✅ **Requires external data** - PhenoBERT repo not part of Phentrieve
- ✅ **May be dataset-specific** - Other converters will be added
- ✅ **Easier to maintain** - Can be updated independently
- ✅ **Clear separation** - Preprocessing vs. runtime functionality

**When to Move to CLI:**
- When conversion becomes a common user workflow
- When we have multiple stable dataset converters
- When we add automatic dataset discovery/download

#### 4.1.2 Directory Structure

**Design Rationale:**

After analyzing Phentrieve's existing structure and NLP corpus best practices, the optimal organization:

1. ✅ **Store in `data/test_texts/`** - Existing location for test corpus data
2. ✅ **Separate source from translations** - Clear distinction between original and derived data
3. ✅ **Language-organized translations** - ISO 639-1 codes (de, fr, es, etc.)
4. ✅ **Parallel corpus structure** - Same organization across languages for easy comparison
5. ✅ **Follows existing patterns** - Matches `data/test_cases/` which already has translations

**Evidence from Existing Phentrieve:**
- `data/test_cases/` already contains multilingual test data
- `data/test_cases/expanded_test_200cases_gemini25translated.json` shows translation support
- `data/test_texts/full_text_hpo_annotations.json` shows single-file format

```
phentrieve/
├── data_processing/
│   ├── __init__.py
│   ├── phenobert_converter.py       # Main converter class
│   ├── annotation_parsers.py         # Annotation format parsers
│   ├── validators.py                 # Validation utilities
│   └── hpo_lookup.py                 # HPO label lookup
│
├── evaluation/
│   ├── dataset_loader.py             # UPDATED: Load benchmark datasets
│   ├── corpus_registry.py            # NEW: Corpus discovery & loading
│   └── ... (existing files)
│
└── cli/
    └── ... (existing CLI commands)

scripts/
├── README.md                         # Scripts documentation
├── convert_phenobert_data.py         # PhenoBERT corpus conversion
├── validate_converted_data.py        # Post-conversion validation
├── generate_dataset_statistics.py    # Dataset statistics report
├── compare_annotation_formats.py     # Compare original vs converted
└── benchmark_conversion_speed.py     # Performance benchmarking

config/
└── phenobert_conversion.yaml         # Conversion configuration

data/
├── test_cases/                       # EXISTING - Short test cases
│   ├── sample_test_cases.json
│   ├── expanded_test_200cases_gemini25translated.json
│   └── ...
│
├── test_texts/                       # EXISTING - Full-text corpus data
│   ├── full_text_hpo_annotations.json    # Existing single file
│   │
│   └── corpora/                      # NEW - Organized corpus collection
│       │
│       ├── phenobert/                # Corpus name
│       │   ├── README.md             # Corpus-specific documentation
│       │   ├── corpus_info.json      # Corpus metadata & catalog
│       │   │
│       │   ├── source/               # Original data (English)
│       │   │   ├── GSC_plus/
│       │   │   │   ├── dataset_info.json
│       │   │   │   └── annotations/
│       │   │   │       ├── gsc_plus_001.json
│       │   │   │       ├── gsc_plus_002.json
│       │   │   │       └── ... (228 files)
│       │   │   │
│       │   │   ├── ID68/
│       │   │   │   ├── dataset_info.json
│       │   │   │   └── annotations/
│       │   │   │       ├── id68_001.json
│       │   │   │       └── ... (68 files)
│       │   │   │
│       │   │   └── GeneReviews/
│       │   │       ├── dataset_info.json
│       │   │       └── annotations/
│       │   │           ├── genereview_001.json
│       │   │           └── ... (10 files)
│       │   │
│       │   └── translations/         # Future translations
│       │       │
│       │       ├── de/               # German translations
│       │       │   ├── GSC_plus/
│       │       │   │   ├── dataset_info.json
│       │       │   │   ├── translation_info.json
│       │       │   │   └── annotations/
│       │       │   │       └── ... (228 files)
│       │       │   ├── ID68/
│       │       │   └── GeneReviews/
│       │       │
│       │       ├── fr/               # French translations
│       │       │   └── ... (same structure)
│       │       │
│       │       ├── es/               # Spanish translations
│       │       │   └── ... (same structure)
│       │       │
│       │       └── nl/               # Dutch translations
│       │           └── ... (same structure)
│       │
│       └── other_corpus/             # Future corpus (e.g., LIRICAL, BioCreative)
│           ├── README.md
│           ├── corpus_info.json
│           ├── source/
│           └── translations/
│
├── hpo_core_data/                    # EXISTING - HPO ontology data
└── ... (other data directories)

tests/
├── fixtures/                         # NEW - Test fixtures
│   ├── corpus_fixtures.py            # Fixtures to load corpus data
│   └── ...
│
├── test_data/                        # EXISTING - Empty, could use
└── ... (existing test directories)
```

#### 4.1.3 Scripts Folder Organization

**Purpose:** Standalone scripts for data processing, validation, and benchmarking that don't belong in the main phentrieve CLI.

**Script Categories:**

1. **Data Conversion** (`convert_*.py`)
   - Convert external datasets to Phentrieve format
   - One script per dataset/format
   - Examples: `convert_phenobert_data.py`, `convert_biocreative_data.py`

2. **Validation** (`validate_*.py`)
   - Validate converted data quality
   - Check schema compliance
   - Examples: `validate_converted_data.py`, `validate_hpo_ids.py`

3. **Analysis** (`analyze_*.py`, `generate_*.py`)
   - Generate statistics and reports
   - Compare datasets
   - Examples: `generate_dataset_statistics.py`, `analyze_annotation_overlap.py`

4. **Benchmarking** (`benchmark_*.py`)
   - Performance testing
   - Quality comparison
   - Examples: `benchmark_conversion_speed.py`, `benchmark_retrieval_accuracy.py`

**Script Guidelines:**
- ✅ Each script should be runnable standalone with `python scripts/script_name.py`
- ✅ Use argparse for CLI arguments
- ✅ Support `--help` flag with clear documentation
- ✅ Include docstring with usage examples
- ✅ Import from `phentrieve` package for shared functionality
- ✅ Write output to `results/` or `data/benchmark_datasets/`
- ✅ Log to console and/or log files
- ✅ Handle errors gracefully

**Example Script Template:**
```python
#!/usr/bin/env python3
"""
Brief description of what this script does.

Usage:
    python scripts/script_name.py --input data.txt --output results.json
    python scripts/script_name.py --help

Requirements:
    - Phentrieve installed (pip install -e .)
    - External data downloaded to data/
"""

import argparse
import logging
from pathlib import Path

from phentrieve.data_processing import SomeUtility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Add arguments...
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run main logic
    # ...


if __name__ == "__main__":
    main()
```

### 4.2 Core Classes

#### 4.2.1 `PhenoBERTConverter`

```python
class PhenoBERTConverter:
    """
    Main converter for PhenoBERT datasets to Phentrieve format.

    Features:
    - Auto-detects annotation formats
    - Validates all conversions
    - Generates detailed reports
    - Configurable via YAML
    """

    def __init__(self, config: dict):
        self.config = config
        self.hpo_lookup = HPOLookup(config["hpo_data_path"])
        self.stats = ConversionStats()

    def convert_dataset(
        self,
        dataset_name: str,
        corpus_dir: Path,
        ann_dir: Path,
        output_dir: Path,
    ) -> ConversionResult:
        """Convert entire dataset to JSON format."""
        pass

    def convert_document(
        self,
        corpus_file: Path,
        ann_file: Path,
        dataset_name: str,
    ) -> dict:
        """Convert single document to JSON format."""
        pass

    def validate_conversion(self, json_doc: dict) -> List[ValidationError]:
        """Validate converted JSON document."""
        pass

    def generate_report(self) -> dict:
        """Generate conversion summary report."""
        pass
```

#### 4.2.2 `AnnotationParser`

```python
class AnnotationParser:
    """Base class for annotation parsers."""

    @abstractmethod
    def parse(self, ann_file: Path) -> List[Annotation]:
        """Parse annotation file."""
        pass


class RawFormatParser(AnnotationParser):
    """Parser for GSC+ raw format: [start::end] HPO_ID | text"""

    def parse(self, ann_file: Path) -> List[Annotation]:
        annotations = []

        with open(ann_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse: [27::42] HP_0000110 | renal dysplasia
                match = re.match(
                    r"\[(\d+)::(\d+)\]\s+(HP_\d+)\s+\|\s+(.+)",
                    line
                )

                if not match:
                    logger.warning(f"Invalid format at line {line_num}: {line}")
                    continue

                start, end, hpo_id, text = match.groups()

                # Normalize HPO ID: HP_NNNN → HP:NNNN
                hpo_id = hpo_id.replace("_", ":")

                annotations.append(Annotation(
                    start=int(start),
                    end=int(end),
                    text=text,
                    hpo_id=hpo_id,
                    confidence=None,
                ))

        return annotations


class ProcessedFormatParser(AnnotationParser):
    """Parser for processed format: start\tend\ttext\tHPO:ID\tconfidence"""

    def parse(self, ann_file: Path) -> List[Annotation]:
        annotations = []

        with open(ann_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) < 4:
                    logger.warning(f"Invalid format at line {line_num}: {line}")
                    continue

                start = int(parts[0])
                end = int(parts[1])
                text = parts[2]
                hpo_id = parts[3]
                confidence = float(parts[4]) if len(parts) > 4 else None

                annotations.append(Annotation(
                    start=start,
                    end=end,
                    text=text,
                    hpo_id=hpo_id,
                    confidence=confidence,
                ))

        return annotations


def detect_and_parse(ann_file: Path) -> List[Annotation]:
    """Auto-detect format and parse."""
    with open(ann_file) as f:
        first_line = f.readline().strip()

    if first_line.startswith("[") and "::" in first_line:
        parser = RawFormatParser()
    elif "\t" in first_line:
        parser = ProcessedFormatParser()
    else:
        raise ValueError(f"Unknown annotation format: {first_line}")

    return parser.parse(ann_file)
```

#### 4.2.3 `SpanValidator`

```python
class SpanValidator:
    """Validates annotation spans against text."""

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.errors = []
        self.warnings = []

    def validate_span(
        self,
        text: str,
        annotation: Annotation,
    ) -> bool:
        """Validate single span."""

        # Check bounds
        if annotation.start < 0 or annotation.end > len(text):
            self.errors.append(
                f"Out of bounds: [{annotation.start}:{annotation.end}] "
                f"(text length: {len(text)})"
            )
            return False

        # Extract actual text
        actual_text = text[annotation.start:annotation.end]
        expected_text = annotation.text

        # Exact match
        if actual_text == expected_text:
            return True

        # Whitespace normalization
        if actual_text.strip() == expected_text.strip():
            self.warnings.append(
                f"Whitespace mismatch at [{annotation.start}:{annotation.end}]: "
                f"expected '{expected_text}', got '{actual_text}'"
            )
            return not self.strict

        # Complete mismatch
        self.errors.append(
            f"Text mismatch at [{annotation.start}:{annotation.end}]: "
            f"expected '{expected_text}', got '{actual_text}'"
        )
        return False

    def validate_all(
        self,
        text: str,
        annotations: List[Annotation],
    ) -> Tuple[List[Annotation], List[str], List[str]]:
        """Validate all annotations, return valid ones and errors/warnings."""

        valid = []

        for ann in annotations:
            if self.validate_span(text, ann):
                valid.append(ann)

        return valid, self.errors, self.warnings
```

#### 4.2.4 `HPOLookup`

```python
class HPOLookup:
    """Lookup HPO term labels from various sources."""

    def __init__(self, hpo_data_path: Path):
        self.hpo_data_path = hpo_data_path
        self.cache = {}
        self._load_hpo_data()

    def _load_hpo_data(self):
        """Load HPO terms from TSV file."""
        hpo_file = self.hpo_data_path / "hpo_terms.tsv"

        if not hpo_file.exists():
            logger.warning(f"HPO data file not found: {hpo_file}")
            return

        with open(hpo_file) as f:
            # Skip header
            next(f)

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    hpo_id = parts[0]
                    label = parts[1]
                    self.cache[hpo_id] = label

        logger.info(f"Loaded {len(self.cache)} HPO terms")

    def get_label(self, hpo_id: str) -> Optional[str]:
        """Get label for HPO ID."""
        return self.cache.get(hpo_id)

    def get_label_with_fallback(self, hpo_id: str) -> str:
        """Get label with fallback to API."""

        # Try cache
        label = self.cache.get(hpo_id)
        if label:
            return label

        # Try API
        try:
            label = self._fetch_from_api(hpo_id)
            if label:
                self.cache[hpo_id] = label
                return label
        except Exception as e:
            logger.warning(f"API lookup failed for {hpo_id}: {e}")

        # Fallback
        return "UNKNOWN"

    def _fetch_from_api(self, hpo_id: str) -> Optional[str]:
        """Fetch label from HPO API."""
        url = f"https://hpo.jax.org/api/hpo/term/{hpo_id}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return data.get("name")

        return None
```

### 4.3 CLI Script

```python
#!/usr/bin/env python3
"""
Convert PhenoBERT corpus datasets to Phentrieve JSON format.

Usage:
    python scripts/convert_phenobert_data.py --config config.yaml
    python scripts/convert_phenobert_data.py --dataset GSC+ --output data/benchmark_datasets
"""

import argparse
import logging
from pathlib import Path
import yaml

from phentrieve.data_processing.phenobert_converter import PhenoBERTConverter


def main():
    parser = argparse.ArgumentParser(
        description="Convert PhenoBERT datasets to Phentrieve format"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default="config/phenobert_conversion.yaml",
        help="Configuration file (default: config/phenobert_conversion.yaml)",
    )

    parser.add_argument(
        "--phenobert-data",
        type=Path,
        help="Path to phenobert/data directory (overrides config)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        choices=["GSC+", "ID-68", "GeneReviews", "all"],
        default="all",
        help="Dataset to convert (default: all)",
    )

    parser.add_argument(
        "--hpo-data",
        type=Path,
        help="Path to HPO data directory (overrides config)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict validation (fail on warnings)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (don't write files)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with CLI arguments
    if args.phenobert_data:
        config["phenobert_data_dir"] = str(args.phenobert_data)
    if args.output:
        config["output_dir"] = str(args.output)
    if args.hpo_data:
        config["hpo_data_path"] = str(args.hpo_data)
    if args.strict:
        config["strict_validation"] = True
    if args.dry_run:
        config["dry_run"] = True

    # Create converter
    converter = PhenoBERTConverter(config)

    # Determine datasets to convert
    if args.dataset == "all":
        datasets = ["GSC+", "ID-68", "GeneReviews"]
    else:
        datasets = [args.dataset]

    # Convert each dataset
    for dataset_name in datasets:
        logger.info(f"Converting dataset: {dataset_name}")

        try:
            result = converter.convert_dataset(dataset_name)

            logger.info(
                f"Converted {result.num_documents} documents, "
                f"{result.num_annotations} annotations"
            )

            if result.errors:
                logger.error(f"Found {len(result.errors)} errors")
                for error in result.errors[:10]:  # Show first 10
                    logger.error(f"  {error}")

            if result.warnings:
                logger.warning(f"Found {len(result.warnings)} warnings")
                for warning in result.warnings[:10]:
                    logger.warning(f"  {warning}")

        except Exception as e:
            logger.error(f"Error converting {dataset_name}: {e}", exc_info=True)

    # Generate report
    report = converter.generate_report()

    logger.info(f"\n{'='*60}")
    logger.info("CONVERSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents: {report['total_documents']}")
    logger.info(f"Total annotations: {report['total_annotations']}")
    logger.info(f"Unique HPO terms: {report['unique_hpo_terms']}")
    logger.info(f"Total errors: {report['total_errors']}")
    logger.info(f"Total warnings: {report['total_warnings']}")
    logger.info(f"{'='*60}\n")

    # Save report
    if not config.get("dry_run"):
        report_file = Path(config["output_dir"]) / "conversion_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
```

### 4.4 Companion Scripts

The `scripts/` folder will contain several companion scripts to support the conversion workflow. These scripts provide validation, analysis, and quality assurance functionality.

#### 4.4.1 `validate_converted_data.py`

**Purpose:** Validate converted JSON files for quality and schema compliance.

**Features:**
- JSON schema validation
- Offset accuracy checking
- HPO ID existence verification
- Text snippet matching
- Statistical analysis

**Usage:**
```bash
# Validate entire dataset
python scripts/validate_converted_data.py \
    --input data/benchmark_datasets/phenobert/GSC_plus/annotations

# Validate with detailed report
python scripts/validate_converted_data.py \
    --input data/benchmark_datasets/phenobert \
    --recursive \
    --output-report results/validation_report.json

# Strict validation (fail on warnings)
python scripts/validate_converted_data.py \
    --input data/benchmark_datasets/phenobert \
    --strict
```

**Implementation Sketch:**
```python
#!/usr/bin/env python3
"""
Validate converted PhenoBERT datasets.

Checks:
- JSON schema compliance
- Offset accuracy
- HPO ID validity
- Text snippet matching
- Required fields present
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def validate_json_file(json_file: Path) -> List[str]:
    """Validate single JSON file, return list of errors."""
    errors = []

    with open(json_file) as f:
        doc = json.load(f)

    # Schema validation
    required_fields = ["doc_id", "language", "full_text", "annotations"]
    for field in required_fields:
        if field not in doc:
            errors.append(f"Missing required field: {field}")

    # Validate annotations
    text = doc.get("full_text", "")
    for ann in doc.get("annotations", []):
        for span in ann.get("evidence_spans", []):
            start = span.get("start_char")
            end = span.get("end_char")
            expected = span.get("text_snippet")

            if start is None or end is None:
                errors.append(f"Missing offsets in annotation {ann.get('hpo_id')}")
                continue

            # Check bounds
            if start < 0 or end > len(text):
                errors.append(f"Out of bounds: [{start}:{end}] for {ann.get('hpo_id')}")
                continue

            # Check text match
            actual = text[start:end]
            if actual != expected:
                errors.append(
                    f"Text mismatch at [{start}:{end}]: "
                    f"expected '{expected}', got '{actual}'"
                )

    return errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-report", "-o", type=Path)

    args = parser.parse_args()

    # Find JSON files
    if args.input.is_file():
        json_files = [args.input]
    elif args.recursive:
        json_files = list(args.input.rglob("*.json"))
    else:
        json_files = list(args.input.glob("*.json"))

    print(f"Validating {len(json_files)} files...")

    # Validate each file
    all_errors = {}
    for json_file in json_files:
        errors = validate_json_file(json_file)
        if errors:
            all_errors[str(json_file)] = errors

    # Report results
    if all_errors:
        print(f"\n❌ Found errors in {len(all_errors)} files:")
        for file, errors in all_errors.items():
            print(f"\n{file}:")
            for error in errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        if args.strict:
            exit(1)
    else:
        print("\n✅ All files validated successfully!")

    # Save report
    if args.output_report:
        report = {
            "total_files": len(json_files),
            "files_with_errors": len(all_errors),
            "total_errors": sum(len(e) for e in all_errors.values()),
            "errors": all_errors,
        }
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output_report}")

if __name__ == "__main__":
    main()
```

#### 4.4.2 `generate_dataset_statistics.py`

**Purpose:** Generate comprehensive statistics for converted datasets.

**Features:**
- Document count and sizes
- Annotation statistics
- HPO term frequency analysis
- Dataset comparison tables
- Visualization generation

**Usage:**
```bash
# Generate statistics for single dataset
python scripts/generate_dataset_statistics.py \
    --input data/benchmark_datasets/phenobert/GSC_plus \
    --output results/gsc_plus_stats.json

# Compare multiple datasets
python scripts/generate_dataset_statistics.py \
    --input data/benchmark_datasets/phenobert \
    --compare \
    --output results/dataset_comparison.json

# With visualization
python scripts/generate_dataset_statistics.py \
    --input data/benchmark_datasets/phenobert \
    --visualize \
    --output-dir results/visualizations
```

#### 4.4.3 `compare_annotation_formats.py`

**Purpose:** Compare original PhenoBERT annotations with converted Phentrieve format.

**Features:**
- Side-by-side format comparison
- Detect any conversion issues
- Verify no data loss
- Report discrepancies

**Usage:**
```bash
# Compare single document
python scripts/compare_annotation_formats.py \
    --original PhenoBERT/phenobert/data/GSC+/ann/gsc_001.ann \
    --converted data/benchmark_datasets/phenobert/GSC_plus/annotations/gsc_plus_001.json

# Batch comparison
python scripts/compare_annotation_formats.py \
    --original-dir PhenoBERT/phenobert/data/GSC+/ann \
    --converted-dir data/benchmark_datasets/phenobert/GSC_plus/annotations \
    --output-report results/format_comparison.json
```

#### 4.4.4 `benchmark_conversion_speed.py`

**Purpose:** Benchmark conversion performance and identify bottlenecks.

**Features:**
- Time each conversion stage
- Measure memory usage
- Generate performance report
- Identify slow documents

**Usage:**
```bash
# Benchmark conversion
python scripts/benchmark_conversion_speed.py \
    --phenobert-data PhenoBERT/phenobert/data \
    --dataset GSC+ \
    --iterations 3 \
    --output results/conversion_benchmark.json
```

#### 4.4.5 `scripts/README.md`

**Purpose:** Documentation for all scripts in the folder.

**Content:**
```markdown
# Phentrieve Scripts

This directory contains standalone scripts for data processing, validation, and benchmarking.

## Overview

Scripts are organized by function:

- **Data Conversion** - Convert external datasets to Phentrieve format
- **Validation** - Validate converted data quality
- **Analysis** - Generate statistics and reports
- **Benchmarking** - Performance and accuracy testing

## Prerequisites

All scripts require Phentrieve to be installed:

```bash
# Install in development mode
cd /path/to/phentrieve
make install-dev
```

## Script Inventory

### Data Conversion

#### `convert_phenobert_data.py`
Convert PhenoBERT corpus datasets (GSC+, ID-68, GeneReviews) to Phentrieve JSON format.

**Usage:**
```bash
python scripts/convert_phenobert_data.py --config config/phenobert_conversion.yaml
```

**See:** `plan/01-active/PHENOBERT-CORPUS-CONVERSION-PLAN.md`

### Validation

#### `validate_converted_data.py`
Validate converted JSON files for quality and schema compliance.

**Usage:**
```bash
python scripts/validate_converted_data.py --input data/benchmark_datasets/phenobert
```

### Analysis

#### `generate_dataset_statistics.py`
Generate comprehensive statistics and comparison tables.

**Usage:**
```bash
python scripts/generate_dataset_statistics.py --input data/benchmark_datasets/phenobert --compare
```

#### `compare_annotation_formats.py`
Compare original and converted annotations to verify no data loss.

**Usage:**
```bash
python scripts/compare_annotation_formats.py --original-dir ... --converted-dir ...
```

### Benchmarking

#### `benchmark_conversion_speed.py`
Benchmark conversion performance and identify bottlenecks.

**Usage:**
```bash
python scripts/benchmark_conversion_speed.py --dataset GSC+ --iterations 3
```

## Writing New Scripts

When adding new scripts:

1. ✅ Use the template from `PHENOBERT-CORPUS-CONVERSION-PLAN.md`
2. ✅ Add shebang: `#!/usr/bin/env python3`
3. ✅ Include docstring with usage examples
4. ✅ Use argparse for CLI arguments
5. ✅ Add to this README
6. ✅ Write tests in `tests/scripts/`

## Common Patterns

### Configuration Loading

```python
import yaml

with open("config/my_config.yaml") as f:
    config = yaml.safe_load(f)
```

### Logging Setup

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
```

### Import from Phentrieve

```python
from phentrieve.data_processing import PhenoBERTConverter
from phentrieve.evaluation import load_dataset
```

## Troubleshooting

**Import errors:**
```bash
# Make sure Phentrieve is installed
make install-dev
```

**Permission errors:**
```bash
# Make script executable
chmod +x scripts/my_script.py
```

**Path errors:**
```bash
# Run from project root
cd /path/to/phentrieve
python scripts/my_script.py
```

## Future Scripts

Planned additions:
- `convert_biocreative_data.py` - BioCreative dataset conversion
- `convert_lirical_data.py` - LIRICAL corpus conversion
- `validate_hpo_ids.py` - HPO ID validation against ontology
- `analyze_annotation_overlap.py` - Inter-annotator agreement
- `benchmark_retrieval_accuracy.py` - End-to-end retrieval benchmarks

## Contributing

When contributing scripts:

1. Follow the script template
2. Add comprehensive documentation
3. Include usage examples
4. Write tests
5. Update this README

See `CONTRIBUTING.md` for more details.
```

---

## 5. Data Quality Assurance

### 5.1 Validation Checks

#### Level 1: File Structure
- ✅ All required files present (corpus + annotations)
- ✅ Filename matching (corpus and annotation pairs)
- ✅ File encoding (UTF-8)
- ✅ File sizes reasonable (not empty, not corrupted)

#### Level 2: Content Validation
- ✅ Text content readable
- ✅ Annotation format parseable
- ✅ HPO IDs valid format (`HP:NNNNNNN`)
- ✅ Offsets are integers
- ✅ Confidence scores in [0, 1] range

#### Level 3: Semantic Validation
- ✅ Offsets within text bounds
- ✅ Text snippets match extracted text
- ✅ HPO IDs exist in ontology
- ✅ No overlapping evidence spans for same term
- ✅ At least one annotation per document

#### Level 4: Schema Validation
- ✅ JSON schema compliance
- ✅ Required fields present
- ✅ Data types correct
- ✅ Nested structures valid

### 5.2 Quality Metrics

**Computed for Each Dataset:**
```python
{
    "dataset_name": "GSC+",
    "num_documents": 228,
    "num_annotations": 1933,
    "num_unique_hpo_terms": 497,
    "avg_annotations_per_doc": 8.5,
    "avg_text_length_chars": 500,
    "avg_text_length_words": 85,
    "min_text_length": 138,
    "max_text_length": 2417,
    "validation": {
        "total_spans": 1933,
        "valid_spans": 1920,
        "invalid_spans": 13,
        "span_accuracy": 0.993,
        "errors": [
            "Out of bounds: [500:520] at GSC_123.txt",
            # ...
        ],
        "warnings": [
            "Whitespace mismatch at [100:115] in GSC_045.txt",
            # ...
        ]
    },
    "hpo_coverage": {
        "total_hpo_ids": 497,
        "found_in_ontology": 495,
        "missing_from_ontology": 2,
        "coverage_rate": 0.996,
        "missing_ids": ["HP:9999999", "HP:8888888"]
    }
}
```

### 5.3 Error Handling Strategy

**Error Categories:**

| Category | Severity | Action |
|----------|----------|--------|
| **File not found** | Critical | Skip document, log error |
| **Encoding error** | Critical | Skip document, log error |
| **Parse error** | High | Skip annotation, log error |
| **Out of bounds** | High | Skip span, log error |
| **Text mismatch** | Medium | Skip span (strict) or keep (lenient) |
| **Missing HPO label** | Low | Use "UNKNOWN", log warning |
| **Whitespace mismatch** | Low | Accept with warning |

**Handling Modes:**
1. **Strict:** Fail conversion on any error
2. **Lenient:** Skip problematic items, continue
3. **Best-effort:** Try to fix common issues automatically

### 5.4 Manual Review Process

**For Documents with Errors:**
1. Generate review list: `conversion_report_errors.txt`
2. Include original text + annotations
3. Highlight problematic spans
4. Provide suggested fixes
5. Allow manual correction via JSON edits

**Review Report Format:**
```
================================================================================
DOCUMENT: gsc_plus_123.txt
ERRORS: 2

Error 1: Out of bounds span
  Offset: [500:520]
  Expected text: "developmental delay"
  Actual text: [OUT OF BOUNDS]
  Suggestion: Check if offset is 0-indexed vs 1-indexed

Error 2: HPO ID not found
  HPO ID: HP:9999999
  Text span: "unusual phenotype"
  Suggestion: Verify HPO ID is correct or update ontology

================================================================================
```

---

## 6. Integration Plan

### 6.1 Output Directory Structure

```
data/benchmark_datasets/phenobert/
├── GSC_plus/
│   ├── annotations/
│   │   ├── gsc_plus_001.json
│   │   ├── gsc_plus_002.json
│   │   └── ... (228 files)
│   ├── metadata.json
│   └── README.md
│
├── ID68/
│   ├── annotations/
│   │   ├── id68_001.json
│   │   ├── id68_002.json
│   │   └── ... (68 files)
│   ├── metadata.json
│   └── README.md
│
├── GeneReviews/
│   ├── annotations/
│   │   ├── genereview_001.json
│   │   ├── genereview_002.json
│   │   └── ... (10 files)
│   ├── metadata.json
│   └── README.md
│
├── dataset_catalog.json
├── conversion_report.json
└── README.md
```

### 6.2 Dataset Catalog Format

```json
{
  "datasets": [
    {
      "id": "phenobert_gsc_plus",
      "name": "PhenoBERT GSC+ (BiolarkGSC+)",
      "type": "clinical_notes",
      "language": "en",
      "num_documents": 228,
      "num_annotations": 1933,
      "num_unique_hpo_terms": 497,
      "avg_length_words": 85,
      "source": "phenobert",
      "source_url": "https://github.com/EclipseCN/PhenoBERT",
      "original_paper": "Groza et al. (2015), refined by Lobo et al. (2017)",
      "license": "To be verified",
      "path": "phenobert/GSC_plus",
      "conversion_date": "2025-01-18T10:30:00Z",
      "converter_version": "1.0.0"
    },
    {
      "id": "phenobert_id68",
      "name": "PhenoBERT ID-68",
      "type": "clinical_notes",
      "language": "en",
      "num_documents": 68,
      "num_annotations": 542,
      "num_unique_hpo_terms": 156,
      "avg_length_words": 120,
      "source": "phenobert",
      "source_url": "https://github.com/EclipseCN/PhenoBERT",
      "description": "Clinical notes from families with intellectual disabilities",
      "license": "To be verified",
      "path": "phenobert/ID68",
      "conversion_date": "2025-01-18T10:45:00Z",
      "converter_version": "1.0.0"
    },
    {
      "id": "phenobert_genereview",
      "name": "PhenoBERT GeneReviews",
      "type": "case_reports",
      "language": "en",
      "num_documents": 10,
      "num_annotations": 89,
      "num_unique_hpo_terms": 45,
      "avg_length_words": 200,
      "source": "phenobert",
      "source_url": "https://github.com/EclipseCN/PhenoBERT",
      "description": "Clinical cases from GeneReviews database",
      "license": "To be verified",
      "path": "phenobert/GeneReviews",
      "conversion_date": "2025-01-18T11:00:00Z",
      "converter_version": "1.0.0"
    }
  ],
  "total_documents": 306,
  "total_annotations": 2564,
  "total_unique_hpo_terms": 698,
  "catalog_version": "1.0.0",
  "generated": "2025-01-18T11:00:00Z"
}
```

### 6.3 Integration with Benchmarking

**Use in Chunking Benchmark:**
```python
# phentrieve/evaluation/chunking_benchmark.py

# Load PhenoBERT datasets
config = ChunkingBenchmarkConfig(
    dataset_ids=[
        "phenobert_gsc_plus",
        "phenobert_id68",
        "phenobert_genereview",
    ],
    chunking_strategies=["balanced", "precise", "adaptive"],
    enable_span_metrics=True,
)

runner = ChunkingBenchmarkRunner(config, retriever, cross_encoder)
results = runner.run_benchmark()
```

**Use in Full-Text Evaluation:**
```python
# phentrieve/evaluation/full_text_runner.py

# Load single document
with open("data/benchmark_datasets/phenobert/GSC_plus/annotations/gsc_plus_001.json") as f:
    doc = json.load(f)

metrics = evaluate_single_document_extraction(
    ground_truth_doc=doc,
    language="en",
    pipeline=pipeline,
    retriever=retriever,
    enable_span_metrics=True,
)
```

---

## 7. Configuration & Reproducibility

### 7.1 Configuration File Schema

**File:** `config/phenobert_conversion.yaml`

```yaml
# PhenoBERT Corpus Conversion Configuration

# Input paths
phenobert_data_dir: "path/to/PhenoBERT/phenobert/data"

# Output paths
output_dir: "data/benchmark_datasets/phenobert"

# HPO data
hpo_data_path: "data/hpo_core_data"
hpo_fallback_api: true  # Use API for missing terms

# Datasets to convert
datasets:
  - name: "GSC+"
    enabled: true
    corpus_subdir: "GSC+/corpus"
    ann_subdir: "GSC+/ann"
    output_name: "GSC_plus"

  - name: "ID-68"
    enabled: true
    corpus_subdir: "ID-68/corpus"
    ann_subdir: "ID-68/ann"
    output_name: "ID68"

  - name: "GeneReviews"
    enabled: true
    corpus_subdir: "GeneReviews/corpus"
    ann_subdir: "GeneReviews/ann"
    output_name: "GeneReviews"

# Validation
validation:
  strict: false  # false = lenient, true = fail on warnings
  check_hpo_ids: true
  check_span_accuracy: true
  max_offset_error: 0  # Allow no offset errors (strict)

# Processing options
processing:
  encoding: "utf-8"
  fallback_encoding: "latin-1"
  normalize_line_endings: true
  strip_whitespace: false  # Keep original formatting

# Output options
output:
  pretty_print: true  # Format JSON with indentation
  indent: 2
  include_confidence: true  # Include confidence scores if present
  include_metadata: true

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/phenobert_conversion.log"
  console_output: true

# Metadata
metadata:
  converter_version: "1.0.0"
  source_attribution: "PhenoBERT (https://github.com/EclipseCN/PhenoBERT)"
  license: "To be verified"
```

### 7.2 Reproducibility Requirements

**Version Control:**
1. Pin all dependencies in `pyproject.toml`
2. Document Python version (3.10+)
3. Include configuration file in git
4. Tag releases for converter script

**Documentation:**
1. Step-by-step setup instructions
2. Example commands for all use cases
3. Troubleshooting guide
4. Expected output examples

**Provenance Tracking:**
- Record source commit SHA for PhenoBERT repo
- Include conversion timestamp in metadata
- Log converter version in output
- Save conversion report with checksums

**Reproducible Workflow:**
```bash
# 1. Clone PhenoBERT repository
git clone https://github.com/EclipseCN/PhenoBERT.git
cd PhenoBERT
git checkout <commit-sha>  # Pin to specific version

# 2. Install Phentrieve
cd /path/to/phentrieve
uv sync

# 3. Configure conversion
cp config/phenobert_conversion.yaml.template config/phenobert_conversion.yaml
# Edit phenobert_data_dir path

# 4. Run conversion
python scripts/convert_phenobert_data.py --config config/phenobert_conversion.yaml

# 5. Validate output
python scripts/validate_converted_data.py data/benchmark_datasets/phenobert

# 6. Generate checksums
find data/benchmark_datasets/phenobert -name "*.json" -type f -exec sha256sum {} \; > checksums.txt
```

### 7.3 Environment Setup

**Dependencies:**
```toml
# pyproject.toml

[project.dependencies]
# Existing dependencies...

# For conversion
pyyaml = "^6.0"
requests = "^2.31.0"  # For HPO API fallback
chardet = "^5.2.0"    # For encoding detection

[project.optional-dependencies]
conversion = [
    "jsonschema>=4.20.0",  # For JSON validation
]
```

**Python Version:**
- Minimum: Python 3.10
- Recommended: Python 3.11

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Test Coverage:**

```python
# tests/unit/data_processing/test_phenobert_converter.py

class TestAnnotationParsers:
    def test_raw_format_parser(self):
        """Test parsing raw GSC+ format."""
        line = "[27::42] HP_0000110 | renal dysplasia"
        parser = RawFormatParser()
        ann = parser.parse_line(line)

        assert ann.start == 27
        assert ann.end == 42
        assert ann.hpo_id == "HP:0000110"
        assert ann.text == "renal dysplasia"

    def test_processed_format_parser(self):
        """Test parsing processed format."""
        line = "9\t17\theadache\tHP:0002315\t1.0"
        parser = ProcessedFormatParser()
        ann = parser.parse_line(line)

        assert ann.start == 9
        assert ann.end == 17
        assert ann.hpo_id == "HP:0002315"
        assert ann.text == "headache"
        assert ann.confidence == 1.0

    def test_hpo_id_normalization(self):
        """Test HPO ID normalization."""
        assert normalize_hpo_id("HP_0000110") == "HP:0000110"
        assert normalize_hpo_id("HP:0000110") == "HP:0000110"


class TestSpanValidator:
    def test_valid_span(self):
        """Test validation of valid span."""
        text = "Patient has renal dysplasia and seizures."
        ann = Annotation(start=12, end=27, text="renal dysplasia", hpo_id="HP:0000110")

        validator = SpanValidator()
        assert validator.validate_span(text, ann) == True

    def test_out_of_bounds_span(self):
        """Test out of bounds detection."""
        text = "Short text"
        ann = Annotation(start=0, end=100, text="too long", hpo_id="HP:0000001")

        validator = SpanValidator()
        assert validator.validate_span(text, ann) == False

    def test_text_mismatch(self):
        """Test text mismatch detection."""
        text = "Patient has renal dysplasia"
        ann = Annotation(start=12, end=27, text="wrong text", hpo_id="HP:0000110")

        validator = SpanValidator(strict=True)
        assert validator.validate_span(text, ann) == False


class TestHPOLookup:
    def test_lookup_cached(self):
        """Test HPO label lookup from cache."""
        lookup = HPOLookup(hpo_data_path=Path("data/hpo_core_data"))
        label = lookup.get_label("HP:0001263")

        assert label == "Global developmental delay"

    def test_lookup_missing(self):
        """Test missing HPO ID."""
        lookup = HPOLookup(hpo_data_path=Path("data/hpo_core_data"))
        label = lookup.get_label("HP:9999999")

        assert label is None
```

### 8.2 Integration Tests

```python
# tests/integration/test_phenobert_conversion.py

class TestPhenoBERTConversionIntegration:
    def test_convert_single_document(self, tmp_path):
        """Test end-to-end conversion of single document."""

        # Create test corpus file
        corpus_file = tmp_path / "test_doc.txt"
        corpus_file.write_text("Patient has renal dysplasia and seizures.")

        # Create test annotation file
        ann_file = tmp_path / "test_doc.ann"
        ann_file.write_text(
            "12\t27\trenal dysplasia\tHP:0000110\t1.0\n"
            "32\t40\tseizures\tHP:0001250\t0.95\n"
        )

        # Convert
        converter = PhenoBERTConverter(config={
            "hpo_data_path": "data/hpo_core_data",
            "output_dir": tmp_path,
        })

        json_doc = converter.convert_document(
            corpus_file=corpus_file,
            ann_file=ann_file,
            dataset_name="test",
        )

        # Validate structure
        assert json_doc["doc_id"] == "test_test_doc"
        assert json_doc["full_text"] == "Patient has renal dysplasia and seizures."
        assert len(json_doc["annotations"]) == 2

        # Validate first annotation
        ann1 = json_doc["annotations"][0]
        assert ann1["hpo_id"] == "HP:0000110"
        assert ann1["label"] == "Renal dysplasia"
        assert len(ann1["evidence_spans"]) == 1
        assert ann1["evidence_spans"][0]["start_char"] == 12
        assert ann1["evidence_spans"][0]["end_char"] == 27

    def test_convert_gsc_plus_subset(self):
        """Test conversion of GSC+ subset (first 10 documents)."""

        config = {
            "phenobert_data_dir": "path/to/PhenoBERT/phenobert/data",
            "output_dir": "test_output",
            "hpo_data_path": "data/hpo_core_data",
        }

        converter = PhenoBERTConverter(config)

        # Convert first 10 documents
        result = converter.convert_dataset(
            dataset_name="GSC+",
            max_documents=10,
        )

        assert result.num_documents == 10
        assert result.num_annotations > 0
        assert len(result.errors) == 0
```

### 8.3 Validation Tests

```python
# tests/validation/test_converted_data_quality.py

class TestConvertedDataQuality:
    def test_all_documents_have_required_fields(self):
        """Test all converted documents have required fields."""

        dataset_dir = Path("data/benchmark_datasets/phenobert/GSC_plus/annotations")

        required_fields = [
            "doc_id", "language", "full_text", "metadata", "annotations"
        ]

        for json_file in dataset_dir.glob("*.json"):
            with open(json_file) as f:
                doc = json.load(f)

            for field in required_fields:
                assert field in doc, f"Missing field '{field}' in {json_file}"

    def test_all_offsets_valid(self):
        """Test all annotation offsets are within text bounds."""

        dataset_dir = Path("data/benchmark_datasets/phenobert/GSC_plus/annotations")

        for json_file in dataset_dir.glob("*.json"):
            with open(json_file) as f:
                doc = json.load(f)

            text = doc["full_text"]
            text_len = len(text)

            for ann in doc["annotations"]:
                for span in ann["evidence_spans"]:
                    start = span["start_char"]
                    end = span["end_char"]

                    assert 0 <= start < text_len, f"Invalid start offset in {json_file}"
                    assert start < end <= text_len, f"Invalid end offset in {json_file}"

    def test_span_text_matches(self):
        """Test annotation text snippets match extracted text."""

        dataset_dir = Path("data/benchmark_datasets/phenobert/GSC_plus/annotations")

        mismatches = []

        for json_file in dataset_dir.glob("*.json"):
            with open(json_file) as f:
                doc = json.load(f)

            text = doc["full_text"]

            for ann in doc["annotations"]:
                for span in ann["evidence_spans"]:
                    expected = span["text_snippet"]
                    actual = text[span["start_char"]:span["end_char"]]

                    if expected != actual:
                        mismatches.append({
                            "file": json_file.name,
                            "hpo_id": ann["hpo_id"],
                            "expected": expected,
                            "actual": actual,
                        })

        assert len(mismatches) == 0, f"Found {len(mismatches)} text mismatches"
```

### 8.4 Test Data

**Create test fixtures:**

```python
# tests/fixtures/phenobert_test_data.py

import pytest
from pathlib import Path

@pytest.fixture
def sample_gsc_corpus_text():
    """Sample GSC+ corpus text."""
    return (
        "The patient presented with global developmental delay, "
        "hypotonia, and seizures. Brain MRI showed ventriculomegaly."
    )

@pytest.fixture
def sample_gsc_raw_annotations():
    """Sample GSC+ raw format annotations."""
    return """[26::51] HP_0001263 | global developmental delay
[53::62] HP_0001252 | hypotonia
[68::76] HP_0001250 | seizures
[96::111] HP_0002119 | ventriculomegaly"""

@pytest.fixture
def sample_processed_annotations():
    """Sample processed format annotations."""
    return """26\t51\tglobal developmental delay\tHP:0001263\t1.0
53\t62\thypotonia\tHP:0001252\t0.98
68\t76\tseizures\tHP:0001250\t1.0
96\t111\tventriculomegaly\tHP:0002119\t0.95"""

@pytest.fixture
def expected_json_output():
    """Expected JSON output structure."""
    return {
        "doc_id": "gsc_plus_test_001",
        "language": "en",
        "source": "phenobert",
        "full_text": "The patient presented with global developmental delay...",
        "metadata": {
            "dataset": "GSC+",
            "text_length_chars": 115,
            "num_annotations": 4,
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
                    }
                ],
            },
            # ... more annotations
        ],
    }
```

---

## 9. Documentation Requirements

### 9.1 User Guide

**File:** `docs/phenobert_conversion_guide.md`

**Contents:**
1. Overview of PhenoBERT datasets
2. Installation and setup
3. Configuration file explanation
4. Step-by-step conversion tutorial
5. Troubleshooting common issues
6. FAQ

### 9.2 Technical Documentation

**File:** `docs/api/phenobert_converter.md`

**Contents:**
1. Module architecture
2. Class and function documentation
3. Annotation format specifications
4. Extension points for custom formats
5. API reference

### 9.3 Dataset Documentation

**File:** `data/benchmark_datasets/phenobert/README.md`

**Contents:**
```markdown
# PhenoBERT Corpus Datasets

## Overview

This directory contains three corpus datasets from PhenoBERT, converted to Phentrieve's JSON annotation format:

1. **GSC+ (BiolarkGSC+):** 228 PubMed abstracts
2. **ID-68:** 68 clinical notes (intellectual disability)
3. **GeneReviews:** 10 clinical cases

## Source

**Repository:** https://github.com/EclipseCN/PhenoBERT
**License:** [To be verified]
**Citation:**
```
[PhenoBERT citation here]
```

## Conversion Details

**Converted:** 2025-01-18
**Converter Version:** 1.0.0
**Script:** `scripts/convert_phenobert_data.py`

## Statistics

| Dataset | Documents | Annotations | Unique HPO Terms |
|---------|-----------|-------------|------------------|
| GSC+ | 228 | 1,933 | 497 |
| ID-68 | 68 | 542 | 156 |
| GeneReviews | 10 | 89 | 45 |
| **Total** | **306** | **2,564** | **698** |

## Directory Structure

```
phenobert/
├── GSC_plus/
│   ├── annotations/          # 228 JSON files
│   ├── metadata.json         # Dataset metadata
│   └── README.md             # Dataset-specific info
├── ID68/
│   ├── annotations/          # 68 JSON files
│   ├── metadata.json
│   └── README.md
├── GeneReviews/
│   ├── annotations/          # 10 JSON files
│   ├── metadata.json
│   └── README.md
├── dataset_catalog.json      # Catalog of all datasets
└── conversion_report.json    # Conversion summary
```

## JSON Format

Each annotation file follows this structure:

```json
{
  "doc_id": "gsc_plus_001",
  "language": "en",
  "source": "phenobert",
  "full_text": "...",
  "metadata": {...},
  "annotations": [...]
}
```

See `docs/phenobert_conversion_guide.md` for full format specification.

## Usage

### Load Single Document

```python
import json

with open("data/benchmark_datasets/phenobert/GSC_plus/annotations/gsc_plus_001.json") as f:
    doc = json.load(f)

print(f"Document: {doc['doc_id']}")
print(f"Text: {doc['full_text']}")
print(f"Annotations: {len(doc['annotations'])}")
```

### Load Entire Dataset

```python
from phentrieve.evaluation.dataset_loader import load_dataset

dataset = load_dataset("phenobert_gsc_plus")

for doc in dataset:
    print(f"Processing: {doc['doc_id']}")
```

### Use in Benchmarking

```python
from phentrieve.evaluation.chunking_benchmark import ChunkingBenchmarkRunner

config = ChunkingBenchmarkConfig(
    dataset_ids=["phenobert_gsc_plus", "phenobert_id68"],
    chunking_strategies=["balanced", "precise"],
)

runner = ChunkingBenchmarkRunner(config, retriever, cross_encoder)
results = runner.run_benchmark()
```

## Validation

All converted documents have been validated:
- ✅ JSON schema compliance
- ✅ Span offset accuracy
- ✅ HPO ID existence
- ✅ Text snippet matching

See `conversion_report.json` for detailed validation results.

## Known Issues

None currently reported.

## Updates

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-18 | 1.0.0 | Initial conversion |
```

---

## 10. Timeline & Milestones

### Phase 1: Setup & Infrastructure (1 day)

**Day 1:**
- [ ] Set up development environment
- [ ] Create module structure
- [ ] Download PhenoBERT repository
- [ ] Create configuration file template
- [ ] Set up logging infrastructure

**Deliverables:**
- ✅ Module skeleton in `phentrieve/data_processing/`
- ✅ Configuration template
- ✅ PhenoBERT data downloaded

### Phase 2: Core Implementation (2 days)

**Day 2:**
- [ ] Implement annotation parsers (raw + processed formats)
- [ ] Implement text loader with encoding detection
- [ ] Implement span validator
- [ ] Write unit tests for parsers

**Day 3:**
- [ ] Implement HPO label lookup
- [ ] Implement JSON constructor
- [ ] Implement main converter class
- [ ] Write integration tests

**Deliverables:**
- ✅ Functional converter for GSC+ dataset
- ✅ Unit tests passing
- ✅ Integration tests passing

### Phase 3: Conversion & Validation (1 day)

**Day 4:**
- [ ] Run conversion on all three datasets
- [ ] Validate converted data
- [ ] Fix any conversion errors
- [ ] Generate quality reports
- [ ] Manual review of sample documents

**Deliverables:**
- ✅ 306 converted JSON files
- ✅ Validation passing
- ✅ Conversion report generated

### Phase 4: Documentation & Integration (1 day)

**Day 5:**
- [ ] Write user guide
- [ ] Write technical documentation
- [ ] Update dataset catalog
- [ ] Create README files
- [ ] Integrate with benchmark infrastructure
- [ ] Test end-to-end workflows

**Deliverables:**
- ✅ Complete documentation
- ✅ Integration with existing benchmarks
- ✅ Ready for production use

---

## Success Criteria

**Must Have:**
- ✅ All 306 documents converted successfully
- ✅ 100% of annotations with valid offsets
- ✅ >95% of HPO IDs found in ontology
- ✅ All validation checks passing
- ✅ Reproducible conversion process
- ✅ Complete documentation

**Should Have:**
- ✅ <5% warnings in conversion report
- ✅ Automated testing with >80% coverage
- ✅ Integration with chunking benchmark
- ✅ Command-line interface

**Nice to Have:**
- ✅ Web-based review interface
- ✅ Automated data quality dashboard
- ✅ Comparison with original PhenoBERT format

---

## Appendix A: Example Converted Document

```json
{
  "doc_id": "gsc_plus_001",
  "language": "en",
  "source": "phenobert",
  "source_id": "gsc_001.txt",
  "full_text": "The patient presented at 3 years of age with global developmental delay, hypotonia, and recurrent seizures. Brain MRI revealed bilateral ventriculomegaly. Physical examination showed dysmorphic facial features including hypertelorism and low-set ears.",
  "metadata": {
    "dataset": "GSC+",
    "original_filename": "gsc_001.txt",
    "text_length_chars": 251,
    "text_length_words": 37,
    "num_annotations": 7,
    "num_unique_hpo_terms": 7,
    "conversion_date": "2025-01-18T10:30:45Z",
    "converter_version": "1.0.0"
  },
  "annotations": [
    {
      "hpo_id": "HP:0001263",
      "label": "Global developmental delay",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 45,
          "end_char": 70,
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
          "start_char": 72,
          "end_char": 81,
          "text_snippet": "hypotonia",
          "confidence": 0.98
        }
      ]
    },
    {
      "hpo_id": "HP:0001250",
      "label": "Seizure",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 97,
          "end_char": 105,
          "text_snippet": "seizures",
          "confidence": 1.0
        }
      ]
    },
    {
      "hpo_id": "HP:0002119",
      "label": "Ventriculomegaly",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 138,
          "end_char": 154,
          "text_snippet": "ventriculomegaly",
          "confidence": 0.95
        }
      ]
    },
    {
      "hpo_id": "HP:0001999",
      "label": "Abnormal facial shape",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 184,
          "end_char": 208,
          "text_snippet": "dysmorphic facial features",
          "confidence": 0.92
        }
      ]
    },
    {
      "hpo_id": "HP:0000316",
      "label": "Hypertelorism",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 219,
          "end_char": 232,
          "text_snippet": "hypertelorism",
          "confidence": 1.0
        }
      ]
    },
    {
      "hpo_id": "HP:0000369",
      "label": "Low-set ears",
      "assertion_status": "affirmed",
      "evidence_spans": [
        {
          "start_char": 237,
          "end_char": 249,
          "text_snippet": "low-set ears",
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

---

## Appendix B: Conversion Report Example

```json
{
  "conversion_summary": {
    "total_datasets": 3,
    "total_documents": 306,
    "total_annotations": 2564,
    "unique_hpo_terms": 698,
    "conversion_date": "2025-01-18T11:00:00Z",
    "converter_version": "1.0.0",
    "processing_time_seconds": 45.3
  },
  "datasets": {
    "GSC+": {
      "num_documents": 228,
      "num_annotations": 1933,
      "unique_hpo_terms": 497,
      "avg_annotations_per_doc": 8.5,
      "avg_text_length_chars": 503,
      "validation": {
        "total_spans": 1933,
        "valid_spans": 1920,
        "invalid_spans": 13,
        "span_accuracy": 0.993,
        "errors": 5,
        "warnings": 8
      }
    },
    "ID-68": {
      "num_documents": 68,
      "num_annotations": 542,
      "unique_hpo_terms": 156,
      "avg_annotations_per_doc": 8.0,
      "avg_text_length_chars": 612,
      "validation": {
        "total_spans": 542,
        "valid_spans": 540,
        "invalid_spans": 2,
        "span_accuracy": 0.996,
        "errors": 2,
        "warnings": 3
      }
    },
    "GeneReviews": {
      "num_documents": 10,
      "num_annotations": 89,
      "unique_hpo_terms": 45,
      "avg_annotations_per_doc": 8.9,
      "avg_text_length_chars": 1024,
      "validation": {
        "total_spans": 89,
        "valid_spans": 89,
        "invalid_spans": 0,
        "span_accuracy": 1.0,
        "errors": 0,
        "warnings": 1
      }
    }
  },
  "hpo_coverage": {
    "total_unique_hpo_ids": 698,
    "found_in_ontology": 696,
    "missing_from_ontology": 2,
    "coverage_rate": 0.997,
    "missing_ids": ["HP:9999999", "HP:8888888"]
  },
  "errors": [
    {
      "file": "gsc_123.txt",
      "error": "Out of bounds: [500:520]",
      "severity": "high"
    },
    {
      "file": "id68_045.txt",
      "error": "Text mismatch at [100:115]",
      "severity": "medium"
    }
  ],
  "warnings": [
    {
      "file": "gsc_067.txt",
      "warning": "Whitespace mismatch at [200:220]",
      "severity": "low"
    }
  ]
}
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-18
**Authors:** Claude Code (AI Assistant)
**Review Status:** Ready for implementation
