# Phentrieve Scripts

Standalone scripts for data processing, conversion, and analysis.

## Available Scripts

### `convert_phenobert_data.py`

Convert PhenoBERT corpus datasets (GSC+, ID-68, GeneReviews) to Phentrieve JSON format.

**Prerequisites:**
1. Download PhenoBERT dataset (see [PHENOBERT-DOWNLOAD-GUIDE.md](./PHENOBERT-DOWNLOAD-GUIDE.md) for reproducible download instructions):
   ```bash
   # For reproducibility, download specific version
   # See PHENOBERT-DOWNLOAD-GUIDE.md for detailed instructions
   ```

2. Ensure Phentrieve is installed:
   ```bash
   make install  # or: uv sync
   ```

3. Prepare HPO data (if not already done):
   ```bash
   phentrieve data prepare
   ```

**Usage:**

```bash
# Convert all datasets (recommended path for test data)
python scripts/convert_phenobert_data.py \
    --phenobert-data path/to/PhenoBERT/phenobert/data \
    --output tests/data/en/phenobert \
    --hpo-data data/hpo_core_data

# Convert specific dataset
python scripts/convert_phenobert_data.py \
    --phenobert-data path/to/PhenoBERT/phenobert/data \
    --output tests/data/en/phenobert \
    --hpo-data data/hpo_core_data \
    --dataset GSC+

# With debug logging
python scripts/convert_phenobert_data.py \
    --phenobert-data path/to/PhenoBERT/phenobert/data \
    --output tests/data/en/phenobert \
    --hpo-data data/hpo_core_data \
    --log-level DEBUG
```

**Output Structure:**

```
tests/data/en/phenobert/         # Test data organized by language
├── GSC_plus/                    # Normalized from 'GSC+'
│   └── annotations/
│       ├── GSC_plus_1003450.json   # Normalized filenames
│       ├── GSC_plus_1003451.json
│       └── ... (228 files)
├── ID_68/                       # Normalized from 'ID-68'
│   └── annotations/
│       ├── ID_68_doc001.json    # Normalized filenames
│       └── ... (68 files)
├── GeneReviews/
│   └── annotations/
│       ├── GeneReviews_doc001.json
│       └── ... (10 files)
└── conversion_report.json
```

**Note:** Dataset and file names are normalized for filesystem compatibility (e.g., `GSC+` → `GSC_plus`, `ID-68` → `ID_68`), while original names are preserved in JSON metadata for traceability.

**Options:**

- `--phenobert-data PATH` - Path to PhenoBERT data directory (required)
- `--output PATH` - Output directory for converted files (required)
- `--hpo-data PATH` - Path to HPO core data directory (required)
- `--dataset {GSC+,ID-68,GeneReviews,all}` - Dataset to convert (default: all)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level (default: INFO)

---

### `convert_raghpo_paper_data.py`

Convert the released RAG-HPO paper benchmark artifacts into the same
per-document JSON fixture shape already used by the converted PhenoBERT
datasets in this repository.

This is intended for the released files from:

- `Test_Cases.csv`
- `RAG-HPO Tests and Data Analysis copy.xlsx`

The converter currently extracts the paper's two benchmark datasets:

- `CSC`
- `GSC`

**Prerequisites:**
1. Ensure Phentrieve is installed:
   ```bash
   make install  # or: uv sync
   ```
2. Prepare HPO data (if not already done):
   ```bash
   phentrieve data prepare
   ```

**Usage:**

```bash
# Convert both released benchmark datasets
python scripts/convert_raghpo_paper_data.py \
    --workbook path/to/'RAG-HPO Tests and Data Analysis copy.xlsx' \
    --test-cases-csv path/to/Test_Cases.csv \
    --hpo-json data/hp.json \
    --output tests/data/en/raghpo_paper

# Convert only the CSC subset
python scripts/convert_raghpo_paper_data.py \
    --workbook path/to/'RAG-HPO Tests and Data Analysis copy.xlsx' \
    --test-cases-csv path/to/Test_Cases.csv \
    --hpo-json data/hp.json \
    --output tests/data/en/raghpo_paper \
    --dataset CSC

# Convert CSC and normalize obsolete HPO IDs to current replacements
python scripts/convert_raghpo_paper_data.py \
    --workbook path/to/'RAG-HPO Tests and Data Analysis copy.xlsx' \
    --test-cases-csv path/to/Test_Cases.csv \
    --hpo-json data/hp.json \
    --output tests/data/en/raghpo_paper \
    --dataset CSC \
    --normalize-obsolete-ids

# Strict current-term mode: also drop obsolete IDs that still have no replacement
python scripts/convert_raghpo_paper_data.py \
    --workbook path/to/'RAG-HPO Tests and Data Analysis copy.xlsx' \
    --test-cases-csv path/to/Test_Cases.csv \
    --hpo-json data/hp.json \
    --output tests/data/en/raghpo_paper \
    --dataset CSC \
    --normalize-obsolete-ids \
    --drop-obsolete-without-replacement

# Download the released source files automatically into a local cache
python scripts/convert_raghpo_paper_data.py \
    --download-source \
    --download-dir tests/data/en/raghpo_paper/source \
    --hpo-json data/hp.json \
    --output tests/data/en/raghpo_paper \
    --dataset CSC \
    --normalize-obsolete-ids \
    --drop-obsolete-without-replacement
```

**Output Structure:**

```
tests/data/en/raghpo_paper/
├── CSC/
│   └── annotations/
│       ├── CSC_1.json
│       └── ...
├── GSC/
│   └── annotations/
│       ├── GSC_1.json
│       └── ...
└── conversion_report.json
```

**Compatibility Notes:**

- Output documents follow the same basic JSON contract as the converted
  PhenoBERT fixtures:
  - `doc_id`
  - `language`
  - `source`
  - `full_text`
  - `metadata`
  - `annotations`
- `evidence_spans` are derived by matching each released
  `hpo_description` phrase back into the source note text.
- Labels can be resolved either from `--hpo-terms` or directly from
  `--hpo-json`. In this repository, `data/hp.json` is sufficient for a full
  real conversion run.
- If a released phrase cannot be found exactly in the note text, conversion
  continues and that annotation receives an empty `evidence_spans` list.
  These cases are recorded in `conversion_report.json`.
- By default, released HPO IDs are preserved exactly as published. If
  `--normalize-obsolete-ids` is enabled, obsolete IDs are replaced with their
  current ontology terms using replacement mappings from the supplied
  `--hpo-json` ontology export, and each normalization is recorded as a warning
  in `conversion_report.json`.
- If `--download-source` is enabled, the CLI downloads the released workbook
  and `Test_Cases.csv` from the upstream RAG-HPO repository into
  `--download-dir` and reuses the cached copies on subsequent runs. The default
  cache path is `tests/data/en/raghpo_paper/source`.
- If `--drop-obsolete-without-replacement` is also enabled, any obsolete term
  that still cannot be mapped to a current HPO ID is omitted from the converted
  annotations and recorded in `conversion_report.json`.

**Options:**

- `--workbook PATH` - Path to `RAG-HPO Tests and Data Analysis copy.xlsx` (required unless `--download-source` is used)
- `--test-cases-csv PATH` - Path to `Test_Cases.csv` (required unless `--download-source` is used)
- `--download-source` - Download the released workbook and `Test_Cases.csv` into a local cache when not available locally
- `--download-dir PATH` - Cache directory for downloaded RAG-HPO source files (default: `tests/data/en/raghpo_paper/source`)
- `--hpo-terms PATH` - Path to `data/hpo_core_data/hpo_terms.tsv` (optional if `--hpo-json` is supplied)
- `--output PATH` - Output directory for converted files (required)
- `--dataset {CSC,GSC,all}` - Dataset subset to convert (default: all)
- `--normalize-obsolete-ids` - Replace obsolete released HPO IDs with current ontology replacements
- `--hpo-json PATH` - Path to an HPO ontology JSON export such as `data/hp.json` (required with `--normalize-obsolete-ids`, and sufficient on its own for label resolution)
- `--drop-obsolete-without-replacement` - When normalizing, drop obsolete IDs that still have no current replacement
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level (default: INFO)

---

### `generate_chunking_variants.py`

Generate ground-truth chunking variants for annotated documents using the Voronoi boundary algorithm. This creates chunks at multiple expansion levels for benchmarking Phentrieve's semantic chunking strategies.

**Algorithm:** Pure annotation-position-based chunking using geometric midpoints. Each annotation gets exclusive territory, and chunks are generated at multiple context levels (0.0 = annotation only, 0.5 = balanced, 1.0 = full territory).

**Prerequisites:**
1. Annotated JSON files (e.g., from `convert_phenobert_data.py`)
2. Files must have `full_text` and `annotations` fields

**Usage:**

```bash
# Process single file
python scripts/generate_chunking_variants.py \
    --input tests/data/en/phenobert/GSC_plus/annotations/GSC_plus_1003450.json

# Process entire directory (recommended)
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --pattern "*/annotations/*.json"

# Custom expansion ratios
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --expansion-ratios 0.0 0.25 0.5 0.75 1.0

# Dry run (preview without writing)
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --dry-run

# Force overwrite existing chunks
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --force
```

**Output:**

The script augments existing JSON files in-place by adding a `chunk_variants` field:

```json
{
  "doc_id": "GSC+_1003450",
  "full_text": "A syndrome of brachydactyly...",
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
            "0.00": {"text": "brachydactyly", "span": [14, 27]},
            "0.50": {"text": "A syndrome of brachydactyly", "span": [0, 27]},
            "1.00": {"text": "A syndrome of brachydactyly (absence...", "span": [0, 35]}
          }
        }
      ]
    }
  }
}
```

**Options:**

- `--input PATH` - Single input file (mutually exclusive with --input-dir)
- `--input-dir PATH` - Input directory for batch processing (mutually exclusive with --input)
- `--pattern GLOB` - File pattern for directory mode (default: `*/annotations/*.json`)
- `--expansion-ratios FLOAT [FLOAT ...]` - Expansion ratios (default: `0.0 0.5 1.0`)
- `--strategy-name STR` - Strategy name for output (default: `voronoi_v1`)
- `--dry-run` - Preview without writing changes
- `--force` - Overwrite existing chunk variants
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level (default: INFO)

**Properties:**

- ✅ **Idempotent:** Running multiple times produces identical output
- ✅ **Language-agnostic:** Works on any language (tested with English, German)
- ✅ **Concept isolation:** Guarantees no annotation overlap
- ✅ **Reproducible:** Full provenance tracking with script version and parameters
- ✅ **Translation-ready:** Re-run on translated texts with same HPO IDs

**Testing:**

```bash
# Run unit tests
make test-scripts

# Run with coverage
pytest scripts/tests/ -v --cov=scripts --cov-report=term-missing

# Run all tests (package + scripts)
make test-all
```

## Example Workflow

```bash
# 1. Clone PhenoBERT repository (first time only)
cd /tmp
git clone https://github.com/EclipseCN/PhenoBERT.git

# 2. Return to Phentrieve directory
cd /path/to/phentrieve

# 3. Run conversion
python scripts/convert_phenobert_data.py \
    --phenobert-data /tmp/PhenoBERT/phenobert/data \
    --output tests/data/en/phenobert \
    --hpo-data data/hpo_core_data

# 4. Check output
ls tests/data/en/phenobert/
cat tests/data/en/phenobert/conversion_report.json

# 5. Verify a sample file
cat tests/data/en/phenobert/GSC_plus/annotations/GSC_plus_*.json | head -n 50
```

## Troubleshooting

**Import errors:**
```bash
# Make sure Phentrieve is installed
make install
# or
uv sync
```

**PhenoBERT data not found:**
```bash
# Verify directory structure
ls /tmp/PhenoBERT/phenobert/data/
# Should show: GSC+/, ID-68/, GeneReviews/
```

**HPO data not found:**
```bash
# Prepare HPO data
phentrieve data prepare

# Verify
ls data/hpo_core_data/
# Should show: hpo_terms.tsv and other files
```

**Permission errors:**
```bash
# Make script executable
chmod +x scripts/convert_phenobert_data.py
```

## Development

To modify or extend the conversion:

1. Edit converter classes in `phentrieve/data_processing/phenobert_converter.py`
2. Edit CLI script in `scripts/convert_phenobert_data.py`
3. Run with debug logging to see detailed output:
   ```bash
   python scripts/convert_phenobert_data.py --log-level DEBUG ...
   ```

See `.planning/active/PHENOBERT-CORPUS-CONVERSION-PLAN.md` for architecture details.
