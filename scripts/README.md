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

---

### `analyze_embedding_ontology.py`

Ontology–embedding fidelity analysis. Loads BioLORD HPO embeddings from the
existing ChromaDB collection (cached to `.npy` on first run), computes four
correlation metrics between the embedding space and the curated HPO DAG, and
produces a timestamped results directory with a summary JSON, a per-term
fidelity CSV, and five plots.

**Prerequisites:**

1. Phentrieve installed with the `analysis` extra:
   ```bash
   uv sync --extra analysis
   ```
2. HPO SQLite DB prepared:
   ```bash
   phentrieve data prepare
   ```
3. BioLORD index built:
   ```bash
   phentrieve index build --model-name FremyCompany/BioLORD-2023-M
   ```

> **Note:** The embedding cache reads the Chroma index location from the `PHENTRIEVE_INDEX_DIR` environment variable (falling back to `~/.phentrieve/hpo_chroma_index/`). When running from a dev checkout whose index lives under the repo's `data/indexes/`, export it first:
>
> ```bash
> export PHENTRIEVE_INDEX_DIR=$PWD/data/indexes
> ```
>
> Run `python scripts/analyze_embedding_ontology.py --help` for all flags.

**Usage:**

```bash
python scripts/analyze_embedding_ontology.py \
    --model-name FremyCompany/BioLORD-2023-M \
    --k 10 \
    --n-pairs 50000
```

Full flag set: see `--help`.

**Outputs** — under `data/results/ontology_fidelity/<model-slug>_<YYYYMMDD-HHMMSS>/`:

| File | Description |
|---|---|
| `summary.json` | Spearman ρ (shortest-path & Resnik), mean per-term fidelity, branch k-NN purity, depth correlation, config echo. |
| `per_term_fidelity.csv` | Per-term fidelity, sorted ascending (worst first). |
| `umap_coords.csv` | UMAP coordinates for each HPO term. |
| `umap_by_branch.png` | UMAP colored by top-level HPO branch. |
| `umap_by_fidelity.png` | UMAP colored by per-term fidelity (RdBu). |
| `umap_interactive.html` | Interactive Plotly; toggle between branch and fidelity colorings. |
| `distance_correlation.png` | Hexbin of embedding cosine vs graph distance (two panels). |
| `branch_fidelity.png` | Per-branch mean fidelity bar chart. |

See [`.planning/specs/2026-04-17-ontology-embedding-fidelity-design.md`](../.planning/specs/2026-04-17-ontology-embedding-fidelity-design.md)
for the full contract and metric definitions.
