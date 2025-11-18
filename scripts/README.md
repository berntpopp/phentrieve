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

See `plan/01-active/PHENOBERT-CORPUS-CONVERSION-PLAN.md` for architecture details.
