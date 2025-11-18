# Phentrieve Scripts

Standalone scripts for data processing, conversion, and analysis.

## Available Scripts

### `convert_phenobert_data.py`

Convert PhenoBERT corpus datasets (GSC+, ID-68, GeneReviews) to Phentrieve JSON format.

**Prerequisites:**
1. Clone PhenoBERT repository:
   ```bash
   git clone https://github.com/EclipseCN/PhenoBERT.git
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
# Convert all datasets
python scripts/convert_phenobert_data.py \
    --phenobert-data path/to/PhenoBERT/phenobert/data \
    --output data/test_texts/phenobert \
    --hpo-data data/hpo_core_data

# Convert specific dataset
python scripts/convert_phenobert_data.py \
    --phenobert-data path/to/PhenoBERT/phenobert/data \
    --output data/test_texts/phenobert \
    --hpo-data data/hpo_core_data \
    --dataset GSC+

# With debug logging
python scripts/convert_phenobert_data.py \
    --phenobert-data path/to/PhenoBERT/phenobert/data \
    --output data/test_texts/phenobert \
    --hpo-data data/hpo_core_data \
    --log-level DEBUG
```

**Output Structure:**

```
data/test_texts/phenobert/
├── GSC_plus/
│   └── annotations/
│       ├── GSC+_doc001.json
│       ├── GSC+_doc002.json
│       └── ... (228 files)
├── ID68/
│   └── annotations/
│       ├── ID-68_doc001.json
│       └── ... (68 files)
├── GeneReviews/
│   └── annotations/
│       ├── GeneReviews_doc001.json
│       └── ... (10 files)
└── conversion_report.json
```

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
    --output data/test_texts/phenobert \
    --hpo-data data/hpo_core_data

# 4. Check output
ls data/test_texts/phenobert/
cat data/test_texts/phenobert/conversion_report.json

# 5. Verify a sample file
cat data/test_texts/phenobert/GSC_plus/annotations/GSC+_*.json | head -n 50
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
