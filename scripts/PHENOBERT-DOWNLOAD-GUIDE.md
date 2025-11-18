# PhenoBERT Data Download Guide

**For Reproducibility**: Always download a specific version/commit of the PhenoBERT dataset.

## Recommended Approach: Download Specific Commit

### Option 1: Download as ZIP (No Git Required)

**Recommended for reproducibility** - Downloads exact snapshot.

```bash
# 1. Choose a specific commit or use latest stable
# Latest as of 2025-01-18: Check https://github.com/EclipseCN/PhenoBERT/commits/main

# 2. Download ZIP of specific commit
# Replace COMMIT_SHA with actual commit hash
COMMIT_SHA="your_commit_sha_here"
wget https://github.com/EclipseCN/PhenoBERT/archive/${COMMIT_SHA}.zip -O phenobert.zip

# 3. Extract
unzip phenobert.zip

# 4. The data will be in: PhenoBERT-${COMMIT_SHA}/phenobert/data/
```

### Option 2: Clone Specific Commit (Git Required)

```bash
# 1. Clone repository
git clone https://github.com/EclipseCN/PhenoBERT.git

# 2. Checkout specific commit for reproducibility
cd PhenoBERT
git checkout COMMIT_SHA_HERE

# 3. Document which commit you're using
git rev-parse HEAD > PHENOBERT_VERSION.txt
git log -1 --format="%ci" >> PHENOBERT_VERSION.txt

# 4. Data is in: phenobert/data/
```

### Option 3: Use Tagged Release (If Available)

```bash
# Check for releases at: https://github.com/EclipseCN/PhenoBERT/releases

# Download specific release
wget https://github.com/EclipseCN/PhenoBERT/archive/refs/tags/vX.Y.Z.zip

# Or clone and checkout tag
git clone https://github.com/EclipseCN/PhenoBERT.git
cd PhenoBERT
git checkout tags/vX.Y.Z
```

## Verify Data Integrity

The converter automatically tracks provenance including:
- Git commit SHA (if downloaded via git)
- Commit date
- Repository URL
- Converter version

This information is saved in `conversion_report.json`.

## Example: Reproducible Workflow

```bash
# 1. Define version to use
PHENOBERT_VERSION="abc123def456"  # Replace with actual commit SHA

# 2. Download specific version
wget "https://github.com/EclipseCN/PhenoBERT/archive/${PHENOBERT_VERSION}.zip" \
    -O phenobert-${PHENOBERT_VERSION}.zip

# 3. Extract
unzip "phenobert-${PHENOBERT_VERSION}.zip"

# 4. Run conversion with version tracking
python scripts/convert_phenobert_data.py \
    --phenobert-data "PhenoBERT-${PHENOBERT_VERSION}/phenobert/data" \
    --output data/test_texts/phenobert \
    --hpo-data data/hpo_core_data

# 5. Conversion report will include:
#    - Source repository URL
#    - Commit SHA (if git repo)
#    - Converter version
#    - Conversion date
```

## What Gets Tracked in Conversion Report

The `conversion_report.json` includes provenance metadata:

```json
{
  "provenance": {
    "converter_version": "1.0.0",
    "conversion_date": "2025-01-18T10:30:00",
    "source": {
      "repository": "https://github.com/EclipseCN/PhenoBERT",
      "version": {
        "commit_sha": "abc123def456...",
        "commit_date": "2024-12-15 14:23:45 +0100",
        "is_dirty": false
      }
    }
  },
  "summary": {
    "total_documents": 306,
    "total_annotations": 2564
  },
  "datasets": { ... }
}
```

## For Publications/Research

When citing the converted dataset, include:
1. PhenoBERT repository commit SHA
2. Phentrieve converter version
3. Conversion date
4. Link to this repository

Example citation metadata:
```
Dataset: PhenoBERT corpus (converted to Phentrieve format)
Source: https://github.com/EclipseCN/PhenoBERT
Source version: commit abc123def456 (2024-12-15)
Converter: Phentrieve v1.0.0
Conversion date: 2025-01-18
Converter repository: https://github.com/berntpopp/phentrieve
```

## Troubleshooting

### If converter reports "Not a git repository"

This happens when downloading as ZIP (not cloned via git).

**This is fine!** The conversion report will note:
```json
"version": {"note": "Not a git repository"}
```

Document the download method manually:
- Save the commit SHA you downloaded
- Save the download date
- Keep the ZIP file or note the exact URL used

### If using git clone and seeing "is_dirty: true"

This means the repository has uncommitted changes. For reproducibility:
```bash
# Check what changed
git status

# If you modified files, either:
# 1. Revert changes: git checkout .
# 2. Or commit them and document the changes
```

## Best Practices

1. **Always use specific commit** - Don't use "latest" or "main" branch
2. **Document the version** - Save commit SHA in your project
3. **Archive the data** - Keep a copy of the original ZIP
4. **Keep conversion report** - It contains all provenance metadata
5. **For papers** - Include commit SHA in methods section
