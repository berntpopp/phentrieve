# HPO Data Release Runbook

## Existing Workflow

- GitHub workflow: `.github/workflows/build-data-bundles.yml`
- Docker bundle consumer: `.github/workflows/docker-publish.yml`
- User-facing Docker documentation: `docs/DOCKER-DEPLOYMENT.md`
- CLI commands: `phentrieve data prepare`, `phentrieve index build`, `phentrieve data bundle create`, `phentrieve data download`

## GitHub Actions Release Build

Run the existing bundle workflow from `main` for the pinned HPO release:

```bash
gh workflow run "Build and Release Data Bundles" --ref main -f hpo_version=v2026-02-16 -f models=all -f index_type=both -f create_release=true
```

The workflow builds the minimal database bundle plus single-vector and multivector
model bundles, then publishes them to `data-v2026-02-16` when
`create_release=true`.

## RTX-Local Full Rebuild Fallback

Use this path if GitHub Actions is too slow, fails from compute limits, or cannot
use the RTX.

Prepare an isolated data root:

```bash
export HPO_VERSION=v2026-02-16
export RUN_ID="hpo-${HPO_VERSION}-full-$(date -u +%Y%m%dT%H%M%SZ)"
export PHENTRIEVE_DATA_ROOT_DIR="$PWD/.runs/$RUN_ID/data"
export PHENTRIEVE_DATA_DIR="$PHENTRIEVE_DATA_ROOT_DIR"
export PHENTRIEVE_INDEX_DIR="$PHENTRIEVE_DATA_ROOT_DIR/indexes"
export PHENTRIEVE_RESULTS_DIR="$PWD/.runs/$RUN_ID/results"
mkdir -p "$PHENTRIEVE_DATA_ROOT_DIR" "$PHENTRIEVE_INDEX_DIR" "$PHENTRIEVE_RESULTS_DIR" "dist/hpo-$HPO_VERSION"
```

Prepare the HPO database:

```bash
uv run phentrieve data prepare \
  --hpo-version "$HPO_VERSION" \
  --force \
  --data-dir "$PHENTRIEVE_DATA_ROOT_DIR" \
  --debug
```

Create the minimal bundle:

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from phentrieve.data_processing.bundle_packager import create_bundle

create_bundle(
    output_dir=Path(f"dist/hpo-{os.environ['HPO_VERSION']}"),
    model_name=None,
    data_dir=Path(os.environ["PHENTRIEVE_DATA_ROOT_DIR"]),
)
PY
```

Build and bundle all models:

```bash
for MODEL in \
  "FremyCompany/BioLORD-2023-M" \
  "BAAI/bge-m3" \
  "sentence-transformers/LaBSE" \
  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  "Alibaba-NLP/gte-multilingual-base" \
  "jinaai/jina-embeddings-v2-base-de" \
  "T-Systems-onsite/cross-en-de-roberta-sentence-transformer" \
  "sentence-transformers/distiluse-base-multilingual-cased-v2"
do
  rm -rf "$PHENTRIEVE_INDEX_DIR"
  mkdir -p "$PHENTRIEVE_INDEX_DIR"

  uv run phentrieve index build \
    --model-name "$MODEL" \
    --recreate \
    --batch-size 256 \
    --data-dir "$PHENTRIEVE_DATA_ROOT_DIR" \
    --index-dir "$PHENTRIEVE_INDEX_DIR" \
    --debug

  uv run phentrieve data bundle create \
    --model "$MODEL" \
    --output-dir "dist/hpo-$HPO_VERSION" \
    --data-dir "$PHENTRIEVE_DATA_ROOT_DIR" \
    --debug

  rm -rf "$PHENTRIEVE_INDEX_DIR"
  mkdir -p "$PHENTRIEVE_INDEX_DIR"

  uv run phentrieve index build \
    --model-name "$MODEL" \
    --multi-vector \
    --recreate \
    --batch-size 256 \
    --data-dir "$PHENTRIEVE_DATA_ROOT_DIR" \
    --index-dir "$PHENTRIEVE_INDEX_DIR" \
    --debug

  uv run phentrieve data bundle create \
    --model "$MODEL" \
    --multi-vector \
    --output-dir "dist/hpo-$HPO_VERSION" \
    --data-dir "$PHENTRIEVE_DATA_ROOT_DIR" \
    --debug

  rm -rf "$PHENTRIEVE_INDEX_DIR"
  mkdir -p "$PHENTRIEVE_INDEX_DIR"
done
```

Create checksums:

```bash
cd "dist/hpo-$HPO_VERSION"
sha256sum "phentrieve-data-$HPO_VERSION"-*.tar.gz > SHA256SUMS.txt
sha256sum "phentrieve-data-$HPO_VERSION"-*-multivec.tar.gz > SHA256SUMS-multivec.txt
cd -
```

Verify the expected bundle count:

```bash
find "dist/hpo-$HPO_VERSION" -maxdepth 1 -type f -name "phentrieve-data-$HPO_VERSION-*.tar.gz" | wc -l
```

Expected output: `19` bundle files: 1 minimal, 9 single-vector, 9 multivector.
