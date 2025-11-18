# Annotation-Based Chunking

Ground-truth chunking for benchmarking using Voronoi boundaries with word alignment.

## Purpose

Create reference chunks based on actual annotation positions to benchmark semantic chunking strategies. Uses pure geometric algorithm (no NLP) - language-agnostic.

## Algorithm

**Voronoi Boundaries:** Each annotation gets exclusive territory via midpoints to neighbors.

```
Text:     [annotation1]    [annotation2]    [annotation3]
Territory: |----T1----|      |----T2----|      |----T3----|
Boundary:  0          mid1  mid1        mid2  mid2      end
```

**Word Alignment:** Boundaries adjusted to complete words (no mid-word cutting).

**Expansion Ratios:**
- `0.0` - Annotation only (minimal context)
- `0.5` - 50% of territory (balanced)
- `1.0` - Full territory (maximal context)

## Usage

```bash
# Process all PhenoBERT data
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --pattern "*/annotations/*.json"

# Single file
python scripts/generate_chunking_variants.py \
    --input tests/data/en/phenobert/GSC_plus/annotations/file.json

# Custom ratios
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --expansion-ratios 0.0 0.25 0.5 0.75 1.0

# Dry run (preview)
python scripts/generate_chunking_variants.py \
    --input-dir tests/data/en/phenobert \
    --dry-run
```

## Output Format

Augments JSON files in-place with `chunk_variants` field:

```json
{
  "doc_id": "GSC+_1003450",
  "full_text": "...",
  "annotations": [...],
  "chunk_variants": {
    "voronoi_v1": {
      "provenance": {
        "script_version": "1.0.0",
        "generated_at": "2025-11-18T14:22:42Z",
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
            "0.00": {"text": "brachydactyly ", "span": [14, 28]},
            "0.50": {"text": "syndrome of brachydactyly and...", "span": [2, 45]},
            "1.00": {"text": "A syndrome of brachydactyly...", "span": [0, 71]}
          }
        }
      ]
    }
  }
}
```

## Properties

- ✅ **Idempotent** - Re-run produces identical output
- ✅ **Word-aligned** - No mid-word cutting
- ✅ **Language-agnostic** - Works with any alphabet
- ✅ **Reproducible** - Full provenance tracking
- ✅ **Concept isolation** - No annotation overlap

## Testing

```bash
# Run script tests
make test-scripts

# Run with coverage
pytest scripts/tests/ -v --cov=scripts

# All tests
make test-all
```

## Implementation

- `scripts/annotation_chunker.py` - Core algorithm (100% coverage)
- `scripts/generate_chunking_variants.py` - CLI script (97% coverage)
- `scripts/shared_utils.py` - Shared utilities
- `scripts/tests/` - 40 tests, all passing

## Limitations

**Skips files with overlapping annotations** (23% of PhenoBERT dataset). Overlapping annotations violate the Voronoi concept isolation principle and must be manually reviewed.

Example error:
```
ERROR - Overlapping annotations detected: annotation 2 overlaps with annotation 3
```

These files need annotation correction before chunking.
