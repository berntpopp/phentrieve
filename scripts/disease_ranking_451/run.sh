#!/usr/bin/env bash
# Rerun the whole issue #451 evaluation from the pinned inputs (PROTOCOL.md).
#   EVAL451_DATA=/path/to/data scripts/disease_ranking_451/run.sh
# Downloads land in their own empty directories under $EVAL451_DATA and are checked against
# manifest.json. Letters are cached in letters/ (the LLM is only called for missing ones,
# which needs GEMINI_API_KEY). Extractions are cached in results/extractions_*.jsonl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="${EVAL451_DATA:-$HOME/development/_eval451}"
VARLENS="${VARLENS_REPO:-$HOME/development/VarLens}"
PY="uv run --project $REPO python -I"

fetch() {  # fetch <dir> <url>...: download into an empty dir unless already present
  local dir="$1"; shift
  [ -d "$dir" ] && return 0
  mkdir -p "$dir"
  for url in "$@"; do (cd "$dir" && curl -sSfLO "$url"); done
}
HPO=https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-09-01
fetch "$DATA/hpo-v2026-09-01" "$HPO/hp.json" "$HPO/phenotype.hpoa" "$HPO/genes_to_phenotype.txt" "$HPO/genes_to_disease.txt"
TR=https://raw.githubusercontent.com/obophenotype/hpo-translations/735ab350d1b1f8ea9e8833f6505c8c973df4280b/babelon
fetch "$DATA/hpo-translations-735ab35" "$TR/hp-de.babelon.tsv" "$TR/hp-de.synonyms.tsv"
fetch "$DATA/phenopacket-store-0.1.27" https://github.com/monarch-initiative/phenopacket-store/releases/download/0.1.27/all_phenopackets.zip
fetch "$DATA/phentrieve-data-v2026-09-01" \
  https://github.com/berntpopp/phentrieve-data/releases/download/hpo-v2026-09-01-r1/phentrieve-data-v2026-09-01-biolord-multivec.tar.gz \
  https://github.com/berntpopp/phentrieve-data/releases/download/hpo-v2026-09-01-r1/SHA256SUMS
fetch "$DATA/rarebench-6f054e0" https://huggingface.co/datasets/chenxz/RareBench/resolve/6f054e04071953ef2c1779b279074245f2ab398c/data.zip

# Verify every pinned input file against manifest.json.
python3 - "$HERE/manifest.json" "$DATA" <<'EOF'
import hashlib, json, sys, pathlib
m, data = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
for d, files in m["inputs_sha256"].items():
    for name, digest in files.items():
        p = data / d / name
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            sys.exit(f"SHA256 mismatch: {p}")
print("inputs verified")
EOF

[ -d "$DATA/phenopacket-store-0.1.27/x" ] || (cd "$DATA/phenopacket-store-0.1.27" && mkdir x && unzip -q all_phenopackets.zip -d x)
[ -d "$DATA/rarebench-6f054e0/x" ] || (cd "$DATA/rarebench-6f054e0" && mkdir x && unzip -q data.zip -d x)
[ -d "$DATA/phentrieve-data-v2026-09-01/data" ] || (mkdir "$DATA/phentrieve-data-v2026-09-01/data" && tar xzf "$DATA"/phentrieve-data-v2026-09-01/*.tar.gz -C "$DATA/phentrieve-data-v2026-09-01/data")

cd "$HERE"
$PY cases.py "$DATA/hpo-v2026-09-01" "$DATA/phenopacket-store-0.1.27/x"
uv run --project "$REPO" python generate.py  # cached letters are reused

# Phentrieve resolves phentrieve.yaml and data/ from the working directory: point it at the bundle.
RUN="$DATA/run"; mkdir -p "$RUN"
cp "$REPO/phentrieve.yaml" "$RUN/"; ln -sfn "$DATA/phentrieve-data-v2026-09-01/data" "$RUN/data"
for lang in de en; do (cd "$RUN" && uv run --project "$REPO" python "$HERE/extract.py" "$lang"); done

ARGS=("$DATA/hpo-v2026-09-01" "$DATA/phentrieve-data-v2026-09-01/data" "$DATA/hpo-translations-735ab35")
$PY analysis.py tune "${ARGS[@]}"
$PY analysis.py test "${ARGS[@]}"
$PY rarebench.py "$DATA/hpo-v2026-09-01" "$DATA/rarebench-6f054e0/x/data"
uv run --project "$REPO" pytest "$REPO/tests/unit/scripts/test_disease_ranking_scorer.py" -q -p no:cacheprovider
