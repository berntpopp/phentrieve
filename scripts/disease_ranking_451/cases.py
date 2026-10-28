"""Select the evaluation cases from phenopacket-store (issue #451).

Single-gene, solved cases whose diagnosis is a gene-linked disease of the scorer index,
stratified by HPOA citation (up to 400 non-cited, cited fill to 600), at most 3 cases
per gene, seeded. The
diagnosed genes are split 80/20 into test/dev (seeded). Writes `cases.jsonl` and
`results/attrition.csv`.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # run with -I: own code only, never the data dirs
import scorer  # noqa: E402

SEED = 451
TOTAL, NON_CITED_MAX = (
    600,
    400,
)  # non-cited first (primary endpoint), cited fill the rest
MAX_PER_GENE = 3
DEV_GENE_SHARE = 0.2
HERE = Path(__file__).resolve().parent


def _age(x: dict | None) -> str | None:
    """ISO 8601 age, gestational age or ontology class label of a phenopacket time element."""
    if not x:
        return None
    if "age" in x:
        return x["age"].get("iso8601duration")
    if "gestationalAge" in x:
        return f"gestational week {x['gestationalAge'].get('weeks')}"
    if "ontologyClass" in x:
        return x["ontologyClass"].get("label")
    return None


def parse(pp: dict) -> dict | None:
    """VarLens parsePhenopacket: one diagnosed gene and >=1 observed HPO term, else None."""
    genes: set[str] = set()
    diseases: list[str] = []
    for interp in pp.get("interpretations") or []:
        dx = interp.get("diagnosis") or {}
        if dx.get("disease", {}).get("id"):
            diseases.append(dx["disease"]["id"])
        for gi in dx.get("genomicInterpretations") or []:
            vd = (gi.get("variantInterpretation") or {}).get(
                "variationDescriptor"
            ) or {}
            sym = (gi.get("gene") or {}).get("symbol") or (
                vd.get("geneContext") or {}
            ).get("symbol")
            if sym:
                genes.add(sym)
    for d in pp.get("diseases") or []:
        if d.get("term", {}).get("id"):
            diseases.append(d["term"]["id"])
    feats = pp.get("phenotypicFeatures") or []
    observed = list(
        dict.fromkeys(
            f["type"]["id"]
            for f in feats
            if not f.get("excluded") and f["type"]["id"].startswith("HP:")
        )
    )
    excluded = [
        t
        for t in dict.fromkeys(f["type"]["id"] for f in feats if f.get("excluded"))
        if t not in observed
    ]
    if len(genes) != 1 or not observed:
        return None
    subject = pp.get("subject") or {}
    onset = next(
        (_age(d.get("onset")) for d in pp.get("diseases") or [] if d.get("onset")), None
    )
    return {
        "id": pp.get("id", ""),
        "gene": genes.pop(),
        "diseases": list(dict.fromkeys(diseases)),
        "pmids": sorted(
            {
                r["id"]
                for r in (pp.get("metaData") or {}).get("externalReferences") or []
                if r.get("id", "").startswith("PMID:")
            }
        ),
        "sex": subject.get("sex"),
        "age": _age(subject.get("timeAtLastEncounter")),
        "onset": onset,
        "observed": observed,
        "excluded": excluded,
    }


def hpoa_references(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if line.startswith(("#", "database_id")) or len(cols) < 5:
            continue
        refs[cols[0]] |= {
            r.strip() for r in cols[4].split(";") if r.startswith("PMID:")
        }
    return refs


def main(hpo_dir: Path, pp_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    in_index = set(ix.disease_ids)
    refs = hpoa_references(hpo_dir / "phenotype.hpoa")
    files = sorted(pp_dir.rglob("*.json"))
    attrition = [("phenopacket files", len(files))]
    parsed = [(f, parse(json.loads(f.read_text(encoding="utf-8")))) for f in files]
    cases = [(f, c) for f, c in parsed if c is not None]
    attrition.append(("single gene, >=1 observed term", len(cases)))
    cases = [
        (f, {**c, "disease": next((d for d in c["diseases"] if d in in_index), None)})
        for f, c in cases
    ]
    cases = [(f, c) for f, c in cases if c["disease"]]
    attrition.append(("diagnosis is a gene-linked disease of the index", len(cases)))
    cases = [(f, c) for f, c in cases if c["pmids"]]
    attrition.append(("has a source PMID", len(cases)))
    for _, c in cases:
        # Registered rule (VarLens): cited when HPOA cites the PMID for any of the case's diagnoses.
        c["cited"] = any(
            p in refs.get(d, set()) for d in c["diseases"] for p in c["pmids"]
        )
    pool = Counter("cited" if c["cited"] else "non_cited" for _, c in cases)
    attrition.append(("eligible non-cited", pool["non_cited"]))
    attrition.append(("eligible cited", pool["cited"]))

    rng = random.Random(SEED)  # noqa: S311 - seeded sampling, not crypto
    rng.shuffle(cases)
    per_gene: Counter[str] = Counter()
    sample = []
    for cited, quota in ((False, NON_CITED_MAX), (True, TOTAL)):
        for f, c in cases:
            if (
                c["cited"] == cited
                and per_gene[c["gene"]] < MAX_PER_GENE
                and len(sample) < quota
            ):
                per_gene[c["gene"]] += 1
                sample.append((f, c))
    genes = sorted(per_gene)
    random.Random(SEED + 1).shuffle(genes)  # noqa: S311
    dev = set(genes[: round(DEV_GENE_SHARE * len(genes))])
    sample.sort(key=lambda fc: fc[1]["id"])
    label = dict(zip(ix.disease_ids, ix.disease_labels, strict=True))
    hp_labels, _, aliases = scorer.parse_hp_json(hpo_dir / "hp.json")
    term = lambda t: {"id": t, "label": hp_labels.get(aliases.get(t, t))}  # noqa: E731
    with (HERE / "cases.jsonl").open("w", encoding="utf-8") as out:
        for f, c in sample:
            c.update(
                file=str(f.relative_to(pp_dir)),
                disease_label=label[c["disease"]],
                split="dev" if c["gene"] in dev else "test",
                observed=[term(t) for t in c["observed"]],
                excluded=[term(t) for t in c["excluded"]],
            )
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        for stratum, cited in (("non-cited", False), ("cited", True)):
            n = sum(c["split"] == split and c["cited"] == cited for _, c in sample)
            attrition.append((f"sampled {split} {stratum}", n))
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results" / "attrition.csv").open("w", newline="") as out:
        csv.writer(out, lineterminator="\n").writerows([("step", "n"), *attrition])
    for row in attrition:
        print(*row, sep="\t")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
