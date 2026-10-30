"""Compare reorder variants (exploratory, issue #451): Hit@1/3/10, exact and gene-level.

    python -I reorder_eval.py <hpo_dir> [split]

Gene-level: a candidate counts when it is a disease of the case's causal gene (OMIM/ORPHA
twins and allelic diseases of one gene). Every results/reorder_*_<lang>.jsonl is evaluated on
the cases it shares with the others, so variants are compared on the same cases.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import scorer  # noqa: E402
from analysis import LANGS, RESULTS, load_jsonl, write_csv  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(hpo_dir: Path, split: str = "dev") -> None:
    ix = scorer.build_index(hpo_dir)
    gene_ds = {g: {ix.disease_ids[d] for d in ds} for g, ds in ix.gene_diseases.items()}
    cases = {
        c["id"]: c for c in load_jsonl(HERE / "cases.jsonl") if c["split"] == split
    }
    rows = []
    for lang in LANGS:
        files = sorted(RESULTS.glob(f"reorder_*_{lang}.jsonl"))
        runs = {
            re.sub(rf"^reorder_|_{lang}\.jsonl$", "", f.name): {
                r["case"]: r for r in load_jsonl(f)
            }
            for f in files
        }
        has_order = {
            k for k, v in runs.items() if any("final" in r for r in v.values())
        }
        base = next((runs[k] for k in sorted(has_order)), None)
        shared = (
            set(cases).intersection(*(set(v) for v in runs.values())) if runs else set()
        )
        for name, recs in [("scorer", None), *runs.items()]:
            exact, gene = [], []
            for cid in sorted(shared):
                c = cases[cid]
                rec = (recs or base or next(iter(runs.values())))[cid]
                e = (
                    rec["scorer_rank"]
                    if recs is None
                    else (rec["reorder_rank"] or rec["scorer_rank"])
                )
                exact.append(e)
                order = rec.get("scorer_order") if recs is None else rec.get("final")
                if (name not in has_order and recs is not None) or (
                    recs is None and base is None
                ):
                    gene.append(float("nan"))  # this run stored no candidate order
                elif order is None:
                    gene.append(e)  # no usable term: no candidate list
                elif order:
                    hits = [
                        i + 1
                        for i, d in enumerate(order)
                        if d in gene_ds.get(c["gene"], {c["disease"]})
                    ]
                    gene.append(min(hits) if hits else e)
                else:
                    gene.append(e)
            n = len(exact)
            row = {"lang": lang, "split": split, "variant": name, "n": n}
            for k in (1, 3, 10):
                row[f"hit{k}"] = sum(r <= k for r in exact) / n if n else float("nan")
                row[f"gene_hit{k}"] = (
                    sum(r <= k for r in gene) / n
                    if n and all(g == g for g in gene)
                    else float("nan")
                )
            rows.append(row)
            print(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
            )
    write_csv(RESULTS / f"reorder_eval_{split}.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "dev")
