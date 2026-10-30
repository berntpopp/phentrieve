"""Single test evaluation of the frozen reorder pipeline (exploratory, issue #451).

    python -I reorder_test.py <hpo_dir>

Arms per language on the non-cited test split (n = 289):
- gold: curated phenopacket terms through the scorer;
- scorer: frozen LLM extraction (EXTRACT_TAG from results/reorder_frozen.json) through the scorer;
- reorder: the frozen reorder of the scorer's top 50 (results/reorder_frozen.json).
Hit@1/3/10 and MRR, exact disease and gene level, PMID-clustered bootstrap 95% CIs (2,000
replicates), paired differences and exact McNemar. Writes results/reorder_test.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from analysis import (  # noqa: E402
    LANGS,
    RESULTS,
    SEED,
    B,
    load_jsonl,
    mcnemar,
    rank_case,
    write_csv,
)

HERE = Path(__file__).resolve().parent
KS = (1, 3, 10)


def main(hpo_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    frozen = json.loads((RESULTS / "reorder_frozen.json").read_text())
    tag = frozen["env"].split("EXTRACT_TAG=")[1].split()[0]
    prompt = frozen["reorder"]["prompt"]
    gene_ds = {g: {ix.disease_ids[d] for d in ds} for g, ds in ix.gene_diseases.items()}
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if c["split"] == "test" and not c["cited"]
    ]
    rows = []
    for lang in LANGS:
        rr = {
            r["case"]: r
            for r in load_jsonl(
                RESULTS / f"reorder_cc-claude-opus-5-5_top50{tag}_{prompt}_{lang}.jsonl"
            )
        }
        missing = [c["id"] for c in cases if c["id"] not in rr]
        assert not missing, f"{lang}: {len(missing)} test cases not reordered"
        ranks: dict[str, dict[str, list[float]]] = {"exact": {}, "gene": {}}
        for c in cases:
            r = rr[c["id"]]
            genes = gene_ds.get(c["gene"], {c["disease"]})
            gold_rank = rank_case(ix, c, {t["id"]: 1.0 for t in c["observed"]})[0]
            s = scorer.score(ix, {t["id"]: 1.0 for t in c["observed"]})
            gold_gene = gold_rank
            if s is not None:
                order = [ix.disease_ids[d] for d in np.argsort(-s, kind="stable")[:500]]
                hits = [i + 1 for i, d in enumerate(order) if d in genes]
                gold_gene = min(hits) if hits else gold_rank
            arms = {
                "gold": (gold_rank, gold_gene),
                "scorer": (r["scorer_rank"], r.get("scorer_order")),
                "reorder": (r["reorder_rank"] or r["scorer_rank"], r.get("final")),
            }
            for arm, (exact, order) in arms.items():
                ranks["exact"].setdefault(arm, []).append(exact)
                if arm == "gold":
                    ranks["gene"].setdefault(arm, []).append(order)
                else:
                    hits = [i + 1 for i, d in enumerate(order or []) if d in genes]
                    ranks["gene"].setdefault(arm, []).append(
                        min(hits) if hits else exact
                    )
        ids, inv = np.unique([c["pmids"][0] for c in cases], return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(ids))]
        for level, by_arm in ranks.items():
            r = {k: np.array(v, dtype=float) for k, v in by_arm.items()}
            rng = np.random.default_rng(SEED)
            reps: dict[tuple, list[float]] = {}
            for _ in range(B):
                idx = np.concatenate(
                    [members[i] for i in rng.integers(0, len(ids), len(ids))]
                )
                for k in KS:
                    h = {a: float((v[idx] <= k).mean()) for a, v in r.items()}
                    for a in r:
                        reps.setdefault((a, k), []).append(h[a])
                    reps.setdefault(("reorder-scorer", k), []).append(
                        h["reorder"] - h["scorer"]
                    )
                    reps.setdefault(("reorder/gold", k), []).append(
                        h["reorder"] / h["gold"] if h["gold"] else np.nan
                    )
            for k in KS:
                point = {a: float((v <= k).mean()) for a, v in r.items()}
                ci = lambda key, reps=reps: np.nanpercentile(reps[key], [2.5, 97.5])  # noqa: E731
                row = {"lang": lang, "level": level, "k": k, "n": len(cases)}
                for a in r:
                    row[a], (row[f"{a}_lo"], row[f"{a}_hi"]) = point[a], ci((a, k))
                row["reorder_minus_scorer"] = point["reorder"] - point["scorer"]
                row["diff_lo"], row["diff_hi"] = ci(("reorder-scorer", k))
                row["mcnemar_p"] = mcnemar(r["reorder"], r["scorer"], k)
                row["reorder_over_gold"] = (
                    point["reorder"] / point["gold"] if point["gold"] else np.nan
                )
                row["ratio_lo"], row["ratio_hi"] = ci(("reorder/gold", k))
                rows.append(row)
            if level == "exact":
                sc, re_ = r["scorer"], r["reorder"]
                print(
                    f"{lang}: #1 lost {int(((sc == 1) & (re_ > 1)).sum())}, gained {int(((sc > 1) & (re_ == 1)).sum())}"
                )
    write_csv(RESULTS / "reorder_test.csv", rows)
    for row in rows:
        print(
            {
                k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
                for k, v in row.items()
            }
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
