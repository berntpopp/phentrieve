"""Post-hoc error analysis (exploratory, not registered): what costs the disease rank? (#451)

    python -I errors.py <hpo_dir>

On the test split, per language and Phentrieve arm (default top-1 and frozen soft):
- counterfactual profiles that separate missing terms from wrong terms;
- false positives: negation failures (an excluded finding extracted as present), generic
  terms, the most frequent wrong terms;
- misses by organ system and by term specificity.
"Related" means ancestor-or-self in either direction within the phenotype subtree.
Writes results/errors_*.csv.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from analysis import (  # noqa: E402
    LANGS,
    RESULTS,
    canonical,
    load_jsonl,
    profiles,
    write_csv,
)

HERE = Path(__file__).resolve().parent


def main(hpo_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    root = ix.index[scorer.ROOT]
    anc = [set(ix.ancestors(t).tolist()) for t in range(len(ix.ids))]
    systems = {
        t: [
            ix.labels[s]
            for s in anc[t]
            if s != root and root in anc[s] and len(anc[s]) == 2
        ]
        for t in range(len(ix.ids))
    }
    frozen = json.loads((RESULTS / "frozen.json").read_text())
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if c["split"] == "test" and not c["cited"]
    ]
    ext = {
        lang: {e["case"]: e for e in load_jsonl(RESULTS / f"extractions_{lang}.jsonl")}
        for lang in LANGS
    }
    n = len(ix.disease_ids)

    def rank(case: dict, terms: set[int]) -> float:
        s = scorer.score(ix, {ix.ids[t]: 1.0 for t in terms})
        return (
            (n + 1) / 2
            if s is None
            else scorer.avg_rank(s, ix.disease_ids.index(case["disease"]))
        )

    def related(a: int, b: int) -> bool:
        return a in anc[b] or b in anc[a]

    rows, fp_terms, miss_rows = [], Counter(), []
    for case in cases:
        gold = canonical(ix, [t["id"] for t in case["observed"]])
        excluded = canonical(ix, [t["id"] for t in case["excluded"]])
        for lang in LANGS:
            arms = profiles(
                case, ext[lang][case["id"]], (frozen["k"], frozen["weighting"])
            )
            for arm in ("top1", "soft"):
                pred = canonical(ix, [t for t, w in arms[arm].items() if w > 0])
                tp = {p for p in pred if any(related(p, g) for g in gold)}
                fp = pred - tp
                found = {g for g in gold if any(related(g, p) for p in pred)}
                neg = {p for p in fp if any(related(p, e) for e in excluded)}
                generic = {
                    p for p in fp if len(anc[p]) <= 4
                }  # within 3 levels of HP:0000118
                rows.append(
                    {
                        "case": case["id"],
                        "lang": lang,
                        "arm": arm,
                        "n_gold": len(gold),
                        "n_pred": len(pred),
                        "n_tp": len(tp),
                        "n_fp": len(fp),
                        "n_fp_negation": len(neg),
                        "n_fp_generic": len(generic),
                        "recall_anc": len(found) / len(gold) if gold else np.nan,
                        "rank_gold": rank(case, gold),
                        "rank_arm": rank(case, pred),
                        "rank_no_fp": rank(
                            case, tp
                        ),  # wrong terms removed: cost of misses only
                        "rank_found_gold": rank(
                            case, found
                        ),  # gold the arm recovered, no noise
                        "rank_gold_plus_fp": rank(
                            case, gold | fp
                        ),  # perfect recall + the arm's noise
                    }
                )
                if arm == "top1" and lang == "de":
                    fp_terms.update(fp)
                    for g in gold:
                        miss_rows.append(
                            {
                                "term": ix.ids[g],
                                "label": ix.labels[g],
                                "depth": len(anc[g]) - 1,
                                "system": "|".join(sorted(systems[g])),
                                "missed": int(g not in found),
                            }
                        )
    write_csv(RESULTS / "errors_cases.csv", rows)
    write_csv(RESULTS / "errors_gold_terms_de.csv", miss_rows)
    total = sum(r["n_pred"] for r in rows if r["arm"] == "top1" and r["lang"] == "de")
    write_csv(
        RESULTS / "errors_top_fp_de.csv",
        [
            {
                "term": ix.ids[t],
                "label": ix.labels[t],
                "letters": c,
                "share_of_top1_terms": c / total,
            }
            for t, c in fp_terms.most_common(40)
        ],
    )

    hit = lambda key, sel: float(np.mean([r[key] <= 10 for r in sel]))  # noqa: E731
    summary = []
    for lang in LANGS:
        for arm in ("top1", "soft"):
            sel = [r for r in rows if r["lang"] == lang and r["arm"] == arm]
            summary.append(
                {
                    "lang": lang,
                    "arm": arm,
                    "n": len(sel),
                    "hit10_gold": hit("rank_gold", sel),
                    "hit10_arm": hit("rank_arm", sel),
                    "hit10_arm_without_fp": hit("rank_no_fp", sel),
                    "hit10_found_gold_only": hit("rank_found_gold", sel),
                    "hit10_gold_plus_arm_fp": hit("rank_gold_plus_fp", sel),
                    "median_pred": float(np.median([r["n_pred"] for r in sel])),
                    "median_fp": float(np.median([r["n_fp"] for r in sel])),
                    "fp_share_negation": sum(r["n_fp_negation"] for r in sel)
                    / max(1, sum(r["n_fp"] for r in sel)),
                    "fp_share_generic": sum(r["n_fp_generic"] for r in sel)
                    / max(1, sum(r["n_fp"] for r in sel)),
                    "mean_recall_anc": float(
                        np.nanmean([r["recall_anc"] for r in sel])
                    ),
                }
            )
    write_csv(RESULTS / "errors_summary.csv", summary)
    for s in summary:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
