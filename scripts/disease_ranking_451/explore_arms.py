"""Exploratory arms (post hoc, not registered), next to the registered ones (issue #451).

    python -I explore_arms.py <hpo_dir>

- top_term: `--top-term-per-chunk`, affirmed aggregated terms (extract.py --top-term-per-chunk)
- llm: LLM full-text backend, terms with status `present` and not excluded (extract_llm.py)
- llm_negfix: the same with the grounded-negation fix of PR #453 (EXTRACT_TAG=_negfix)

Per language and split: Hit@10 and median rank of every arm, the exploratory arms' Hit@10
differences to gold / default / soft and ratio to gold (PMID-clustered bootstrap), and term
precision / recall / unrelated share / negation failures. Writes results/explore_arms.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from analysis import (
    LANGS,
    RESULTS,
    SEED,
    B,
    canonical,
    load_jsonl,
    metrics,
    profiles,
    rank_case,
    write_csv,
)  # noqa: E402

HERE = Path(__file__).resolve().parent
EXTRA = {
    "top_term": ("_top_term", lambda t: t[1] == "affirmed"),
    "llm": ("_llm", lambda t: t[1] == "present" and not t[2]),
    # LLM backend with the grounded negation fix (PR #453)
    "llm_negfix": ("_llm_negfix", lambda t: t[1] == "present" and not t[2]),
}


def main(hpo_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    anc = [set(ix.ancestors(t).tolist()) for t in range(len(ix.ids))]
    related = lambda a, b: a in anc[b] or b in anc[a]  # noqa: E731
    frozen = json.loads((RESULTS / "frozen.json").read_text())
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if not (c["split"] == "test" and c["cited"])
    ]
    out = []
    for lang in LANGS:
        default = {
            e["case"]: e for e in load_jsonl(RESULTS / f"extractions_{lang}.jsonl")
        }
        extra = {
            arm: {
                e["case"]: e
                for e in load_jsonl(RESULTS / f"extractions_{lang}{suffix}.jsonl")
            }
            for arm, (suffix, _) in EXTRA.items()
            if (RESULTS / f"extractions_{lang}{suffix}.jsonl").exists()
        }
        for split in ("dev", "test"):
            sel = [
                c
                for c in cases
                if c["split"] == split and all(c["id"] in e for e in extra.values())
            ]
            ranks: dict[str, list[float]] = {}
            stats: dict[str, list[dict]] = {}
            for c in sel:
                arms = profiles(c, default[c["id"]], (frozen["k"], frozen["weighting"]))
                for arm, rows in extra.items():
                    keep = EXTRA[arm][1]
                    arms[arm] = {t[0]: 1.0 for t in rows[c["id"]]["terms"] if keep(t)}
                gold = canonical(ix, [t["id"] for t in c["observed"]])
                excluded = canonical(ix, [t["id"] for t in c["excluded"]])
                for arm, w in arms.items():
                    ranks.setdefault(arm, []).append(rank_case(ix, c, w)[0])
                    pred = canonical(ix, [t for t, v in w.items() if v > 0])
                    tp = {p for p in pred if any(related(p, g) for g in gold)}
                    fp = pred - tp
                    stats.setdefault(arm, []).append(
                        {
                            "n": len(pred),
                            "p_exact": len(pred & gold) / len(pred) if pred else np.nan,
                            "r_exact": len(pred & gold) / len(gold) if gold else np.nan,
                            "p_anc": len(tp) / len(pred) if pred else np.nan,
                            "r_anc": sum(any(related(g, p) for p in pred) for g in gold)
                            / len(gold)
                            if gold
                            else np.nan,
                            "fp_negation": sum(
                                any(related(p, e) for e in excluded) for p in fp
                            ),
                            "fp": len(fp),
                        }
                    )
            r = {k: np.array(v) for k, v in ranks.items()}
            ids, inv = np.unique([c["pmids"][0] for c in sel], return_inverse=True)
            members = [np.flatnonzero(inv == i) for i in range(len(ids))]
            rng = np.random.default_rng(SEED)
            reps: dict[str, list[float]] = {}
            for _ in range(B):
                idx = np.concatenate(
                    [members[i] for i in rng.integers(0, len(ids), len(ids))]
                )
                h = {k: float((v[idx] <= 10).mean()) for k, v in r.items()}
                for arm in r:
                    reps.setdefault(f"{arm}_hit10", []).append(h[arm])
                for arm in extra:
                    for base in ("gold", "top1", "soft"):
                        reps.setdefault(f"{arm}-{base}", []).append(h[arm] - h[base])
                    reps.setdefault(f"{arm}/gold", []).append(
                        h[arm] / h["gold"] if h["gold"] else np.nan
                    )
            hit = {k: metrics(v)["hit10"] for k, v in r.items()}
            for arm in r:
                s = stats[arm]
                row = {
                    "lang": lang,
                    "split": split,
                    "arm": arm,
                    "n": len(r[arm]),
                    "hit10": hit[arm],
                    "hit10_lo": float(np.percentile(reps[f"{arm}_hit10"], 2.5)),
                    "hit10_hi": float(np.percentile(reps[f"{arm}_hit10"], 97.5)),
                    "median_rank": metrics(r[arm])["median_rank"],
                    "hit1": metrics(r[arm])["hit1"],
                    "hit3": metrics(r[arm])["hit3"],
                    "mrr": metrics(r[arm])["mrr"],
                    "ratio_to_gold": hit[arm] / hit["gold"],
                    "median_terms": float(np.median([x["n"] for x in s])),
                    **{
                        k: float(np.nanmean([x[k] for x in s]))
                        for k in ("p_exact", "r_exact", "p_anc", "r_anc")
                    },
                    "fp_share_negation": sum(x["fp_negation"] for x in s)
                    / max(1, sum(x["fp"] for x in s)),
                }
                if arm in extra:
                    for base in ("gold", "top1", "soft"):
                        v = reps[f"{arm}-{base}"]
                        row[f"diff_{base}"] = hit[arm] - hit[base]
                        row[f"diff_{base}_lo"], row[f"diff_{base}_hi"] = np.percentile(
                            v, [2.5, 97.5]
                        )
                    row["ratio_lo"], row["ratio_hi"] = np.percentile(
                        reps[f"{arm}/gold"], [2.5, 97.5]
                    )
                out.append(row)
    keys = list(dict.fromkeys(k for row in out for k in row))
    write_csv(
        RESULTS / "explore_arms.csv", [{k: row.get(k, "") for k in keys} for row in out]
    )
    for row in out:
        print(
            {
                k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
                for k, v in row.items()
            }
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
