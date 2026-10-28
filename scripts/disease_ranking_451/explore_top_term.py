"""Exploratory arm (post hoc, not registered): Phentrieve's existing `--top-term-per-chunk`.

    python -I explore_top_term.py <hpo_dir>

Affirmed aggregated terms of the run with top_term_per_chunk=True (extract.py
--top-term-per-chunk), scored like the top-1 arm. Reported on dev and on non-cited test
next to the registered arms; it cannot change the registered decision.
Writes results/explore_top_term.csv.
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
    load_jsonl,
    metrics,
    profiles,
    rank_case,
    write_csv,
)  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(hpo_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
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
        top = {
            e["case"]: e
            for e in load_jsonl(RESULTS / f"extractions_{lang}_top_term.jsonl")
        }
        for split in ("dev", "test"):
            sel = [c for c in cases if c["split"] == split]
            ranks: dict[str, list[float]] = {
                "gold": [],
                "top1": [],
                "soft": [],
                "top_term": [],
            }
            terms: dict[str, list[int]] = {k: [] for k in ranks}
            for c in sel:
                arms = profiles(c, default[c["id"]], (frozen["k"], frozen["weighting"]))
                arms["top_term"] = {
                    t[0]: 1.0 for t in top[c["id"]]["terms"] if t[1] == "affirmed"
                }
                for arm, w in arms.items():
                    ranks[arm].append(rank_case(ix, c, w)[0])
                    terms[arm].append(len(w))
            r = {k: np.array(v) for k, v in ranks.items()}
            pmids = np.array([c["pmids"][0] for c in sel])
            ids, inv = np.unique(pmids, return_inverse=True)
            members = [np.flatnonzero(inv == i) for i in range(len(ids))]
            rng = np.random.default_rng(SEED)
            reps = {
                k: []
                for k in (
                    "top_term",
                    "top_term-gold",
                    "top_term-top1",
                    "top_term-soft",
                    "ratio",
                )
            }
            for _ in range(B):
                idx = np.concatenate(
                    [members[i] for i in rng.integers(0, len(ids), len(ids))]
                )
                h = {k: float((v[idx] <= 10).mean()) for k, v in r.items()}
                reps["top_term"].append(h["top_term"])
                for b in ("gold", "top1", "soft"):
                    reps[f"top_term-{b}"].append(h["top_term"] - h[b])
                reps["ratio"].append(h["top_term"] / h["gold"])
            ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in reps.items()}
            row = {"lang": lang, "split": split, "n": len(sel)}
            for arm in r:
                row[f"{arm}_hit10"] = metrics(r[arm])["hit10"]
                row[f"{arm}_median_rank"] = metrics(r[arm])["median_rank"]
                row[f"{arm}_median_terms"] = float(np.median(terms[arm]))
            for k, (lo, hi) in ci.items():
                point = (
                    row["top_term_hit10"]
                    if k == "top_term"
                    else (
                        row["top_term_hit10"] / row["gold_hit10"]
                        if k == "ratio"
                        else row["top_term_hit10"] - row[f"{k.split('-')[1]}_hit10"]
                    )
                )
                row[f"{k}_point"], row[f"{k}_lo"], row[f"{k}_hi"] = point, lo, hi
            out.append(row)
            print(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
            )
    write_csv(RESULTS / "explore_top_term.csv", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
