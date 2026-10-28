"""Why scorer Hit@1 drops with the grounded-negation fix (exploratory, issue #451).

    python -I negfix_effect.py <hpo_dir>

On non-cited test cases: the share of gold-excluded terms that are annotated to the true
disease (the term or a descendant), and per language the true #1 lost / gained when the
fixed extraction replaces the old one, with how many losses involve removed wrongly-present
findings that are features of the true disease. Writes results/negfix_effect.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import scorer  # noqa: E402
from analysis import LANGS, RESULTS, load_jsonl, write_csv  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(hpo_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if c["split"] == "test" and not c["cited"]
    ]

    def annotated(d: int, hpo_id: str) -> bool:
        if hpo_id not in ix.index:
            return False
        t = ix.index[hpo_id]
        a, b = ix.dis_ptr[d], ix.dis_ptr[d + 1]
        return any(t in set(ix.ancestors(x).tolist()) for x in ix.ann_term[a:b])

    def present(e: dict) -> set[str]:
        return {t[0] for t in e["terms"] if t[1] == "present" and not t[2]}

    def rank(terms: set[str], d: int) -> float:
        s = scorer.score(ix, dict.fromkeys(terms, 1.0))
        return scorer.avg_rank(s, d) if s is not None else float(len(ix.disease_ids))

    total = sum(len(c["excluded"]) for c in cases)
    on_true = sum(
        annotated(ix.disease_ids.index(c["disease"]), t["id"])
        for c in cases
        for t in c["excluded"]
    )
    rows = [
        {
            "lang": "all",
            "metric": "gold_excluded_annotated_to_true_disease",
            "n": on_true,
            "of": total,
        }
    ]
    print(
        f"gold-excluded terms annotated to the true disease: {on_true}/{total} = {on_true / total:.0%}"
    )
    for lang in LANGS:
        old = {
            e["case"]: e for e in load_jsonl(RESULTS / f"extractions_{lang}_llm.jsonl")
        }
        new = {
            e["case"]: e
            for e in load_jsonl(RESULTS / f"extractions_{lang}_llm_negfix.jsonl")
        }
        lost = gained = explained = 0
        for c in cases:
            d = ix.disease_ids.index(c["disease"])
            po, pn = present(old[c["id"]]), present(new[c["id"]])
            ro, rn = rank(po, d), rank(pn, d)
            if ro == 1 and rn > 1:
                lost += 1
                excluded = {t["id"] for t in c["excluded"]}
                explained += any(annotated(d, t) for t in (po - pn) & excluded)
            elif ro > 1 and rn == 1:
                gained += 1
        rows += [
            {"lang": lang, "metric": "true_top1_lost", "n": lost, "of": len(cases)},
            {
                "lang": lang,
                "metric": "lost_with_removed_true_disease_features",
                "n": explained,
                "of": lost,
            },
            {"lang": lang, "metric": "true_top1_gained", "n": gained, "of": len(cases)},
        ]
        print(
            f"{lang}: #1 lost {lost} (removed true-disease features in {explained}), gained {gained}"
        )
    write_csv(RESULTS / "negfix_effect.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
