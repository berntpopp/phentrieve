"""Score the arms, tune soft profiles on dev, evaluate test once (issue #451; PROTOCOL.md).

    python -I analysis.py tune <hpo_dir> <phentrieve_data_dir> <translations_dir>
    python -I analysis.py test <hpo_dir> <phentrieve_data_dir> <translations_dir>

`tune` scores dev only and writes results/frozen.json. `test` refuses to run without it,
scores the test split once and writes the case-level ranks, metrics, paired differences and
diagnostics to results/.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LANGS = ("de", "en")
GRID = [(k, w) for k in (1, 3, 5) for w in ("raw", 0.05, 0.1, 0.2)]
B, SEED = 2000, 451
HITS = (1, 3, 10)


def load_jsonl(path: Path) -> list[dict]:
    """JSON lines; falls back to `<path>.gz` (large result files are committed gzipped)."""
    gz = path.with_name(path.name + ".gz")
    if not path.exists() and gz.exists():
        import gzip

        text = gzip.decompress(gz.read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def soft_weights(chunks: list, k: int, weighting: str | float) -> dict[str, float]:
    """Top-k affirmed candidates per chunk; the chunk's unit weight is shared by raw score or softmax(T)."""
    w: dict[str, float] = {}
    for matches in chunks:
        top = sorted((m for m in matches if m[2] == "affirmed"), key=lambda m: -m[1])[
            :k
        ]
        if not top:
            continue
        s = np.array([m[1] for m in top])
        share = s / s.sum() if weighting == "raw" else np.exp((s - s.max()) / weighting)
        share = share / share.sum()
        for (hpo_id, _, _), v in zip(top, share, strict=True):
            w[hpo_id] = max(w.get(hpo_id, 0.0), float(v))
    return w


def profiles(
    case: dict, ext: dict, config: tuple[int, str | float] | None
) -> dict[str, dict[str, float]]:
    arms = {
        "gold": {t["id"]: 1.0 for t in case["observed"]},
        "top1": {t[0]: 1.0 for t in ext["terms"] if t[1] == "affirmed"},
    }
    if config is None:
        for k, w in GRID:
            arms[f"soft_k{k}_{w}"] = soft_weights(ext["chunks"], k, w)
    else:
        arms["soft"] = soft_weights(ext["chunks"], *config)
    return arms


def rank_case(
    ix: scorer.PhenotypeIndex, case: dict, weights: dict[str, float]
) -> tuple[float, float, float]:
    """Disease rank, gene rank (NaN when the gene has no scored disease) and scoring seconds."""
    start = time.perf_counter()
    s = scorer.score(ix, weights)
    n = len(ix.disease_ids)
    if s is None:  # no usable term: every disease ties
        return (n + 1) / 2, (len(ix.gene_diseases) + 1) / 2, time.perf_counter() - start
    d_rank = scorer.avg_rank(s, ix.disease_ids.index(case["disease"]))
    genes = scorer.gene_scores(ix, s)
    g_rank = math.nan
    if case["gene"] in genes:
        names = list(genes)
        g_rank = scorer.avg_rank(
            np.array([genes[g] for g in names]), names.index(case["gene"])
        )
    return d_rank, g_rank, time.perf_counter() - start


def score_split(ix, cases, extractions, split: str, config) -> list[dict]:
    rows = []
    for case in (c for c in cases if c["split"] == split):
        for lang in LANGS:
            ext = extractions[lang][case["id"]]
            for arm, weights in profiles(case, ext, config).items():
                d, g, secs = rank_case(ix, case, weights)
                rows.append(
                    {
                        "case": case["id"],
                        "lang": lang,
                        "split": split,
                        "cited": int(case["cited"]),
                        "pmid": case["pmids"][0],
                        "gene": case["gene"],
                        "disease": case["disease"],
                        "arm": arm,
                        "n_terms": len(weights),
                        "disease_rank": d,
                        "gene_rank": g,
                        "extract_seconds": ext["seconds"],
                        "score_seconds": round(secs, 4),
                    }
                )
    return rows


def tune(ix, cases, extractions) -> None:
    rows = score_split(ix, cases, extractions, "dev", None)
    write_csv(RESULTS / "ranks_dev.csv", rows)
    table = []
    for k, w in GRID:
        r = np.array([x["disease_rank"] for x in rows if x["arm"] == f"soft_k{k}_{w}"])
        table.append(
            {
                "k": k,
                "weighting": w,
                "hit10": float((r <= 10).mean()),
                "mrr": float((1 / r).mean()),
            }
        )
    order = {"raw": 0, 0.05: 1, 0.1: 2, 0.2: 3}
    best = max(
        table, key=lambda t: (t["hit10"], t["mrr"], -t["k"], -order[t["weighting"]])
    )
    write_csv(RESULTS / "tuning_dev.csv", table)
    frozen = {
        "k": best["k"],
        "weighting": best["weighting"],
        "dev_hit10": best["hit10"],
        "dev_mrr": best["mrr"],
    }
    (RESULTS / "frozen.json").write_text(json.dumps(frozen, indent=1) + "\n")
    for t in table:
        print(t)
    print("frozen:", frozen)


# --- statistics -----------------------------------------------------------------------


def metrics(r: np.ndarray) -> dict[str, float]:
    out = {f"hit{k}": float((r <= k).mean()) for k in HITS}
    out.update(mrr=float((1 / r).mean()), median_rank=float(np.median(r)))
    return out


def bootstrap(ranks: dict[str, np.ndarray], clusters: np.ndarray, best: str) -> dict:
    """Cluster bootstrap over PMIDs: percentile CIs of each arm's metrics, paired differences, ratio best/gold."""
    rng = np.random.default_rng(SEED)
    ids, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(ids))]
    reps = defaultdict(list)
    pairs = [("top1", "gold"), ("soft", "gold"), ("soft", "top1")]
    for _ in range(B):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(ids), len(ids))])
        m = {arm: metrics(r[idx]) for arm, r in ranks.items()}
        for arm, vals in m.items():
            for key, v in vals.items():
                reps[(arm, key)].append(v)
        for a, b in pairs:
            for key in ("hit1", "hit3", "hit10", "mrr"):
                reps[(f"{a}-{b}", key)].append(m[a][key] - m[b][key])
        reps[("ratio", "hit10")].append(
            m[best]["hit10"] / m["gold"]["hit10"] if m["gold"]["hit10"] else math.nan
        )
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in reps.items()
    }


def mcnemar(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ha, hb = a <= k, b <= k
    x, n = int((ha & ~hb).sum()), int((ha ^ hb).sum())
    return float(binomtest(x, n, 0.5).pvalue) if n else 1.0


# --- diagnostics ----------------------------------------------------------------------


def label_sets(
    hpo_data_db: Path, translations: Path, ids: set[str]
) -> dict[str, dict[str, set[str]]]:
    """English (Phentrieve HPO DB: label + synonyms) and German (hpo-translations) strings per term."""
    en: dict[str, set[str]] = defaultdict(set)
    con = sqlite3.connect(hpo_data_db)
    for tid, label, syn in con.execute("SELECT id, label, synonyms FROM hpo_terms"):
        if tid in ids:
            en[tid] = {label.lower(), *(s.lower() for s in json.loads(syn or "[]"))}
    con.close()
    de: dict[str, set[str]] = defaultdict(set)
    with (translations / "hp-de.babelon.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["predicate_id"] == "rdfs:label" and row["translation_value"]:
                de[row["subject_id"]].add(row["translation_value"].lower())
    with (translations / "hp-de.synonyms.tsv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["subject_id"].startswith("HP:"):
                de[row["subject_id"]] |= {
                    s.lower() for s in row["translation_value"].split("|") if s
                }
    return {"en": en, "de": {t: de.get(t, set()) | en.get(t, set()) for t in ids}}


def verbatim_rate(text: str, gold: list[str], strings: dict[str, set[str]]) -> float:
    low = text.lower()
    hit = [
        any(re.search(rf"(?<!\w){re.escape(s)}(?!\w)", low) for s in strings.get(t, ()))
        for t in gold
    ]
    return sum(hit) / len(gold) if gold else math.nan


def canonical(ix: scorer.PhenotypeIndex, ids) -> set[int]:
    root = ix.index[scorer.ROOT]
    return {ix.index[t] for t in ids if t in ix.index and ix.index[t] != root}


def precision_recall(ix, pred: set[int], gold: set[int]) -> dict[str, float]:
    anc = {t: set(ix.ancestors(t).tolist()) for t in pred | gold}
    related = lambda a, b: a in anc[b] or b in anc[a]  # noqa: E731
    nan = math.nan
    return {
        "p_exact": len(pred & gold) / len(pred) if pred else nan,
        "r_exact": len(pred & gold) / len(gold) if gold else nan,
        "p_anc": sum(any(related(p, g) for g in gold) for p in pred) / len(pred)
        if pred
        else nan,
        "r_anc": sum(any(related(g, p) for p in pred) for g in gold) / len(gold)
        if gold
        else nan,
    }


def evaluate(ix, cases, extractions, hpo_data_db: Path, translations: Path) -> None:
    frozen_path = RESULTS / "frozen.json"
    if not frozen_path.exists():
        sys.exit(
            "results/frozen.json missing: run `tune` first (PROTOCOL.md: no tuning on test)"
        )
    frozen = json.loads(frozen_path.read_text())
    config = (frozen["k"], frozen["weighting"])
    rows = score_split(ix, cases, extractions, "test", config)
    by_case = {c["id"]: c for c in cases}

    # Leakage (verbatim gold labels) per letter, and term precision/recall per arm.
    gold_ids = {t["id"] for c in cases for t in c["observed"]}
    strings = label_sets(hpo_data_db, translations, gold_ids)
    leak, pr_rows = {}, []
    for c in cases:
        gold = canonical(ix, [t["id"] for t in c["observed"]])
        gold_ids_case = [ix.ids[t] for t in gold]
        for lang in LANGS:
            letter = json.loads(
                (
                    HERE
                    / "letters"
                    / lang
                    / (re.sub(r"[^\w.-]", "_", c["id"]) + ".json")
                ).read_text()
            )
            leak[(c["id"], lang)] = verbatim_rate(
                letter["text"], gold_ids_case, strings[lang]
            )
            if c["split"] != "test":
                continue
            for arm, w in profiles(c, extractions[lang][c["id"]], config).items():
                if arm != "gold":
                    pr = precision_recall(
                        ix, canonical(ix, [t for t, v in w.items() if v > 0]), gold
                    )
                    pr_rows.append(
                        {
                            "case": c["id"],
                            "lang": lang,
                            "cited": int(c["cited"]),
                            "arm": arm,
                            **pr,
                        }
                    )
    for r in rows:
        r["verbatim_rate"] = leak[(r["case"], r["lang"])]
    write_csv(RESULTS / "ranks_test.csv", rows)
    write_csv(RESULTS / "term_pr_test.csv", pr_rows)
    write_csv(
        RESULTS / "leakage.csv",
        [
            {
                "case": k[0],
                "lang": k[1],
                "split": by_case[k[0]]["split"],
                "cited": int(by_case[k[0]]["cited"]),
                "verbatim_rate": v,
            }
            for k, v in leak.items()
        ],
    )

    metric_rows, diff_rows = [], []
    for lang in LANGS:
        for subset, keep in (("non_cited", lambda r: not r["cited"]),):
            sel = [r for r in rows if r["lang"] == lang and keep(r)]
            arms = {
                arm: [r for r in sel if r["arm"] == arm]
                for arm in ("gold", "top1", "soft")
            }
            order = [r["case"] for r in arms["gold"]]
            assert all([r["case"] for r in v] == order for v in arms.values())  # paired
            ranks = {
                a: np.array([r["disease_rank"] for r in v]) for a, v in arms.items()
            }
            best = (
                "soft"
                if metrics(ranks["soft"])["hit10"] > metrics(ranks["top1"])["hit10"]
                else "top1"
            )
            ci = bootstrap(ranks, np.array([r["pmid"] for r in arms["gold"]]), best)
            for arm, r in ranks.items():
                g = np.array([x["gene_rank"] for x in arms[arm]])
                g = g[~np.isnan(g)]
                m = metrics(r)
                metric_rows.append(
                    {
                        "lang": lang,
                        "subset": subset,
                        "arm": arm,
                        "n": len(r),
                        **{f"{key}": v for key, v in m.items()},
                        **{f"{key}_lo": ci[(arm, key)][0] for key in m},
                        **{f"{key}_hi": ci[(arm, key)][1] for key in m},
                        "gene_hit10": float((g <= 10).mean()),
                        "median_terms": float(
                            np.median([x["n_terms"] for x in arms[arm]])
                        ),
                        "median_extract_seconds": float(
                            np.median([x["extract_seconds"] for x in arms[arm]])
                        ),
                        "median_score_seconds": float(
                            np.median([x["score_seconds"] for x in arms[arm]])
                        ),
                    }
                )
            for a, b in (("top1", "gold"), ("soft", "gold"), ("soft", "top1")):
                for key in ("hit1", "hit3", "hit10", "mrr"):
                    da, db = metrics(ranks[a])[key], metrics(ranks[b])[key]
                    k = int(key[3:]) if key.startswith("hit") else None
                    diff_rows.append(
                        {
                            "lang": lang,
                            "subset": subset,
                            "comparison": f"{a}-{b}",
                            "metric": key,
                            "diff": da - db,
                            "lo": ci[(f"{a}-{b}", key)][0],
                            "hi": ci[(f"{a}-{b}", key)][1],
                            "mcnemar_p": mcnemar(ranks[a], ranks[b], k)
                            if k
                            else np.nan,
                        }
                    )
            gold10 = metrics(ranks["gold"])["hit10"]
            diff_rows.append(
                {
                    "lang": lang,
                    "subset": subset,
                    "comparison": f"{best}/gold",
                    "metric": "hit10_ratio",
                    "diff": metrics(ranks[best])["hit10"] / gold10
                    if gold10
                    else np.nan,
                    "lo": ci[("ratio", "hit10")][0],
                    "hi": ci[("ratio", "hit10")][1],
                    "mcnemar_p": np.nan,
                }
            )
    write_csv(RESULTS / "metrics_test.csv", metric_rows)
    write_csv(RESULTS / "differences_test.csv", diff_rows)

    # Hit@10 by verbatim-rate tertile: per language, test cases sorted by rate, split into thirds.
    tert = []
    for lang in LANGS:
        ids = sorted(
            {r["case"] for r in rows if r["lang"] == lang},
            key=lambda c: (leak[(c, lang)], c),
        )
        for t, part in enumerate(np.array_split(np.array(ids), 3)):
            part_set = set(part.tolist())
            rates = [leak[(c, lang)] for c in part_set]
            for arm in ("gold", "top1", "soft"):
                rr = np.array(
                    [
                        r["disease_rank"]
                        for r in rows
                        if r["lang"] == lang
                        and r["arm"] == arm
                        and r["case"] in part_set
                    ]
                )
                tert.append(
                    {
                        "lang": lang,
                        "arm": arm,
                        "tertile": t + 1,
                        "rate_min": min(rates),
                        "rate_max": max(rates),
                        "n": len(rr),
                        "hit10": float((rr <= 10).mean()),
                    }
                )
    write_csv(RESULTS / "verbatim_tertiles_test.csv", tert)
    print(json.dumps({"frozen": frozen}, indent=1))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(stage: str, hpo_dir: Path, data_dir: Path, translations: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    cases = load_jsonl(HERE / "cases.jsonl")
    # PoC scope (PROTOCOL.md deviation 2026-10-28): cited test cases are not used.
    cases = [c for c in cases if not (c["split"] == "test" and c["cited"])]
    extractions = {
        lang: {e["case"]: e for e in load_jsonl(RESULTS / f"extractions_{lang}.jsonl")}
        for lang in LANGS
    }
    for lang in LANGS:
        missing = [c["id"] for c in cases if c["id"] not in extractions[lang]]
        assert not missing, f"{lang}: {len(missing)} letters not extracted"
    if stage == "tune":
        tune(ix, cases, extractions)
    elif stage == "test":
        evaluate(ix, cases, extractions, data_dir / "hpo_data.db", translations)
    else:
        sys.exit(f"unknown stage {stage}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
