"""Port of VarLens phenotype scoring v2 at its shipped setting (issue #451).

Source of truth: VarLens `src/shared/phenotype/{hpo-sources,resnik,phenotype-index,
phenotype-score}.ts`, `src/main/services/phenotype/hpo-similarity-db.ts` and
`src/shared/config/hpo-scoring.config.ts`. Shipped setting only: tau = 1, background
correction off, exclusions `conflictsOnly` (excluded terms never change the score).

    sim(a, b) = IC(MICA(a, b)), IC from phenotype.hpoa (diseases propagate to ancestors)
    r(p, d)   = max_t sim(p, t) * (1 - a_f + a_f * f(d, t)) / IC(p)
    Fwd(d)    = sum_p w_p IC(p) r(p, d) / sum_p w_p IC(p)
    Rev(d)    = sum_t f(d, t) max_p sim(p, t) / IC(t) / (f0 + sum_t f(d, t))
    S(d)      = gamma Fwd(d) + (1 - gamma) Rev(d);  S(gene) = max over its diseases

`w_p` is 1 for a plain term and the key weight (2) for a key term. The soft profile
of the evaluation only passes other weights (see `score`); nothing else changes.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = "HP:0000118"
ALPHA_F, F0, GAMMA, KEY_WEIGHT = 0.25, 1.0, 0.55, 2.0
FRACTION_PRIOR = (0.8, 1.2)
FREQUENCY_TERMS = {
    "HP:0040280": 1.0,
    "HP:0040281": 0.895,
    "HP:0040282": 0.545,
    "HP:0040283": 0.17,
    "HP:0040284": 0.025,
}
MISSING_F = 0.35
EXCLUDED_FREQUENCY = "HP:0040285"
_HP = re.compile(r"^(?:http://purl\.obolibrary\.org/obo/HP_|HP:)(\d{7})$")
_ALT_ID = "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId"
_REPLACED_BY = "http://purl.obolibrary.org/obo/IAO_0100001"


def _hp(iri: str) -> str | None:
    m = _HP.match(iri)
    return f"HP:{m.group(1)}" if m else None


def parse_hp_json(
    path: Path,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """Labels of current terms, is_a parents, alias -> current term (parseHpJson)."""
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    labels: dict[str, str] = {}
    raw_alias: dict[str, str] = {}
    for node in graph["nodes"]:
        tid = _hp(node["id"])
        if tid is None or node.get("type") != "CLASS":
            continue
        meta = node.get("meta") or {}
        deprecated = meta.get("deprecated") is True
        for pv in meta.get("basicPropertyValues") or []:
            other = _hp(pv.get("val", ""))
            if other is None:
                continue
            if not deprecated and pv["pred"] == _ALT_ID:
                raw_alias[other] = tid
            if deprecated and pv["pred"] == _REPLACED_BY:
                raw_alias[tid] = other
        if not deprecated:
            labels[tid] = node.get("lbl") or tid
    aliases: dict[str, str] = {}
    for alias in raw_alias:
        seen: set[str] = set()
        term: str | None = alias
        while term is not None and term not in labels and term not in seen:
            seen.add(term)
            term = raw_alias.get(term)
        if term is not None and term in labels and alias not in labels:
            aliases[alias] = term
    parents: dict[str, list[str]] = defaultdict(list)
    for e in graph.get("edges") or []:
        if e["pred"] != "is_a":
            continue
        sub, obj = _hp(e["sub"]), _hp(e["obj"])
        if sub in labels and obj in labels:
            parents[sub].append(obj)  # type: ignore[index]
    return labels, dict(parents), aliases


def phenotype_closure(
    labels: dict[str, str], parents: dict[str, list[str]], aliases: dict[str, str]
) -> dict[str, set[str]]:
    """term (or alias) -> ancestors-or-self, restricted to the HP:0000118 subtree."""
    closure: dict[str, set[str]] = {}

    def visit(term: str) -> set[str]:
        if term in closure:
            return closure[term]
        own = {term}
        for p in parents.get(term, []):
            own |= visit(p)
        closure[term] = own
        return own

    import sys

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    for t in labels:
        visit(t)
    for alias, term in aliases.items():
        closure[alias] = closure.get(term, {term})
    inside = {t for t, anc in closure.items() if ROOT in anc}
    return {t: anc & inside for t, anc in closure.items() if t in inside}


def parse_hpoa(
    path: Path, aliases: dict[str, str]
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """disease -> {term: f} for positive aspect-P pairs, and disease labels (parseHpoaDiseases)."""
    labels: dict[str, str] = {}
    pairs: dict[str, dict[str, list]] = defaultdict(
        dict
    )  # [m, n, other_f | None, absent]
    seen: set[tuple[str, ...]] = set()
    cols: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if cols is None:
            cols = fields
            continue
        row = dict(zip(cols, fields + [""] * (len(cols) - len(fields)), strict=False))
        if row["aspect"] != "P":
            continue
        term = aliases.get(row["hpo_id"], row["hpo_id"])
        key = (
            row["database_id"],
            row["qualifier"],
            term,
            row["reference"],
            row["evidence"],
            row["onset"],
            row["frequency"],
            row["sex"],
            row["modifier"],
        )
        if key in seen:
            continue
        seen.add(key)
        labels.setdefault(row["database_id"], row["disease_name"])
        ev = pairs[row["database_id"]].setdefault(term, [0, 0, None, False])
        freq = row["frequency"]
        frac = re.fullmatch(r"(\d+)/(\d+)", freq)
        if row["qualifier"] == "NOT":
            ev[3] = True
        elif frac and int(frac.group(2)) > 0:
            ev[0] += min(int(frac.group(1)), int(frac.group(2)))
            ev[1] += int(frac.group(2))
        else:
            other: float | str
            if freq in ("", "-"):
                other = MISSING_F
            elif freq in FREQUENCY_TERMS:
                other = FREQUENCY_TERMS[freq]
            elif freq == EXCLUDED_FREQUENCY:
                other = "absent"
            elif pct := re.fullmatch(r"(\d+(?:\.\d+)?)%", freq):
                f = min(1.0, float(pct.group(1)) / 100)
                other = f if f > 0 else "absent"
            else:
                other = MISSING_F
            if other == "absent":
                ev[3] = True
            elif ev[2] is None or other > ev[2]:  # type: ignore[operator]
                ev[2] = other
    a, b = FRACTION_PRIOR
    terms: dict[str, dict[str, float]] = {}
    for disease, by_term in pairs.items():
        for term, (m, n, other, _absent) in by_term.items():
            f = (m + a) / (n + a + b) if m > 0 else other
            if f is not None:
                terms.setdefault(disease, {})[term] = f
    return terms, labels


def _f32(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float64)


@dataclass
class PhenotypeIndex:
    ids: list[str]
    index: dict[str, int]  # term id or alias -> term index
    labels: list[str]
    ic: np.ndarray  # float32-rounded annotation IC (lanes)
    self_ic: np.ndarray  # max IC over ancestors-or-self (normaliser)
    anc_ptr: np.ndarray  # CSR: ancestors-or-self of term t
    anc_idx: np.ndarray
    disease_ids: list[str]
    disease_labels: list[str]
    dis_ptr: np.ndarray  # CSR over annotations
    ann_dis: np.ndarray
    ann_term: np.ndarray
    ann_f: np.ndarray
    ann_factor: np.ndarray
    freq_sum: np.ndarray
    gene_diseases: dict[str, list[int]]

    def ancestors(self, t: int) -> np.ndarray:
        return self.anc_idx[self.anc_ptr[t] : self.anc_ptr[t + 1]]


def build_index(hpo_dir: Path) -> PhenotypeIndex:
    """Same tables as VarLens's buildHpoSimilarityDb + buildPhenotypeIndex (gene-linked diseases)."""
    labels, parents, aliases = parse_hp_json(hpo_dir / "hp.json")
    closure = phenotype_closure(labels, parents, aliases)
    hpoa, disease_labels = parse_hpoa(hpo_dir / "phenotype.hpoa", aliases)
    # IC(t) = -ln(n(t)/N) over diseases with >=1 positive phenotype-subtree term.
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for terms in hpoa.values():
        reached: set[str] = set()
        for t in terms:
            reached |= closure.get(t, set())
        if reached:
            total += 1
            for anc in reached:
                counts[anc] += 1
    ic_of = {t: -math.log(n / total) for t, n in counts.items()}
    # genes_to_phenotype: gene -> diseases with a positive annotation.
    gene_dis: dict[str, set[str]] = defaultdict(set)
    cols: list[str] | None = None
    for line in (
        (hpo_dir / "genes_to_phenotype.txt").read_text(encoding="utf-8").splitlines()
    ):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if cols is None:
            cols = fields
            continue
        row = dict(zip(cols, fields, strict=False))
        if row["gene_symbol"] and row["disease_id"] in hpoa:
            gene_dis[row["gene_symbol"]].add(row["disease_id"])
    linked: dict[str, dict[str, float]] = {}
    for diseases in gene_dis.values():
        for d in diseases:
            if d in linked:
                continue
            merged: dict[str, float] = {}
            for raw, f in hpoa[d].items():
                term = aliases.get(raw, raw)
                if term in closure and f > merged.get(term, -1.0):
                    merged[term] = f
            if merged:
                linked[d] = merged
    # Phenotype subtree terms; topological order so parents come first.
    ids = sorted(t for t in labels if t in closure)
    index = {t: i for i, t in enumerate(ids)}
    for alias, term in aliases.items():
        if term in index and alias not in index:
            index[alias] = index[term]
    ic = _f32([ic_of.get(t, 0.0) for t in ids])
    order: list[int] = []
    done: set[str] = set()

    def emit(t: str) -> None:
        if t in done:
            return
        for p in parents.get(t, []):
            if p in index:
                emit(p)
        done.add(t)
        order.append(index[t])

    for t in ids:
        emit(t)
    self_ic = np.zeros(len(ids))
    for t in order:
        ps = [index[p] for p in parents.get(ids[t], []) if p in index]
        self_ic[t] = max([ic[t], *(self_ic[p] for p in ps)])
    anc = [sorted(index[a] for a in closure[t]) for t in ids]
    anc_ptr = np.cumsum([0, *map(len, anc)])
    anc_idx = np.fromiter((a for row in anc for a in row), dtype=np.int64)
    disease_ids = sorted(linked)
    d_index = {d: i for i, d in enumerate(disease_ids)}
    ann_term: list[int] = []
    ann_f: list[float] = []
    ptr = [0]
    for d in disease_ids:
        for term, f in linked[d].items():
            t = index[term]
            if self_ic[t] > 0:
                ann_term.append(t)
                ann_f.append(f)
        ptr.append(len(ann_term))
    f32 = _f32(ann_f)
    dis_ptr = np.asarray(ptr)
    return PhenotypeIndex(
        ids=ids,
        index=index,
        labels=[labels[t] for t in ids],
        ic=ic,
        self_ic=self_ic,
        anc_ptr=anc_ptr,
        anc_idx=anc_idx,
        disease_ids=disease_ids,
        disease_labels=[disease_labels.get(d, d) for d in disease_ids],
        dis_ptr=dis_ptr,
        ann_dis=np.repeat(np.arange(len(disease_ids)), np.diff(dis_ptr)),
        ann_term=np.asarray(ann_term, dtype=np.int64),
        ann_f=f32,
        ann_factor=_f32(1 - ALPHA_F + ALPHA_F * f32),
        freq_sum=np.bincount(
            np.repeat(np.arange(len(disease_ids)), np.diff(dis_ptr)),
            f32,
            len(disease_ids),
        ),
        gene_diseases={
            g: sorted(d_index[d] for d in ds if d in d_index)
            for g, ds in gene_dis.items()
            if any(d in d_index for d in ds)
        },
    )


def prepare(
    ix: PhenotypeIndex, weights: dict[str, float]
) -> tuple[list[int], list[float]]:
    """Canonicalise present terms, drop ancestors of other present terms (descendant keeps the max weight)."""
    w: dict[int, float] = {}
    for hpo_id, weight in weights.items():
        t = ix.index.get(hpo_id)
        if t is not None and ix.self_ic[t] > 0:
            w[t] = max(w.get(t, 0.0), weight)
    terms = list(w)
    weight = [w[t] for t in terms]
    dropped = [False] * len(terms)
    for i, q in enumerate(terms):
        anc = set(ix.ancestors(q).tolist())
        for j, p in enumerate(terms):
            if j != i and p in anc:
                dropped[j] = True
                weight[i] = max(weight[i], weight[j])
    keep = [i for i in range(len(terms)) if not dropped[i]]
    return [terms[i] for i in keep], [weight[i] for i in keep]


def score(ix: PhenotypeIndex, weights: dict[str, float]) -> np.ndarray | None:
    """S(d) for every indexed disease; `weights` maps present HPO ids to w_p. None without a usable term."""
    present, w = prepare(ix, weights)
    if not present:
        return None
    k = len(present)
    in_anc = np.zeros((len(ix.ids), k), dtype=bool)
    for lane, p in enumerate(present):
        in_anc[ix.ancestors(p), lane] = True
    # sim[t, l] = max IC over Anc(t) & Anc(p_l)
    vals = ix.ic[ix.anc_idx][:, None] * in_anc[ix.anc_idx]
    sim = np.maximum.reduceat(vals, ix.anc_ptr[:-1], axis=0)
    p_idx = np.asarray(present)
    lane_w = np.asarray(w) * ix.self_ic[p_idx]
    lane_w = lane_w / lane_w.sum() / ix.self_ic[p_idx]
    with np.errstate(divide="ignore"):
        inv_ic = np.where(ix.self_ic > 0, 1 / ix.self_ic, 0.0)
    reverse = sim.max(axis=1) * inv_ic
    nonempty = np.diff(ix.dis_ptr) > 0
    starts = ix.dis_ptr[:-1][nonempty]
    lane_max = np.zeros((len(ix.disease_ids), k))
    lane_max[nonempty] = np.maximum.reduceat(
        sim[ix.ann_term] * ix.ann_factor[:, None], starts, axis=0
    )
    rev = np.bincount(ix.ann_dis, ix.ann_f * reverse[ix.ann_term], len(ix.disease_ids))
    fwd = lane_max @ lane_w
    return GAMMA * fwd + (1 - GAMMA) * rev / (F0 + ix.freq_sum)


def gene_scores(ix: PhenotypeIndex, disease_scores: np.ndarray) -> dict[str, float]:
    return {g: float(disease_scores[ds].max()) for g, ds in ix.gene_diseases.items()}


def avg_rank(values: np.ndarray, i: int) -> float:
    """1-based rank of values[i], ties averaged; scores rounded to 1e-12 so summation-order noise never splits a tie."""
    v = np.round(values, 12)
    return float((v > v[i]).sum() + ((v == v[i]).sum() + 1) / 2)
