"""Conformance of the VarLens v2 scorer port (issue #451) against VarLens's own output."""

from __future__ import annotations

import json
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

EVAL_DIR = Path(__file__).resolve().parents[3] / "scripts" / "disease_ranking_451"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import scorer  # noqa: E402

HPO_DIR = (
    Path(os.environ.get("EVAL451_DATA", "/home/bernt-popp/development/_eval451"))
    / "hpo-v2026-09-01"
)
REFERENCE = json.loads(
    (EVAL_DIR / "conformance" / "varlens_reference.json").read_text()
)

needs_hpo = pytest.mark.skipif(
    not (HPO_DIR / "phenotype.hpoa").exists(),
    reason="HPO v2026-09-01 release files not present",
)


@cache
def _index() -> scorer.PhenotypeIndex:
    return scorer.build_index(HPO_DIR)


def _weights(terms: list[dict]) -> dict[str, float]:
    w: dict[str, float] = {}
    for t in terms:
        if not t["excluded"]:  # conflictsOnly: excluded terms never score
            w[t["hpoId"]] = max(
                w.get(t["hpoId"], 0.0), scorer.KEY_WEIGHT if t["key"] else 1.0
            )
    return w


@needs_hpo
def test_universe_matches_varlens() -> None:
    ix = _index()
    assert len(ix.disease_ids) == REFERENCE["n_diseases"]
    assert len(ix.gene_diseases) == REFERENCE["n_genes"]


@needs_hpo
@pytest.mark.parametrize("ref", REFERENCE["profiles"], ids=lambda r: r["id"])
def test_scores_and_ranks_match_varlens(ref: dict) -> None:
    ix = _index()
    s = scorer.score(ix, _weights(ref["terms"]))
    assert s is not None
    d = ix.disease_ids.index(ref["disease"])
    assert s[d] == pytest.approx(ref["disease_score"], abs=1e-5)
    assert scorer.avg_rank(s, d) == ref["disease_rank"]
    by_id = dict(zip(ix.disease_ids, s, strict=True))
    for disease, expected in ref["top_diseases"]:
        assert by_id[disease] == pytest.approx(expected, abs=1e-5)
    genes = scorer.gene_scores(ix, s)
    names = list(genes)
    values = np.array([genes[g] for g in names])
    for gene, expected in ref["top_genes"]:
        assert genes[gene] == pytest.approx(expected, abs=1e-5)
    if ref["gene_rank"] is not None:
        assert scorer.avg_rank(values, names.index(ref["gene"])) == ref["gene_rank"]


def test_reference_has_enough_profiles() -> None:
    assert len(REFERENCE["profiles"]) >= 20


def test_soft_weights_share_a_chunk_unit_weight() -> None:
    import analysis

    chunks = [
        [
            ["HP:A", 0.9, "affirmed"],
            ["HP:B", 0.8, "affirmed"],
            ["HP:C", 0.95, "negated"],
        ],
        [["HP:B", 0.85, "affirmed"]],
        [],
    ]
    assert analysis.soft_weights(chunks, 1, "raw") == {"HP:A": 1.0, "HP:B": 1.0}
    raw = analysis.soft_weights(chunks, 3, "raw")
    assert raw["HP:A"] == pytest.approx(0.9 / 1.7)
    assert raw["HP:B"] == 1.0  # a term takes its maximum weight over chunks
    assert "HP:C" not in raw  # negated candidates never enter
    sharp = analysis.soft_weights(chunks, 3, 0.05)
    assert sharp["HP:A"] == pytest.approx(1 / (1 + np.exp(-0.1 / 0.05)))
