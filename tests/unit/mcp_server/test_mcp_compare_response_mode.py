"""D5 -- compare_hpo_terms honours response_mode: at standard/full it returns the
MICA, per-term depth/IC, labels and subsumer path so the score is explainable;
minimal/compact stay the lean 4-field payload.
"""

import pytest

from api.mcp.service_adapters import compare_hpo_terms_service
from phentrieve.evaluation.metrics import load_hpo_graph_data

# Related pair from the assessment (HP:0000787 vs HP:0004724).
T1, T2 = "HP:0000787", "HP:0004724"


def _ontology_available(*term_ids: str) -> bool:
    """The prepared HPO graph is absent on the fast CI matrix; skip there."""
    try:
        _ancestors, depths = load_hpo_graph_data()
    except Exception:
        return False
    return all(t in depths for t in term_ids)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _ontology_available(T1, T2),
        reason="HPO ontology graph unavailable; run `phentrieve data prepare`",
    ),
]


def _compare(**kw):
    return compare_hpo_terms_service(term1_id=T1, term2_id=T2, formula="hybrid", **kw)


def test_compact_payload_stays_lean():
    res = _compare(include_lca_details=False)
    assert "lca_details" not in res
    assert set(res) >= {"term1_id", "term2_id", "formula_used", "similarity_score"}


def test_standard_payload_explains_the_score():
    res = _compare(include_lca_details=True)
    details = res["lca_details"]
    assert details["mica"]["hpo_id"]
    assert "lca_depth" in details
    assert details["term1"]["depth"] is not None
    assert details["term2"]["depth"] is not None
    # D2: the structural proxy is honestly labelled normalized_depth, not ic_proxy
    # (it is depth/max_depth, not corpus information content).
    assert details["term1"]["normalized_depth"] is not None
    assert "ic_proxy" not in details["term1"]
    assert "path_length" in details
