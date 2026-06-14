"""D5 -- compare_hpo_terms honours response_mode: at standard/full it returns the
MICA, per-term depth/IC, labels and subsumer path so the score is explainable;
minimal/compact stay the lean 4-field payload.
"""

import pytest

from api.mcp.service_adapters import compare_hpo_terms_service

pytestmark = pytest.mark.unit

# Related pair from the assessment (HP:0000787 vs HP:0004724).
T1, T2 = "HP:0000787", "HP:0004724"


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
    assert details["term1"]["ic_proxy"] is not None
    assert "path_length" in details
