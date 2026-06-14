"""Unit tests for api.mcp.next_commands (workflow hint builders)."""

from __future__ import annotations

from api.mcp.next_commands import (
    after_chunk,
    after_compare,
    after_extract,
    after_search,
    cmd,
    default_error_next_commands,
)


def test_cmd_shape():
    assert cmd("phentrieve_search_hpo_terms", text="x") == {
        "tool": "phentrieve_search_hpo_terms",
        "arguments": {"text": "x"},
    }


def test_after_search_points_to_compare_with_two_hits():
    hints = after_search([{"hpo_id": "HP:0001250"}, {"hpo_id": "HP:0002133"}])
    assert hints[0]["tool"] == "phentrieve_compare_hpo_terms"
    assert hints[0]["arguments"]["term1_id"] == "HP:0001250"


def test_after_search_empty_suggests_capabilities():
    hints = after_search([])
    assert hints
    assert hints[0]["tool"] == "phentrieve_get_capabilities"


def test_after_extract_builds_phenopacket_payload():
    # aggregated_hpo_terms use the "id" key (search results use "hpo_id"); both work.
    hints = after_extract([{"id": "HP:1", "name": "x", "status": "affirmed"}])
    assert hints[0]["tool"] == "phentrieve_export_phenopacket"
    assert hints[0]["arguments"]["phenotypes"][0]["hpo_id"] == "HP:1"
    assert hints[0]["arguments"]["phenotypes"][0]["label"] == "x"


def test_after_extract_empty():
    assert after_extract([])[0]["tool"] == "phentrieve_get_capabilities"


def _aggregated_terms(n: int) -> list[dict[str, object]]:
    return [
        {
            "hpo_id": f"HP:{i:07d}",
            "label": f"term{i}",
            "assertion": "affirmed",
            "score": 0.9,
        }
        for i in range(n)
    ]


def test_after_extract_caps_and_compacts_prefill_under_lean_modes():
    """R2: under minimal/compact the export prefill is capped to <=5 terms and
    omits label (still executable -- _coerce_export_phenotype falls back to
    hpo_id), so _meta does not carry the full 25-term copy."""
    aggregated = _aggregated_terms(25)

    compact = after_extract(aggregated, mode="compact")
    phenos = compact[0]["arguments"]["phenotypes"]
    assert len(phenos) == 5
    assert "label" not in phenos[0]
    assert phenos[0]["hpo_id"] == "HP:0000000"
    assert phenos[0]["assertion"] == "affirmed"
    assert phenos[0]["score"] == 0.9

    minimal = after_extract(aggregated, mode="minimal")
    assert len(minimal[0]["arguments"]["phenotypes"]) == 5


def test_after_extract_standard_keeps_full_list_with_label():
    aggregated = _aggregated_terms(25)
    standard = after_extract(aggregated, mode="standard")
    phenos = standard[0]["arguments"]["phenotypes"]
    assert len(phenos) == 25
    assert phenos[0]["label"] == "term0"


def test_after_compare_and_chunk():
    # after_compare cross-checks the same pair with the alternate formula (L2:
    # executable, no free-text placeholder).
    hint = after_compare("HP:1", "HP:2", "hybrid")[0]
    assert hint["tool"] == "phentrieve_compare_hpo_terms"
    assert hint["arguments"]["term1_id"] == "HP:1"
    assert hint["arguments"]["formula"] == "simple_resnik_like"
    assert after_chunk([{"text": "abc"}])[0]["arguments"]["text"] == "abc"
    assert after_chunk([])[0]["tool"] == "phentrieve_get_capabilities"


def _all_text_args(hints):
    return [h["arguments"]["text"] for h in hints if "text" in h.get("arguments", {})]


def test_next_commands_have_no_unfilled_text_placeholders():
    """L2: no next_commands free-text argument is an unfillable <placeholder>."""
    samples = [
        *after_search([]),
        *after_search([{"hpo_id": "HP:0001250", "label": "Seizure"}]),
        *after_search([{"hpo_id": "HP:0001250"}, {"hpo_id": "HP:0002133"}]),
        *after_compare("HP:1", "HP:2", "hybrid"),
        *after_extract([{"id": "HP:1", "name": "x", "assertion": "affirmed"}]),
    ]
    for text in _all_text_args(samples):
        assert "<" not in text and ">" not in text


def test_after_search_single_hit_exports_real_term():
    hints = after_search([{"hpo_id": "HP:0001250", "label": "Seizure"}])
    assert hints[0]["tool"] == "phentrieve_export_phenopacket"
    pheno = hints[0]["arguments"]["phenotypes"][0]
    assert pheno["hpo_id"] == "HP:0001250"
    assert pheno["label"] == "Seizure"


def test_default_error_next_commands():
    hints = default_error_next_commands("phentrieve_compare_hpo_terms")
    tools = {h["tool"] for h in hints}
    assert "phentrieve_get_capabilities" in tools
    assert "phentrieve_diagnostics" in tools
