"""Unit tests for api.mcp.arg_help (aliases, did-you-mean, signatures)."""

from __future__ import annotations

from api.mcp.arg_help import (
    describe_constraints,
    did_you_mean,
    normalize_alias_args,
    tool_signature,
)


def test_normalize_alias_applies_known_synonym():
    new_args, applied = normalize_alias_args(["text", "language"], {"query": "ataxia"})
    assert new_args == {"text": "ataxia"}
    assert applied == [("query", "text")]


def test_normalize_alias_explicit_canonical_wins():
    new_args, applied = normalize_alias_args(["text"], {"text": "a", "query": "b"})
    assert new_args == {"text": "a"}
    assert applied == []


def test_normalize_alias_skips_when_canonical_not_a_param():
    # 'term1' -> 'term1_id', but tool only has 'text'
    new_args, applied = normalize_alias_args(["text"], {"term1": "HP:1"})
    assert new_args == {"term1": "HP:1"}
    assert applied == []


def test_did_you_mean_uses_alias_then_fuzzy():
    assert did_you_mean("limit", ["num_results", "text"]) == "num_results"
    assert did_you_mean("num_result", ["num_results", "text"]) == "num_results"
    assert did_you_mean("zzz", ["num_results", "text"]) is None


def test_describe_constraints_enum_and_range():
    enum_schema = {"enum": ["hybrid", "simple_resnik_like"]}
    allowed, human = describe_constraints(enum_schema)
    assert allowed == ["hybrid", "simple_resnik_like"]
    assert "must be one of" in human

    range_schema = {"minimum": 1, "maximum": 50}
    allowed, human = describe_constraints(range_schema)
    assert allowed == ["1..50"]
    assert "between" in human

    assert describe_constraints({"type": "string"}) is None


def test_tool_signature_orders_required_first():
    schema = {
        "properties": {"text": {}, "response_mode": {}, "num_results": {}},
        "required": ["text"],
    }
    sig = tool_signature("phentrieve_search_hpo_terms", schema)
    assert sig == "phentrieve_search_hpo_terms(text, response_mode=, num_results=)"
