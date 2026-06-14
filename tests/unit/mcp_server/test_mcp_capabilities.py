"""Unit tests for api.mcp.capabilities (descriptor + content hash)."""

from __future__ import annotations

import json

from api.mcp.capabilities import build_capabilities, capabilities_version


def test_version_is_prefixed_sha256_short():
    v = capabilities_version()
    assert v.startswith("sha256:")
    assert len(v) == len("sha256:") + 16


def test_version_is_deterministic():
    assert capabilities_version() == capabilities_version()


def test_build_capabilities_core_fields():
    cap = build_capabilities()
    assert cap["server"] == "phentrieve"
    assert cap["transport"] == "streamable_http"
    assert cap["research_use_only"] is True
    assert set(cap["response_modes"]["modes"]) == {
        "minimal",
        "compact",
        "standard",
        "full",
    }
    assert "error_codes" in cap and "not_found" in cap["error_codes"]
    assert cap["tool_count"] == 8
    assert cap["capabilities_version"].startswith("sha256:")


def test_descriptor_chars_matches_serialized_body():
    cap = build_capabilities()
    body = {
        k: v
        for k, v in cap.items()
        if k not in ("capabilities_version", "descriptor_hash", "descriptor_chars")
    }
    assert cap["descriptor_chars"] == len(json.dumps(body, sort_keys=True, default=str))


def test_details_sections_expand():
    cap = build_capabilities(details=["sample_calls", "argument_aliases"])
    assert "sample_calls" in cap
    assert "argument_aliases" in cap
    # capabilities_version is the stable warm-cache key (M1): it stays equal to
    # base/_meta across details; the per-descriptor content hash differs instead.
    assert cap["capabilities_version"] == capabilities_version()
    assert cap["descriptor_hash"] != capabilities_version()
