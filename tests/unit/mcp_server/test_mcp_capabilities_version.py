"""M1: capabilities_version is a stable warm-cache key that matches
_meta.capabilities_version regardless of the details expansion."""

import pytest

from api.mcp.capabilities import build_capabilities, capabilities_version

pytestmark = pytest.mark.unit


def test_capabilities_version_stable_across_details():
    base = build_capabilities()
    detailed = build_capabilities(details=["sample_calls", "argument_aliases"])
    assert base["capabilities_version"] == capabilities_version()
    # The detailed descriptor must report the SAME capabilities_version as base
    # (and therefore as _meta) so a warm client never re-fetches forever.
    assert detailed["capabilities_version"] == capabilities_version()


def test_detailed_descriptor_exposes_its_own_content_hash():
    base = build_capabilities()
    detailed = build_capabilities(details=["sample_calls"])
    # A distinct content hash is still available for clients that want it.
    assert "descriptor_hash" in detailed
    assert detailed["descriptor_hash"] != base["descriptor_hash"]
