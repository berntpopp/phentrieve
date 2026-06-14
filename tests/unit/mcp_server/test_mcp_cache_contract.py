"""D8 -- the capability descriptor documents one canonical cache key and what
descriptor_hash is for, so the warm-cache contract is unambiguous to consumers.
"""

import pytest

from api.mcp.capabilities import build_capabilities

pytestmark = pytest.mark.unit


def test_capabilities_documents_cache_key_contract():
    body = build_capabilities()
    assert "capabilities_version" in body
    assert "descriptor_hash" in body
    contract = body.get("cache_contract", "")
    assert "capabilities_version" in contract
    assert "descriptor_hash" in contract
    assert "canonical" in contract.lower() or "warm" in contract.lower()


def test_cache_contract_present_when_detailed():
    body = build_capabilities(details=["sample_calls"])
    assert "cache_contract" in body
