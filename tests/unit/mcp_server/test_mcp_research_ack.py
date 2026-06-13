"""Unit tests for research-use acknowledgement parity in hosted mode."""

from __future__ import annotations

import pytest

import api.config as api_config
from api.mcp.envelope import McpToolError
from api.mcp.tools._common import require_research_ack


def test_ack_required_in_hosted_mode(monkeypatch):
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_PUBLIC_HOSTED_MODE", True, raising=False
    )
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_REQUIRE_RESEARCH_ACK", False, raising=False
    )
    with pytest.raises(McpToolError) as ei:
        require_research_ack(acknowledged=False)
    assert ei.value.error_code == "invalid_input"
    assert ei.value.details.get("field") == "research_use_acknowledged"


def test_ack_satisfied(monkeypatch):
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_PUBLIC_HOSTED_MODE", True, raising=False
    )
    require_research_ack(acknowledged=True)  # no raise


def test_ack_required_via_require_research_ack_flag(monkeypatch):
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_PUBLIC_HOSTED_MODE", False, raising=False
    )
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_REQUIRE_RESEARCH_ACK", True, raising=False
    )
    with pytest.raises(McpToolError):
        require_research_ack(acknowledged=False)


def test_ack_not_required_when_unconfigured(monkeypatch):
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_PUBLIC_HOSTED_MODE", False, raising=False
    )
    monkeypatch.setattr(
        api_config, "PHENTRIEVE_REQUIRE_RESEARCH_ACK", False, raising=False
    )
    require_research_ack(acknowledged=False)  # no raise
