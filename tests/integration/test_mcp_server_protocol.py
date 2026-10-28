"""Integration test: drive the Phentrieve MCP server via the FastMCP client.

Exercises a real protocol round-trip (initialize -> list_tools -> call_tool)
over the in-memory transport, asserting the discovery surface and the Family B
envelope contract.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


def _structured(result):
    # fastmcp Client CallToolResult exposes structured_content (and .data).
    data = getattr(result, "structured_content", None)
    if data is None:
        data = getattr(result, "data", None)
    return data


def test_protocol_roundtrip_surface_and_envelope(tmp_path, monkeypatch):
    from fastmcp import Client

    from api.mcp.facade import create_phentrieve_mcp
    from phentrieve.evaluation.metrics import load_hpo_graph_data
    from tests.unit.tiny_hpo_db import use_hpo_data_dir, write_tiny_hpo_db

    # A tiny prepared ontology makes "not found" independent of whether
    # `phentrieve data prepare` has run (missing data is temporarily_unavailable).
    use_hpo_data_dir(monkeypatch, tmp_path / "data")
    write_tiny_hpo_db(tmp_path / "data")

    async def run():
        mcp = create_phentrieve_mcp()
        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools}
            assert len(names) == 8
            assert "phentrieve_get_capabilities" in names
            for t in tools:
                assert t.annotations is not None
                assert t.annotations.readOnlyHint is True

            cap = _structured(await client.call_tool("phentrieve_get_capabilities", {}))
            assert cap["success"] is True
            assert cap["capabilities_version"].startswith("sha256:")
            assert cap["_meta"]["unsafe_for_clinical_use"] is True

            err = _structured(
                await client.call_tool(
                    "phentrieve_compare_hpo_terms",
                    {"term1_id": "HP:0000001", "term2_id": "HP:9999999"},
                )
            )
            assert err["success"] is False
            assert err["error_code"] == "not_found"

    try:
        asyncio.run(run())
    finally:
        load_hpo_graph_data.cache_clear()


def test_protocol_initialize_exposes_instructions():
    from fastmcp import Client

    from api.mcp.facade import create_phentrieve_mcp

    async def run():
        mcp = create_phentrieve_mcp()
        async with Client(mcp) as client:
            result = client.initialize_result
            assert result is not None
            assert "response_mode" in (result.instructions or "")

    asyncio.run(run())
