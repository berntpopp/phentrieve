from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def _ensure_external_mcp_sdk() -> None:
    import sys
    from pathlib import Path

    removed_paths = [
        path for path in sys.path if Path(path).as_posix().endswith("/tests/unit")
    ]
    for path in removed_paths:
        sys.path.remove(path)
    sys.modules.pop("mcp", None)

    try:
        import mcp.server.fastmcp  # noqa: F401
    finally:
        sys.path[:0] = removed_paths


@pytest.mark.integration
def test_mcp_http_tools_list_smoke(monkeypatch) -> None:
    _ensure_external_mcp_sdk()
    monkeypatch.setenv("ENABLE_MCP_HTTP", "true")

    import api.main as api_main

    monkeypatch.setattr(
        api_main,
        "get_sbert_model_dependency",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        api_main,
        "get_dense_retriever_dependency",
        AsyncMock(return_value=MagicMock()),
    )

    app = api_main.create_app()
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }
    with TestClient(
        app,
        raise_server_exceptions=False,
        base_url="http://127.0.0.1:8000",
    ) as client:
        initialize = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
            headers=headers,
        )
        assert initialize.status_code in {200, 202}
        session_headers = dict(headers)
        if session_id := initialize.headers.get("mcp-session-id"):
            session_headers["mcp-session-id"] = session_id

        initialized = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
            headers=session_headers,
        )
        assert initialized.status_code in {200, 202}

        tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers=session_headers,
        )

    assert tools.status_code == 200
    payload = tools.json()
    tool_names = {tool["name"] for tool in payload["result"]["tools"]}
    assert "phentrieve.extract_hpo_terms" in tool_names
    assert "phentrieve.extract_hpo_terms_llm" in tool_names
