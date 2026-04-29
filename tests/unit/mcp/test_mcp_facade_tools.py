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


def test_facade_registers_first_class_tools() -> None:
    _ensure_external_mcp_sdk()

    from api.mcp.facade import create_phentrieve_mcp

    mcp = create_phentrieve_mcp()
    tool_names = set(mcp._tool_manager._tools.keys())

    assert "phentrieve.extract_hpo_terms" in tool_names
    assert "phentrieve.extract_hpo_terms_llm" in tool_names
    assert "phentrieve.get_server_capabilities" in tool_names
