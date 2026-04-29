"""Unit tests for MCP server factory."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


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


# Check if MCP dependencies are installed
try:
    import pydantic_settings

    MCP_AVAILABLE = bool(pydantic_settings)
except ImportError:
    MCP_AVAILABLE = False


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP dependencies not installed")
class TestCreateMcpServer:
    """Tests for create_mcp_server factory function."""

    def test_create_mcp_server_with_mock(self):
        """Test create_mcp_server with mocked FastApiMCP."""
        import sys

        # Clear any cached imports to ensure clean state
        modules_to_clear = [k for k in list(sys.modules.keys()) if "api.mcp" in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

        mock_mcp_class = MagicMock()
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.tools = []
        mock_mcp_class.return_value = mock_mcp_instance

        with patch.dict(
            "sys.modules",
            {
                "fastapi_mcp": MagicMock(FastApiMCP=mock_mcp_class),
                "mcp": MagicMock(types=MagicMock()),
            },
        ):
            # Re-import after patching
            from api.mcp.server import MCP_ALLOWED_OPERATIONS, create_mcp_server

            mock_app = MagicMock()
            result = create_mcp_server(mock_app)

            # Verify FastApiMCP was called with correct arguments
            mock_mcp_class.assert_called_once()
            call_kwargs = mock_mcp_class.call_args[1]
            assert call_kwargs["name"] == "phentrieve"
            assert call_kwargs["include_operations"] == MCP_ALLOWED_OPERATIONS
            assert result == mock_mcp_instance


class TestMcpCliCommands:
    """Tests for MCP CLI commands."""

    def test_mcp_check_installed_returns_false_without_package(self):
        """Test that _check_mcp_installed returns False when package not installed."""
        import sys

        from phentrieve.cli.mcp_commands import _check_mcp_installed

        with patch.dict(sys.modules, {"fastapi_mcp": None, "mcp.server.fastmcp": None}):
            # When sys.modules[name] is None, import raises ImportError
            result = _check_mcp_installed()
            assert result is False

    def test_mcp_check_installed_returns_true_with_package(self):
        """Test that _check_mcp_installed returns True when package is installed."""
        with patch.dict(
            "sys.modules",
            {
                "fastapi_mcp": MagicMock(),
                "mcp": MagicMock(),
                "mcp.server": MagicMock(),
                "mcp.server.fastmcp": MagicMock(),
            },
        ):
            from phentrieve.cli.mcp_commands import _check_mcp_installed

            assert _check_mcp_installed() is True

    def test_mcp_check_installed_requires_official_mcp_sdk(self):
        """HTTP MCP support requires fastapi-mcp and the official MCP SDK."""
        with patch.dict(
            "sys.modules",
            {"fastapi_mcp": MagicMock(), "mcp.server.fastmcp": None},
        ):
            from phentrieve.cli.mcp_commands import _check_mcp_installed

            assert _check_mcp_installed() is False


def test_mcp_info_prefers_http_configuration():
    import sys

    from typer.testing import CliRunner

    from phentrieve.cli.mcp_commands import app

    runner = CliRunner()
    with patch.dict(
        sys.modules,
        {
            "fastapi_mcp": MagicMock(),
            "mcp": MagicMock(),
            "mcp.server": MagicMock(),
            "mcp.server.fastmcp": MagicMock(),
        },
    ):
        result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert '"type": "http"' in result.output
    assert "phentrieve.extract_hpo_terms_llm" in result.output
    assert "process_clinical_text" not in result.output
    assert "http://127.0.0.1:8734/mcp" in result.output
    assert "Streamable HTTP" in result.output
    assert "/sse" not in result.output


def test_mount_phentrieve_mcp_facade_uses_idempotent_subapp_mount():
    _ensure_external_mcp_sdk()

    from fastapi import FastAPI
    from starlette.routing import Mount

    from api.mcp.server import mount_phentrieve_mcp_facade

    app = FastAPI()

    mount_phentrieve_mcp_facade(app)
    mount_phentrieve_mcp_facade(app)

    mounts = [
        route
        for route in app.routes
        if isinstance(route, Mount) and route.path == "/mcp"
    ]
    assert len(mounts) == 1
    assert hasattr(app.state, "phentrieve_mcp_session_manager")
