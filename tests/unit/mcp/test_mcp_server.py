"""Unit tests for MCP server factory."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

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
        mock_mcp_class.return_value = mock_mcp_instance

        with patch.dict(
            "sys.modules", {"fastapi_mcp": MagicMock(FastApiMCP=mock_mcp_class)}
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

        # Remove fastapi_mcp from sys.modules if cached, then mock import failure
        with patch.dict(sys.modules, {"fastapi_mcp": None}):
            # When sys.modules[name] is None, import raises ImportError
            result = _check_mcp_installed()
            assert result is False

    def test_mcp_check_installed_returns_true_with_package(self):
        """Test that _check_mcp_installed returns True when package is installed."""
        with patch.dict("sys.modules", {"fastapi_mcp": MagicMock()}):
            from phentrieve.cli.mcp_commands import _check_mcp_installed

            assert _check_mcp_installed() is True


def test_mcp_info_prefers_http_configuration():
    import sys

    from typer.testing import CliRunner

    from phentrieve.cli.mcp_commands import app

    runner = CliRunner()
    with patch.dict(sys.modules, {"fastapi_mcp": MagicMock()}):
        result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert '"type": "http"' in result.output
    assert "http://127.0.0.1:8734/mcp" in result.output
    assert "Streamable HTTP" in result.output
    assert "/sse" not in result.output
