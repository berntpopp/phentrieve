"""Unit tests for MCP server configuration."""

import os
from unittest.mock import patch

import pytest

# Check if MCP dependencies are installed
try:
    from pydantic_settings import BaseSettings  # noqa: F401

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP dependencies not installed")
class TestMCPSettings:
    """Tests for MCPSettings configuration class."""

    def test_default_settings(self):
        """Test that default settings are applied correctly."""
        from api.mcp.config import MCPSettings

        settings = MCPSettings()

        assert settings.name == "phentrieve"
        assert "HPO" in settings.description
        assert settings.host == "127.0.0.1"
        assert settings.port == 8734

    def test_env_override_name(self):
        """Test that PHENTRIEVE_MCP_NAME env var overrides default."""
        with patch.dict(os.environ, {"PHENTRIEVE_MCP_NAME": "custom-server"}):
            from api.mcp.config import MCPSettings

            settings = MCPSettings()
            assert settings.name == "custom-server"

    def test_env_override_port(self):
        """Test that PHENTRIEVE_MCP_PORT env var overrides default."""
        with patch.dict(os.environ, {"PHENTRIEVE_MCP_PORT": "9000"}):
            from api.mcp.config import MCPSettings

            settings = MCPSettings()
            assert settings.port == 9000

    def test_env_override_host(self):
        """Test that PHENTRIEVE_MCP_HOST env var overrides default."""
        with patch.dict(os.environ, {"PHENTRIEVE_MCP_HOST": "0.0.0.0"}):
            from api.mcp.config import MCPSettings

            settings = MCPSettings()
            assert settings.host == "0.0.0.0"


class TestMCPAllowedOperations:
    """Tests for MCP allowed operations configuration."""

    def test_allowed_operations_list(self):
        """Test that MCP_ALLOWED_OPERATIONS contains expected operations."""
        from api.mcp.server import MCP_ALLOWED_OPERATIONS

        assert isinstance(MCP_ALLOWED_OPERATIONS, list)
        assert len(MCP_ALLOWED_OPERATIONS) == 3

        # Check for expected operations
        assert "query_hpo_terms" in MCP_ALLOWED_OPERATIONS
        assert "process_clinical_text" in MCP_ALLOWED_OPERATIONS
        assert "calculate_term_similarity" in MCP_ALLOWED_OPERATIONS

    def test_operation_names_valid_format(self):
        """Test that operation names follow MCP best practices (no hyphens)."""
        from api.mcp.server import MCP_ALLOWED_OPERATIONS

        for op in MCP_ALLOWED_OPERATIONS:
            # Must start with letter
            assert op[0].isalpha(), f"Operation {op} must start with letter"
            # Only letters, numbers, underscores
            assert all(c.isalnum() or c == "_" for c in op), (
                f"Operation {op} has invalid characters"
            )


class TestMCPModule:
    """Tests for MCP module exports."""

    def test_module_exports(self):
        """Test that module exports expected symbols."""
        from api.mcp import MCP_ALLOWED_OPERATIONS, create_mcp_server

        assert callable(create_mcp_server)
        assert isinstance(MCP_ALLOWED_OPERATIONS, list)
