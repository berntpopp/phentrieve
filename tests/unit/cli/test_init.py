"""Unit tests for CLI __init__.py module.

Tests for the main CLI entry point:
- Version callback
- Main callback
- App configuration

Following best practices:
- Lazy imports to avoid slow test collection
- Mock external dependencies
- Test success/failure paths
- Clear Arrange-Act-Assert structure
"""

import pytest
import typer

# NOTE: Do NOT import CLI functions at module level!
# They trigger slow torch/transformers imports during test collection.
# Import them inside test functions instead.

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for version_callback()
# =============================================================================


class TestVersionCallback:
    """Test version_callback() function."""

    def test_displays_version_when_true(self, mocker):
        """Test version callback displays version and exits when value is True."""
        from phentrieve.cli import __version__, version_callback

        # Arrange
        mock_echo = mocker.patch("typer.echo")

        # Act & Assert - should raise Exit
        with pytest.raises(typer.Exit) as exc_info:
            version_callback(True)

        # Verify exit code is 0 (success)
        assert exc_info.value.exit_code == 0

        # Verify version was displayed
        mock_echo.assert_called_once()
        assert __version__ in mock_echo.call_args[0][0]
        assert "Phentrieve CLI version:" in mock_echo.call_args[0][0]

    def test_does_nothing_when_false(self, mocker):
        """Test version callback does nothing when value is False."""
        from phentrieve.cli import version_callback

        # Arrange
        mock_echo = mocker.patch("typer.echo")

        # Act - should not raise Exit
        result = version_callback(False)

        # Assert
        assert result is None
        mock_echo.assert_not_called()


# =============================================================================
# Tests for main_callback()
# =============================================================================


class TestMainCallback:
    """Test main_callback() function."""

    def test_main_callback_executes_without_error(self, mocker):
        """Test main callback executes successfully."""
        from phentrieve.cli import main_callback

        # Arrange - Mock version_callback to prevent exit
        mocker.patch("phentrieve.cli.version_callback", return_value=None)

        # Act - should not raise any exception
        result = main_callback(version=False)

        # Assert - callback returns None (does nothing except handle --version)
        assert result is None

    def test_main_callback_with_version_true(self, mocker):
        """Test main callback with version=True triggers version_callback."""
        from phentrieve.cli import main_callback, version_callback

        # Arrange
        mock_echo = mocker.patch("typer.echo")

        # Act & Assert - version_callback will raise Exit
        with pytest.raises(typer.Exit):
            # This simulates what happens when --version is passed
            version_callback(True)

        mock_echo.assert_called_once()


# =============================================================================
# Tests for CLI app configuration
# =============================================================================


class TestCliAppConfiguration:
    """Test CLI app setup and configuration."""

    def test_app_is_typer_instance(self):
        """Test that app is a properly configured Typer instance."""
        from phentrieve.cli import app

        # Assert
        assert isinstance(app, typer.Typer)
        assert app.info.name == "phentrieve"
        assert "HPO" in app.info.help

    def test_version_is_accessible(self):
        """Test that __version__ is accessible and is a string."""
        from phentrieve.cli import __version__

        # Assert
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        # Version should follow semantic versioning pattern (e.g., "0.1.0")
        assert "." in __version__

    def test_all_command_groups_registered(self):
        """Test that all command groups are registered with the app."""
        from phentrieve.cli import app

        # Get registered command names
        command_names = {cmd.name for cmd in app.registered_groups}

        # Assert all expected groups are registered
        expected_groups = {"data", "index", "text", "benchmark", "similarity"}
        assert expected_groups.issubset(command_names)

    def test_query_command_registered(self):
        """Test that query command is registered directly."""
        from phentrieve.cli import app

        # Get registered command names
        command_names = {cmd.name for cmd in app.registered_commands}

        # Assert query is registered
        assert "query" in command_names
