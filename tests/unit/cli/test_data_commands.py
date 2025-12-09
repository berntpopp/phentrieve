"""Unit tests for data CLI commands.

Tests for data management commands:
- prepare_hpo_data: HPO data preparation and preprocessing

Following best practices:
- Mock external dependencies (orchestrators, logging)
- Test CLI argument handling
- Test success/failure paths with appropriate exit codes
- Clear Arrange-Act-Assert structure
"""

import pytest
import typer

# NOTE: Do NOT import CLI functions at module level!
# They trigger slow torch/transformers imports during test collection.
# Import them inside test functions instead.

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for prepare_hpo_data()
# =============================================================================


class TestPrepareHpoData:
    """Test prepare_hpo_data() command."""

    def test_prepares_data_successfully_with_defaults(self, mocker):
        """Test successful HPO data preparation with default parameters."""
        # Import inside test to avoid slow collection
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=True,
        )
        mock_echo = mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act
        prepare_hpo_data()

        # Assert
        mock_setup_logging.assert_called_once_with(debug=False)
        mock_echo.assert_called_once_with("Starting HPO data preparation (latest)...")
        mock_orchestrate.assert_called_once_with(
            debug=False,
            force_update=False,
            data_dir_override=None,
            include_obsolete=False,  # Issue #133: Default filters obsolete terms
            hpo_version=None,  # Default to latest
        )

        # Check success message
        success_call = mock_secho.call_args
        assert "completed successfully" in success_call.args[0]
        assert success_call.kwargs["fg"] == typer.colors.GREEN

    def test_prepares_data_with_debug_enabled(self, mocker):
        """Test data preparation with debug logging enabled."""
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        prepare_hpo_data(debug=True)

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        mock_orchestrate.assert_called_once()
        assert mock_orchestrate.call_args.kwargs["debug"] is True

    def test_prepares_data_with_force_update(self, mocker):
        """Test data preparation with force update flag."""
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        prepare_hpo_data(force=True)

        # Assert
        mock_setup_logging.assert_called_once()
        mock_orchestrate.assert_called_once_with(
            debug=False,
            force_update=True,
            data_dir_override=None,
            include_obsolete=False,  # Issue #133: Default filters obsolete terms
            hpo_version=None,  # Default to latest
        )

    def test_prepares_data_with_custom_data_dir(self, mocker):
        """Test data preparation with custom data directory."""
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        prepare_hpo_data(data_dir="/custom/data/path")

        # Assert
        mock_setup_logging.assert_called_once()
        mock_orchestrate.assert_called_once_with(
            debug=False,
            force_update=False,
            data_dir_override="/custom/data/path",
            include_obsolete=False,  # Issue #133: Default filters obsolete terms
            hpo_version=None,  # Default to latest
        )

    def test_prepares_data_with_all_options(self, mocker):
        """Test data preparation with all optional parameters."""
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        prepare_hpo_data(
            debug=True,
            force=True,
            data_dir="/custom/path",
            include_obsolete=True,  # Issue #133: Test include_obsolete flag
        )

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        mock_orchestrate.assert_called_once_with(
            debug=True,
            force_update=True,
            data_dir_override="/custom/path",
            include_obsolete=True,  # Issue #133: Passed through
            hpo_version=None,  # Default to latest
        )

    def test_preparation_fails_with_error(self, mocker):
        """Test preparation failure when orchestrator returns False."""
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=False,  # Failure
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            prepare_hpo_data()

        assert exc_info.value.exit_code == 1

        # Check error message
        error_call = mock_secho.call_args
        assert "failed" in error_call.args[0]
        assert error_call.kwargs["fg"] == typer.colors.RED

    def test_preparation_displays_start_message(self, mocker):
        """Test that preparation displays start message."""
        from phentrieve.cli.data_commands import prepare_hpo_data

        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.data_processing.hpo_parser.orchestrate_hpo_preparation",
            return_value=True,
        )
        mock_echo = mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        prepare_hpo_data()

        # Assert
        assert any(
            "Starting HPO data preparation" in str(call.args[0])
            for call in mock_echo.call_args_list
        )
