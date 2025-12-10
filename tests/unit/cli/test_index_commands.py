"""Unit tests for index CLI commands.

Tests for index management commands:
- build_index: Vector index building with various configurations

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
# Tests for build_index()
# =============================================================================


class TestBuildIndex:
    """Test build_index() command."""

    def test_builds_index_successfully_with_defaults(self, mocker):
        """Test successful index building with default parameters."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mock_echo = mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act
        build_index()

        # Assert
        mock_setup_logging.assert_called_once_with(debug=False)
        mock_echo.assert_called_once_with(
            "Starting single-vector index building process"
        )
        mock_orchestrate.assert_called_once_with(
            model_name_arg=None,
            all_models=False,
            batch_size=100,
            trust_remote_code=False,
            device_override=None,
            recreate=False,
            debug=False,
            multi_vector=False,
            index_dir_override=None,
            data_dir_override=None,
        )

        # Check success message
        success_call = mock_secho.call_args
        assert "completed successfully" in success_call.args[0]
        assert success_call.kwargs["fg"] == typer.colors.GREEN

    def test_builds_index_with_specific_model(self, mocker):
        """Test building index with specific model name."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Assert
        mock_setup_logging.assert_called_once()
        mock_orchestrate.assert_called_once()
        assert (
            mock_orchestrate.call_args.kwargs["model_name_arg"]
            == "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_builds_index_for_all_models(self, mocker):
        """Test building index for all benchmark models."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(all_models=True)

        # Assert
        mock_setup_logging.assert_called_once()
        mock_orchestrate.assert_called_once_with(
            model_name_arg=None,
            all_models=True,
            batch_size=100,
            trust_remote_code=False,
            device_override=None,
            recreate=False,
            debug=False,
            multi_vector=False,
            index_dir_override=None,
            data_dir_override=None,
        )

    def test_builds_index_with_recreate_flag(self, mocker):
        """Test building index with recreate flag."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(recreate=True)

        # Assert
        mock_setup_logging.assert_called_once()
        assert mock_orchestrate.call_args.kwargs["recreate"] is True

    def test_builds_index_with_custom_batch_size(self, mocker):
        """Test building index with custom batch size."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(batch_size=50)

        # Assert
        mock_setup_logging.assert_called_once()
        assert mock_orchestrate.call_args.kwargs["batch_size"] == 50

    def test_builds_index_with_trust_remote_code(self, mocker):
        """Test building index with trust_remote_code flag."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(trust_remote_code=True)

        # Assert
        mock_setup_logging.assert_called_once()
        assert mock_orchestrate.call_args.kwargs["trust_remote_code"] is True

    def test_builds_index_with_cpu_flag(self, mocker):
        """Test building index with CPU flag (device override)."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(cpu=True)

        # Assert
        mock_setup_logging.assert_called_once()
        # CPU flag should set device_override to "cpu"
        assert mock_orchestrate.call_args.kwargs["device_override"] == "cpu"

    def test_builds_index_with_debug_mode(self, mocker):
        """Test building index with debug logging enabled."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(debug=True)

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        assert mock_orchestrate.call_args.kwargs["debug"] is True

    def test_builds_index_with_all_options(self, mocker):
        """Test building index with all optional parameters."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(
            model_name="test-model",
            all_models=True,
            recreate=True,
            batch_size=200,
            trust_remote_code=True,
            cpu=True,
            debug=True,
        )

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        mock_orchestrate.assert_called_once_with(
            model_name_arg="test-model",
            all_models=True,
            batch_size=200,
            trust_remote_code=True,
            device_override="cpu",
            recreate=True,
            debug=True,
            multi_vector=False,
            index_dir_override=None,
            data_dir_override=None,
        )

    def test_index_building_fails_with_error(self, mocker):
        """Test index building failure when orchestrator returns False."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=False,  # Failure
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            build_index()

        assert exc_info.value.exit_code == 1

        # Check error message
        error_call = mock_secho.call_args
        assert "failed" in error_call.args[0]
        assert error_call.kwargs["fg"] == typer.colors.RED

    def test_index_building_displays_start_message(self, mocker):
        """Test that index building displays start message."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mock_echo = mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index()

        # Assert
        mock_echo.assert_called_once_with(
            "Starting single-vector index building process"
        )

    def test_device_override_none_when_cpu_false(self, mocker):
        """Test that device_override is None when CPU flag is False."""
        from phentrieve.cli.index_commands import build_index

        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.indexing.chromadb_orchestrator.orchestrate_index_building",
            return_value=True,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        build_index(cpu=False)

        # Assert
        # When cpu=False, device_override should be None (not "cpu")
        assert mock_orchestrate.call_args.kwargs["device_override"] is None
