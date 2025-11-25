"""Unit tests for benchmark CLI commands.

Tests for benchmark-related commands:
- run_benchmarks: Benchmark execution with model evaluation
- compare_benchmarks: Comparison of multiple benchmark results
- visualize_benchmarks: Generation of visualization charts

Following best practices:
- Mock external dependencies (orchestrators, logging)
- Test CLI argument handling and validation
- Test success/failure paths with appropriate exit codes
- Clear Arrange-Act-Assert structure
"""

import pytest
import typer

from phentrieve.cli.benchmark_commands import (
    compare_benchmarks,
    run_benchmarks,
    visualize_benchmarks,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for run_benchmarks()
# =============================================================================


class TestRunBenchmarks:
    """Test run_benchmarks() command."""

    def test_runs_benchmark_successfully_single_model(self, mocker):
        """Test successful benchmark run for single model."""
        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.benchmark_orchestrator.orchestrate_benchmark",
            return_value={"model": "test-model", "mrr": 0.85},
        )
        mock_echo = mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act
        run_benchmarks(
            test_file="test.json",
            model_name="test-model",
            debug=True,
        )

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        mock_echo.assert_called_with("Starting benchmark evaluation...")
        mock_orchestrate.assert_called_once()

        # Check orchestrate was called with correct args
        call_kwargs = mock_orchestrate.call_args.kwargs
        assert call_kwargs["test_file"] == "test.json"
        assert call_kwargs["model_name"] == "test-model"
        assert call_kwargs["debug"] is True

        # Check success message
        mock_secho.assert_called_once()
        success_call = mock_secho.call_args
        assert "completed successfully" in success_call.args[0]
        assert success_call.kwargs["fg"] == typer.colors.GREEN

    def test_runs_benchmark_successfully_multiple_models(self, mocker):
        """Test successful benchmark run for multiple models."""
        # Arrange
        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.benchmark_orchestrator.orchestrate_benchmark",
            return_value=[
                {"model": "model1", "mrr": 0.85},
                {"model": "model2", "mrr": 0.90},
            ],
        )
        mock_secho = mocker.patch("typer.secho")

        # Act
        run_benchmarks(
            model_list="model1,model2",
            similarity_threshold=0.5,
        )

        # Assert
        mock_setup_logging.assert_called_once_with(debug=False)
        mock_orchestrate.assert_called_once()

        # Check success message mentions multiple models
        success_call = mock_secho.call_args.args[0]
        assert "2 models" in success_call
        assert "completed successfully" in success_call

    def test_runs_benchmark_with_all_options(self, mocker):
        """Test benchmark with all optional parameters."""
        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.benchmark_orchestrator.orchestrate_benchmark",
            return_value={"model": "test", "mrr": 0.8},
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        run_benchmarks(
            test_file="test.json",
            model_name="test-model",
            model_list="model1,model2",
            all_models=True,
            similarity_threshold=0.2,
            cpu=True,
            detailed=True,
            output="results.csv",
            create_sample=True,
            trust_remote_code=True,
            enable_reranker=True,
            reranker_model="cross-encoder",
            rerank_count=20,
            similarity_formula="simple_resnik_like",
            debug=True,
        )

        # Assert
        call_kwargs = mock_orchestrate.call_args.kwargs
        assert call_kwargs["test_file"] == "test.json"
        assert call_kwargs["model_name"] == "test-model"
        assert call_kwargs["model_list"] == "model1,model2"
        assert call_kwargs["all_models"] is True
        assert call_kwargs["similarity_threshold"] == 0.2
        assert call_kwargs["cpu"] is True
        assert call_kwargs["detailed"] is True
        assert call_kwargs["output"] == "results.csv"
        assert call_kwargs["create_sample"] is True
        assert call_kwargs["trust_remote_code"] is True
        assert call_kwargs["enable_reranker"] is True
        assert call_kwargs["reranker_model"] == "cross-encoder"
        assert call_kwargs["rerank_count"] == 20
        assert call_kwargs["similarity_formula"] == "simple_resnik_like"
        assert call_kwargs["debug"] is True

    def test_benchmark_fails_with_no_results(self, mocker):
        """Test benchmark failure when no results returned."""
        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.evaluation.benchmark_orchestrator.orchestrate_benchmark",
            return_value=None,  # No results
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            run_benchmarks(model_name="test-model")

        assert exc_info.value.exit_code == 1

        # Check error message was displayed
        error_call = mock_secho.call_args
        assert "failed" in error_call.args[0] or "no results" in error_call.args[0]
        assert error_call.kwargs["fg"] == typer.colors.RED

    def test_benchmark_fails_with_empty_results(self, mocker):
        """Test benchmark failure when empty results returned."""
        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.evaluation.benchmark_orchestrator.orchestrate_benchmark",
            return_value=[],  # Empty list
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            run_benchmarks(model_name="test-model")

        assert exc_info.value.exit_code == 1
        mock_secho.assert_called()

    def test_benchmark_with_default_parameters(self, mocker):
        """Test benchmark with default parameter values."""
        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.benchmark_orchestrator.orchestrate_benchmark",
            return_value={"model": "test", "mrr": 0.8},
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        run_benchmarks()

        # Assert - Check defaults are passed correctly
        call_kwargs = mock_orchestrate.call_args.kwargs
        assert call_kwargs["test_file"] == ""
        assert call_kwargs["model_name"] == ""
        assert call_kwargs["model_list"] == ""
        assert call_kwargs["all_models"] is False
        assert call_kwargs["similarity_threshold"] == 0.1
        assert call_kwargs["cpu"] is False
        assert call_kwargs["detailed"] is False
        assert call_kwargs["output"] == ""
        assert call_kwargs["create_sample"] is False
        assert call_kwargs["trust_remote_code"] is False
        assert call_kwargs["enable_reranker"] is False
        assert call_kwargs["rerank_count"] == 10
        assert call_kwargs["similarity_formula"] == "hybrid"
        assert call_kwargs["debug"] is False


# =============================================================================
# Tests for compare_benchmarks()
# =============================================================================


class TestCompareBenchmarks:
    """Test compare_benchmarks() command."""

    def test_compares_benchmarks_successfully(self, mocker):
        """Test successful benchmark comparison."""
        # Arrange
        import pandas as pd

        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_df = pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "mrr": [0.85, 0.90],
                "hit_rate": [0.95, 0.98],
            }
        )
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=mock_df,
        )
        mock_echo = mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act
        compare_benchmarks(
            summaries_dir="/results",
            output_csv="comparison.csv",
            debug=True,
        )

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        mock_echo.assert_any_call("Comparing benchmark results...")
        mock_orchestrate.assert_called_once_with(
            summaries_dir="/results",
            output_csv="comparison.csv",
            visualize=False,
            debug=True,
        )

        # Check success message
        success_call = mock_secho.call_args
        assert "completed successfully" in success_call.args[0]
        assert success_call.kwargs["fg"] == typer.colors.GREEN

    def test_compares_benchmarks_with_defaults(self, mocker):
        """Test benchmark comparison with default parameters."""
        # Arrange
        import pandas as pd

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_df = pd.DataFrame({"model": ["model1"], "mrr": [0.85]})
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=mock_df,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        compare_benchmarks()

        # Assert
        call_kwargs = mock_orchestrate.call_args.kwargs
        assert call_kwargs["summaries_dir"] is None
        assert call_kwargs["output_csv"] is None
        assert call_kwargs["visualize"] is False
        assert call_kwargs["debug"] is False

    def test_comparison_fails_with_none_dataframe(self, mocker):
        """Test comparison failure when None returned."""
        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=None,
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            compare_benchmarks()

        assert exc_info.value.exit_code == 1
        error_call = mock_secho.call_args
        assert "failed" in error_call.args[0] or "not found" in error_call.args[0]
        assert error_call.kwargs["fg"] == typer.colors.RED

    def test_comparison_fails_with_empty_dataframe(self, mocker):
        """Test comparison failure when empty DataFrame returned."""
        # Arrange
        import pandas as pd

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=pd.DataFrame(),  # Empty DataFrame
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            compare_benchmarks()

        assert exc_info.value.exit_code == 1
        mock_secho.assert_called()

    def test_comparison_displays_dataframe(self, mocker):
        """Test comparison displays DataFrame with proper formatting."""
        # Arrange
        import pandas as pd

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_df = pd.DataFrame(
            {
                "model": ["model1", "model2", "model3"],
                "mrr": [0.85, 0.90, 0.88],
            }
        )
        mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=mock_df,
        )
        mock_echo = mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        compare_benchmarks()

        # Assert - Check DataFrame was displayed
        assert any(
            "Benchmark Comparison" in str(call.args[0])
            for call in mock_echo.call_args_list
        )
        assert any(
            "Models compared: 3" in str(call.args[0])
            for call in mock_echo.call_args_list
        )


# =============================================================================
# Tests for visualize_benchmarks()
# =============================================================================


class TestVisualizeBenchmarks:
    """Test visualize_benchmarks() command."""

    def test_visualizes_benchmarks_successfully(self, mocker):
        """Test successful visualization generation."""
        # Arrange
        import pandas as pd

        mock_setup_logging = mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_df = pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "mrr": [0.85, 0.90],
            }
        )
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=mock_df,
        )
        mock_echo = mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act
        visualize_benchmarks(
            summaries_dir="/results",
            metrics="mrr,hit_rate",
            output_dir="/visualizations",
            debug=True,
        )

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)
        mock_echo.assert_any_call("Generating visualizations from benchmark results...")
        mock_orchestrate.assert_called_once_with(
            summaries_dir="/results",
            output_csv=None,
            visualize=True,
            output_dir="/visualizations",
            metrics="mrr,hit_rate",
            debug=True,
        )

        # Check success message
        success_call = mock_secho.call_args
        assert "generated successfully" in success_call.args[0]
        assert success_call.kwargs["fg"] == typer.colors.GREEN

    def test_visualizes_with_default_parameters(self, mocker):
        """Test visualization with default parameters."""
        # Arrange
        import pandas as pd

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_df = pd.DataFrame({"model": ["model1"], "mrr": [0.85]})
        mock_orchestrate = mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=mock_df,
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        visualize_benchmarks()

        # Assert
        call_kwargs = mock_orchestrate.call_args.kwargs
        assert call_kwargs["summaries_dir"] is None
        assert call_kwargs["output_csv"] is None
        assert call_kwargs["visualize"] is True  # Always True for visualize
        assert call_kwargs["output_dir"] is None
        assert call_kwargs["metrics"] == "all"
        assert call_kwargs["debug"] is False

    def test_visualization_fails_with_none_dataframe(self, mocker):
        """Test visualization failure when None returned."""
        # Arrange
        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=None,
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            visualize_benchmarks()

        assert exc_info.value.exit_code == 1
        error_call = mock_secho.call_args
        assert "failed" in error_call.args[0] or "not found" in error_call.args[0]
        assert error_call.kwargs["fg"] == typer.colors.RED

    def test_visualization_fails_with_empty_dataframe(self, mocker):
        """Test visualization failure when empty DataFrame returned."""
        # Arrange
        import pandas as pd

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=pd.DataFrame(),
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            visualize_benchmarks()

        assert exc_info.value.exit_code == 1
        mock_secho.assert_called()

    def test_visualization_shows_model_count(self, mocker):
        """Test visualization displays number of models visualized."""
        # Arrange
        import pandas as pd

        mocker.patch("phentrieve.utils.setup_logging_cli")
        mock_df = pd.DataFrame(
            {
                "model": ["model1", "model2", "model3", "model4"],
                "mrr": [0.85, 0.90, 0.88, 0.92],
            }
        )
        mocker.patch(
            "phentrieve.evaluation.comparison_orchestrator.orchestrate_benchmark_comparison",
            return_value=mock_df,
        )
        mock_echo = mocker.patch("typer.echo")
        mocker.patch("typer.secho")

        # Act
        visualize_benchmarks()

        # Assert - Check model count was displayed
        assert any("4 models" in str(call.args[0]) for call in mock_echo.call_args_list)
