"""Unit tests for similarity CLI commands.

Tests for HPO term similarity calculation commands:
- hpo_similarity_cli: Calculate semantic similarity between HPO terms
- _ensure_cli_hpo_label_cache: HPO label cache initialization

Following best practices:
- Mock external dependencies (data loading, logging)
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
# Tests for _ensure_cli_hpo_label_cache()
# =============================================================================


class TestEnsureCliHpoLabelCache:
    """Test _ensure_cli_hpo_label_cache() helper function."""

    def test_initializes_cache_from_hpo_terms(self, mocker):
        """Test cache initialization from HPO terms data."""
        # Arrange
        # Reset the global cache
        import phentrieve.cli.similarity_commands as sim_module

        sim_module._cli_hpo_label_cache = None

        mock_load_terms = mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Term 1"},
                {"id": "HP:0000002", "label": "Term 2"},
            ],
        )

        # Act
        cache = sim_module._ensure_cli_hpo_label_cache()

        # Assert
        mock_load_terms.assert_called_once()
        assert len(cache) == 2
        assert cache["HP:0000001"] == "Term 1"
        assert cache["HP:0000002"] == "Term 2"

    def test_returns_empty_cache_when_no_data(self, mocker):
        """Test cache initialization with no HPO terms data."""
        # Arrange
        import phentrieve.cli.similarity_commands as sim_module

        sim_module._cli_hpo_label_cache = None

        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[],  # No data
        )

        # Act
        cache = sim_module._ensure_cli_hpo_label_cache()

        # Assert
        assert cache == {}

    def test_handles_load_error_gracefully(self, mocker):
        """Test error handling when loading HPO terms fails."""
        # Arrange
        import phentrieve.cli.similarity_commands as sim_module

        sim_module._cli_hpo_label_cache = None

        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            side_effect=Exception("Load failed"),
        )

        # Act
        cache = sim_module._ensure_cli_hpo_label_cache()

        # Assert
        # Should return empty cache rather than crashing
        assert cache == {}

    def test_returns_existing_cache_if_already_loaded(self, mocker):
        """Test returns existing cache without reloading."""
        # Arrange
        import phentrieve.cli.similarity_commands as sim_module

        sim_module._cli_hpo_label_cache = {"HP:0000001": "Cached Term"}

        mock_load_terms = mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms"
        )

        # Act
        cache = sim_module._ensure_cli_hpo_label_cache()

        # Assert
        mock_load_terms.assert_not_called()  # Should not reload
        assert cache["HP:0000001"] == "Cached Term"


# =============================================================================
# Tests for hpo_similarity_cli()
# =============================================================================


class TestHpoSimilarityCli:
    """Test hpo_similarity_cli() command."""

    def setup_method(self):
        """Reset global cache before each test."""
        import phentrieve.cli.similarity_commands as sim_module

        sim_module._cli_hpo_label_cache = None

    def test_calculates_similarity_successfully_with_defaults(self, mocker):
        """Test successful similarity calculation with default parameters."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli
        from phentrieve.evaluation.metrics import SimilarityFormula

        # Arrange - Mock at the point of USE (similarity_commands), not point of definition
        mock_setup_logging = mocker.patch(
            "phentrieve.cli.similarity_commands.setup_logging_cli"
        )
        mock_load_graph = mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {
                    "HP:0000001": {"HP:0000001"},
                    "HP:0000002": {"HP:0000002"},
                },  # ancestors
                {"HP:0000001": 1, "HP:0000002": 1},  # depths
            ),
        )
        mock_load_terms = mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Seizure"},
                {"id": "HP:0000002", "label": "Tremor"},
            ],
        )
        mock_calc_similarity = mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.75,
        )
        mock_find_lca = mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=("HP:0000118", 2),
        )
        mock_echo = mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        # Assert
        mock_setup_logging.assert_called_once_with(debug=False)
        mock_load_graph.assert_called_once()
        mock_load_terms.assert_called_once()
        mock_calc_similarity.assert_called_once()
        assert (
            mock_calc_similarity.call_args.kwargs["formula"] == SimilarityFormula.HYBRID
        )
        mock_find_lca.assert_called_once()

        # Check output messages
        assert any(
            "HPO Term Similarity" in str(call.args[0])
            for call in mock_echo.call_args_list
        )
        assert any("0.75" in str(call.args[0]) for call in mock_secho.call_args_list)

    def test_calculates_similarity_with_different_formula(self, mocker):
        """Test similarity calculation with different formula."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli
        from phentrieve.evaluation.metrics import SimilarityFormula

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Term 1"},
                {"id": "HP:0000002", "label": "Term 2"},
            ],
        )
        mock_calc = mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.5,
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=("HP:0000118", 2),
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(
            term1_id="HP:0000001",
            term2_id="HP:0000002",
            formula_str="simple_resnik_like",
        )

        # Assert
        assert (
            mock_calc.call_args.kwargs["formula"]
            == SimilarityFormula.SIMPLE_RESNIK_LIKE
        )

    def test_calculates_similarity_with_debug_mode(self, mocker):
        """Test similarity calculation with debug logging enabled."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mock_setup_logging = mocker.patch(
            "phentrieve.cli.similarity_commands.setup_logging_cli"
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.5,
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=("HP:0000118", 2),
        )
        mocker.patch("typer.echo")
        mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002", debug=True)

        # Assert
        mock_setup_logging.assert_called_once_with(debug=True)

    def test_exits_when_graph_data_not_found(self, mocker):
        """Test exits with error when graph data is not found."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(None, None),  # No data found
        )
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        assert exc_info.value.exit_code == 1
        assert any(
            "not found" in str(call.args[0]).lower()
            for call in mock_secho.call_args_list
        )

    def test_exits_when_graph_data_empty(self, mocker):
        """Test exits with error when graph data is empty."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=({}, {}),  # Empty data
        )
        mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        assert exc_info.value.exit_code == 1

    def test_exits_when_graph_data_loading_fails(self, mocker):
        """Test exits with error when loading graph data raises exception."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            side_effect=Exception("Load failed"),
        )
        mock_secho = mocker.patch("typer.secho")

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        assert exc_info.value.exit_code == 1
        assert any(
            "failed" in str(call.args[0]).lower() for call in mock_secho.call_args_list
        )

    def test_exits_when_term_not_found_in_ontology(self, mocker):
        """Test exits with error when term not found in ontology."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}},
                {"HP:0000001": 1},  # Only HP:0000001 exists
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            hpo_similarity_cli(
                term1_id="HP:0000001", term2_id="HP:9999999"
            )  # Invalid term

        assert exc_info.value.exit_code == 1
        assert any(
            "not found" in str(call.args[0]).lower()
            for call in mock_secho.call_args_list
        )

    def test_exits_when_both_terms_not_found(self, mocker):
        """Test exits with error when both terms not found in ontology."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=({"HP:0000001": {"HP:0000001"}}, {"HP:0000001": 1}),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[],
        )
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            hpo_similarity_cli(term1_id="HP:9999998", term2_id="HP:9999999")

        assert exc_info.value.exit_code == 1
        # Should show error for both terms
        error_calls = [call.args[0] for call in mock_secho.call_args_list]
        assert sum("not found" in str(call).lower() for call in error_calls) >= 2

    def test_exits_when_similarity_calculation_fails(self, mocker):
        """Test exits with error when similarity calculation raises exception."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            side_effect=Exception("Calculation failed"),
        )
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act & Assert
        with pytest.raises(typer.Exit) as exc_info:
            hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        assert exc_info.value.exit_code == 1
        assert any(
            "error" in str(call.args[0]).lower() for call in mock_secho.call_args_list
        )

    def test_displays_lca_when_found(self, mocker):
        """Test displays LCA information when found."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[
                {"id": "HP:0000001", "label": "Term 1"},
                {"id": "HP:0000118", "label": "Phenotypic abnormality"},
            ],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.5,
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=("HP:0000118", 2),  # LCA found
        )
        mock_echo = mocker.patch("typer.echo")
        mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        # Assert
        output_lines = [str(call.args[0]) for call in mock_echo.call_args_list]
        assert any("Lowest Common Ancestor" in line for line in output_lines)
        assert any("HP:0000118" in line for line in output_lines)
        assert any("LCA Depth: 2" in line for line in output_lines)

    def test_displays_no_lca_when_not_found(self, mocker):
        """Test displays appropriate message when LCA not found."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.0,
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=(None, -1),  # No LCA found
        )
        mock_echo = mocker.patch("typer.echo")
        mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        # Assert
        output_lines = [str(call.args[0]) for call in mock_echo.call_args_list]
        assert any("Not found" in line or "unrelated" in line for line in output_lines)

    def test_color_codes_high_similarity(self, mocker):
        """Test uses green color for high similarity scores."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.9,  # High similarity
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=("HP:0000118", 2),
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        # Assert
        # Find the secho call with similarity score
        score_call = [
            call for call in mock_secho.call_args_list if "0.9" in str(call.args[0])
        ][0]
        assert score_call.kwargs["fg"] == typer.colors.BRIGHT_GREEN

    def test_color_codes_medium_similarity(self, mocker):
        """Test uses yellow color for medium similarity scores."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.3,  # Medium similarity
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=("HP:0000118", 2),
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        # Assert
        score_call = [
            call for call in mock_secho.call_args_list if "0.3" in str(call.args[0])
        ][0]
        assert score_call.kwargs["fg"] == typer.colors.YELLOW

    def test_color_codes_zero_similarity(self, mocker):
        """Test uses white color for zero similarity scores."""
        from phentrieve.cli.similarity_commands import hpo_similarity_cli

        # Arrange
        mocker.patch("phentrieve.cli.similarity_commands.setup_logging_cli")
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_graph_data",
            return_value=(
                {"HP:0000001": {"HP:0000001"}, "HP:0000002": {"HP:0000002"}},
                {"HP:0000001": 1, "HP:0000002": 1},
            ),
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.load_hpo_terms",
            return_value=[{"id": "HP:0000001", "label": "Term"}],
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.calculate_semantic_similarity",
            return_value=0.0,  # Zero similarity
        )
        mocker.patch(
            "phentrieve.cli.similarity_commands.find_lowest_common_ancestor",
            return_value=(None, -1),
        )
        mocker.patch("typer.echo")
        mock_secho = mocker.patch("typer.secho")
        mocker.patch(
            "phentrieve.cli.similarity_commands.normalize_id", side_effect=lambda x: x
        )

        # Act
        hpo_similarity_cli(term1_id="HP:0000001", term2_id="HP:0000002")

        # Assert
        score_call = [
            call for call in mock_secho.call_args_list if "0.0" in str(call.args[0])
        ][0]
        assert score_call.kwargs["fg"] == typer.colors.WHITE
