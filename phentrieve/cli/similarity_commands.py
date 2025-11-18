"""
Similarity calculation tools for HPO terms.

This module provides CLI commands for calculating semantic similarity between HPO terms
using the Human Phenotype Ontology graph structure.
"""

import logging
from typing import Annotated, Optional

import typer

from phentrieve.config import DEFAULT_SIMILARITY_FORMULA  # Default formula
from phentrieve.data_processing.document_creator import (
    load_hpo_terms,
)  # For fetching labels

# Import necessary Phentrieve components
from phentrieve.evaluation.metrics import (
    SimilarityFormula,  # For type hinting and choices
    calculate_semantic_similarity,
    find_lowest_common_ancestor,
    load_hpo_graph_data,  # This handles its own caching
)
from phentrieve.utils import normalize_id, setup_logging_cli

logger = logging.getLogger(__name__)
app = typer.Typer(name="similarity", help="Tools for HPO term similarity calculations.")

# Simple in-memory cache for HPO labels for CLI (loaded once per command run if needed)
_cli_hpo_label_cache: Optional[dict[str, str]] = None


def _ensure_cli_hpo_label_cache() -> dict[str, str]:
    """
    Loads HPO term labels into a cache if not already loaded.

    Uses the optimized get_label_map() from HPODatabase for efficient label-only loading.
    Falls back to load_hpo_terms() if database method fails.
    """
    global _cli_hpo_label_cache
    if _cli_hpo_label_cache is None:
        logger.info("CLI: Initializing HPO label cache for similarity command...")
        try:
            # Try optimized database label loading first
            from phentrieve.config import DEFAULT_HPO_DB_FILENAME
            from phentrieve.data_processing.hpo_database import HPODatabase
            from phentrieve.utils import get_default_data_dir

            data_dir = get_default_data_dir()
            db_path = data_dir / DEFAULT_HPO_DB_FILENAME

            if db_path.exists():
                db = HPODatabase(db_path)
                _cli_hpo_label_cache = db.get_label_map()
                db.close()
                logger.info(
                    f"CLI: HPO label cache initialized with {len(_cli_hpo_label_cache)} terms from database."
                )
            else:
                # Fallback to loading all terms if database not found
                logger.warning(
                    "CLI: Database not found, falling back to load_hpo_terms()..."
                )
                hpo_terms_data = load_hpo_terms()
                if not hpo_terms_data:
                    logger.warning(
                        "CLI: No HPO terms data found for label lookup. Labels will be 'Unknown Label'."
                    )
                    _cli_hpo_label_cache = {}
                else:
                    _cli_hpo_label_cache = {
                        term_data["id"]: term_data["label"]
                        for term_data in hpo_terms_data
                        if term_data.get("id") and term_data.get("label")
                    }
                    logger.info(
                        f"CLI: HPO label cache initialized with {len(_cli_hpo_label_cache)} terms."
                    )
        except Exception as e:
            logger.error(f"CLI: Failed to load HPO terms for labels: {e}")
            _cli_hpo_label_cache = {}  # Ensure it's initialized to prevent repeated attempts
    return _cli_hpo_label_cache


@app.command("calculate", help="Calculate semantic similarity between two HPO terms.")
def hpo_similarity_cli(
    term1_id: Annotated[
        str,
        typer.Argument(
            help="First HPO Term ID (e.g., HP:0000123). Case-sensitive as per HPO standard."
        ),
    ],
    term2_id: Annotated[
        str,
        typer.Argument(
            help="Second HPO Term ID (e.g., HP:0000456). Case-sensitive as per HPO standard."
        ),
    ],
    formula_str: Annotated[
        str,
        typer.Option(
            "--formula",
            "-f",
            help="Similarity formula to use ('hybrid' or 'simple_resnik_like').",
            case_sensitive=False,  # Allows user to type 'hybrid' or 'Hybrid'
        ),
    ] = DEFAULT_SIMILARITY_FORMULA,  # Default from config.py
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging.")
    ] = False,
):
    """
    Calculates and displays the semantic similarity between two HPO terms,
    optionally showing Lowest Common Ancestor (LCA) details.
    """
    setup_logging_cli(debug=debug)

    # Convert formula string to enum
    formula = SimilarityFormula.from_string(formula_str)
    logger.info(
        f"CLI: Request to calculate similarity for '{term1_id}' and '{term2_id}' using formula '{formula.value}'"
    )

    # 1. Load necessary HPO graph data (ancestors and depths)
    # load_hpo_graph_data() handles its own caching.
    try:
        ancestors, depths = load_hpo_graph_data()
        if not ancestors or not depths:
            typer.secho(
                "CLI Error: Core HPO graph data (ancestors.pkl or depths.pkl) not found or failed to load.",
                fg=typer.colors.RED,
            )
            typer.secho(
                "Please ensure Phentrieve data is prepared by running 'phentrieve data prepare'.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"CLI Error: Failed to load HPO graph data: {e}", fg=typer.colors.RED
        )
        logger.error("CLI: Exception during HPO graph data loading.", exc_info=True)
        raise typer.Exit(code=1)

    # 2. Load HPO term labels for richer output
    hpo_labels = _ensure_cli_hpo_label_cache()

    # 3. Normalize input HPO IDs (though typically users should provide correct format)
    norm_term1 = normalize_id(term1_id)
    norm_term2 = normalize_id(term2_id)

    term1_display_label = hpo_labels.get(norm_term1, "Label not found")
    term2_display_label = hpo_labels.get(norm_term2, "Label not found")

    # 4. Validate that terms exist in the loaded ontology data (depths map is a good proxy)
    error_messages = []
    if norm_term1 not in depths:
        error_messages.append(
            f"Term '{norm_term1}' (from input '{term1_id}') not found in the HPO ontology data."
        )
    if norm_term2 not in depths:
        error_messages.append(
            f"Term '{norm_term2}' (from input '{term2_id}') not found in the HPO ontology data."
        )

    if error_messages:
        for msg in error_messages:
            typer.secho(f"CLI Error: {msg}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 5. Calculate similarity using the core function from metrics.py
    try:
        # The 'formula' parameter is already a SimilarityFormula enum instance due to Typer's conversion.
        similarity_score = calculate_semantic_similarity(
            norm_term1, norm_term2, formula=formula
        )

        # 6. Find LCA for contextual output
        lca_id_str, lca_depth_val = find_lowest_common_ancestor(
            norm_term1, norm_term2, ancestors_dict=ancestors
        )
        lca_display_label = hpo_labels.get(lca_id_str, "") if lca_id_str else ""

    except Exception as e:
        typer.secho(
            f"CLI Error: An unexpected error occurred during similarity calculation: {e}",
            fg=typer.colors.RED,
        )
        logger.error(
            f"CLI: Unexpected error calculating similarity for {norm_term1} & {norm_term2}.",
            exc_info=True,
        )
        raise typer.Exit(code=1)

    # 7. Print formatted output
    typer.echo("\n--- HPO Term Similarity ---")
    typer.echo(f"Term 1: {norm_term1} ({term1_display_label})")
    typer.echo(f"Term 2: {norm_term2} ({term2_display_label})")
    typer.echo(f"Formula Used: {formula.value}")
    typer.secho(
        f"Semantic Similarity Score: {similarity_score:.4f}",
        fg=(
            typer.colors.BRIGHT_GREEN
            if similarity_score >= 0.5
            else typer.colors.YELLOW
            if similarity_score > 0
            else typer.colors.WHITE
        ),
    )

    if lca_id_str and lca_depth_val != -1:
        typer.echo(f"Lowest Common Ancestor (LCA): {lca_id_str} ({lca_display_label})")
        typer.echo(f"LCA Depth: {lca_depth_val}")
    else:
        typer.echo("Lowest Common Ancestor (LCA): Not found (terms may be unrelated).")
