"""Data-related commands for Phentrieve CLI.

This module contains commands for managing HPO data:
- prepare: Download and prepare HPO data locally
- download: Download pre-built data bundles from GitHub Releases
- bundle: Create and manage data bundles
- status: Show current data installation status
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

# Create the Typer app for this command group
app = typer.Typer()


@app.command("prepare")
def prepare_hpo_data(
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Force update even if files exist")
    ] = False,
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Custom directory for HPO data storage"),
    ] = None,
    include_obsolete: Annotated[
        bool,
        typer.Option(
            "--include-obsolete",
            help="Include obsolete HPO terms (for analysis). Default: filter out.",
        ),
    ] = False,
):
    """Prepare HPO data for indexing.

    Downloads the HPO ontology data, extracts terms, and precomputes
    graph properties needed for similarity calculations.

    By default, obsolete HPO terms are filtered out to prevent retrieval
    errors (Issue #133). Use --include-obsolete for analysis purposes.
    """
    from phentrieve.data_processing.hpo_parser import orchestrate_hpo_preparation
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    typer.echo("Starting HPO data preparation...")
    if include_obsolete:
        typer.echo("Note: Including obsolete terms (--include-obsolete flag set)")

    success = orchestrate_hpo_preparation(
        debug=debug,
        force_update=force,
        data_dir_override=data_dir,
        include_obsolete=include_obsolete,
    )

    if success:
        typer.secho(
            "HPO data preparation completed successfully!", fg=typer.colors.GREEN
        )
    else:
        typer.secho("HPO data preparation failed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("download")
def download_bundle(
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name or slug (e.g., 'biolord', 'FremyCompany/BioLORD-2023-M'). "
            "Use 'minimal' for database only.",
        ),
    ] = "biolord",
    hpo_version: Annotated[
        Optional[str],
        typer.Option("--hpo-version", help="Specific HPO version (default: latest)"),
    ] = None,
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Custom directory for data installation"),
    ] = None,
    skip_verify: Annotated[
        bool,
        typer.Option("--skip-verify", help="Skip checksum verification"),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Download pre-built HPO data bundle from GitHub Releases.

    Pre-built bundles contain the HPO database and optionally pre-computed
    vector indexes for faster setup (Issue #117).

    Available models: biolord, bge-m3, labse, minimal (db only)

    Examples:
        phentrieve data download --model biolord
        phentrieve data download --model minimal
        phentrieve data download --model bge-m3 --hpo-version v2025-03-03
    """
    from phentrieve.data_processing.bundle_downloader import (
        download_and_extract_bundle,
        find_bundle,
    )
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    # Handle model name
    model_name = None if model == "minimal" else model

    typer.echo(
        f"Searching for bundle: model={model or 'minimal'}, hpo_version={hpo_version or 'latest'}..."
    )

    # Find bundle first to show info
    bundle = find_bundle(model_name=model_name, hpo_version=hpo_version)
    if not bundle:
        typer.secho(
            "No matching bundle found. Try 'phentrieve data list-bundles' to see available bundles.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Found: {bundle.name} ({bundle.size / 1024 / 1024:.1f} MB)")

    # Progress callback for download
    def progress_callback(downloaded: int, total: int) -> None:
        percent = (downloaded / total) * 100 if total > 0 else 0
        typer.echo(f"\rDownloading: {percent:.1f}%", nl=False)

    target_dir = Path(data_dir) if data_dir else None

    typer.echo("Starting download...")
    manifest = download_and_extract_bundle(
        model_name=model_name,
        hpo_version=hpo_version,
        target_data_dir=target_dir,
        verify_checksums=not skip_verify,
        progress_callback=progress_callback,
    )

    typer.echo("")  # Newline after progress

    if manifest:
        typer.secho(
            f"✓ Successfully installed HPO {manifest.hpo_version} "
            f"({manifest.active_terms} active terms)",
            fg=typer.colors.GREEN,
        )
        if manifest.model:
            typer.echo(f"  Model: {manifest.model.name}")
    else:
        typer.secho("Download failed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command("list-bundles")
def list_bundles(
    include_prereleases: Annotated[
        bool,
        typer.Option("--prereleases", help="Include pre-release versions"),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum releases to list"),
    ] = 5,
):
    """List available pre-built data bundles from GitHub Releases.

    Shows available bundles with their HPO versions and model types.
    """
    from phentrieve.data_processing.bundle_downloader import list_available_releases

    typer.echo("Fetching available bundles from GitHub Releases...\n")

    releases = list_available_releases(
        include_prereleases=include_prereleases,
        limit=limit,
    )

    if not releases:
        typer.secho("No releases found with data bundles.", fg=typer.colors.YELLOW)
        typer.echo("\nTo build bundles locally:")
        typer.echo("  phentrieve data prepare")
        typer.echo(
            "  phentrieve index build --model-name 'FremyCompany/BioLORD-2023-M'"
        )
        typer.echo(
            "  phentrieve data bundle create --model 'FremyCompany/BioLORD-2023-M'"
        )
        return

    for release in releases:
        typer.secho(f"\n{release.tag_name}", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"  Published: {release.published_at[:10]}")

        for bundle in release.bundles:
            size_mb = bundle.size / 1024 / 1024
            typer.echo(
                f"  • {bundle.model_slug or 'unknown'}: "
                f"{bundle.name} ({size_mb:.1f} MB)"
            )


@app.command("status")
def data_status(
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Data directory to check"),
    ] = None,
):
    """Show current data installation status.

    Displays information about installed HPO data and bundles.
    """
    from phentrieve.config import DEFAULT_HPO_DB_FILENAME
    from phentrieve.data_processing.bundle_downloader import (
        check_for_updates,
        get_installed_bundle_info,
    )
    from phentrieve.utils import get_default_data_dir

    target_dir = Path(data_dir) if data_dir else get_default_data_dir()

    typer.secho("Data Status", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Data directory: {target_dir}")
    typer.echo("")

    # Check for database
    db_path = target_dir / DEFAULT_HPO_DB_FILENAME
    if db_path.exists():
        db_size = db_path.stat().st_size / 1024 / 1024
        typer.secho(
            f"✓ Database: {db_path.name} ({db_size:.1f} MB)", fg=typer.colors.GREEN
        )
    else:
        typer.secho("✗ Database: Not found", fg=typer.colors.RED)

    # Check for indexes
    index_dir = target_dir / "indexes"
    if index_dir.exists():
        indexes = list(index_dir.iterdir())
        if indexes:
            typer.secho(f"✓ Indexes: {len(indexes)} found", fg=typer.colors.GREEN)
            for idx in indexes:
                typer.echo(f"    • {idx.name}")
        else:
            typer.secho("○ Indexes: None built", fg=typer.colors.YELLOW)
    else:
        typer.secho("○ Indexes: None built", fg=typer.colors.YELLOW)

    # Check manifest
    manifest = get_installed_bundle_info(target_dir)
    if manifest:
        typer.echo("")
        typer.secho("Bundle Info:", fg=typer.colors.CYAN)
        typer.echo(f"  HPO Version: {manifest.hpo_version}")
        typer.echo(f"  Active Terms: {manifest.active_terms}")
        typer.echo(f"  Created: {manifest.created_at[:10]}")
        if manifest.model:
            typer.echo(f"  Model: {manifest.model.name}")

        # Check for updates
        update_available, message = check_for_updates(target_dir)
        if update_available:
            typer.secho(f"\n↑ {message}", fg=typer.colors.YELLOW)


# Bundle subcommand group
bundle_app = typer.Typer(help="Create and manage data bundles")
app.add_typer(bundle_app, name="bundle")


@bundle_app.command("create")
def create_bundle_cmd(
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Output directory for bundle"),
    ] = "./dist",
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name for index bundle (omit for minimal bundle)",
        ),
    ] = None,
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Source data directory"),
    ] = None,
    include_hp_json: Annotated[
        bool,
        typer.Option("--include-hp-json", help="Include original hp.json file"),
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging")
    ] = False,
):
    """Create a data bundle for distribution.

    Creates a tar.gz bundle containing the HPO database and optionally
    pre-computed vector indexes.

    Examples:
        # Create minimal bundle (database only)
        phentrieve data bundle create

        # Create bundle with BioLORD index
        phentrieve data bundle create --model 'FremyCompany/BioLORD-2023-M'
    """
    from phentrieve.data_processing.bundle_packager import create_bundle
    from phentrieve.utils import setup_logging_cli

    setup_logging_cli(debug=debug)

    output_path = Path(output_dir)
    source_dir = Path(data_dir) if data_dir else None

    bundle_type = f"with model {model}" if model else "minimal (database only)"
    typer.echo(f"Creating {bundle_type} bundle...")

    try:
        bundle_path = create_bundle(
            output_dir=output_path,
            model_name=model,
            data_dir=source_dir,
            include_hpo_json=include_hp_json,
        )

        bundle_size = bundle_path.stat().st_size / 1024 / 1024
        typer.secho(
            f"✓ Created bundle: {bundle_path} ({bundle_size:.1f} MB)",
            fg=typer.colors.GREEN,
        )

    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        typer.echo("\nMake sure to run 'phentrieve data prepare' first.")
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@bundle_app.command("list")
def list_local_bundles(
    data_dir: Annotated[
        Optional[str],
        typer.Option("--data-dir", help="Data directory to check"),
    ] = None,
):
    """List bundles that can be created from local data.

    Shows which embedding models have indexes ready for bundling.
    """
    from phentrieve.data_processing.bundle_packager import list_available_bundles

    source_dir = Path(data_dir) if data_dir else None

    typer.secho("Available Bundles", fg=typer.colors.CYAN, bold=True)
    typer.echo("")

    bundles = list_available_bundles(data_dir=source_dir)

    for bundle in bundles:
        status = bundle["status"]
        model_slug = bundle["model_slug"]
        model_name = bundle["model_name"] or "(minimal - database only)"

        if status == "ready":
            typer.secho(f"✓ {model_slug}", fg=typer.colors.GREEN)
            typer.echo(f"    {model_name}")
        elif status == "missing_db":
            typer.secho(f"✗ {model_slug}", fg=typer.colors.RED)
            typer.echo(f"    {model_name}")
            typer.echo("    Missing: Run 'phentrieve data prepare'")
        else:
            typer.secho(f"○ {model_slug}", fg=typer.colors.YELLOW)
            typer.echo(f"    {model_name}")
            typer.echo(
                f"    Missing: Run 'phentrieve index build --model-name \"{model_name}\"'"
            )
