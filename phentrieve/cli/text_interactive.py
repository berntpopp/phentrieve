"""Interactive text processing mode for Phentrieve CLI.

This module contains the interactive text mode command for real-time
HPO term extraction with rich formatting and visualization.
"""

import logging
import time
import traceback
from typing import Annotated, Any, Optional

import typer

from phentrieve.cli.utils import resolve_chunking_pipeline_config
from phentrieve.config import DEFAULT_MODEL, DEFAULT_RERANKER_MODEL
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline
from phentrieve.utils import setup_logging_cli

logger = logging.getLogger(__name__)


def _display_interactive_text_results(
    raw_text: str,
    processed_chunks: list[dict],
    chunk_results: list[dict[str, Any]],
    aggregated_results: list[dict[str, Any]],
    show_annotations_above: bool = True,
) -> None:
    """Display text processing results with rich formatting.

    Shows chunks in a grid layout (2-3 per row depending on console width)
    with HPO annotations above or below each chunk, styled like annotated text.

    Args:
        raw_text: The original input text
        processed_chunks: List of processed text chunks with text and metadata
        chunk_results: List of chunk-level results with HPO matches per chunk
        aggregated_results: List of aggregated HPO terms across all chunks
        show_annotations_above: If True, show annotations above the chunk, else below
    """
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    console_width = console.width

    # Determine how many chunks per row based on console width
    if console_width >= 160:
        chunks_per_row = 3
    elif console_width >= 100:
        chunks_per_row = 2
    else:
        chunks_per_row = 1

    # Calculate width for each chunk panel
    panel_width = (console_width - (chunks_per_row + 1) * 2) // chunks_per_row

    # Create a mapping from chunk_idx to matches for easy lookup
    chunk_matches_map: dict[int, list[dict[str, Any]]] = {}
    for result in chunk_results:
        chunk_idx = result.get("chunk_idx")
        if chunk_idx is not None:
            chunk_matches_map[chunk_idx] = result.get("matches", [])

    # Display header
    console.print()

    # Build chunk panels
    chunk_panels = []
    for i, chunk_data in enumerate(processed_chunks):
        chunk_text = chunk_data.get("text", str(chunk_data))
        matches = chunk_matches_map.get(i, [])

        # Build annotation text (compact format)
        annotation_lines = []
        if matches:
            for match in matches[:5]:  # Limit to top 5 matches per chunk
                hpo_id = match.get("id", "")
                name = match.get("name", "")
                score = match.get("score", 0.0)
                # Format: "HP:0001234 Term Name (0.85)"
                annotation_lines.append(
                    f"[cyan]{hpo_id}[/cyan] [green]{name}[/green] [dim]({score:.2f})[/dim]"
                )

        # Create annotation text
        if annotation_lines:
            annotation_text = Text.from_markup("\n".join(annotation_lines))
        else:
            annotation_text = Text("—", style="dim italic")

        # Create chunk text
        styled_chunk = Text(chunk_text)

        # Determine panel style based on matches
        if matches:
            border_style = "green"
        else:
            border_style = "dim"

        # Build the panel content based on annotation position
        if show_annotations_above:
            # Annotation box above, then chunk text
            content = Group(
                Panel(
                    annotation_text,
                    border_style="blue",
                    padding=(0, 1),
                    title="[dim]annotations[/dim]" if matches else None,
                ),
                Panel(
                    styled_chunk,
                    border_style=border_style,
                    padding=(0, 1),
                ),
            )
        else:
            # Chunk text first, then annotation box below
            content = Group(
                Panel(
                    styled_chunk,
                    border_style=border_style,
                    padding=(0, 1),
                ),
                Panel(
                    annotation_text,
                    border_style="blue",
                    padding=(0, 1),
                    title="[bold blue]HPO Terms[/bold blue]"
                    if matches
                    else "[dim]no matches[/dim]",
                    title_align="left",
                ),
            )

        chunk_panels.append(content)

    # Display chunks in a grid layout
    for row_start in range(0, len(chunk_panels), chunks_per_row):
        row_end = min(row_start + chunks_per_row, len(chunk_panels))
        row_panels = chunk_panels[row_start:row_end]

        # Create a table for this row
        row_table = Table.grid(padding=(0, 1), expand=True)
        for _ in range(len(row_panels)):
            row_table.add_column(width=panel_width)

        row_table.add_row(*row_panels)
        console.print(row_table)
        console.print()  # Spacing between rows


def interactive_text_mode(
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Predefined chunking strategy (simple, semantic, detailed, sliding_window, sliding_window_cleaned, sliding_window_punct_cleaned, sliding_window_punct_conj_cleaned)",
        ),
    ] = "sliding_window_punct_conj_cleaned",
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            "-ws",
            help="Sliding window size in tokens (only for sliding_window strategy)",
        ),
    ] = 3,
    step_size: Annotated[
        int,
        typer.Option(
            "--step-size",
            "-ss",
            help="Sliding window step size in tokens (only for sliding_window strategy)",
        ),
    ] = 1,
    split_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Similarity threshold for splitting (0-1, only for sliding_window strategy)",
        ),
    ] = 0.5,
    min_segment_length: Annotated[
        int,
        typer.Option(
            "--min-segment",
            "-ms",
            help="Minimum segment length in words (only for sliding_window strategy)",
        ),
    ] = 2,
    semantic_chunker_model: Annotated[
        Optional[str],
        typer.Option(
            "--semantic-model",
            "--s-model",
            help=f"Model name for semantic chunking (default: {DEFAULT_MODEL})",
        ),
    ] = DEFAULT_MODEL,
    retrieval_model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model name for HPO term retrieval"),
    ] = None,
    chunk_retrieval_threshold: Annotated[
        float,
        typer.Option(
            "--chunk-retrieval-threshold",
            "-crt",
            help="Minimum similarity score for HPO term matches per chunk (0.0-1.0)",
        ),
    ] = 0.3,
    num_results: Annotated[
        int,
        typer.Option(
            "--num-results",
            "-n",
            help="Maximum number of HPO terms to return per chunk",
        ),
    ] = 5,
    no_assertion_detection: Annotated[
        bool,
        typer.Option(
            "--no-assertion",
            help="Disable assertion detection (treat all chunks as affirmed)",
        ),
    ] = False,
    assertion_preference: Annotated[
        str,
        typer.Option(
            "--assertion-preference",
            help="Assertion detection strategy preference (dependency, keyword, any_negative)",
        ),
    ] = "dependency",
    enable_reranker: Annotated[
        bool,
        typer.Option(
            "--enable-reranker",
            "--rerank",
            help="Enable cross-encoder reranking of results",
        ),
    ] = False,
    reranker_model: Annotated[
        Optional[str],
        typer.Option(
            "--reranker-model",
            help="Cross-encoder model for reranking (if reranking enabled)",
        ),
    ] = None,
    annotations_above: Annotated[
        bool,
        typer.Option(
            "--annotations-above",
            "-aa",
            help="Show annotations above chunks instead of below",
        ),
    ] = False,
    aggregated_term_confidence: Annotated[
        float,
        typer.Option(
            "--aggregated-term-confidence",
            "-atc",
            help="Minimum confidence score for aggregated HPO terms",
        ),
    ] = 0.35,
    top_term_per_chunk: Annotated[
        bool,
        typer.Option(
            "--top-term-per-chunk",
            help="Keep only the top term per chunk",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
) -> None:
    """Interactive text processing mode for HPO term extraction.

    This command enables an interactive session where you can enter clinical text
    and see the chunking, annotations, and HPO term matches displayed with rich
    formatting. Each chunk is highlighted with its annotations shown in a box
    below the chunk text.

    Example usage:
        phentrieve text interactive
        phentrieve text interactive -l de --annotations-above
        phentrieve text interactive -s semantic --enable-reranker
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    setup_logging_cli(debug=debug)

    # Display welcome message
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Phentrieve Interactive Text Analysis[/bold cyan]\n\n"
            "Enter clinical text to analyze for HPO terms.\n"
            "Each chunk will be displayed with its HPO annotations.\n\n"
            "[dim]Commands:[/dim]\n"
            "  • Type your text and press Enter to analyze\n"
            "  • Type 'q' or 'quit' to exit\n"
            "  • Type '!multi' to enter multi-line mode (end with empty line)\n"
            "  • Type '!toggle' to toggle annotation position (below/above)\n"
            "  • Type '!p' to export results as JSON phenopacket",
            border_style="cyan",
        )
    )
    console.print()

    # Get chunking pipeline configuration
    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file=None,
        strategy_arg=strategy,
        window_size=window_size,
        step_size=step_size,
        threshold=split_threshold,
        min_segment_length=min_segment_length,
    )

    # Assertion detection configuration
    assertion_config = {
        "disable": no_assertion_detection,
        "strategy_preference": assertion_preference,
    }

    # Determine if we need a semantic model for chunking
    needs_semantic_model = False
    for chunk_config in chunking_pipeline_config:
        chunk_type = (
            chunk_config.get("type") if isinstance(chunk_config, dict) else chunk_config
        )
        if chunk_type in [
            "semantic",
            "pre_chunk_semantic_grouper",
            "sliding_window_semantic",
            "sliding_window",
        ]:
            needs_semantic_model = True
            break

    # Determine model names
    semantic_model_name = semantic_chunker_model or DEFAULT_MODEL
    retrieval_model_name = retrieval_model or DEFAULT_MODEL

    # Check for GPU availability
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Optimization: If same model is needed for both chunking and retrieval,
    # load it once and share the instance
    use_same_model = (
        needs_semantic_model and semantic_model_name == retrieval_model_name
    )

    # Initialize models once at startup
    console.print("[dim]Loading models...[/dim]")

    # Load the SBERT model for chunking if needed
    sbert_model_for_chunking = None
    if needs_semantic_model:
        from phentrieve.embeddings import load_embedding_model

        try:
            if use_same_model:
                console.print(
                    f"[dim]Loading model (shared): {semantic_model_name}[/dim]"
                )
            else:
                console.print(
                    f"[dim]Loading chunking model: {semantic_model_name}[/dim]"
                )

            sbert_model_for_chunking = load_embedding_model(
                model_name=semantic_model_name,
                device=device if use_same_model else None,
            )
        except Exception as e:
            console.print(f"[red]Error loading chunking model: {e}[/red]")
            raise typer.Exit(code=1)

    # Load the retrieval model
    try:
        from phentrieve.embeddings import load_embedding_model

        if use_same_model and sbert_model_for_chunking is not None:
            st_model = sbert_model_for_chunking
        else:
            console.print(f"[dim]Loading retrieval model: {retrieval_model_name}[/dim]")
            st_model = load_embedding_model(
                model_name=retrieval_model_name,
                device=device,
            )

        retriever = DenseRetriever.from_model_name(
            model=st_model, model_name=retrieval_model_name
        )

        if not retriever:
            console.print("[red]Failed to initialize retriever[/red]")
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error initializing retriever: {e}[/red]")
        raise typer.Exit(code=1)

    # Load cross-encoder if reranking is enabled
    cross_encoder = None
    if enable_reranker:
        try:
            from sentence_transformers import CrossEncoder

            reranker_to_use = reranker_model or DEFAULT_RERANKER_MODEL
            console.print(f"[dim]Loading reranker: {reranker_to_use}[/dim]")
            cross_encoder = CrossEncoder(reranker_to_use)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load reranker: {e}[/yellow]")

    console.print("[green]Models loaded successfully![/green]")
    console.print()

    # Track annotation position
    show_annotations_above = annotations_above

    # Store last results for export
    last_chunk_results = None
    last_processed_chunks = None

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]Enter text[/bold cyan]")

            # Handle commands
            if user_input.lower() in ["q", "quit", "exit"]:
                console.print("[dim]Exiting...[/dim]")
                break

            if user_input.lower() == "!toggle":
                show_annotations_above = not show_annotations_above
                position = "above" if show_annotations_above else "below"
                console.print(
                    f"[cyan]Annotations will now appear {position} chunks[/cyan]"
                )
                continue

            if user_input.lower() == "!p":
                if last_chunk_results is None or last_processed_chunks is None:
                    console.print(
                        "[yellow]No results to export. Process some text first.[/yellow]"
                    )
                    continue

                # Export as phenopacket JSON
                try:
                    from phentrieve.phenopackets.utils import format_as_phenopacket_v2

                    # Build chunk_results with chunk_text for phenopacket export
                    chunk_results_with_text = []
                    for i, chunk_result in enumerate(last_chunk_results):
                        chunk_text = ""
                        if i < len(last_processed_chunks):
                            chunk_data = last_processed_chunks[i]
                            if isinstance(chunk_data, str):
                                chunk_text = chunk_data
                            else:
                                chunk_text = chunk_data.get("text", str(chunk_data))

                        chunk_results_with_text.append(
                            {
                                "chunk_idx": chunk_result.get("chunk_idx", i),
                                "chunk_text": chunk_text,
                                "matches": chunk_result.get("matches", []),
                            }
                        )

                    phenopacket_json = format_as_phenopacket_v2(
                        chunk_results=chunk_results_with_text
                    )
                    console.print("[green]Phenopacket JSON:[/green]")
                    console.print(phenopacket_json)
                except Exception as e:
                    console.print(f"[red]Error exporting phenopacket: {e}[/red]")
                    if debug:
                        traceback.print_exc()
                continue

            if user_input.lower() == "!multi":
                console.print(
                    "[dim]Enter your text (press Enter twice to finish):[/dim]"
                )
                lines = []
                while True:
                    line = Prompt.ask("", default="")
                    if line == "":
                        break
                    lines.append(line)
                user_input = "\n".join(lines)

            if not user_input.strip():
                continue

            # Process the text
            start_time = time.time()

            # Create the pipeline for this text
            try:
                pipeline = TextProcessingPipeline(
                    language=language,
                    chunking_pipeline_config=chunking_pipeline_config,
                    assertion_config=assertion_config,
                    sbert_model_for_semantic_chunking=sbert_model_for_chunking,
                )
            except Exception as e:
                console.print(f"[red]Error creating pipeline: {e}[/red]")
                if debug:
                    traceback.print_exc()
                continue

            # Process the text through the pipeline
            try:
                processed_chunks = pipeline.process(user_input)
            except Exception as e:
                console.print(f"[red]Error processing text: {e}[/red]")
                if debug:
                    traceback.print_exc()
                continue

            # Extract HPO terms from the processed chunks
            try:
                # Extract text and assertion statuses from processed chunks
                text_chunks: list[str] = []
                assertion_statuses: list[str | None] = []
                for chunk in processed_chunks:
                    if isinstance(chunk, str):
                        text_chunks.append(chunk)
                        assertion_statuses.append(None)
                    else:
                        text_chunks.append(chunk.get("text", str(chunk)))
                        status = chunk.get("status")
                        if status is not None and hasattr(status, "value"):
                            status = status.value
                        assertion_statuses.append(
                            str(status) if status is not None else None
                        )

                aggregated_results, chunk_results = orchestrate_hpo_extraction(
                    text_chunks=text_chunks,
                    retriever=retriever,
                    num_results_per_chunk=num_results,
                    chunk_retrieval_threshold=chunk_retrieval_threshold,
                    cross_encoder=cross_encoder,
                    language=language,
                    top_term_per_chunk=top_term_per_chunk,
                    min_confidence_for_aggregated=aggregated_term_confidence,
                    assertion_statuses=assertion_statuses,
                )
            except Exception as e:
                console.print(f"[red]Error extracting HPO terms: {e}[/red]")
                if debug:
                    traceback.print_exc()
                continue

            elapsed_time = time.time() - start_time

            # Store results for export
            last_chunk_results = chunk_results
            last_processed_chunks = processed_chunks

            # Display the results with rich formatting
            _display_interactive_text_results(
                raw_text=user_input,
                processed_chunks=processed_chunks,
                chunk_results=chunk_results,
                aggregated_results=aggregated_results,
                show_annotations_above=show_annotations_above,
            )

            console.print(
                f"[dim]Processed {len(processed_chunks)} chunks in {elapsed_time:.2f}s[/dim]"
            )

        except KeyboardInterrupt:
            console.print("\n[dim]Exiting...[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if debug:
                traceback.print_exc()
