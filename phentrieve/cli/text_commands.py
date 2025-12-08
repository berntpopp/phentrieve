"""Text processing commands for Phentrieve CLI.

This module contains commands for text processing and HPO term extraction.
"""

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Annotated, Optional

import typer

# NOTE: SentenceTransformer is NOT imported at module level to avoid slow startup.
# Importing sentence_transformers loads PyTorch/CUDA (18+ seconds), which should
# only happen when commands actually need ML models, not for --help or --version.
# The import is done inside command functions where the model is actually used.
from phentrieve.cli.utils import load_text_from_input, resolve_chunking_pipeline_config
from phentrieve.config import DEFAULT_MODEL
from phentrieve.retrieval.dense_retriever import DenseRetriever
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

logger = logging.getLogger(__name__)

# Create the Typer app for this command group
app = typer.Typer()


@app.command()
def interactive(
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
    from phentrieve.cli.text_interactive import interactive_text_mode

    interactive_text_mode(
        language=language,
        strategy=strategy,
        window_size=window_size,
        step_size=step_size,
        split_threshold=split_threshold,
        min_segment_length=min_segment_length,
        semantic_chunker_model=semantic_chunker_model,
        retrieval_model=retrieval_model,
        chunk_retrieval_threshold=chunk_retrieval_threshold,
        num_results=num_results,
        no_assertion_detection=no_assertion_detection,
        assertion_preference=assertion_preference,
        enable_reranker=enable_reranker,
        reranker_model=reranker_model,
        annotations_above=annotations_above,
        aggregated_term_confidence=aggregated_term_confidence,
        top_term_per_chunk=top_term_per_chunk,
        debug=debug,
    )


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            "-i",
            help="Enable interactive mode for continuous text processing",
        ),
    ] = False,
):
    """Text processing commands for clinical text analysis."""
    if ctx.invoked_subcommand is None:
        if interactive:
            # Call interactive mode directly
            from phentrieve.cli.text_interactive import interactive_text_mode

            interactive_text_mode()
        else:
            # Show help if no subcommand and not interactive
            typer.echo(ctx.get_help())
            raise typer.Exit()


@app.command("process")
def process_text_for_hpo_command(
    text: str = typer.Argument(
        None,
        help="Text to process for HPO term extraction (can be a string or file path). Not required in interactive mode.",
    ),
    input_file: Annotated[
        Optional[Path],
        typer.Option(
            "--input-file", "-i", help="File to read text from instead of command line"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    chunking_pipeline_config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config-file",
            "-c",
            help="Path to YAML or JSON file with chunking pipeline configuration",
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Predefined chunking strategy. 'simple': paragraph then sentence. 'semantic': paragraph, sentence, then semantic splitting of sentences. 'detailed': paragraph, sentence, punctuation splitting, then semantic splitting of fragments. 'sliding_window': customizable semantic sliding window. 'sliding_window_cleaned': sliding window with final chunk cleaning. 'sliding_window_punct_cleaned': sliding window with punctuation splitting and final cleaning. 'sliding_window_punct_conj_cleaned': sliding window with punctuation, conjunction splitting, and final cleaning (choices: simple, semantic, detailed, sliding_window, sliding_window_cleaned, sliding_window_punct_cleaned, sliding_window_punct_conj_cleaned)",
        ),
    ] = "sliding_window_punct_conj_cleaned",  # Default: advanced chunking with punctuation and conjunction splitting
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
            help=f"Model name for semantic chunking (for sliding_window strategy, default: {DEFAULT_MODEL})",
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
            help="Minimum similarity score for an HPO term to be considered a match for an individual text chunk (0.0-1.0)",
        ),
    ] = 0.3,
    num_results: Annotated[
        int,
        typer.Option(
            "--num-results",
            "-n",
            help="Maximum number of HPO terms to return per query",
        ),
    ] = 10,
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
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "-o",
            help="Output format for results (json_lines, rich_json_summary, csv_hpo_list, phenopacket_v2_json)",
        ),
    ] = "json_lines",
    cross_language_hpo_retrieval: Annotated[
        bool,
        typer.Option(
            "--cross-language-hpo-retrieval",
            "--xlhpo",
            help="Use cross-language HPO term retrieval (if text is not in English)",
        ),
    ] = True,
    use_cached_index: Annotated[
        bool,
        typer.Option(
            "--use-cached-index",
            help="Use cached HPO term index if available",
        ),
    ] = True,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
    chunk_confidence: Annotated[
        float,
        typer.Option(
            "--chunk-confidence",
            "-cc",
            help="Minimum confidence score for a term to be included in a chunk result",
        ),
    ] = 0.2,
    aggregated_term_confidence: Annotated[
        float,
        typer.Option(
            "--aggregated-term-confidence",
            "-atc",
            help="Minimum confidence score for an aggregated HPO term to be included in the final results",
        ),
    ] = 0.35,
    top_term_per_chunk: Annotated[
        bool,
        typer.Option(
            "--top-term-per-chunk",
            help="Keep only the top term per chunk",
        ),
    ] = False,
    include_details: Annotated[
        bool,
        typer.Option(
            "--include-details",
            "-d",
            help="Include HPO term definitions and synonyms in output",
        ),
    ] = False,
) -> None:
    """Process clinical text to extract HPO terms.

    This command processes clinical texts through a chunking pipeline and assertion
    detection, then extracts HPO terms from each chunk. Results are aggregated to provide
    a comprehensive set of terms with evidence.

    Example usage:
    - Process direct text input: phentrieve text process "Patient has severe headaches"
    - Process from file with semantic chunking: phentrieve text process -s semantic -i clinical_note.txt
    - Process German text with specialized model: phentrieve text process -l de -m "Jina-v2-base-de" -i german_note.txt
    """
    import time

    from phentrieve.config import (
        DEFAULT_LANGUAGE,
        DEFAULT_MODEL,
    )
    from phentrieve.text_processing.pipeline import TextProcessingPipeline
    from phentrieve.utils import detect_language, setup_logging_cli

    # Use module-level logging import (line 8) instead of re-importing
    logger = logging.getLogger(__name__)
    start_time = time.time()
    setup_logging_cli(debug=debug)

    # Load the raw text using helper function
    raw_text = load_text_from_input(text, input_file)

    # Detect or set the language
    if not language:
        # Try to auto-detect the language
        try:
            language = detect_language(raw_text, default_lang=DEFAULT_LANGUAGE)
            typer.echo(f"Auto-detected language: {language}")
        except ImportError:
            language = DEFAULT_LANGUAGE
            typer.echo(f"Using default language: {language}")

    # Get chunking pipeline configuration using helper function
    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file=chunking_pipeline_config_file,
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

    # Output the configuration
    logger.debug(f"Chunking pipeline config: {chunking_pipeline_config}")
    logger.debug(f"Assertion config: {assertion_config}")

    # Determine the bi-encoder model to use
    model_name = retrieval_model or DEFAULT_MODEL
    logger.info(f"Using model: {model_name}")

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

    # Determine model names for chunking and retrieval
    semantic_model_name = semantic_chunker_model or DEFAULT_MODEL
    retrieval_model_name = retrieval_model or DEFAULT_MODEL

    # Check for GPU availability
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Optimization: If same model is needed for both chunking and retrieval,
    # load it once and share the instance (avoids double loading into VRAM)
    use_same_model = (
        needs_semantic_model and semantic_model_name == retrieval_model_name
    )

    # Load the SBERT model if needed for semantic chunking
    sbert_model_for_chunking = None
    if needs_semantic_model:
        # Lazy import - only load heavy ML dependencies when actually needed
        from phentrieve.embeddings import load_embedding_model

        if use_same_model:
            # Load once and share for both purposes (memory optimization)
            typer.echo(
                f"Loading sentence transformer model (shared for chunking and retrieval): {semantic_model_name}..."
            )
        else:
            typer.echo(
                f"Loading sentence transformer model for chunking: {semantic_model_name}..."
            )

        try:
            sbert_model_for_chunking = load_embedding_model(
                model_name=semantic_model_name,
                device=device
                if use_same_model
                else None,  # Use GPU if sharing, CPU-only for chunking-only
            )
            logger.debug(
                f"Successfully loaded model for chunking: {semantic_model_name}"
            )
        except Exception as e:
            typer.secho(
                f"Error loading semantic chunker model '{semantic_model_name}': {e!s}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    # Initialize text processing pipeline
    try:
        pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model_for_chunking,
        )
    except Exception as e:
        typer.secho(f"Error creating pipeline: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Process the text through the pipeline
    try:
        processed_chunks = pipeline.process(raw_text)
    except Exception as e:
        typer.secho(f"Error processing text: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Processed {len(processed_chunks)} text chunks.")

    # Extract HPO terms from the processed chunks
    try:
        # Initialize the retriever
        try:
            # Lazy import - only load heavy ML dependencies when actually needed
            from phentrieve.embeddings import load_embedding_model

            # Load the SentenceTransformer model (reuses cached instance if already loaded)
            if use_same_model and sbert_model_for_chunking is not None:
                # Reuse the model we already loaded for chunking (memory optimization)
                logger.info(
                    f"Reusing cached model instance for retrieval: {retrieval_model_name}"
                )
                st_model = sbert_model_for_chunking
            else:
                # Load the model fresh (or from cache if previously loaded)
                logger.info(
                    f"Loading SentenceTransformer model for retrieval: {retrieval_model_name}"
                )
                st_model = load_embedding_model(
                    model_name=retrieval_model_name,
                    device=device,
                )

            # Initialize the retriever with the model
            retriever = DenseRetriever.from_model_name(
                model=st_model, model_name=retrieval_model_name
            )

            if not retriever:
                typer.secho(
                    f"Failed to initialize retriever for model: {retrieval_model_name}",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
        except Exception as e:
            typer.secho(f"Error initializing retriever: {e!s}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        # Create cross-encoder if reranking is enabled
        cross_encoder = None
        if enable_reranker:
            try:
                from sentence_transformers import CrossEncoder

                from phentrieve.config import DEFAULT_RERANKER_MODEL

                reranker_to_use = reranker_model or DEFAULT_RERANKER_MODEL
                logger.info(f"Loading reranker: {reranker_to_use}")
                cross_encoder = CrossEncoder(reranker_to_use)
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")

        # Extract text and assertion statuses from processed chunks
        text_chunks: list[str] = []
        assertion_statuses: list[str | None] = []
        for chunk in processed_chunks:
            # Handle both formats: string or dict with text/status
            if isinstance(chunk, str):
                text_chunks.append(chunk)
                assertion_statuses.append(None)
            else:
                text_chunks.append(chunk.get("text", str(chunk)))
                # Convert AssertionStatus enum to string if present
                status = chunk.get("status")
                if status is not None and hasattr(status, "value"):
                    status = status.value
                # Ensure status is str or None
                assertion_statuses.append(str(status) if status is not None else None)

        logger.debug(
            f"Extracted {len(text_chunks)} text chunks with assertion statuses"
        )

        chunk_results, aggregated_results = orchestrate_hpo_extraction(
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
        typer.secho(f"Error extracting HPO terms: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Enrich with HPO term details if requested
    if include_details:
        from phentrieve.retrieval.details_enrichment import enrich_results_with_details

        def enrich_with_field_adaptation(
            results: list[dict],
            fields_to_add: dict[str, str],
            fields_to_remove: set[str],
        ) -> list[dict]:
            """Adapt result fields for enrichment, then clean up temporary fields.

            This helper supports two input shapes:
            - Flat results: List[dict] where each dict is an HPO hit (has keys like 'id'/'hpo_id' or 'name'/'label')
            - Chunked results: List[dict] where each dict represents a chunk and contains a 'matches' list

            For chunked results we flatten all matches, enrich them, then re-attach enriched details to the
            original chunk structure.

            Args:
                results: List of result dictionaries to enrich
                fields_to_add: Mapping of source_field -> target_field to add temporarily
                fields_to_remove: Set of field names to remove after enrichment

            Returns:
                Enriched results with temporary fields removed
            """
            if not results:
                return []

            # Detect chunked shape (presence of 'matches' key in first item)
            first = results[0]
            if (
                isinstance(first, dict)
                and "matches" in first
                and isinstance(first["matches"], list)
            ):
                # Flatten matches for enrichment
                flat_matches = []
                match_map = []  # list of tuples (chunk_idx, match_idx)
                for ci, chunk in enumerate(results):
                    for mi, match in enumerate(chunk.get("matches", [])):
                        # Create shallow copy and add temporary fields
                        mcopy = {**match}
                        for source, target in fields_to_add.items():
                            if source in mcopy:
                                mcopy[target] = mcopy[source]
                        # Always ensure HPO enrichment key exists: prefer explicit mapping, fallback from 'id'
                        if "hpo_id" not in mcopy and "id" in mcopy:
                            mcopy["hpo_id"] = mcopy["id"]
                        flat_matches.append(mcopy)
                        match_map.append((ci, mi))

                # Enrich flattened matches
                enriched_flat = enrich_results_with_details(flat_matches)

                # Re-attach enriched details back to original structure
                for (ci, mi), enriched in zip(match_map, enriched_flat):
                    # update the original match dict in-place with definition/synonyms
                    try:
                        results[ci]["matches"][mi].update(
                            {
                                k: v
                                for k, v in enriched.items()
                                if k not in fields_to_remove
                            }
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to re-attach enriched data for chunk {ci}, match {mi}: {e}"
                        )
                        continue

                return results

            # Flat shape: adapt each result dict with temporary fields and enrich directly
            adapted = []
            for r in results:
                rcopy = {**r}
                for source, target in fields_to_add.items():
                    if source in rcopy:
                        rcopy[target] = rcopy[source]
                # Also ensure hpo_id exists when input uses 'id'
                if "hpo_id" not in rcopy and "id" in rcopy:
                    rcopy["hpo_id"] = rcopy["id"]
                adapted.append(rcopy)

            enriched_list = enrich_results_with_details(adapted)
            # Remove temporary fields, keeping definition and synonyms
            cleaned = [
                {k: v for k, v in r.items() if k not in fields_to_remove}
                for r in enriched_list
            ]
            return cleaned

        # Enrich aggregated_results (format: {hpo_id, name, ...})
        # Need to add: name -> label (for enrichment)
        # Need to remove: label (after enrichment)
        if aggregated_results:
            aggregated_results = enrich_with_field_adaptation(
                aggregated_results,
                fields_to_add={"name": "label"},
                fields_to_remove={"label"},
            )

        # Enrich chunk_results (format: {id, name, ...})
        # Need to add: id -> hpo_id, name -> label (for enrichment)
        # Need to remove: hpo_id, label (after enrichment)
        if chunk_results:
            chunk_results = enrich_with_field_adaptation(
                chunk_results,
                fields_to_add={"id": "hpo_id", "name": "label"},
                fields_to_remove={"hpo_id", "label"},
            )

    # Output results in the specified format
    # Log debug information about the results
    logger.debug(f"Aggregated results count: {len(aggregated_results)}")
    if chunk_results and len(chunk_results) > 0:
        logger.debug(f"First chunk result keys: {list(chunk_results[0].keys())}")
        logger.debug(f"First chunk result sample: {chunk_results[0]}")

    # Call the formatting function with model information
    _format_and_output_results(
        aggregated_results,
        chunk_results,
        processed_chunks,
        language,
        output_format,
        embedding_model=retrieval_model,
        reranker_model=reranker_model if enable_reranker else None,
        input_text=raw_text,
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")


@app.command("chunk")
def chunk_text_command(
    text: Annotated[
        Optional[str],
        typer.Argument(
            help="Text to chunk (optional, will read from stdin if not provided)"
        ),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option(
            "--input-file", "-i", help="File to read text from instead of command line"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    chunking_pipeline_config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config-file",
            "-c",
            help="Path to YAML or JSON file with chunking pipeline configuration",
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Predefined chunking strategy. 'simple': paragraph then sentence. 'semantic': paragraph, sentence, then semantic splitting of sentences. 'detailed': paragraph, sentence, punctuation splitting, then semantic splitting of fragments. 'sliding_window': customizable semantic sliding window. 'sliding_window_cleaned': sliding window with final chunk cleaning. 'sliding_window_punct_cleaned': sliding window with punctuation splitting and final cleaning. 'sliding_window_punct_conj_cleaned': sliding window with punctuation, conjunction splitting, and final cleaning (choices: simple, semantic, detailed, sliding_window, sliding_window_cleaned, sliding_window_punct_cleaned, sliding_window_punct_conj_cleaned; default: sliding_window)",
        ),
    ] = "sliding_window",
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
            "--model",
            "-m",
            help=f"Model name for semantic chunker (if using semantic or sliding_window strategy, default: {DEFAULT_MODEL})",
        ),
    ] = DEFAULT_MODEL,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format", "-o", help="Output format for chunks (lines, json_lines)"
        ),
    ] = "lines",
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
):
    """Chunk text using configurable chunking strategies.

    This command processes text through a chunking pipeline, which can include
    paragraph splitting, sentence segmentation, semantic chunking, and fine-grained
    punctuation-based splitting. The output is the resulting text chunks.

    Example usage:
    - Simple paragraph+sentence chunking: phentrieve text chunk "My text here"
    - Semantic chunking: phentrieve text chunk -s semantic -m "FremyCompany/BioLORD-2023-M" -i clinical_note.txt
    """

    from phentrieve.config import (
        DEFAULT_LANGUAGE,
        DEFAULT_MODEL,
    )
    from phentrieve.text_processing.pipeline import TextProcessingPipeline
    from phentrieve.utils import detect_language, setup_logging_cli

    setup_logging_cli(debug=debug)

    # Load the raw text using helper function
    raw_text = load_text_from_input(text, input_file)

    # Detect or set the language
    if not language:
        # Try to auto-detect the language
        try:
            language = detect_language(raw_text, default_lang=DEFAULT_LANGUAGE)
            typer.echo(f"Auto-detected language: {language}")
        except ImportError:
            language = DEFAULT_LANGUAGE
            typer.echo(f"Using default language: {language}")

    # Get chunking pipeline configuration using helper function
    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file=chunking_pipeline_config_file,
        strategy_arg=strategy,
        window_size=window_size,
        step_size=step_size,
        threshold=split_threshold,
        min_segment_length=min_segment_length,
    )

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

    # Load the SBERT model if needed
    sbert_model = None
    if needs_semantic_model:
        # Lazy import - only load heavy ML dependencies when actually needed
        from phentrieve.embeddings import load_embedding_model

        model_name = semantic_chunker_model or DEFAULT_MODEL
        typer.echo(f"Loading sentence transformer model: {model_name}...")
        try:
            # Use cached model loading (reuses model if already loaded in this process)
            sbert_model = load_embedding_model(
                model_name=model_name,
            )
            logger.debug(f"Successfully loaded SentenceTransformer model: {model_name}")
            logger.debug(f"Model type: {type(sbert_model)}")
        except Exception as e:
            typer.secho(
                f"Error loading model '{model_name}': {e!s}", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

    # Empty assertion config to disable assertion detection for this command
    assertion_config = {"disable": True}

    # Create the pipeline
    try:
        logger.debug(f"Creating pipeline with model: {sbert_model is not None}")
        logger.debug(f"Chunking config: {chunking_pipeline_config}")
        logger.debug(f"Language: {language}")
        logger.debug(f"Assertion config: {assertion_config}")

        pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model,
        )
        logger.debug("Successfully created TextProcessingPipeline")
    except Exception as e:
        typer.secho(f"Error creating pipeline: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Process the text
    try:
        processed_chunks = pipeline.process(raw_text)
    except Exception as e:
        typer.secho(f"Error processing text: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Output the chunks in the requested format
    if output_format == "lines":
        for i, chunk_data in enumerate(processed_chunks):
            typer.echo(f"[{i + 1}] {chunk_data['text']}")
    elif output_format == "json_lines":
        for chunk_data in processed_chunks:
            # Ensure Enum values are serialized properly
            chunk_json = {
                "text": chunk_data["text"],
                "source_indices": chunk_data["source_indices"],
            }
            typer.echo(json.dumps(chunk_json))
    else:
        typer.secho(
            f"Error: Unknown output format '{output_format}'. "
            f"Supported formats: lines, json_lines",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Summary
    typer.secho(
        f"\nText chunking completed. {len(processed_chunks)} chunks generated.",
        fg=typer.colors.GREEN,
    )


def _format_and_output_results(
    aggregated_results: list[dict],
    chunk_results: list[dict],
    processed_chunks: list[dict],
    language: str,
    output_format: str,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    input_text: Optional[str] = None,
) -> None:
    """Format and output the HPO extraction results according to the specified format.

    Args:
        aggregated_results: The aggregated HPO term results (already filtered by min_confidence)
        chunk_results: The chunk-level results
        processed_chunks: The processed text chunks
        language: The language of the text
        output_format: The output format (json_lines, rich_json_summary, csv_hpo_list)
        embedding_model: Name of embedding model used for retrieval
        reranker_model: Name of reranker model used (if enabled)
        input_text: Original input text for phenopacket metadata
    """
    if output_format == "phenopacket_v2_json":
        from phentrieve.phenopackets.utils import format_as_phenopacket_v2

        # Note: In this function, the variable naming is swapped from the orchestrator:
        # - 'aggregated_results' here is actually 'chunk_results' from orchestrator
        #   (contains chunk_idx, chunk_text, and matches per chunk)
        # - 'chunk_results' here is actually 'aggregated_results' from orchestrator
        # We use 'aggregated_results' which has the per-chunk structure with text evidence
        phenopacket = format_as_phenopacket_v2(
            chunk_results=aggregated_results,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            input_text=input_text,
        )
        typer.echo(phenopacket)
        return  # early exit

    typer.echo(f"Formatting results in {output_format} format...")

    if output_format == "json_lines":
        # Output each chunk and its matches as a JSON object per line
        for chunk_result in chunk_results:
            # Convert assertion_status to string if it's an enum
            if (
                "assertion_status" in chunk_result
                and chunk_result["assertion_status"] is not None
            ):
                if hasattr(chunk_result["assertion_status"], "value"):
                    chunk_result["assertion_status"] = chunk_result[
                        "assertion_status"
                    ].value
                else:
                    chunk_result["assertion_status"] = str(
                        chunk_result["assertion_status"]
                    )
            typer.echo(json.dumps(chunk_result))

        # Output aggregated results as a final JSON object
        typer.echo(json.dumps({"aggregated_hpo_terms": aggregated_results}))

    elif output_format == "rich_json_summary":
        # First let's convert any AssertionStatus enums to strings
        for result in chunk_results:
            if "assertion_status" in result and result["assertion_status"] is not None:
                if hasattr(result["assertion_status"], "value"):
                    result["assertion_status"] = result["assertion_status"].value
                else:
                    result["assertion_status"] = str(result["assertion_status"])

        # Create a nicely formatted JSON summary
        summary = {
            "document": {
                "language": language,
                "total_chunks": len(processed_chunks),
                "total_hpo_terms": len(chunk_results),  # Use chunk_results here instead
                "hpo_terms": [
                    {
                        "hpo_id": result["id"],
                        "name": result["name"],
                        "confidence": (
                            float(result["score"])
                            if isinstance(result["score"], (int, float))
                            else 0.0
                        ),
                        "status": (
                            result["assertion_status"]
                            if "assertion_status" in result
                            else None
                        ),
                        "evidence_count": (
                            len(result["chunks"]) if "chunks" in result else 0
                        ),
                        "top_evidence": (
                            f"Chunk {result['chunks'][0]}"
                            if result.get("chunks")
                            else ""
                        ),
                    }
                    for result in chunk_results  # Use chunk_results here instead
                ],
            }
        }
        # Format the JSON nicely
        formatted_json = json.dumps(summary, indent=2, ensure_ascii=False)
        typer.echo(formatted_json)

    elif output_format == "csv_hpo_list":
        # Create a CSV with HPO terms and basic info
        output = StringIO()
        fieldnames = ["hpo_id", "name", "confidence", "status", "evidence_count"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in aggregated_results:
            writer.writerow(
                {
                    "hpo_id": r["hpo_id"],
                    "name": r["name"],
                    "confidence": r["confidence"],
                    "status": r["status"],
                    "evidence_count": r["evidence_count"],
                }
            )

        typer.echo(output.getvalue())

    else:
        typer.secho(
            f"Error: Unknown output format '{output_format}'. "
            f"Supported formats: json_lines, rich_json_summary, csv_hpo_list, phenopacket_v2_json",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Summary
    # For rich_json_summary format, use the chunk_results length for HPO term count
    # as it contains the full list of HPO terms used in the JSON output
    if output_format == "rich_json_summary":
        hpo_term_count = len(chunk_results)
    else:
        hpo_term_count = len(aggregated_results)

    typer.secho(
        f"\nText processing completed. Found {hpo_term_count} HPO terms "
        f"across {len(processed_chunks)} text chunks.",
        fg=typer.colors.GREEN,
    )
