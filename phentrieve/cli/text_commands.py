"""Text processing commands for Phentrieve CLI.

This module contains commands for text processing and HPO term extraction.
"""

import csv
from io import StringIO
import json
import logging
import yaml
from pathlib import Path
from typing import Optional, List, Dict
from typing_extensions import Annotated

import typer

logger = logging.getLogger(__name__)

from phentrieve.cli.utils import load_text_from_input, resolve_chunking_pipeline_config
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

# Create the Typer app for this command group
app = typer.Typer()


@app.command("process")
def process_text_for_hpo_command(
    text: Annotated[
        Optional[str],
        typer.Argument(
            help="Text to process (optional, will read from stdin if not provided)"
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
            help="Predefined chunking strategy (simple, semantic, detailed)",
        ),
    ] = "semantic",  # Changed default to semantic for better chunks
    semantic_chunker_model: Annotated[
        Optional[str],
        typer.Option(
            "--semantic-model",
            "--s-model",
            help="Model name for semantic chunker (if using semantic strategy)",
        ),
    ] = None,
    retrieval_model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model name for HPO term retrieval"),
    ] = None,
    similarity_threshold: Annotated[
        float,
        typer.Option(
            "--similarity-threshold",
            "--threshold",
            help="Minimum similarity score for HPO term matches",
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
    ] = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    monolingual_reranker_model: Annotated[
        Optional[str],
        typer.Option(
            "--monolingual-reranker-model",
            help="Language-specific cross-encoder model for monolingual reranking",
        ),
    ] = "ml6team/cross-encoder-mmarco-german-distilbert-base",
    reranker_mode: Annotated[
        str,
        typer.Option(
            "--reranker-mode",
            help="Mode for re-ranking: 'cross-lingual' or 'monolingual'",
        ),
    ] = "cross-lingual",
    translation_dir: Annotated[
        Optional[str],
        typer.Option(
            "--translation-dir",
            help="Directory with HPO translations in target language (required for monolingual mode)",
        ),
    ] = None,
    rerank_count: Annotated[
        int,
        typer.Option(
            "--rerank-count",
            help="Number of candidates to consider for re-ranking",
        ),
    ] = 50,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "-o",
            help="Output format for results (json_lines, rich_json_summary, csv_hpo_list)",
        ),
    ] = "rich_json_summary",
    min_confidence: Annotated[
        float,
        typer.Option(
            "--min-confidence",
            "--min-conf",
            help="Minimum confidence threshold for HPO terms in the results",
        ),
    ] = 0.0,
    top_term_per_chunk: Annotated[
        bool,
        typer.Option(
            "--top-term-per-chunk",
            "--top-only",
            help="Only include the highest-scored HPO term for each chunk",
        ),
    ] = False,
    cpu: Annotated[
        bool,
        typer.Option("--cpu", help="Force CPU usage even if GPU is available"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging"),
    ] = False,
):
    """Process clinical text to extract HPO terms.

    This command processes clinical texts through a chunking pipeline and assertion
    detection, then extracts HPO terms from each chunk. Results are aggregated to provide
    a comprehensive set of phenotype terms from the entire document.

    Example usage:
    - Basic processing: phentrieve text process "Patient presents with hearing loss."
    - From file with semantic chunking: phentrieve text process -i clinical_note.txt -s semantic -m "FremyCompany/BioLORD-2023-M"
    """
    from sentence_transformers import SentenceTransformer

    from phentrieve.config import (
        DEFAULT_CHUNK_PIPELINE_CONFIG,
        DEFAULT_MODEL,
        DEFAULT_LANGUAGE,
        DEFAULT_ASSERTION_CONFIG,
        DEFAULT_TRANSLATIONS_SUBDIR,
    )
    from phentrieve.text_processing.pipeline import TextProcessingPipeline
    from phentrieve.text_processing.assertion_detection import AssertionStatus
    from phentrieve.retrieval.dense_retriever import DenseRetriever
    from phentrieve.retrieval import reranker
    from phentrieve.utils import (
        setup_logging_cli,
        resolve_data_path,
        load_translation_text,
        detect_language,
    )

    setup_logging_cli(debug=debug)

    # Determine device
    device = "cpu" if cpu else None

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

    # Load chunking pipeline configuration using helper function
    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file, strategy
    )

    # Configure assertion detection
    assertion_config = dict(DEFAULT_ASSERTION_CONFIG)
    if no_assertion_detection:
        assertion_config["disable"] = True
    else:
        assertion_config["preference"] = assertion_preference

    # Resolve translation directory path if in monolingual mode
    if reranker_mode == "monolingual" and translation_dir is None:
        translation_dir = resolve_data_path(DEFAULT_TRANSLATIONS_SUBDIR)
        typer.echo(f"Using default translation directory: {translation_dir}")

    # Determine if we need a semantic model
    needs_semantic_model = any(
        chunk_config.get("type") in ["semantic", "pre_chunk_semantic_grouper"]
        for chunk_config in chunking_pipeline_config
    )

    # Load models
    sbert_model = None
    retrieval_sbert_model = None
    cross_encoder = None

    try:
        # Load semantic chunking model if needed
        if needs_semantic_model:
            model_name = semantic_chunker_model or DEFAULT_MODEL
            typer.echo(f"Loading semantic chunking model: {model_name}")
            sbert_model = SentenceTransformer(model_name, device=device)

        # Load HPO retrieval model
        retrieval_model_name = retrieval_model or DEFAULT_MODEL
        typer.echo(f"Loading retrieval model: {retrieval_model_name}")
        retrieval_sbert_model = SentenceTransformer(retrieval_model_name, device=device)

        # Initialize text processing pipeline
        typer.echo("Initializing text processing pipeline...")
        pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model,
        )

        # Let the models load silently (no extra echo here)

        # Initialize the DenseRetriever
        typer.echo("Loading HPO term index...")
        retriever = DenseRetriever.from_model_name(
            model=retrieval_sbert_model,  # Pass pre-loaded model
            model_name=retrieval_model_name,
            min_similarity=similarity_threshold,
        )

        # Initialize cross-encoder for reranking if enabled
        if enable_reranker:
            # Determine which reranker model to use based on the mode
            if reranker_mode == "monolingual" and language != "en":
                # For non-English in monolingual mode, use language-specific model
                ce_model_name = monolingual_reranker_model
                typer.echo(f"Loading monolingual reranker model: {ce_model_name}")
            else:
                # For cross-lingual mode or English, use the default reranker
                ce_model_name = reranker_model
                typer.echo(f"Loading cross-lingual reranker model: {ce_model_name}")

            cross_encoder = reranker.load_cross_encoder(ce_model_name, device=device)

        # Process text into chunks
        typer.echo("Processing text into chunks...")
        processed_chunks = pipeline.process(raw_text)
        # Debug log the structure of processed chunks
        logger.info(
            f"Chunk structure: {processed_chunks[0].keys() if processed_chunks else 'No chunks'}"
        )
        text_chunks = [chunk["text"] for chunk in processed_chunks]
        # Get assertion status from the 'status' key
        assertion_statuses = [str(chunk["status"].value) for chunk in processed_chunks]

        # Use the orchestrator to process HPO extraction
        typer.echo("Starting HPO term extraction via orchestrator...")
        unique_terms, chunk_results, all_terms = orchestrate_hpo_extraction(
            text_chunks=text_chunks,
            retriever=retriever,
            num_results_per_chunk=num_results,
            similarity_threshold_per_chunk=similarity_threshold,
            cross_encoder=cross_encoder,
            translation_dir_path=Path(translation_dir) if translation_dir else None,
            language=language,
            reranker_mode=reranker_mode,
            top_term_per_chunk=top_term_per_chunk,
            min_confidence=min_confidence,
            assertion_statuses=assertion_statuses,
        )

        typer.echo(
            f"Extraction complete - found {len(unique_terms)} HPO terms across {len(processed_chunks)} chunks"
        )

        # Output results in the requested format
        if output_format == "json_lines":
            for result in unique_terms:
                typer.echo(json.dumps(result))
        elif output_format == "rich_json_summary":
            output_data = {
                "meta": {
                    "input": {
                        "text": raw_text,
                        "language": language,
                        "strategy": strategy,
                        "min_confidence": similarity_threshold,
                        "top_term_per_chunk": top_term_per_chunk,
                        "assertion_detection": not no_assertion_detection,
                        "assertion_preference": assertion_preference,
                        "reranker_enabled": enable_reranker,
                        "reranker_mode": reranker_mode if enable_reranker else None,
                    },
                    "models": {
                        "retrieval": retriever.model_name,
                        "reranker": reranker_model if enable_reranker else None,
                        "monolingual_reranker": (
                            monolingual_reranker_model if enable_reranker else None
                        ),
                    },
                },
                "chunks": [
                    {
                        "text": chunk["text"],
                        "status": str(chunk["status"].value),
                        "source_indices": chunk["source_indices"],
                    }
                    for chunk in processed_chunks
                ],
                "total_chunks": len(processed_chunks),
                "total_terms": len(unique_terms),
                "terms": unique_terms,
            }
            typer.echo(json.dumps(output_data, indent=2))
        elif output_format == "csv_hpo_list":
            # Output just the HPO IDs as a comma-separated list
            hpo_ids = [result["hpo_id"] for result in unique_terms]
            typer.echo(",".join(hpo_ids))

        return unique_terms

    except Exception as e:
        typer.secho(f"Error processing text: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


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
            help="Predefined chunking strategy (simple, semantic, detailed)",
        ),
    ] = "simple",
    semantic_chunker_model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name for semantic chunker (if using semantic strategy)",
        ),
    ] = None,
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
    from sentence_transformers import SentenceTransformer

    from phentrieve.config import (
        DEFAULT_CHUNK_PIPELINE_CONFIG,
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
        chunking_pipeline_config_file, strategy
    )

    # Determine if we need a semantic model
    needs_semantic_model = any(
        chunk_config.get("type") in ["semantic", "pre_chunk_semantic_grouper"]
        for chunk_config in chunking_pipeline_config
    )

    # Load the SBERT model if needed
    sbert_model = None
    if needs_semantic_model:
        model_name = semantic_chunker_model or DEFAULT_MODEL
        typer.echo(f"Loading sentence transformer model: {model_name}...")
        try:
            sbert_model = SentenceTransformer(model_name)
        except Exception as e:
            typer.secho(
                f"Error loading model '{model_name}': {str(e)}", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)

    # Empty assertion config to disable assertion detection for this command
    assertion_config = {"disable": True}

    # Create the pipeline
    try:
        pipeline = TextProcessingPipeline(
            language=language,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            sbert_model_for_semantic_chunking=sbert_model,
        )
    except Exception as e:
        typer.secho(f"Error creating pipeline: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Process the text
    try:
        processed_chunks = pipeline.process(raw_text)
    except Exception as e:
        typer.secho(f"Error processing text: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Output the chunks in the requested format
    if output_format == "lines":
        for i, chunk_data in enumerate(processed_chunks):
            typer.echo(f"[{i+1}] {chunk_data['text']}")
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
    aggregated_results: List[Dict],
    chunk_results: List[Dict],
    processed_chunks: List[Dict],
    language: str,
    output_format: str,
) -> None:
    """Format and output the HPO extraction results according to the specified format.

    Args:
        aggregated_results: The aggregated HPO term results (already filtered by min_confidence)
        chunk_results: The chunk-level results
        processed_chunks: The processed text chunks
        language: The language of the text
        output_format: The output format (json_lines, rich_json_summary, csv_hpo_list)
    """
    typer.echo(f"Formatting results in {output_format} format...")

    if output_format == "json_lines":
        # Output each chunk and its matches as a JSON object per line
        for chunk_result in chunk_results:
            typer.echo(json.dumps(chunk_result))

        # Output aggregated results as a final JSON object
        typer.echo(json.dumps({"aggregated_hpo_terms": aggregated_results}))

    elif output_format == "rich_json_summary":
        # Create a nicely formatted JSON summary
        summary = {
            "document": {
                "language": language,
                "total_chunks": len(processed_chunks),
                "total_hpo_terms": len(aggregated_results),
                "hpo_terms": [
                    {
                        "hpo_id": result["hpo_id"],
                        "name": result["name"],
                        "confidence": result["confidence"],
                        "status": result["status"],
                        "evidence_count": len(result["evidence"]),
                        "top_evidence": (
                            result["evidence"][0]["chunk_text"]
                            if result["evidence"]
                            else ""
                        ),
                    }
                    for result in aggregated_results
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
            f"Supported formats: json_lines, rich_json_summary, csv_hpo_list",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Summary
    typer.secho(
        f"\nText processing completed. Found {len(aggregated_results)} HPO terms "
        f"across {len(processed_chunks)} text chunks.",
        fg=typer.colors.GREEN,
    )
