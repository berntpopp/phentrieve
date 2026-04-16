"""Text processing commands for Phentrieve CLI.

This module contains commands for text processing and HPO term extraction.
"""

import csv
import json
import logging
import os
from io import StringIO
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

# NOTE: SentenceTransformer is NOT imported at module level to avoid slow startup.
# Importing sentence_transformers loads PyTorch/CUDA (18+ seconds), which should
# only happen when commands actually need ML models, not for --help or --version.
# The import is done inside command functions where the model is actually used.
from phentrieve.cli.utils import load_text_from_input, resolve_chunking_pipeline_config
from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    DEFAULT_MODEL,
    DEFAULT_SPLITTING_THRESHOLD,
    DEFAULT_STEP_SIZE_TOKENS,
    DEFAULT_WINDOW_SIZE_TOKENS,
)

logger = logging.getLogger(__name__)


def get_llm_provider(**kwargs: Any) -> Any:
    """Lazy wrapper around the shared LLM provider factory."""
    from phentrieve.llm.provider import get_llm_provider as _get_llm_provider

    return _get_llm_provider(**kwargs)


class TwoPhaseLLMPipeline:
    """Lazy wrapper that preserves the module-level patch point for tests."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from phentrieve.llm.pipeline import TwoPhaseLLMPipeline as _TwoPhaseLLMPipeline

        return _TwoPhaseLLMPipeline(*args, **kwargs)


def run_full_text_service(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy wrapper around the shared full-text service."""
    from phentrieve.text_processing.full_text_service import (
        run_full_text_service as _run_full_text_service,
    )

    return _run_full_text_service(*args, **kwargs)


def _run_llm_backend(*, text: str, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible CLI wrapper around the shared LLM backend."""
    from phentrieve.text_processing.full_text_service import run_llm_backend

    kwargs.setdefault("provider_factory", get_llm_provider)
    kwargs.setdefault("pipeline_factory", TwoPhaseLLMPipeline)
    return dict(run_llm_backend(text=text, **kwargs))


def _stable_chunks_to_chunk_level_results(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert stable processed chunks into the legacy chunk-result shape."""
    chunk_level_results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        if isinstance(chunk_id, int) and chunk_id > 0:
            chunk_idx = chunk_id - 1
        else:
            chunk_idx = idx

        chunk_level_results.append(
            {
                "chunk_idx": chunk_idx,
                "chunk_text": chunk.get("text", ""),
                "matches": chunk.get("hpo_matches", []),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            }
        )

    return chunk_level_results


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
    ] = DEFAULT_CHUNKING_STRATEGY,
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            "-ws",
            help="Sliding window size in tokens (only for sliding_window strategy)",
        ),
    ] = DEFAULT_WINDOW_SIZE_TOKENS,
    step_size: Annotated[
        int,
        typer.Option(
            "--step-size",
            "-ss",
            help="Sliding window step size in tokens (only for sliding_window strategy)",
        ),
    ] = DEFAULT_STEP_SIZE_TOKENS,
    split_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Similarity threshold for splitting (0-1, only for sliding_window strategy)",
        ),
    ] = DEFAULT_SPLITTING_THRESHOLD,
    min_segment_length: Annotated[
        int,
        typer.Option(
            "--min-segment",
            "-ms",
            help="Minimum segment length in words (only for sliding_window strategy)",
        ),
    ] = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    semantic_chunker_model: Annotated[
        str | None,
        typer.Option(
            "--semantic-model",
            "--s-model",
            help=f"Model name for semantic chunking (default: {DEFAULT_MODEL})",
        ),
    ] = DEFAULT_MODEL,
    retrieval_model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name for HPO term retrieval"),
    ] = None,
    chunk_retrieval_threshold: Annotated[
        float,
        typer.Option(
            "--chunk-retrieval-threshold",
            "-crt",
            help="Minimum similarity score for HPO term matches per chunk (0.0-1.0)",
        ),
    ] = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
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
        str | None,
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
    ] = DEFAULT_MIN_CONFIDENCE_AGGREGATED,
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
        Path | None,
        typer.Option(
            "--input-file", "-i", help="File to read text from instead of command line"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    chunking_pipeline_config_file: Annotated[
        Path | None,
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
    ] = DEFAULT_CHUNKING_STRATEGY,
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            "-ws",
            help="Sliding window size in tokens (only for sliding_window strategy)",
        ),
    ] = DEFAULT_WINDOW_SIZE_TOKENS,
    step_size: Annotated[
        int,
        typer.Option(
            "--step-size",
            "-ss",
            help="Sliding window step size in tokens (only for sliding_window strategy)",
        ),
    ] = DEFAULT_STEP_SIZE_TOKENS,
    split_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Similarity threshold for splitting (0-1, only for sliding_window strategy)",
        ),
    ] = DEFAULT_SPLITTING_THRESHOLD,
    min_segment_length: Annotated[
        int,
        typer.Option(
            "--min-segment",
            "-ms",
            help="Minimum segment length in words (only for sliding_window strategy)",
        ),
    ] = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    semantic_chunker_model: Annotated[
        str | None,
        typer.Option(
            "--semantic-model",
            "--s-model",
            help=f"Model name for semantic chunking (for sliding_window strategy, default: {DEFAULT_MODEL})",
        ),
    ] = DEFAULT_MODEL,
    retrieval_model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name for HPO term retrieval"),
    ] = None,
    extraction_backend: Annotated[
        Literal["standard", "llm"],
        typer.Option(
            "--extraction-backend",
            help="Choose full-text extraction backend: standard or llm.",
        ),
    ] = "standard",
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="LLM model for full-text extraction.",
        ),
    ] = None,
    llm_mode: Annotated[
        Literal["two_phase"],
        typer.Option(
            "--llm-mode",
            help="LLM extraction mode.",
        ),
    ] = "two_phase",
    llm_internal_mode: Annotated[
        Literal["whole_document_legacy", "whole_document_grounded"],
        typer.Option(
            "--llm-internal-mode",
            help="Internal grounding mode for the LLM backend.",
        ),
    ] = "whole_document_grounded",
    chunk_retrieval_threshold: Annotated[
        float,
        typer.Option(
            "--chunk-retrieval-threshold",
            "-crt",
            help="Minimum similarity score for an HPO term to be considered a match for an individual text chunk (0.0-1.0)",
        ),
    ] = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
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
        str | None,
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
    ] = DEFAULT_MIN_CONFIDENCE_AGGREGATED,
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

    from phentrieve.config import DEFAULT_LANGUAGE, DEFAULT_MODEL
    from phentrieve.utils import detect_language, setup_logging_cli

    logger = logging.getLogger(__name__)
    start_time = time.time()
    setup_logging_cli(debug=debug)

    raw_text = load_text_from_input(text, input_file)

    if not language:
        try:
            language = detect_language(raw_text, default_lang=DEFAULT_LANGUAGE)
            typer.echo(f"Auto-detected language: {language}", err=True)
        except ImportError:
            language = DEFAULT_LANGUAGE
            typer.echo(f"Using default language: {language}", err=True)

    chunking_pipeline_config = resolve_chunking_pipeline_config(
        chunking_pipeline_config_file=chunking_pipeline_config_file,
        strategy_arg=strategy,
        window_size=window_size,
        step_size=step_size,
        threshold=split_threshold,
        min_segment_length=min_segment_length,
    )

    assertion_config = {
        "disable": no_assertion_detection,
        "preference": assertion_preference,
    }
    logger.debug(f"Chunking pipeline config: {chunking_pipeline_config}")
    logger.debug(f"Assertion config: {assertion_config}")

    if extraction_backend == "llm" and not (
        llm_model or os.getenv("PHENTRIEVE_LLM_MODEL")
    ):
        typer.secho(
            "Error processing text: No LLM model configured. "
            "Provide --llm-model or set PHENTRIEVE_LLM_MODEL.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info("Using full-text extraction backend: %s", extraction_backend)
    if extraction_backend == "llm":
        logger.debug(
            "LLM backend configuration: model=%s, mode=%s, internal_mode=%s, language=%s",
            llm_model or os.getenv("PHENTRIEVE_LLM_MODEL"),
            llm_mode,
            llm_internal_mode,
            language,
        )

    model_name = retrieval_model or DEFAULT_MODEL
    logger.info("Using model: %s", model_name)

    sbert_model_for_chunking = None
    if extraction_backend == "standard":
        needs_semantic_model = False
        for chunk_config in chunking_pipeline_config:
            chunk_type = (
                chunk_config.get("type")
                if isinstance(chunk_config, dict)
                else chunk_config
            )
            if chunk_type in [
                "semantic",
                "pre_chunk_semantic_grouper",
                "sliding_window_semantic",
                "sliding_window",
            ]:
                needs_semantic_model = True
                break

        if needs_semantic_model:
            from phentrieve.embeddings import load_embedding_model

            semantic_model_name = semantic_chunker_model or DEFAULT_MODEL
            typer.echo(
                f"Loading sentence transformer model for chunking: {semantic_model_name}...",
                err=True,
            )
            try:
                sbert_model_for_chunking = load_embedding_model(
                    model_name=semantic_model_name,
                )
                logger.debug(
                    "Successfully loaded model for chunking: %s",
                    semantic_model_name,
                )
            except Exception as e:
                typer.secho(
                    f"Error loading semantic chunker model '{semantic_model_name}': {e!s}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)

    cross_encoder = None
    if extraction_backend == "standard" and enable_reranker:
        try:
            from sentence_transformers import CrossEncoder

            from phentrieve.config import DEFAULT_RERANKER_MODEL

            reranker_to_use = reranker_model or DEFAULT_RERANKER_MODEL
            logger.info("Loading reranker: %s", reranker_to_use)
            cross_encoder = CrossEncoder(reranker_to_use)
        except Exception as e:
            logger.warning("Failed to load cross-encoder: %s", e)

    try:
        service_result = run_full_text_service(
            text=raw_text,
            extraction_backend=extraction_backend,
            language=language,
            llm_model=llm_model,
            llm_mode=llm_mode,
            llm_internal_mode=llm_internal_mode,
            chunking_pipeline_config=chunking_pipeline_config,
            assertion_config=assertion_config,
            include_positions=True,
            model_name=model_name,
            retrieval_model_name=model_name,
            sbert_model_for_semantic_chunking=sbert_model_for_chunking,
            num_results_per_chunk=num_results,
            chunk_retrieval_threshold=chunk_retrieval_threshold,
            cross_encoder=cross_encoder,
            top_term_per_chunk=top_term_per_chunk,
            min_confidence_for_aggregated=aggregated_term_confidence,
            include_details=include_details,
        )
    except Exception as e:
        typer.secho(f"Error processing text: {e!s}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    meta = service_result["meta"]
    terms = service_result["aggregated_hpo_terms"]
    chunks = service_result["processed_chunks"]

    if meta.get("extraction_backend") == "llm":
        llm_model = meta.get("llm_model")
        llm_mode = meta.get("llm_mode")
        token_input = meta.get("token_input")
        token_output = meta.get("token_output")
        if llm_model or llm_mode:
            note_parts: list[str] = []
            if llm_model:
                note_parts.append(f"model={llm_model}")
            if llm_mode:
                note_parts.append(f"mode={llm_mode}")
            if token_input is not None or token_output is not None:
                note_parts.append(
                    f"tokens_in={token_input if token_input is not None else 'unknown'}"
                )
                note_parts.append(
                    f"tokens_out={token_output if token_output is not None else 'unknown'}"
                )
            logger.info("LLM metadata: %s", ", ".join(note_parts))
        if token_input is not None or token_output is not None:
            logger.info(
                "LLM token usage: input=%s output=%s",
                token_input if token_input is not None else "unknown",
                token_output if token_output is not None else "unknown",
            )

    logger.info(
        "Full-text extraction completed: backend=%s, terms=%d, chunks=%d",
        meta.get("extraction_backend", extraction_backend),
        len(terms),
        len(chunks),
    )

    chunk_level_results = _stable_chunks_to_chunk_level_results(chunks)

    logger.debug("Aggregated results count: %s", len(terms))
    if chunk_level_results:
        logger.debug("First chunk result keys: %s", list(chunk_level_results[0].keys()))
        logger.debug("First chunk result sample: %s", chunk_level_results[0])

    _format_and_output_results(
        chunk_level_results,
        terms,
        chunks,
        language,
        output_format,
        embedding_model=model_name,
        reranker_model=reranker_model if enable_reranker else None,
        input_text=raw_text,
    )

    elapsed_time = time.time() - start_time
    logger.info("Total processing time: %.2f seconds", elapsed_time)


@app.command("chunk")
def chunk_text_command(
    text: Annotated[
        str | None,
        typer.Argument(
            help="Text to chunk (optional, will read from stdin if not provided)"
        ),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option(
            "--input-file", "-i", help="File to read text from instead of command line"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language of the text (en, de, etc.)"),
    ] = "en",
    chunking_pipeline_config_file: Annotated[
        Path | None,
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
    ] = DEFAULT_CHUNKING_STRATEGY,
    window_size: Annotated[
        int,
        typer.Option(
            "--window-size",
            "-ws",
            help="Sliding window size in tokens (only for sliding_window strategy)",
        ),
    ] = DEFAULT_WINDOW_SIZE_TOKENS,
    step_size: Annotated[
        int,
        typer.Option(
            "--step-size",
            "-ss",
            help="Sliding window step size in tokens (only for sliding_window strategy)",
        ),
    ] = DEFAULT_STEP_SIZE_TOKENS,
    split_threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Similarity threshold for splitting (0-1, only for sliding_window strategy)",
        ),
    ] = DEFAULT_SPLITTING_THRESHOLD,
    min_segment_length: Annotated[
        int,
        typer.Option(
            "--min-segment",
            "-ms",
            help="Minimum segment length in words (only for sliding_window strategy)",
        ),
    ] = DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
    semantic_chunker_model: Annotated[
        str | None,
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
            typer.echo(f"Auto-detected language: {language}", err=True)
        except ImportError:
            language = DEFAULT_LANGUAGE
            typer.echo(f"Using default language: {language}", err=True)

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
        typer.echo(f"Loading sentence transformer model: {model_name}...", err=True)
        try:
            # Use cached model loading (reuses model if already loaded in this process)
            sbert_model = load_embedding_model(
                model_name=model_name,
            )
            logger.debug(f"Successfully loaded SentenceTransformer model: {model_name}")
            logger.debug(f"Model type: {type(sbert_model)}")
        except Exception as e:
            typer.secho(
                f"Error loading model '{model_name}': {e!s}",
                fg=typer.colors.RED,
                err=True,
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
        typer.secho(f"Error creating pipeline: {e!s}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Process the text
    try:
        processed_chunks = pipeline.process(raw_text)
    except Exception as e:
        typer.secho(f"Error processing text: {e!s}", fg=typer.colors.RED, err=True)
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
            err=True,
        )
        raise typer.Exit(code=1)

    # Summary
    typer.secho(
        f"\nText chunking completed. {len(processed_chunks)} chunks generated.",
        fg=typer.colors.GREEN,
        err=True,
    )


def _format_and_output_results(
    chunk_level_results: list[dict],
    term_level_results: list[dict],
    processed_chunks: list[dict],
    language: str,
    output_format: str,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    input_text: str | None = None,
) -> None:
    """Format and output the HPO extraction results according to the specified format.

    Args:
        chunk_level_results: Per-chunk results with chunk_idx, chunk_text, matches,
            start_char, end_char (from orchestrator's second return value)
        term_level_results: Per-term aggregated results with id, name, score, chunks
            (from orchestrator's first return value)
        processed_chunks: The processed text chunks from pipeline
        language: The language of the text
        output_format: The output format (json_lines, rich_json_summary, csv_hpo_list)
        embedding_model: Name of embedding model used for retrieval
        reranker_model: Name of reranker model used (if enabled)
        input_text: Original input text for phenopacket metadata
    """
    if output_format == "phenopacket_v2_json":
        from phentrieve.phenopackets.utils import format_as_phenopacket_v2

        # chunk_level_results has chunk_idx, chunk_text, matches, start_char, end_char
        phenopacket = format_as_phenopacket_v2(
            chunk_results=chunk_level_results if chunk_level_results else None,
            aggregated_results=term_level_results if not chunk_level_results else None,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            input_text=input_text,
        )
        typer.echo(phenopacket)
        return  # early exit

    typer.echo(f"Formatting results in {output_format} format...", err=True)

    if output_format == "json_lines":
        # Output each term and its info as a JSON object per line
        for term_result in term_level_results:
            # Convert assertion_status to string if it's an enum
            if (
                "assertion_status" in term_result
                and term_result["assertion_status"] is not None
            ):
                if hasattr(term_result["assertion_status"], "value"):
                    term_result["assertion_status"] = term_result[
                        "assertion_status"
                    ].value
                else:
                    term_result["assertion_status"] = str(
                        term_result["assertion_status"]
                    )
            typer.echo(json.dumps(term_result))

        # Output chunk-level results with positions as final JSON object when present.
        # LLM mode can legitimately return no chunks, so skip the preview in that case.
        if chunk_level_results:
            typer.echo(json.dumps({"aggregated_hpo_terms": chunk_level_results}))

    elif output_format == "rich_json_summary":
        # First convert any AssertionStatus enums to strings
        for result in term_level_results:
            if "assertion_status" in result and result["assertion_status"] is not None:
                if hasattr(result["assertion_status"], "value"):
                    result["assertion_status"] = result["assertion_status"].value
                else:
                    result["assertion_status"] = str(result["assertion_status"])

        # Build chunks info with positions from chunk_level_results
        chunks_info = [
            {
                "chunk_idx": chunk_data.get("chunk_idx", 0),
                "text": chunk_data.get("chunk_text", ""),
                "start_char": chunk_data.get("start_char", -1),
                "end_char": chunk_data.get("end_char", -1),
            }
            for chunk_data in chunk_level_results
        ]

        summary = {
            "document": {
                "language": language,
                "total_chunks": len(processed_chunks),
                "total_hpo_terms": len(term_level_results),
                "chunks": chunks_info,
                "hpo_terms": [
                    {
                        "hpo_id": result["id"],
                        "name": result["name"],
                        "confidence": (
                            float(result["score"])
                            if isinstance(result.get("score"), (int, float))
                            else 0.0
                        ),
                        "status": result.get("assertion_status", result.get("status")),
                        "evidence_count": (
                            len(result["chunks"])
                            if isinstance(result.get("chunks"), list)
                            else int(bool(result.get("evidence")))
                        ),
                        "top_evidence_chunk": (
                            result["chunks"][0]
                            if isinstance(result.get("chunks"), list)
                            and result["chunks"]
                            else -1
                        ),
                    }
                    for result in term_level_results
                ],
            }
        }
        # Format the JSON nicely
        formatted_json = json.dumps(summary, indent=2, ensure_ascii=False)
        typer.echo(formatted_json)

    elif output_format == "csv_hpo_list":
        # Create a CSV with HPO terms and chunk positions
        output = StringIO()
        fieldnames = [
            "hpo_id",
            "name",
            "confidence",
            "status",
            "chunk_idx",
            "chunk_text",
            "start_char",
            "end_char",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Build a lookup from chunk_idx to positions
        chunk_positions = {
            chunk_data.get("chunk_idx", -1): {
                "chunk_text": chunk_data.get("chunk_text", ""),
                "start_char": chunk_data.get("start_char", -1),
                "end_char": chunk_data.get("end_char", -1),
            }
            for chunk_data in chunk_level_results
        }

        for term in term_level_results:
            hpo_id = term.get("hpo_id") or term.get("id", "")
            confidence = term.get("confidence") or term.get("score", 0.0)
            status = term.get("status") or term.get("assertion_status", "")
            # Get chunk index from the term's evidence
            chunk_idx = term.get("chunks", [-1])[0] if term.get("chunks") else -1
            chunk_info = chunk_positions.get(chunk_idx, {})

            writer.writerow(
                {
                    "hpo_id": hpo_id,
                    "name": term.get("name", ""),
                    "confidence": confidence,
                    "status": status,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_info.get("chunk_text", ""),
                    "start_char": chunk_info.get("start_char", -1),
                    "end_char": chunk_info.get("end_char", -1),
                }
            )

        typer.echo(output.getvalue())

    else:
        typer.secho(
            f"Error: Unknown output format '{output_format}'. "
            f"Supported formats: json_lines, rich_json_summary, csv_hpo_list, phenopacket_v2_json",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Summary
    typer.secho(
        f"\nText processing completed. Found {len(term_level_results)} HPO terms "
        f"across {len(processed_chunks)} text chunks.",
        fg=typer.colors.GREEN,
        err=True,
    )
