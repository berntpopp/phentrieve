"""CLI commands for LLM-based annotation.

This module provides commands for annotating clinical text using LLMs,
running benchmarks, and comparing annotation modes.

Commands:
    phentrieve llm annotate   - Annotate clinical text
    phentrieve llm benchmark  - Run benchmarks
    phentrieve llm compare    - Compare modes
    phentrieve llm models     - List available models
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

logger = logging.getLogger(__name__)

# Default LLM model, overridable via environment variable
DEFAULT_LLM_MODEL = os.environ.get("PHENTRIEVE_LLM_MODEL", "github/gpt-4o")

app = typer.Typer(
    name="llm",
    help="LLM-based HPO annotation commands.",
    no_args_is_help=True,
)


def _check_litellm_installed() -> None:
    """Check if LiteLLM is installed and provide helpful error if not."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        typer.echo(
            "Error: LiteLLM is not installed.\n\n"
            "The LLM annotation feature requires LiteLLM. Install with:\n"
            "    pip install litellm\n\n"
            "Or install Phentrieve with LLM support:\n"
            "    pip install phentrieve[llm]",
            err=True,
        )
        raise typer.Exit(1)


@app.command(name="annotate")
def annotate(
    text: Annotated[
        Optional[str],
        typer.Argument(
            help="Clinical text to annotate. If not provided, reads from stdin or --input file."
        ),
    ] = None,
    input_file: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            help="Input file containing clinical text.",
            exists=True,
            readable=True,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use (e.g., github/gpt-4o, gemini/gemini-1.5-pro). "
            "Default can be set via PHENTRIEVE_LLM_MODEL env var.",
        ),
    ] = DEFAULT_LLM_MODEL,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Annotation mode: direct, tool_term, or tool_text.",
        ),
    ] = "tool_text",
    language: Annotated[
        str,
        typer.Option(
            "--language",
            "-l",
            help="Language code (en, de, es, fr, nl) or 'auto' for detection.",
        ),
    ] = "auto",
    postprocess: Annotated[
        Optional[str],
        typer.Option(
            "--postprocess",
            "-p",
            help="Comma-separated post-processing steps: validation,refinement,assertion_review",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json, text, or phenopacket.",
        ),
    ] = "text",
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path. If not provided, prints to stdout.",
        ),
    ] = None,
    include_details: Annotated[
        bool,
        typer.Option(
            "--include-details",
            help="Include HPO term definitions and synonyms.",
        ),
    ] = False,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-t",
            help="Sampling temperature (0.0 = deterministic).",
            min=0.0,
            max=2.0,
        ),
    ] = 0.0,
    no_validate: Annotated[
        bool,
        typer.Option(
            "--no-validate",
            help="Skip HPO ID validation against database.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug logging to show intermediate steps (tool results, filtering, etc.).",
        ),
    ] = False,
) -> None:
    """
    Annotate clinical text with HPO terms using an LLM.

    Examples:
        # Direct text annotation
        phentrieve llm annotate "Patient has seizures and hypotonia"

        # Use specific model and mode
        phentrieve llm annotate "Krampfanfälle und Muskelhypotonie" \\
            --model gemini/gemini-1.5-flash --mode tool_text --language de

        # With post-processing
        phentrieve llm annotate "seizures, no cardiac issues" \\
            --postprocess validation,refinement

        # From file to phenopacket
        phentrieve llm annotate -i clinical_note.txt --format phenopacket -o output.json
    """
    _check_litellm_installed()

    # Enable debug logging if requested
    if debug:
        llm_logger = logging.getLogger("phentrieve.llm")
        llm_logger.setLevel(logging.DEBUG)
        # Add a handler if none exists to ensure debug output is visible
        if not llm_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            )
            llm_logger.addHandler(handler)
        typer.echo("Debug mode enabled - showing intermediate steps", err=True)

    # Get input text
    if text:
        input_text = text
    elif input_file:
        input_text = input_file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read()
    else:
        typer.echo("Error: No input text provided.", err=True)
        typer.echo(
            "Provide text as argument, via --input file, or pipe from stdin.",
            err=True,
        )
        raise typer.Exit(1)

    # Parse mode
    from phentrieve.llm import AnnotationMode, LLMAnnotationPipeline, PostProcessingStep

    try:
        annotation_mode = AnnotationMode(mode)
    except ValueError:
        typer.echo(
            f"Error: Invalid mode '{mode}'. Use: direct, tool_term, or tool_text",
            err=True,
        )
        raise typer.Exit(1)

    # Parse post-processing steps
    postprocess_steps: list[PostProcessingStep] = []
    if postprocess:
        for step_name in postprocess.split(","):
            step_name = step_name.strip()
            try:
                postprocess_steps.append(PostProcessingStep(step_name))
            except ValueError:
                typer.echo(
                    f"Warning: Unknown post-processing step '{step_name}', skipping.",
                    err=True,
                )

    # Create pipeline and run
    try:
        pipeline = LLMAnnotationPipeline(
            model=model,
            temperature=temperature,
            validate_hpo_ids=not no_validate,
        )

        typer.echo(f"Annotating with {model} (mode: {mode})...", err=True)

        result = pipeline.run(
            text=input_text,
            mode=annotation_mode,
            language=language,
            postprocess=postprocess_steps if postprocess_steps else None,
        )

        # Enrich with details if requested
        if include_details:
            result = _enrich_with_details(result)

        # Format output
        output = _format_output(result, output_format)

        # Write output
        if output_file:
            output_file.write_text(output, encoding="utf-8")
            typer.echo(f"Output written to: {output_file}", err=True)
        else:
            typer.echo(output)

    except Exception as e:
        logger.exception("Annotation failed")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="benchmark")
def benchmark(
    test_file: Annotated[
        Path,
        typer.Option(
            "--test-file",
            "-t",
            help="Path to benchmark test file (JSON).",
            exists=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use. Default can be set via PHENTRIEVE_LLM_MODEL env var.",
        ),
    ] = DEFAULT_LLM_MODEL,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Annotation mode: direct, tool_term, or tool_text.",
        ),
    ] = "tool_text",
    postprocess: Annotated[
        Optional[str],
        typer.Option(
            "--postprocess",
            "-p",
            help="Comma-separated post-processing steps.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for benchmark results.",
        ),
    ] = Path("data/results/llm"),
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            "-n",
            help="Limit number of test cases (for quick testing).",
        ),
    ] = None,
) -> None:
    """
    Run LLM annotation benchmark.

    Evaluates LLM annotation performance against a gold standard test set
    using the existing Phentrieve evaluation metrics.

    Examples:
        # Basic benchmark
        phentrieve llm benchmark --test-file tests/data/benchmarks/german/tiny_v1.json

        # With specific model and mode
        phentrieve llm benchmark -t tests/data/benchmarks/german/70cases_gemini_v1.json \\
            --model gemini/gemini-1.5-pro --mode tool_text

        # Quick test with limit
        phentrieve llm benchmark -t test.json --limit 5
    """
    _check_litellm_installed()

    from phentrieve.data_processing.test_data_loader import load_test_data
    from phentrieve.llm import AnnotationMode, LLMAnnotationPipeline, PostProcessingStep

    try:
        annotation_mode = AnnotationMode(mode)
    except ValueError:
        typer.echo(f"Error: Invalid mode '{mode}'", err=True)
        raise typer.Exit(1)

    # Parse post-processing steps
    postprocess_steps: list[PostProcessingStep] = []
    if postprocess:
        for step_name in postprocess.split(","):
            step_name = step_name.strip()
            try:
                postprocess_steps.append(PostProcessingStep(step_name))
            except ValueError:
                typer.echo(f"Warning: Unknown step '{step_name}'", err=True)

    # Load test data
    typer.echo(f"Loading test data from: {test_file}", err=True)
    test_cases = load_test_data(str(test_file))

    if test_cases is None:
        typer.echo("Error: Failed to load test data.", err=True)
        raise typer.Exit(1)

    if limit:
        test_cases = test_cases[:limit]
        typer.echo(f"Limited to {limit} test cases", err=True)

    typer.echo(f"Running benchmark with {len(test_cases)} cases...", err=True)

    # Create pipeline
    pipeline = LLMAnnotationPipeline(model=model)

    # Run benchmark
    results = []
    for i, case in enumerate(test_cases):
        typer.echo(f"  [{i + 1}/{len(test_cases)}] Processing...", err=True)

        result = pipeline.run(
            text=case.get("input_text", case.get("text", "")),
            mode=annotation_mode,
            language="auto",
            postprocess=postprocess_steps if postprocess_steps else None,
        )

        # Compare to expected
        expected_ids = set(case.get("expected_hpo_ids", []))
        predicted_ids = set(result.hpo_ids)

        results.append(
            {
                "case_id": case.get("id", i),
                "input_text": result.input_text[:100] + "...",
                "expected": list(expected_ids),
                "predicted": list(predicted_ids),
                "precision": len(expected_ids & predicted_ids) / len(predicted_ids)
                if predicted_ids
                else 0,
                "recall": len(expected_ids & predicted_ids) / len(expected_ids)
                if expected_ids
                else 0,
                "processing_time": result.processing_time_seconds,
            }
        )

    # Calculate overall metrics
    total_precision = (
        sum(r["precision"] for r in results) / len(results) if results else 0
    )
    total_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"benchmark_{model.replace('/', '_')}_{mode}.json"

    benchmark_output = {
        "model": model,
        "mode": mode,
        "postprocess": [s.value for s in postprocess_steps],
        "metrics": {
            "precision": total_precision,
            "recall": total_recall,
            "f1": f1,
            "num_cases": len(results),
        },
        "results": results,
    }

    results_file.write_text(json.dumps(benchmark_output, indent=2), encoding="utf-8")

    # Display summary
    typer.echo("\n=== Benchmark Results ===", err=True)
    typer.echo(f"Model: {model}", err=True)
    typer.echo(f"Mode: {mode}", err=True)
    typer.echo(f"Cases: {len(results)}", err=True)
    typer.echo(f"Precision: {total_precision:.3f}", err=True)
    typer.echo(f"Recall: {total_recall:.3f}", err=True)
    typer.echo(f"F1: {f1:.3f}", err=True)
    typer.echo(f"\nResults saved to: {results_file}", err=True)


@app.command(name="compare")
def compare(
    test_file: Annotated[
        Path,
        typer.Option(
            "--test-file",
            "-t",
            help="Path to benchmark test file (JSON).",
            exists=True,
        ),
    ],
    models: Annotated[
        str,
        typer.Option(
            "--models",
            "-m",
            help="Comma-separated list of models to compare. "
            "Default can be set via PHENTRIEVE_LLM_MODEL env var.",
        ),
    ] = DEFAULT_LLM_MODEL,
    modes: Annotated[
        str,
        typer.Option(
            "--modes",
            help="Comma-separated modes: direct,tool_term,tool_text",
        ),
    ] = "direct,tool_term,tool_text",
    postprocess_configs: Annotated[
        str,
        typer.Option(
            "--postprocess",
            "-p",
            help="Comma-separated post-processing configs to compare (e.g., 'none,validation,all').",
        ),
    ] = "none",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for comparison results.",
        ),
    ] = Path("data/results/llm"),
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            "-n",
            help="Limit test cases per configuration.",
        ),
    ] = None,
) -> None:
    """
    Compare multiple LLM annotation configurations.

    Run benchmarks across different models, modes, and post-processing
    configurations to find the best setup.

    Examples:
        # Compare modes
        phentrieve llm compare -t test.json --modes direct,tool_term,tool_text

        # Compare models
        phentrieve llm compare -t test.json \\
            --models github/gpt-4o,gemini/gemini-1.5-flash

        # Full comparison
        phentrieve llm compare -t test.json \\
            --models github/gpt-4o,gemini/gemini-1.5-pro \\
            --modes direct,tool_text \\
            --postprocess none,validation
    """
    _check_litellm_installed()

    model_list = [m.strip() for m in models.split(",")]
    mode_list = [m.strip() for m in modes.split(",")]
    postprocess_list = [p.strip() for p in postprocess_configs.split(",")]

    total_configs = len(model_list) * len(mode_list) * len(postprocess_list)
    typer.echo(f"Comparing {total_configs} configurations...", err=True)

    comparison_results: list[dict[str, Any]] = []
    current = 0

    for model in model_list:
        for mode in mode_list:
            for pp_config in postprocess_list:
                current += 1
                typer.echo(
                    f"\n[{current}/{total_configs}] {model} / {mode} / postprocess={pp_config}",
                    err=True,
                )

                # Build postprocess arg
                if pp_config == "none":
                    pp_arg = None
                elif pp_config == "all":
                    pp_arg = "validation,refinement,assertion_review"
                else:
                    pp_arg = pp_config

                # Run benchmark (reuse the benchmark function logic)
                # For brevity, we'll call the internal logic here
                try:
                    from phentrieve.data_processing.test_data_loader import (
                        load_test_data,
                    )
                    from phentrieve.llm import (
                        AnnotationMode,
                        LLMAnnotationPipeline,
                        PostProcessingStep,
                    )

                    annotation_mode = AnnotationMode(mode)
                    loaded_cases = load_test_data(str(test_file))

                    if loaded_cases is None:
                        raise ValueError("Failed to load test data")

                    cases_to_process = loaded_cases[:limit] if limit else loaded_cases

                    postprocess_steps: list[PostProcessingStep] = []
                    if pp_arg:
                        for step_name in pp_arg.split(","):
                            try:
                                postprocess_steps.append(
                                    PostProcessingStep(step_name.strip())
                                )
                            except ValueError:
                                pass

                    pipeline = LLMAnnotationPipeline(model=model)

                    metrics: dict[str, list[float]] = {
                        "precision": [],
                        "recall": [],
                        "time": [],
                    }
                    for case in cases_to_process:
                        result = pipeline.run(
                            text=case.get("input_text", case.get("text", "")),
                            mode=annotation_mode,
                            language="auto",
                            postprocess=postprocess_steps
                            if postprocess_steps
                            else None,
                        )

                        expected = set(case.get("expected_hpo_ids", []))
                        predicted = set(result.hpo_ids)

                        prec = (
                            len(expected & predicted) / len(predicted)
                            if predicted
                            else 0
                        )
                        rec = (
                            len(expected & predicted) / len(expected) if expected else 0
                        )

                        metrics["precision"].append(prec)
                        metrics["recall"].append(rec)
                        metrics["time"].append(result.processing_time_seconds or 0)

                    avg_p = (
                        sum(metrics["precision"]) / len(metrics["precision"])
                        if metrics["precision"]
                        else 0
                    )
                    avg_r = (
                        sum(metrics["recall"]) / len(metrics["recall"])
                        if metrics["recall"]
                        else 0
                    )
                    f1 = (
                        2 * avg_p * avg_r / (avg_p + avg_r)
                        if (avg_p + avg_r) > 0
                        else 0
                    )

                    comparison_results.append(
                        {
                            "model": model,
                            "mode": mode,
                            "postprocess": pp_config,
                            "precision": avg_p,
                            "recall": avg_r,
                            "f1": f1,
                            "avg_time": sum(metrics["time"]) / len(metrics["time"])
                            if metrics["time"]
                            else 0,
                        }
                    )

                    typer.echo(f"  P={avg_p:.3f} R={avg_r:.3f} F1={f1:.3f}", err=True)

                except Exception as e:
                    typer.echo(f"  Error: {e}", err=True)
                    comparison_results.append(
                        {
                            "model": model,
                            "mode": mode,
                            "postprocess": pp_config,
                            "error": str(e),
                        }
                    )

    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = output_dir / "comparison.json"
    comparison_file.write_text(
        json.dumps(comparison_results, indent=2), encoding="utf-8"
    )

    # Display summary table
    typer.echo("\n=== Comparison Summary ===", err=True)
    typer.echo(
        f"{'Model':<30} {'Mode':<12} {'PP':<15} {'P':<8} {'R':<8} {'F1':<8}", err=True
    )
    typer.echo("-" * 90, err=True)

    for comp_result in sorted(
        comparison_results, key=lambda x: x.get("f1", 0), reverse=True
    ):
        if "error" in comp_result:
            typer.echo(
                f"{comp_result['model']:<30} {comp_result['mode']:<12} {comp_result['postprocess']:<15} ERROR",
                err=True,
            )
        else:
            typer.echo(
                f"{comp_result['model']:<30} {comp_result['mode']:<12} {comp_result['postprocess']:<15} "
                f"{comp_result['precision']:<8.3f} {comp_result['recall']:<8.3f} {comp_result['f1']:<8.3f}",
                err=True,
            )

    typer.echo(f"\nResults saved to: {comparison_file}", err=True)


@app.command(name="models")
def list_models() -> None:
    """
    List available LLM models.

    Shows model presets organized by provider, with authentication status.
    """
    from phentrieve.llm import get_available_models

    models = get_available_models()

    auth_vars = {
        "github": "GITHUB_TOKEN",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    typer.echo("Available LLM Models")
    typer.echo("=" * 50)

    # Show current default model
    env_model = os.environ.get("PHENTRIEVE_LLM_MODEL")
    if env_model:
        typer.echo(f"\nDefault model (from PHENTRIEVE_LLM_MODEL): {env_model}")
    else:
        typer.echo(f"\nDefault model: {DEFAULT_LLM_MODEL}")

    for provider, model_list in models.items():
        # Check auth status
        auth_var = auth_vars.get(provider)
        if auth_var:
            has_auth = bool(os.environ.get(auth_var))
            status = "[OK]" if has_auth else f"[needs {auth_var}]"
        else:
            status = "[local]"

        typer.echo(f"\n{provider.upper()} {status}")
        for m in model_list:
            typer.echo(f"  - {m}")

    typer.echo("\n\nUsage:")
    typer.echo("  phentrieve llm annotate 'text' --model github/gpt-4o")
    typer.echo("\nTo set default model:")
    typer.echo("  export PHENTRIEVE_LLM_MODEL=gemini/gemini-2.0-flash")
    typer.echo("\nTo set authentication:")
    typer.echo("  export GITHUB_TOKEN=ghp_...")
    typer.echo("  export GEMINI_API_KEY=...")


def _format_output(result: Any, format_type: str) -> str:
    """Format annotation result for output."""

    if format_type == "json":
        return json.dumps(result.to_dict(), indent=2)

    elif format_type == "phenopacket":
        try:
            from phentrieve.phenopackets.utils import format_as_phenopacket_v2

            # Convert annotations to the format expected by phenopacket formatter
            aggregated_results = []
            for ann in result.annotations:
                aggregated_results.append(
                    {
                        "hpo_id": ann.hpo_id,
                        "term_name": ann.term_name,
                        "assertion": ann.assertion.value,
                        "score": ann.confidence,
                        "evidence_text": ann.evidence_text,
                    }
                )

            # format_as_phenopacket_v2 returns a JSON string
            phenopacket_json = format_as_phenopacket_v2(
                aggregated_results=aggregated_results,
                embedding_model=result.model,
                input_text=result.input_text,
            )
            # It's already a JSON string, just return it
            return phenopacket_json
        except ImportError:
            return json.dumps({"error": "Phenopacket formatter not available"})

    else:  # text format
        lines = []
        lines.append(f"Model: {result.model}")
        lines.append(f"Mode: {result.mode.value}")
        lines.append(f"Language: {result.language}")
        if result.processing_time_seconds:
            lines.append(f"Time: {result.processing_time_seconds:.2f}s")
        lines.append("")

        if result.annotations:
            lines.append("Annotations:")
            for ann in result.annotations:
                status = f"[{ann.assertion.value.upper()}]"
                conf = f"({ann.confidence:.2f})"
                lines.append(f"  {ann.hpo_id}: {ann.term_name} {status} {conf}")
                if ann.evidence_text:
                    lines.append(f'    Evidence: "{ann.evidence_text}"')
        else:
            lines.append("No annotations found.")

        # Add token usage section
        if result.token_usage.api_calls > 0:
            lines.append("")
            lines.append("Token Usage:")
            lines.append(f"  Input tokens:  {result.token_usage.prompt_tokens:,}")
            lines.append(f"  Output tokens: {result.token_usage.completion_tokens:,}")
            lines.append(f"  Total tokens:  {result.token_usage.total_tokens:,}")
            lines.append(f"  API calls:     {result.token_usage.api_calls}")

        return "\n".join(lines)


def _enrich_with_details(result: Any) -> Any:
    """Add HPO definitions and synonyms to annotations."""
    try:
        from pathlib import Path

        from phentrieve.config import DEFAULT_HPO_DB_FILENAME
        from phentrieve.data_processing.hpo_database import HPODatabase
        from phentrieve.utils import get_default_data_dir

        # Search multiple locations for the HPO database
        candidates = [
            get_default_data_dir() / DEFAULT_HPO_DB_FILENAME,  # User config dir
            Path.cwd() / "data" / DEFAULT_HPO_DB_FILENAME,  # Project ./data
            Path(__file__).resolve().parents[2]
            / "data"
            / DEFAULT_HPO_DB_FILENAME,  # Package root
        ]

        db_path = None
        for candidate in candidates:
            if candidate.exists():
                db_path = candidate
                break

        if db_path is None:
            return result

        db = HPODatabase(db_path)
        hpo_ids = [a.hpo_id for a in result.annotations]
        terms_by_id: dict[str, dict[str, Any]] = db.get_terms_by_ids(hpo_ids)

        for ann in result.annotations:
            if ann.hpo_id in terms_by_id:
                term_data = terms_by_id[ann.hpo_id]
                ann.definition = term_data.get("definition")
                ann.synonyms = term_data.get("synonyms", [])

    except Exception as e:
        logger.warning("Failed to enrich with details: %s", e)

    return result
