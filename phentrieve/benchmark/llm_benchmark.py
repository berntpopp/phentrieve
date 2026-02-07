"""LLM annotation benchmark for document-level HPO extraction evaluation.

This module evaluates LLM-based annotation modes (DIRECT, TOOL_TERM, TOOL_TEXT)
against PhenoBERT gold-standard datasets or simple benchmark JSON files.

Key components:
- LLMBenchmarkConfig: Configuration for benchmark parameters
- LLMBenchmark: Main benchmark runner with assertion-aware metrics
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from phentrieve.benchmark.data_loader import (
    LLM_ASSERTION_TO_BENCHMARK,
    load_phenobert_data,
    parse_gold_terms,
)
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    CorpusMetrics,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMBenchmarkConfig:
    """Configuration for LLM annotation benchmark."""

    model: str = "github/gpt-4o"
    modes: list[str] = field(default_factory=lambda: ["tool_text"])
    dataset: str = "all"
    include_assertions: bool = True
    postprocess: str | None = None
    temperature: float = 0.0
    limit: int | None = None
    averaging: str = "micro"
    bootstrap_ci: bool = False
    bootstrap_samples: int = 1000
    validate_hpo_ids: bool = True
    debug: bool = False


class LLMBenchmark:
    """Main benchmark runner for LLM-based HPO annotation.

    Evaluates LLM annotation modes against gold-standard datasets,
    computing assertion-aware and ID-only metrics using the same
    evaluation framework as ``ExtractionBenchmark``.
    """

    def __init__(self, config: LLMBenchmarkConfig) -> None:
        self.config = config

    def _setup_debug_logging(self) -> None:
        """Enable DEBUG-level logging for phentrieve.llm and benchmark loggers."""
        import sys

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("[%(levelname)s %(name)s] %(message)s"))

        # Attach handler to the two top-level parent loggers and stop propagation
        # so messages are not duplicated by the root logger or other handlers.
        for name in ("phentrieve.llm", "phentrieve.benchmark"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.DEBUG)
            lg.handlers.clear()
            lg.addHandler(handler)
            lg.propagate = False

    def run(
        self,
        test_path: Path,
        output_dir: Path,
    ) -> dict[str, CorpusMetrics]:
        """Run the benchmark on all configured modes.

        Args:
            test_path: Path to PhenoBERT directory or simple JSON file.
            output_dir: Directory for saving results and predictions.

        Returns:
            Dict mapping mode name to CorpusMetrics.
        """
        from phentrieve.llm import (
            AnnotationMode,
            LLMAnnotationPipeline,
            PostProcessingStep,
            TokenUsage,
            estimate_cost,
        )

        if self.config.debug:
            self._setup_debug_logging()

        # Load test data
        logger.debug("[BENCHMARK] Loading test data from: %s", test_path)
        t0 = time.time()
        test_data = self._load_test_data(test_path)
        documents = test_data["documents"]
        logger.debug(
            "[BENCHMARK] Loaded %d documents in %.2fs (metadata: %s)",
            len(documents),
            time.time() - t0,
            test_data.get("metadata", {}),
        )

        if self.config.limit:
            documents = documents[: self.config.limit]
            logger.debug("[BENCHMARK] Limited to %d documents", len(documents))

        # Parse post-processing steps
        postprocess_steps: list[PostProcessingStep] = []
        if self.config.postprocess:
            for step_name in self.config.postprocess.split(","):
                step_name = step_name.strip()
                try:
                    postprocess_steps.append(PostProcessingStep(step_name))
                except ValueError:
                    logger.warning(f"Unknown post-processing step: {step_name}")
        logger.debug(
            "[BENCHMARK] Post-processing steps: %s",
            [s.value for s in postprocess_steps] if postprocess_steps else "none",
        )

        # Create pipeline
        logger.debug(
            "[BENCHMARK] Creating pipeline: model=%s, temperature=%s, validate=%s",
            self.config.model,
            self.config.temperature,
            self.config.validate_hpo_ids,
        )
        t0 = time.time()
        pipeline = LLMAnnotationPipeline(
            model=self.config.model,
            temperature=self.config.temperature,
            validate_hpo_ids=self.config.validate_hpo_ids,
        )
        logger.debug(
            "[BENCHMARK] Pipeline created in %.2fs (provider: %s)",
            time.time() - t0,
            pipeline.provider.provider,
        )

        all_mode_metrics: dict[str, CorpusMetrics] = {}

        logger.debug("[BENCHMARK] Modes to evaluate: %s", self.config.modes)

        for mode_str in self.config.modes:
            try:
                annotation_mode = AnnotationMode(mode_str)
            except ValueError:
                logger.error(f"Invalid mode: {mode_str}")
                continue

            print(
                f"\n--- Mode: {mode_str} ({len(documents)} documents) ---",
                file=sys.stderr,
            )
            logger.debug("[BENCHMARK] Starting mode: %s", mode_str)

            results: list[ExtractionResult] = []
            id_only_results: list[ExtractionResult] = []
            total_token_usage = TokenUsage()
            prediction_records: list[dict[str, Any]] = []
            doc_stats: list[dict[str, Any]] = []
            mode_start_time = time.time()

            for idx, doc in enumerate(documents):
                doc_id = doc["id"]
                text = doc["text"]
                logger.debug(
                    "[BENCHMARK] [%d/%d] Processing document %s (%d chars)",
                    idx + 1,
                    len(documents),
                    doc_id,
                    len(text),
                )
                start_time = time.time()
                try:
                    result = pipeline.run(
                        text=text,
                        mode=annotation_mode,
                        language="auto",
                        postprocess=postprocess_steps if postprocess_steps else None,
                    )
                except Exception:
                    logger.exception(f"Failed to process document {doc_id}")
                    continue
                processing_time = time.time() - start_time
                logger.debug(
                    "[BENCHMARK] [%d/%d] Document %s done in %.2fs — "
                    "%d annotations, language=%s, tokens=%d, tool_calls=%d",
                    idx + 1,
                    len(documents),
                    doc_id,
                    processing_time,
                    len(result.annotations),
                    result.language,
                    result.token_usage.total_tokens,
                    len(result.tool_calls),
                )

                # Accumulate token usage
                total_token_usage.merge(result.token_usage)

                # Convert LLM annotations to (hpo_id, assertion) tuples
                # Use result.annotations (all) not result.hpo_ids (affirmed only)
                predicted_with_assertions: list[tuple[str, str]] = []
                predicted_ids_only: list[tuple[str, str]] = []
                for ann in result.annotations:
                    assertion = LLM_ASSERTION_TO_BENCHMARK.get(
                        ann.assertion.value, "PRESENT"
                    )
                    predicted_with_assertions.append((ann.hpo_id, assertion))
                    predicted_ids_only.append((ann.hpo_id, "PRESENT"))

                # Parse gold standard
                gold = parse_gold_terms(doc["gold_hpo_terms"])
                gold_ids_only = [(hpo_id, "PRESENT") for hpo_id, _ in gold]

                # Assertion-aware results
                results.append(
                    ExtractionResult(
                        doc_id=doc_id,
                        predicted=predicted_with_assertions,
                        gold=gold,
                    )
                )

                # ID-only results (ignore assertion status)
                id_only_results.append(
                    ExtractionResult(
                        doc_id=doc_id,
                        predicted=predicted_ids_only,
                        gold=gold_ids_only,
                    )
                )

                # Per-document metrics for progress reporting
                doc_pred_set = set(predicted_with_assertions)
                doc_gold_set = set(gold)
                doc_tp = len(doc_pred_set & doc_gold_set)
                doc_fp = len(doc_pred_set - doc_gold_set)
                doc_fn = len(doc_gold_set - doc_pred_set)
                doc_p = doc_tp / (doc_tp + doc_fp) if (doc_tp + doc_fp) > 0 else 0.0
                doc_r = doc_tp / (doc_tp + doc_fn) if (doc_tp + doc_fn) > 0 else 0.0
                doc_f1 = (
                    2 * doc_p * doc_r / (doc_p + doc_r) if (doc_p + doc_r) > 0 else 0.0
                )

                # Per-document cost estimation
                doc_cost = estimate_cost(result.token_usage, self.config.model)
                doc_cost_str = f"${doc_cost['total_cost']:.4f}" if doc_cost else ""
                llm_t = result.token_usage.llm_time_seconds
                tool_t = result.token_usage.tool_time_seconds
                timing_parts = f"LLM: {llm_t:.1f}s, Tool: {tool_t:.1f}s"

                print(
                    f"  [{idx + 1}/{len(documents)}] {doc_id} — "
                    f"{len(predicted_with_assertions)} pred, {len(gold)} gold, "
                    f"P={doc_p:.3f} R={doc_r:.3f} F1={doc_f1:.3f}\n"
                    f"       Time: {processing_time:.1f}s ({timing_parts}) | "
                    f"Tokens: {result.token_usage.total_tokens:,}"
                    + (f" | Cost: {doc_cost_str}" if doc_cost_str else ""),
                    file=sys.stderr,
                )

                # Print per-event timing breakdown
                # Build lookup of postprocess timing from timing events
                pp_timing: dict[str, float] = {}
                if result.token_usage.timing_events:
                    for event in result.token_usage.timing_events:
                        if event.category == "postprocess":
                            # Extract step name from label like "postprocess: validation (...)"
                            for ps in result.post_processing_stats:
                                if ps.step in event.label:
                                    pp_timing[ps.step] = event.duration_seconds
                        else:
                            print(
                                f"         {event.category:>11}: {event.duration_seconds:>6.2f}s  {event.label}",
                                file=sys.stderr,
                            )

                # Print per-document post-processing stats (with timing folded in)
                if result.post_processing_stats:
                    for ps in result.post_processing_stats:
                        pp_parts: list[str] = [
                            f"{ps.annotations_in}->{ps.annotations_out}"
                        ]
                        if ps.removed:
                            pp_parts.append(f"{ps.removed} removed")
                        if ps.added:
                            pp_parts.append(f"{ps.added} added")
                        if ps.assertions_changed:
                            pp_parts.append(
                                f"{ps.assertions_changed} assertions changed"
                            )
                        if ps.terms_refined:
                            pp_parts.append(f"{ps.terms_refined} terms refined")
                        step_time = pp_timing.get(ps.step)
                        time_str = f" ({step_time:.1f}s)" if step_time else ""
                        print(
                            f"      >> {ps.step}: {', '.join(pp_parts)}{time_str}",
                            file=sys.stderr,
                        )

                doc_stats.append(
                    {
                        "doc_id": doc_id,
                        "time": processing_time,
                        "llm_time": llm_t,
                        "tool_time": tool_t,
                        "tokens": result.token_usage.total_tokens,
                        "cost": doc_cost,
                        "pp_stats": [s.to_dict() for s in result.post_processing_stats],
                    }
                )

                # Build prediction record in PhenoBERT format
                prediction_records.append(
                    self._convert_to_phenobert_format(
                        doc_id=doc_id,
                        text=text,
                        result=result,
                        processing_time=processing_time,
                        estimated_cost=doc_cost,
                    )
                )

            if not results:
                logger.warning(f"No results for mode {mode_str}")
                continue

            # Calculate metrics
            logger.debug(
                "[BENCHMARK] Calculating metrics for mode %s (%d results, averaging=%s)",
                mode_str,
                len(results),
                self.config.averaging,
            )
            t0 = time.time()
            evaluator = CorpusExtractionMetrics(averaging=self.config.averaging)

            assertion_metrics = evaluator.calculate_all_metrics(results)
            id_only_metrics = evaluator.calculate_all_metrics(id_only_results)

            # Bootstrap CI if requested
            if self.config.bootstrap_ci:
                ci = evaluator.bootstrap_confidence_intervals(
                    results, n_bootstrap=self.config.bootstrap_samples
                )
                assertion_metrics = CorpusMetrics(
                    micro=assertion_metrics.micro,
                    macro=assertion_metrics.macro,
                    weighted=assertion_metrics.weighted,
                    confidence_intervals=ci,
                )
            logger.debug("[BENCHMARK] Metrics calculated in %.2fs", time.time() - t0)

            mode_elapsed = time.time() - mode_start_time
            avg_time = mode_elapsed / len(results) if results else 0

            # Per-document summary table
            if doc_stats:
                self._print_doc_summary_table(doc_stats)

            print(
                f"\n  Done: {len(results)} documents in {mode_elapsed:.1f}s "
                f"({avg_time:.1f}s avg)",
                file=sys.stderr,
            )
            print(
                f"  Tokens: {total_token_usage.total_tokens:,} total "
                f"({total_token_usage.prompt_tokens:,} prompt + "
                f"{total_token_usage.completion_tokens:,} completion), "
                f"{total_token_usage.api_calls} API calls",
                file=sys.stderr,
            )
            print(
                f"  Timing: {total_token_usage.llm_time_seconds:.1f}s LLM + "
                f"{total_token_usage.tool_time_seconds:.1f}s tool = "
                f"{total_token_usage.llm_time_seconds + total_token_usage.tool_time_seconds:.1f}s "
                f"(wall-clock: {mode_elapsed:.1f}s)",
                file=sys.stderr,
            )
            total_cost = estimate_cost(total_token_usage, self.config.model)
            if total_cost:
                print(
                    f"  Cost: ${total_cost['total_cost']:.4f} "
                    f"(input: ${total_cost['input_cost']:.4f}, "
                    f"output: ${total_cost['output_cost']:.4f})",
                    file=sys.stderr,
                )

            # Aggregate post-processing impact table
            if doc_stats and any(d["pp_stats"] for d in doc_stats):
                self._print_postprocessing_table(doc_stats)

            all_mode_metrics[mode_str] = assertion_metrics

            # Save results
            logger.debug(
                "[BENCHMARK] Saving results for mode %s to %s", mode_str, output_dir
            )
            t0 = time.time()
            self._save_results(
                mode_str=mode_str,
                output_dir=output_dir,
                results=results,
                assertion_metrics=assertion_metrics,
                id_only_metrics=id_only_metrics,
                token_usage=total_token_usage,
                prediction_records=prediction_records,
                metadata=test_data.get("metadata", {}),
                estimated_cost=total_cost,
            )
            logger.debug("[BENCHMARK] Results saved in %.2fs", time.time() - t0)

        return all_mode_metrics

    def _load_test_data(self, test_path: Path) -> dict[str, Any]:
        """Load test dataset from JSON file or PhenoBERT directory."""
        if test_path.is_dir():
            return load_phenobert_data(test_path, dataset=self.config.dataset)
        else:
            # Simple JSON format: [{text, expected_hpo_ids}]
            with open(test_path) as f:
                raw_data = json.load(f)

            # Normalize simple format to documents format
            if isinstance(raw_data, list):
                documents = []
                for i, case in enumerate(raw_data):
                    expected = case.get("expected_hpo_ids", [])
                    gold_terms = [
                        {"id": hpo_id, "assertion": "PRESENT"} for hpo_id in expected
                    ]
                    documents.append(
                        {
                            "id": case.get("id", str(i)),
                            "text": case.get("input_text", case.get("text", "")),
                            "gold_hpo_terms": gold_terms,
                            "source_dataset": "simple_json",
                        }
                    )
                return {
                    "metadata": {
                        "dataset_name": test_path.stem,
                        "source": "simple_json",
                        "total_documents": len(documents),
                    },
                    "documents": documents,
                }
            else:
                # Already in {metadata, documents} format
                return raw_data  # type: ignore[no-any-return]

    def _convert_to_phenobert_format(
        self,
        doc_id: str,
        text: str,
        result: Any,
        processing_time: float,
        estimated_cost: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Convert LLM annotation result to PhenoBERT-compatible format.

        Args:
            doc_id: Document identifier.
            text: Original document text.
            result: AnnotationResult from the LLM pipeline.
            processing_time: Time taken to process the document.

        Returns:
            PhenoBERT-format prediction dict.
        """
        annotations = []
        for ann in result.annotations:
            evidence_spans = []
            if ann.evidence_text:
                span: dict[str, Any] = {"text_snippet": ann.evidence_text}
                if ann.evidence_start is not None:
                    span["start_char"] = ann.evidence_start
                if ann.evidence_end is not None:
                    span["end_char"] = ann.evidence_end
                evidence_spans.append(span)

            annotations.append(
                {
                    "hpo_id": ann.hpo_id,
                    "label": ann.term_name,
                    "assertion_status": ann.assertion.value,
                    "evidence_spans": evidence_spans,
                }
            )

        return {
            "doc_id": doc_id,
            "language": result.language,
            "source": "llm_annotation",
            "full_text": text,
            "metadata": {
                "model": result.model,
                "mode": result.mode.value,
                "temperature": result.temperature,
                "token_usage": result.token_usage.to_dict(),
                "processing_time_seconds": processing_time,
                "timing_breakdown": {
                    "total_seconds": round(processing_time, 3),
                    "llm_seconds": round(result.token_usage.llm_time_seconds, 3),
                    "tool_seconds": round(result.token_usage.tool_time_seconds, 3),
                },
                "estimated_cost": estimated_cost,
            },
            "annotations": annotations,
        }

    def _print_doc_summary_table(self, doc_stats: list[dict[str, Any]]) -> None:
        """Print a per-document timing/cost summary table."""
        has_cost = any(d["cost"] is not None for d in doc_stats)
        # Header
        header = f"  {'Document':<28} {'Time':>6} {'LLM':>6} {'Tool':>6} {'Tokens':>8}"
        if has_cost:
            header += f" {'Cost':>9}"
        print(f"\n{header}", file=sys.stderr)

        # Rows
        for d in doc_stats:
            row = (
                f"  {d['doc_id']:<28} {d['time']:>5.1f}s {d['llm_time']:>5.1f}s "
                f"{d['tool_time']:>5.1f}s {d['tokens']:>8,}"
            )
            if has_cost and d["cost"]:
                row += f" ${d['cost']['total_cost']:>8.4f}"
            print(row, file=sys.stderr)

        # Separator + totals
        sep_len = 62 + (10 if has_cost else 0)
        print(f"  {'─' * sep_len}", file=sys.stderr)

        total_time = sum(d["time"] for d in doc_stats)
        total_llm = sum(d["llm_time"] for d in doc_stats)
        total_tool = sum(d["tool_time"] for d in doc_stats)
        total_tokens = sum(d["tokens"] for d in doc_stats)
        n = len(doc_stats)

        total_row = (
            f"  {'Total':<28} {total_time:>5.1f}s {total_llm:>5.1f}s "
            f"{total_tool:>5.1f}s {total_tokens:>8,}"
        )
        if has_cost:
            total_cost = sum(d["cost"]["total_cost"] for d in doc_stats if d["cost"])
            total_row += f" ${total_cost:>8.4f}"
        print(total_row, file=sys.stderr)

        avg_row = (
            f"  {'Average':<28} {total_time / n:>5.1f}s {total_llm / n:>5.1f}s "
            f"{total_tool / n:>5.1f}s {total_tokens // n:>8,}"
        )
        if has_cost:
            avg_cost = sum(d["cost"]["total_cost"] for d in doc_stats if d["cost"]) / n
            avg_row += f" ${avg_cost:>8.4f}"
        print(avg_row, file=sys.stderr)

    def _print_postprocessing_table(self, doc_stats: list[dict[str, Any]]) -> None:
        """Print an aggregate post-processing impact table."""
        # Collect unique step names in order of first appearance
        step_names: list[str] = []
        for d in doc_stats:
            for ps in d["pp_stats"]:
                if ps["step"] not in step_names:
                    step_names.append(ps["step"])

        if not step_names:
            return

        # Aggregate per step
        agg: dict[str, dict[str, int]] = {}
        for d in doc_stats:
            for ps in d["pp_stats"]:
                name = ps["step"]
                if name not in agg:
                    agg[name] = {
                        "docs": 0,
                        "in": 0,
                        "out": 0,
                        "removed": 0,
                        "added": 0,
                        "assertions_changed": 0,
                        "terms_refined": 0,
                    }
                agg[name]["docs"] += 1
                agg[name]["in"] += ps["annotations_in"]
                agg[name]["out"] += ps["annotations_out"]
                agg[name]["removed"] += ps["removed"]
                agg[name]["added"] += ps["added"]
                agg[name]["assertions_changed"] += ps["assertions_changed"]
                agg[name]["terms_refined"] += ps["terms_refined"]

        # Determine which columns have non-zero values
        show_removed = any(v["removed"] for v in agg.values())
        show_added = any(v["added"] for v in agg.values())
        show_assert = any(v["assertions_changed"] for v in agg.values())
        show_refined = any(v["terms_refined"] for v in agg.values())

        # Header
        header = f"\n  {'Step':<20} {'Docs':>5} {'In':>5} {'Out':>5}"
        if show_removed:
            header += f" {'Removed':>8}"
        if show_added:
            header += f" {'Added':>6}"
        if show_assert:
            header += f" {'Assert':>8}"
        if show_refined:
            header += f" {'Refined':>8}"
        print(header, file=sys.stderr)

        # Rows
        for name in step_names:
            c = agg[name]
            row = f"  {name:<20} {c['docs']:>5} {c['in']:>5} {c['out']:>5}"
            if show_removed:
                row += f" {c['removed']:>8}"
            if show_added:
                row += f" {c['added']:>6}"
            if show_assert:
                row += f" {c['assertions_changed']:>8}"
            if show_refined:
                row += f" {c['terms_refined']:>8}"
            print(row, file=sys.stderr)

    def _save_results(
        self,
        mode_str: str,
        output_dir: Path,
        results: list[ExtractionResult],
        assertion_metrics: CorpusMetrics,
        id_only_metrics: CorpusMetrics,
        token_usage: Any,
        prediction_records: list[dict[str, Any]],
        metadata: dict[str, Any],
        estimated_cost: dict[str, float] | None = None,
    ) -> None:
        """Save benchmark results and predictions to disk."""
        # Save per-document predictions in PhenoBERT format
        predictions_dir = output_dir / "predictions" / mode_str
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for record in prediction_records:
            pred_file = predictions_dir / f"{record['doc_id']}.json"
            with open(pred_file, "w") as f:
                json.dump(record, f, indent=2)

        # Save aggregate metrics
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = metrics_dir / f"benchmark_{mode_str}.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "model": self.config.model,
                        "mode": mode_str,
                        "temperature": self.config.temperature,
                        "postprocess": self.config.postprocess,
                        "timestamp": datetime.now().isoformat(),
                        "dataset": metadata,
                        "num_documents": len(results),
                    },
                    "assertion_aware_metrics": {
                        "micro": assertion_metrics.micro,
                        "macro": assertion_metrics.macro,
                        "weighted": assertion_metrics.weighted,
                        "confidence_intervals": assertion_metrics.confidence_intervals,
                    },
                    "id_only_metrics": {
                        "micro": id_only_metrics.micro,
                        "macro": id_only_metrics.macro,
                        "weighted": id_only_metrics.weighted,
                    },
                    "token_usage": token_usage.to_dict(),
                    "timing_breakdown": {
                        "llm_seconds": round(token_usage.llm_time_seconds, 3),
                        "tool_seconds": round(token_usage.tool_time_seconds, 3),
                        "events": [e.to_dict() for e in token_usage.timing_events],
                    },
                    "estimated_cost": estimated_cost,
                },
                f,
                indent=2,
            )

        logger.info(f"Results for mode '{mode_str}' saved to {output_dir}")
