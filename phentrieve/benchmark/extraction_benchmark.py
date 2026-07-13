"""Extraction benchmark for document-level HPO extraction evaluation.

This module provides tools for evaluating HPO term extraction from clinical
documents against gold-standard annotations. It calculates precision, recall,
and F1 scores with support for assertion status (present/absent).

Key components:
- ExtractionConfig: Configuration for benchmark parameters
- HPOExtractor: Wrapper for the HPO extraction pipeline
- ExtractionBenchmark: Main benchmark runner with metrics calculation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phentrieve.benchmark.data_loader import (
    ASSERTION_STATUS_MAP,
    load_benchmark_data,
    parse_gold_terms,
)
from phentrieve.benchmark.result_store import (
    RunLayout,
    sha256_path,
    write_json,
    write_jsonl,
    write_manifest,
)
from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MULTI_VECTOR,
    get_sliding_window_punct_conj_cleaned_config,
)
from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    CorpusMetrics,
    ExtractionResult,
    OntologyAwareCorpusMetrics,
    serialize_ontology_metrics,
)
from phentrieve.utils import calculate_similarity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from phentrieve.evaluation.ontology_credit import OntologyCreditConfig
    from phentrieve.retrieval.dense_retriever import DenseRetriever
    from phentrieve.text_processing.pipeline import TextProcessingPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for extraction benchmark."""

    model_name: str = "BAAI/bge-m3"
    language: str = "en"
    num_results_per_chunk: int = 3
    chunk_retrieval_threshold: float = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
    min_confidence_for_aggregated: float = DEFAULT_MIN_CONFIDENCE_AGGREGATED
    multi_vector: bool = DEFAULT_MULTI_VECTOR
    top_term_per_chunk: bool = False
    averaging: str = "micro"
    scoring_mode: str = "strict"  # strict | present-only
    include_assertions: bool = True
    relaxed_matching: bool = False
    bootstrap_ci: bool = True
    bootstrap_samples: int = 1000
    bootstrap_seed: int | None = 12345
    dataset: str = "all"
    detailed_output: bool = False
    ontology_aware_metrics: bool = False
    ontology_semantic_floor: float = 0.30
    ontology_similarity_formula: str = "hybrid"


def _build_ontology_credit_config(config: ExtractionConfig) -> OntologyCreditConfig:
    """Build ontology-aware metric config from extraction benchmark settings."""
    from phentrieve.evaluation.ontology_credit import build_ontology_credit_config

    return build_ontology_credit_config(
        semantic_floor=config.ontology_semantic_floor,
        similarity_formula=config.ontology_similarity_formula,
    )


def validate_hpo_graph_available() -> None:
    from phentrieve.evaluation.ontology_credit import (
        validate_hpo_graph_available as validate,
    )

    validate()


class HPOExtractor:
    """Wrapper for HPO extraction pipeline."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._pipeline: TextProcessingPipeline | None = None
        self._retriever: DenseRetriever | None = None
        self._sbert_model: SentenceTransformer | None = None

    def _lazy_init(self):
        """Lazily initialize heavy components."""
        if self._pipeline is not None:
            return

        from phentrieve.embeddings import load_embedding_model
        from phentrieve.retrieval.dense_retriever import DenseRetriever
        from phentrieve.text_processing.pipeline import TextProcessingPipeline

        logger.info(f"Loading embedding model: {self.config.model_name}")
        self._sbert_model = load_embedding_model(self.config.model_name)

        logger.info("Initializing text processing pipeline")
        self._pipeline = TextProcessingPipeline(
            language=self.config.language,
            chunking_pipeline_config=get_sliding_window_punct_conj_cleaned_config(),
            assertion_config={
                "disable": not self.config.include_assertions,
                "strategy_preference": "dependency",
            },
            sbert_model_for_semantic_chunking=self._sbert_model,
        )

        logger.info("Initializing dense retriever")
        self._retriever = DenseRetriever.from_model_name(
            model=self._sbert_model,
            model_name=self.config.model_name,
            multi_vector=self.config.multi_vector,
        )

    def extract(self, text: str) -> list[tuple[str, str]]:
        """
        Extract HPO terms from clinical text.

        Args:
            text: Clinical text to process

        Returns:
            List of (hpo_id, assertion) tuples
        """
        results, _ = self.extract_with_details(text)
        return results

    def extract_with_details(
        self, text: str
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        """
        Extract HPO terms with full chunk-level details for analysis.

        Args:
            text: Clinical text to process

        Returns:
            Tuple of:
            - List of (hpo_id, assertion) tuples
            - Details dict with chunk_results, aggregated_results, and processed_chunks
        """
        from phentrieve.text_processing.hpo_extraction_orchestrator import (
            orchestrate_hpo_extraction,
        )

        self._lazy_init()
        if self._pipeline is None or self._retriever is None:
            raise RuntimeError(
                f"Failed to initialize extraction components for model "
                f"'{self.config.model_name}'. Ensure the ChromaDB index exists "
                f"and the embedding model is available."
            )

        # Process text into chunks
        processed_chunks = self._pipeline.process(text)

        if not processed_chunks:
            return [], {
                "chunk_results": [],
                "aggregated_results": [],
                "processed_chunks": [],
            }

        text_chunks = [chunk["text"] for chunk in processed_chunks]
        assertion_statuses = [chunk["status"].value for chunk in processed_chunks]

        # Extract HPO terms - NOW CAPTURE chunk_results!
        orchestration_result = orchestrate_hpo_extraction(
            text_chunks=text_chunks,
            retriever=self._retriever,
            num_results_per_chunk=self.config.num_results_per_chunk,
            chunk_retrieval_threshold=self.config.chunk_retrieval_threshold,
            language=self.config.language,
            top_term_per_chunk=self.config.top_term_per_chunk,
            min_confidence_for_aggregated=self.config.min_confidence_for_aggregated,
            assertion_statuses=assertion_statuses,
            include_details=False,
        )
        aggregated_results = orchestration_result.aggregated_results
        chunk_results = orchestration_result.chunk_results

        # Convert to (hpo_id, assertion) tuples using module-level mapping
        results = []
        for term in aggregated_results:
            hpo_id = term["id"]
            status = term.get("status", term.get("assertion_status", "affirmed"))
            assertion = ASSERTION_STATUS_MAP.get(status, "PRESENT")
            results.append((hpo_id, assertion))

        # Build details for analysis
        details = {
            "chunk_results": chunk_results,
            "aggregated_results": aggregated_results,
            "processed_chunks": [
                {
                    "chunk_idx": i,
                    "text": chunk["text"],
                    "assertion_status": chunk["status"].value,
                    "start_char": chunk.get("start_char", -1),
                    "end_char": chunk.get("end_char", -1),
                }
                for i, chunk in enumerate(processed_chunks)
            ],
            "raw_query_results": orchestration_result.raw_query_results,
        }

        return results, details


class ExtractionBenchmark:
    """Main benchmark runner for document-level extraction."""

    def __init__(
        self,
        model_name: str,
        config: ExtractionConfig | None = None,
    ):
        self.model_name = model_name
        self.config = config or ExtractionConfig(model_name=model_name)
        self.extractor = HPOExtractor(self.config)

    def run_benchmark(
        self,
        test_file: Path,
        output_dir: Path,
        config_overrides: dict[str, Any] | None = None,
        run_layout: RunLayout | None = None,
    ) -> CorpusMetrics:
        """Run extraction benchmark on test dataset."""
        # Create config copy and apply overrides (don't mutate original)
        config = dataclass_replace(self.config)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        ontology_config = None
        if config.ontology_aware_metrics:
            ontology_config = _build_ontology_credit_config(config)
            validate_hpo_graph_available()

        # Update extractor with new config for this run
        self.extractor.config = config

        # Load test data
        test_data = self._load_test_data(test_file, config)

        # Process each document
        results: list[ExtractionResult] = []
        detailed_results: list[dict[str, Any]] = []
        canonical_details: dict[str, dict[str, Any]] = {}
        total_docs = len(test_data["documents"])
        benchmark_start = time.perf_counter()

        for idx, doc in enumerate(test_data["documents"]):
            logger.info(f"Processing document {idx + 1}/{total_docs}: {doc['id']}")

            # Extract HPO terms (with details only if detailed_output enabled)
            document_start = time.perf_counter()
            if config.detailed_output or run_layout is not None:
                extracted, extraction_details = self.extractor.extract_with_details(
                    doc["text"]
                )
            else:
                extracted = self.extractor.extract(doc["text"])
                extraction_details = {}
            canonical_details[str(doc["id"])] = {
                "document": doc,
                "details": extraction_details,
                "elapsed_seconds": time.perf_counter() - document_start,
            }

            # Parse gold standard
            gold = self._parse_gold_terms(doc["gold_hpo_terms"])

            results.append(
                ExtractionResult(
                    doc_id=doc["id"],
                    predicted=extracted,
                    gold=gold,
                )
            )

            # Build detailed analysis only if enabled
            if config.detailed_output:
                detailed_results.append(
                    self._build_detailed_analysis(
                        doc_id=doc["id"],
                        full_text=doc["text"],
                        extracted=extracted,
                        gold_terms=doc["gold_hpo_terms"],
                        extraction_details=extraction_details,
                    )
                )

        from phentrieve.evaluation.extraction_metrics import normalize_for_scoring

        results = normalize_for_scoring(results, config.scoring_mode)

        # Calculate metrics
        evaluator = CorpusExtractionMetrics(averaging=config.averaging)
        metrics = evaluator.calculate_all_metrics(results)
        ontology_metrics = None
        if ontology_config is not None:
            ontology_metrics = evaluator.calculate_ontology_aware_metrics(
                results,
                config=ontology_config,
            )
            metrics.ontology_metrics = ontology_metrics
            if config.detailed_output:
                self._attach_ontology_detailed_results(
                    detailed_results,
                    ontology_metrics.document_metrics,
                )

        # Calculate bootstrap CI if requested
        if config.bootstrap_ci:
            ci = evaluator.bootstrap_confidence_intervals(
                results,
                n_bootstrap=config.bootstrap_samples,
                seed=config.bootstrap_seed,
            )
            metrics = CorpusMetrics(
                micro=metrics.micro,
                macro=metrics.macro,
                weighted=metrics.weighted,
                confidence_intervals=ci,
                ontology_metrics=ontology_metrics,
            )

        # Save results (pass detailed_results only if enabled)
        summary = self._save_results(
            results,
            metrics,
            output_dir,
            test_data.get("metadata", {}),
            config,
            ontology_metrics,
            detailed_results if config.detailed_output else None,
        )

        if run_layout is not None:
            self._save_canonical_run(
                run_layout=run_layout,
                results=results,
                summary=summary,
                config=config,
                dataset_metadata=test_data.get("metadata", {}),
                canonical_details=canonical_details,
                elapsed_seconds=time.perf_counter() - benchmark_start,
                test_file=test_file,
            )

        return metrics

    def _load_test_data(
        self, test_path: Path, config: ExtractionConfig
    ) -> dict[str, Any]:
        """Load test dataset from JSON file or PhenoBERT directory."""
        return load_benchmark_data(test_path, dataset=config.dataset)

    def _parse_gold_terms(self, gold_hpo_terms: list[dict]) -> list[tuple[str, str]]:
        """Parse gold HPO terms into (id, assertion) tuples."""
        return parse_gold_terms(gold_hpo_terms)

    def _find_chunk_position_in_text(
        self, chunk_text: str, full_text: str, search_start: int = 0
    ) -> tuple[int, int]:
        """Find the position of a chunk in the full text.

        Args:
            chunk_text: The chunk text to find
            full_text: The full document text
            search_start: Start searching from this position (for duplicate handling)

        Returns:
            Tuple of (start_char, end_char) or (-1, -1) if not found
        """
        # Normalize whitespace for matching
        normalized_chunk = " ".join(chunk_text.split())

        # Try exact match first
        pos = full_text.find(chunk_text, search_start)
        if pos != -1:
            return (pos, pos + len(chunk_text))

        # Try with normalized whitespace
        normalized_full = " ".join(full_text.split())
        pos = normalized_full.find(normalized_chunk)
        if pos != -1:
            # Map back to original position (approximate)
            return (pos, pos + len(normalized_chunk))

        # Try case-insensitive
        pos = full_text.lower().find(chunk_text.lower(), search_start)
        if pos != -1:
            return (pos, pos + len(chunk_text))

        return (-1, -1)

    def _build_detailed_analysis(
        self,
        doc_id: str,
        full_text: str,
        extracted: list[tuple[str, str]],
        gold_terms: list[dict],
        extraction_details: dict[str, Any],
    ) -> dict[str, Any]:
        """Build detailed analysis with TP/FP/FN breakdown for a document.

        Args:
            doc_id: Document identifier
            full_text: Full document text
            extracted: List of (hpo_id, assertion) tuples predicted
            gold_terms: List of gold term dicts with id, label, assertion, evidence_spans
            extraction_details: Details from extract_with_details()

        Returns:
            Detailed analysis dict with TP/FP/FN and chunk info
        """
        # Build lookup maps
        predicted_ids = {hpo_id for hpo_id, _ in extracted}
        gold_ids = {term["id"] for term in gold_terms}
        gold_by_id = {term["id"]: term for term in gold_terms}

        # Build aggregated results lookup
        aggregated_by_id = {
            term["id"]: term
            for term in extraction_details.get("aggregated_results", [])
        }

        # Use pipeline-provided positions if available, otherwise fall back to search
        chunk_positions: dict[int, tuple[int, int]] = {}
        last_end = 0
        for chunk_info in extraction_details.get("processed_chunks", []):
            chunk_idx = chunk_info["chunk_idx"]

            # Prefer pipeline-provided positions (more accurate)
            start = chunk_info.get("start_char", -1)
            end = chunk_info.get("end_char", -1)

            # Fall back to string search if positions not provided
            if start < 0 or end < 0:
                chunk_text = chunk_info["text"]
                start, end = self._find_chunk_position_in_text(
                    chunk_text, full_text, last_end
                )

            chunk_positions[chunk_idx] = (start, end)
            if end > 0:
                last_end = end  # Continue searching from where we left off

        # Helper to enrich chunk info with position
        def enrich_chunk(cr: dict, match: dict) -> dict:
            chunk_idx = cr["chunk_idx"]
            start, end = chunk_positions.get(chunk_idx, (-1, -1))
            return {
                "chunk_idx": chunk_idx,
                "chunk_text": cr["chunk_text"],
                "score": match.get("score", 0),
                "start_char": start,
                "end_char": end,
            }

        # Calculate TP, FP, FN
        tp_ids = predicted_ids & gold_ids
        fp_ids = predicted_ids - gold_ids
        fn_ids = gold_ids - predicted_ids

        # Build detailed breakdowns
        true_positives = []
        for hpo_id in tp_ids:
            gold_term = gold_by_id.get(hpo_id, {})
            agg_term = aggregated_by_id.get(hpo_id, {})
            # Find the chunks that matched this term
            chunk_results = extraction_details.get("chunk_results", [])
            matching_chunks = []
            for cr in chunk_results:
                for match in cr.get("matches", []):
                    if match.get("id") == hpo_id:
                        matching_chunks.append(enrich_chunk(cr, match))
            true_positives.append(
                {
                    "hpo_id": hpo_id,
                    "label": gold_term.get("label", agg_term.get("name", "")),
                    "gold_evidence_spans": gold_term.get("evidence_spans", []),
                    "predicted_score": agg_term.get("score", 0),
                    "predicted_avg_score": agg_term.get("avg_score", 0),
                    "matching_chunks": matching_chunks,
                }
            )

        false_positives = []
        for hpo_id in fp_ids:
            agg_term = aggregated_by_id.get(hpo_id, {})
            # Find the chunks that caused this false positive
            chunk_results = extraction_details.get("chunk_results", [])
            matching_chunks = []
            for cr in chunk_results:
                for match in cr.get("matches", []):
                    if match.get("id") == hpo_id:
                        matching_chunks.append(enrich_chunk(cr, match))
            false_positives.append(
                {
                    "hpo_id": hpo_id,
                    "label": agg_term.get("name", ""),
                    "predicted_score": agg_term.get("score", 0),
                    "predicted_avg_score": agg_term.get("avg_score", 0),
                    "matching_chunks": matching_chunks,
                }
            )

        false_negatives = []
        for hpo_id in fn_ids:
            gold_term = gold_by_id.get(hpo_id, {})
            false_negatives.append(
                {
                    "hpo_id": hpo_id,
                    "label": gold_term.get("label", ""),
                    "gold_evidence_spans": gold_term.get("evidence_spans", []),
                }
            )

        # Enrich all_chunks with positions
        all_chunks = []
        for chunk_info in extraction_details.get("processed_chunks", []):
            chunk_idx = chunk_info["chunk_idx"]
            start, end = chunk_positions.get(chunk_idx, (-1, -1))
            all_chunks.append(
                {
                    **chunk_info,
                    "start_char": start,
                    "end_char": end,
                }
            )

        return {
            "doc_id": doc_id,
            "full_text": full_text,
            "text_length": len(full_text),
            "num_chunks": len(all_chunks),
            "metrics": {
                "true_positives": len(tp_ids),
                "false_positives": len(fp_ids),
                "false_negatives": len(fn_ids),
                "precision": len(tp_ids) / len(predicted_ids) if predicted_ids else 0,
                "recall": len(tp_ids) / len(gold_ids) if gold_ids else 0,
            },
            "analysis": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
            "all_chunks": all_chunks,
        }

    def _attach_ontology_detailed_results(
        self,
        detailed_results: list[dict[str, Any]],
        document_metrics: list[Any],
    ) -> None:
        metrics_by_doc_id = {metric.doc_id: metric for metric in document_metrics}
        for detailed_result in detailed_results:
            ontology_metric = metrics_by_doc_id.get(detailed_result["doc_id"])
            if ontology_metric is None:
                continue
            detailed_result["ontology_metrics"] = (
                self._serialize_document_ontology_metrics(
                    ontology_metric,
                    detailed_result,
                )
            )

    def _serialize_document_ontology_metrics(
        self,
        ontology_metric: Any,
        detailed_result: dict[str, Any],
    ) -> dict[str, Any]:
        predicted_labels, gold_labels = self._ontology_label_lookups(detailed_result)

        return {
            "counts": {
                "predictions": ontology_metric.prediction_count,
                "gold": ontology_metric.gold_count,
                "strict_tp": ontology_metric.strict_tp,
            },
            "soft": {
                "tp": ontology_metric.soft_tp,
                "fp": ontology_metric.soft_fp,
                "fn": ontology_metric.soft_fn,
                "precision": ontology_metric.soft_precision,
                "recall": ontology_metric.soft_recall,
                "f1": ontology_metric.soft_f1,
            },
            "partial": {
                "precision": ontology_metric.partial_precision,
                "recall": ontology_metric.partial_recall,
                "f1": ontology_metric.partial_f1,
            },
            "matches": [
                {
                    "predicted": self._serialize_ontology_annotation(
                        match.predicted,
                        predicted_labels,
                    ),
                    "gold": self._serialize_ontology_annotation(
                        match.gold,
                        gold_labels,
                    ),
                    "assertion": match.predicted[1],
                    "match_kind": match.credit.match_kind.value,
                    "credit": match.credit.credit,
                    "semantic_similarity": match.credit.semantic_similarity,
                    "distance": match.credit.distance,
                }
                for match in ontology_metric.matches
            ],
            "unmatched_predictions": [
                self._serialize_ontology_annotation(annotation, predicted_labels)
                for annotation in ontology_metric.unmatched_predictions
            ],
            "unmatched_gold": [
                self._serialize_ontology_annotation(annotation, gold_labels)
                for annotation in ontology_metric.unmatched_gold
            ],
        }

    def _ontology_label_lookups(
        self,
        detailed_result: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, str]]:
        predicted_labels: dict[str, str] = {}
        gold_labels: dict[str, str] = {}
        analysis = detailed_result.get("analysis", {})

        for item in analysis.get("true_positives", []):
            hpo_id = item["hpo_id"]
            label = item.get("label", "")
            predicted_labels[hpo_id] = label
            gold_labels[hpo_id] = label

        for item in analysis.get("false_positives", []):
            predicted_labels[item["hpo_id"]] = item.get("label", "")

        for item in analysis.get("false_negatives", []):
            gold_labels[item["hpo_id"]] = item.get("label", "")

        return predicted_labels, gold_labels

    def _serialize_ontology_annotation(
        self,
        annotation: tuple[str, str],
        labels_by_id: dict[str, str],
    ) -> dict[str, Any]:
        hpo_id, assertion = annotation
        return {
            "hpo_id": hpo_id,
            "label": labels_by_id.get(hpo_id, ""),
            "assertion": assertion,
        }

    def _save_results(
        self,
        results: list[ExtractionResult],
        metrics: CorpusMetrics,
        output_dir: Path,
        dataset_metadata: dict,
        config: ExtractionConfig,
        ontology_metrics: OntologyAwareCorpusMetrics | None = None,
        detailed_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Save benchmark results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = [
            {
                "doc_id": r.doc_id,
                "predicted": r.predicted,
                "gold": r.gold,
            }
            for r in results
        ]

        # Save detailed results
        results_file = output_dir / "extraction_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "model": self.model_name,
                        "timestamp": datetime.now().isoformat(),
                        "config": {
                            "model_name": config.model_name,
                            "language": config.language,
                            "averaging": config.averaging,
                            "include_assertions": config.include_assertions,
                            "relaxed_matching": config.relaxed_matching,
                            "chunk_retrieval_threshold": config.chunk_retrieval_threshold,
                            "min_confidence_for_aggregated": config.min_confidence_for_aggregated,
                            "ontology_aware_metrics": config.ontology_aware_metrics,
                            "ontology_semantic_floor": config.ontology_semantic_floor,
                            "ontology_similarity_formula": config.ontology_similarity_formula,
                            "scoring_mode": config.scoring_mode,
                            "multi_vector": config.multi_vector,
                        },
                        "dataset": dataset_metadata,
                    },
                    "results": serializable_results,
                    "corpus_metrics": {
                        "micro": metrics.micro,
                        "macro": metrics.macro,
                        "weighted": metrics.weighted,
                        "confidence_intervals": metrics.confidence_intervals,
                        **(
                            {
                                "ontology_metrics": serialize_ontology_metrics(
                                    ontology_metrics
                                )
                            }
                            if ontology_metrics is not None
                            else {}
                        ),
                    },
                },
                f,
                indent=2,
            )

        # Save summary metrics
        summary = {
            "model": self.model_name,
            "scoring_mode": config.scoring_mode,
            "micro_f1": metrics.micro.get("f1", 0),
            "micro_precision": metrics.micro.get("precision", 0),
            "micro_recall": metrics.micro.get("recall", 0),
            "macro_f1": metrics.macro.get("f1", 0),
            "macro_precision": metrics.macro.get("precision", 0),
            "macro_recall": metrics.macro.get("recall", 0),
            "weighted_f1": metrics.weighted.get("f1", 0),
            "weighted_precision": metrics.weighted.get("precision", 0),
            "weighted_recall": metrics.weighted.get("recall", 0),
        }
        if ontology_metrics is not None:
            summary.update(
                {
                    "soft_micro_f1": ontology_metrics.soft.micro.get("f1", 0),
                    "soft_micro_precision": ontology_metrics.soft.micro.get(
                        "precision", 0
                    ),
                    "soft_micro_recall": ontology_metrics.soft.micro.get("recall", 0),
                    "partial_micro_f1": ontology_metrics.partial.micro.get("f1", 0),
                    "partial_micro_precision": ontology_metrics.partial.micro.get(
                        "precision", 0
                    ),
                    "partial_micro_recall": ontology_metrics.partial.micro.get(
                        "recall", 0
                    ),
                }
            )

        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed analysis with chunks (NEW!)
        if detailed_results:
            detailed_file = output_dir / "extraction_detailed_analysis.json"
            with open(detailed_file, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "model": self.model_name,
                            "timestamp": datetime.now().isoformat(),
                            "description": "Detailed extraction analysis with chunk-level info",
                        },
                        "documents": detailed_results,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Detailed analysis saved to {detailed_file}")

        logger.info(f"Results saved to {output_dir}")
        return summary

    def _save_canonical_run(
        self,
        *,
        run_layout: RunLayout,
        results: list[ExtractionResult],
        summary: dict[str, Any],
        config: ExtractionConfig,
        dataset_metadata: dict[str, Any],
        canonical_details: dict[str, dict[str, Any]],
        elapsed_seconds: float,
        test_file: Path,
    ) -> None:
        """Write analysis-first extraction artifacts for one structured run."""
        terms: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []

        for result in results:
            captured = canonical_details[result.doc_id]
            document = captured["document"]
            details = captured["details"]
            predicted = dict(result.predicted)
            gold = dict(result.gold)
            predicted_ids = set(predicted)
            gold_ids = set(gold)
            raw_by_id: dict[str, dict[str, Any]] = {}

            processed_chunks = details.get("processed_chunks", [])
            thresholded_chunks = {
                int(chunk.get("chunk_idx", index)): chunk
                for index, chunk in enumerate(details.get("chunk_results", []))
            }
            raw_query_results = details.get("raw_query_results", [])
            for chunk_index, processed_chunk in enumerate(processed_chunks):
                raw_result = (
                    raw_query_results[chunk_index]
                    if chunk_index < len(raw_query_results)
                    else {}
                )
                metadata_rows = (raw_result.get("metadatas") or [[]])[0]
                similarity_rows = (raw_result.get("similarities") or [[]])[0]
                distance_rows = (raw_result.get("distances") or [[]])[0]
                accepted_matches = thresholded_chunks.get(chunk_index, {}).get(
                    "matches", []
                )
                accepted_ids = {
                    str(match.get("id") or match.get("hpo_id") or "")
                    for match in accepted_matches
                }
                candidates = []
                for rank, metadata in enumerate(metadata_rows, start=1):
                    hpo_id = str(metadata.get("hpo_id") or metadata.get("id") or "")
                    if rank - 1 < len(similarity_rows):
                        score = float(similarity_rows[rank - 1])
                    elif rank - 1 < len(distance_rows):
                        score = calculate_similarity(float(distance_rows[rank - 1]))
                    else:
                        score = 0.0
                    candidate = {
                        "rank": rank,
                        "hpo_id": hpo_id,
                        "label": metadata.get("label", metadata.get("name", "")),
                        "score": score,
                        "passes_threshold": score >= config.chunk_retrieval_threshold,
                        "used_in_aggregation": hpo_id in accepted_ids,
                    }
                    if "component_scores" in metadata:
                        candidate["component_scores"] = metadata["component_scores"]
                    candidates.append(candidate)
                    previous = raw_by_id.get(hpo_id)
                    passed_any = bool(candidate["passes_threshold"])
                    used_any = bool(candidate["used_in_aggregation"])
                    if previous is not None:
                        passed_any = passed_any or bool(previous["passes_threshold"])
                        used_any = used_any or bool(previous["used_in_aggregation"])
                    if previous is None or score > previous["score"]:
                        raw_by_id[hpo_id] = {
                            **candidate,
                            "chunk_id": chunk_index,
                            "passes_threshold": passed_any,
                            "used_in_aggregation": used_any,
                        }
                    else:
                        previous["passes_threshold"] = passed_any
                        previous["used_in_aggregation"] = used_any

                chunks.append(
                    {
                        "doc_id": result.doc_id,
                        "chunk_id": chunk_index,
                        "text": processed_chunk.get("text", ""),
                        "start_char": processed_chunk.get("start_char", -1),
                        "end_char": processed_chunk.get("end_char", -1),
                        "assertion_status": processed_chunk.get("assertion_status"),
                        "candidates": candidates,
                    }
                )

            labels = {
                str(term.get("id") or term.get("hpo_id") or ""): term.get("label", "")
                for term in document.get("gold_hpo_terms", [])
            }
            aggregated = {
                str(term.get("id") or term.get("hpo_id") or ""): term
                for term in details.get("aggregated_results", [])
            }
            pipeline_predicted_ids = set(aggregated)
            for hpo_id in sorted(
                gold_ids | predicted_ids | pipeline_predicted_ids | set(raw_by_id)
            ):
                if hpo_id in gold_ids and hpo_id in predicted_ids:
                    outcome = "tp"
                elif hpo_id in predicted_ids:
                    outcome = "fp"
                elif hpo_id in gold_ids:
                    outcome = "fn"
                else:
                    outcome = "filtered"
                raw = raw_by_id.get(hpo_id, {})
                aggregate = aggregated.get(hpo_id, {})
                if hpo_id in predicted_ids:
                    filter_stage = None
                elif hpo_id in pipeline_predicted_ids:
                    filter_stage = "scoring_mode"
                elif raw.get("used_in_aggregation"):
                    filter_stage = "aggregation_confidence"
                elif raw.get("passes_threshold"):
                    filter_stage = "chunk_selection"
                elif raw:
                    filter_stage = "chunk_threshold"
                else:
                    filter_stage = "not_retrieved"
                pipeline_status = aggregate.get(
                    "assertion_status", aggregate.get("status")
                )
                terms.append(
                    {
                        "doc_id": result.doc_id,
                        "hpo_id": hpo_id,
                        "label": labels.get(
                            hpo_id,
                            aggregate.get("name", raw.get("label", "")),
                        ),
                        "outcome": outcome,
                        "is_gold": hpo_id in gold_ids,
                        "is_predicted": hpo_id in predicted_ids,
                        "is_pipeline_prediction": hpo_id in pipeline_predicted_ids,
                        "is_evaluated_prediction": hpo_id in predicted_ids,
                        "filter_stage": filter_stage,
                        "gold_assertion": gold.get(hpo_id),
                        "predicted_assertion": predicted.get(hpo_id),
                        "pipeline_assertion": ASSERTION_STATUS_MAP.get(
                            str(pipeline_status),
                            pipeline_status,
                        ),
                        "rank": raw.get("rank"),
                        "raw_score": raw.get("score"),
                        "final_score": aggregate.get("score"),
                        "source_chunk_ids": [
                            (
                                chunk.get("chunk_idx")
                                if isinstance(chunk, dict)
                                else chunk
                            )
                            for chunk in aggregate.get("chunks", [])
                            if isinstance(chunk, int)
                            or (
                                isinstance(chunk, dict)
                                and isinstance(chunk.get("chunk_idx"), int)
                            )
                        ],
                    }
                )

            tp = len(predicted_ids & gold_ids)
            fp = len(predicted_ids - gold_ids)
            fn = len(gold_ids - predicted_ids)
            cases.append(
                {
                    "doc_id": result.doc_id,
                    "text": document.get("text", ""),
                    "expected_hpo_ids": sorted(gold_ids),
                    "pipeline_predicted_hpo_ids": sorted(pipeline_predicted_ids),
                    "predicted_hpo_ids": sorted(predicted_ids),
                    "metrics": {"tp": tp, "fp": fp, "fn": fn},
                    "elapsed_seconds": captured["elapsed_seconds"],
                    "status": "complete",
                }
            )

        canonical_summary = {
            **summary,
            "run_id": run_layout.run_id,
            "benchmark_type": "extraction",
            "dataset_name": dataset_metadata.get("dataset_name", config.dataset),
            "elapsed_seconds": elapsed_seconds,
        }
        write_json(run_layout.summary_path, canonical_summary)
        write_jsonl(run_layout.terms_path, terms)
        write_jsonl(run_layout.cases_path, cases)
        write_jsonl(run_layout.chunks_path, chunks)
        write_manifest(
            run_layout,
            {
                "status": "complete",
                "elapsed_seconds": elapsed_seconds,
                "dataset": {
                    **dataset_metadata,
                    "path": str(test_file.resolve()),
                    "sha256": sha256_path(test_file),
                },
                "config": {
                    "scoring_mode": config.scoring_mode,
                    "chunk_retrieval_threshold": config.chunk_retrieval_threshold,
                    "min_confidence_for_aggregated": (
                        config.min_confidence_for_aggregated
                    ),
                    "num_results_per_chunk": config.num_results_per_chunk,
                    "multi_vector": config.multi_vector,
                    "include_assertions": config.include_assertions,
                },
                "counts": {
                    "documents": len(cases),
                    "terms": len(terms),
                    "chunks": len(chunks),
                },
            },
        )
