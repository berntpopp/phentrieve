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
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phentrieve.evaluation.extraction_metrics import (
    CorpusExtractionMetrics,
    CorpusMetrics,
    ExtractionResult,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from phentrieve.retrieval.dense_retriever import DenseRetriever
    from phentrieve.text_processing.pipeline import TextProcessingPipeline

logger = logging.getLogger(__name__)

# Mapping from internal assertion status to benchmark format
ASSERTION_STATUS_MAP: dict[str, str] = {
    "affirmed": "PRESENT",
    "negated": "ABSENT",
    "uncertain": "UNCERTAIN",
    "normal": "PRESENT",  # normal findings are present
}


@dataclass
class ExtractionConfig:
    """Configuration for extraction benchmark."""

    model_name: str = "BAAI/bge-m3"
    language: str = "en"
    num_results_per_chunk: int = 3  # Reduced from 10 for better precision
    chunk_retrieval_threshold: float = 0.5  # Raised from 0.3
    min_confidence_for_aggregated: float = 0.5  # Raised from 0.35
    top_term_per_chunk: bool = False  # Only keep best match per chunk
    averaging: str = "micro"
    include_assertions: bool = True
    relaxed_matching: bool = False
    bootstrap_ci: bool = True
    bootstrap_samples: int = 1000
    dataset: str = "all"  # For PhenoBERT: all, GSC_plus, ID_68, GeneReviews
    detailed_output: bool = False  # Generate detailed chunk-level analysis JSON


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
            chunking_pipeline_config=[
                {"type": "paragraph"},
                {"type": "sentence"},
                {
                    "type": "sliding_window",
                    "config": {
                        "window_size_tokens": 3,
                        "step_size_tokens": 1,
                        "splitting_threshold": 0.5,
                        "min_split_segment_length_words": 2,
                    },
                },
                {"type": "final_chunk_cleaner"},
            ],
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
        aggregated_results, chunk_results = orchestrate_hpo_extraction(
            text_chunks=text_chunks,
            retriever=self._retriever,
            num_results_per_chunk=self.config.num_results_per_chunk,
            chunk_retrieval_threshold=self.config.chunk_retrieval_threshold,
            cross_encoder=None,
            language=self.config.language,
            top_term_per_chunk=self.config.top_term_per_chunk,
            min_confidence_for_aggregated=self.config.min_confidence_for_aggregated,
            assertion_statuses=assertion_statuses,
            include_details=False,
        )

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
                }
                for i, chunk in enumerate(processed_chunks)
            ],
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
    ) -> CorpusMetrics:
        """Run extraction benchmark on test dataset."""
        # Create config copy and apply overrides (don't mutate original)
        config = dataclass_replace(self.config)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Update extractor with new config for this run
        self.extractor.config = config

        # Load test data
        test_data = self._load_test_data(test_file)

        # Process each document
        results: list[ExtractionResult] = []
        detailed_results: list[dict[str, Any]] = []
        total_docs = len(test_data["documents"])

        for idx, doc in enumerate(test_data["documents"]):
            logger.info(f"Processing document {idx + 1}/{total_docs}: {doc['id']}")

            # Extract HPO terms (with details only if detailed_output enabled)
            if config.detailed_output:
                extracted, extraction_details = self.extractor.extract_with_details(
                    doc["text"]
                )
            else:
                extracted = self.extractor.extract(doc["text"])
                extraction_details = {}

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

        # Calculate metrics
        evaluator = CorpusExtractionMetrics(averaging=self.config.averaging)
        metrics = evaluator.calculate_all_metrics(results)

        # Calculate bootstrap CI if requested
        if self.config.bootstrap_ci:
            ci = evaluator.bootstrap_confidence_intervals(
                results, n_bootstrap=self.config.bootstrap_samples
            )
            metrics = CorpusMetrics(
                micro=metrics.micro,
                macro=metrics.macro,
                weighted=metrics.weighted,
                confidence_intervals=ci,
            )

        # Save results (pass detailed_results only if enabled)
        self._save_results(
            results,
            metrics,
            output_dir,
            test_data.get("metadata", {}),
            detailed_results if config.detailed_output else None,
        )

        return metrics

    def _load_test_data(self, test_path: Path) -> dict[str, Any]:
        """Load test dataset from JSON file or PhenoBERT directory."""
        if test_path.is_dir():
            return self._load_phenobert_data(test_path)
        else:
            with open(test_path) as f:
                return json.load(f)  # type: ignore[no-any-return]

    def _load_phenobert_data(self, base_dir: Path) -> dict[str, Any]:
        """Load PhenoBERT-format data from directory structure.

        Expected structure:
        base_dir/
            GSC_plus/annotations/*.json
            ID_68/annotations/*.json
            GeneReviews/annotations/*.json
        """
        # Determine which subdirectories to load
        dataset_dirs = {
            "GSC_plus": base_dir / "GSC_plus" / "annotations",
            "ID_68": base_dir / "ID_68" / "annotations",
            "GeneReviews": base_dir / "GeneReviews" / "annotations",
        }

        if self.config.dataset != "all":
            # Load only the specified dataset
            if self.config.dataset not in dataset_dirs:
                raise ValueError(
                    f"Unknown dataset: {self.config.dataset}. "
                    f"Available: {list(dataset_dirs.keys())}"
                )
            dataset_dirs = {self.config.dataset: dataset_dirs[self.config.dataset]}

        documents = []
        total_annotations = 0

        for dataset_name, annotations_dir in dataset_dirs.items():
            if not annotations_dir.exists():
                logger.warning(f"Dataset directory not found: {annotations_dir}")
                continue

            for json_file in sorted(annotations_dir.glob("*.json")):
                with open(json_file) as f:
                    doc_data = json.load(f)

                # Convert PhenoBERT format to benchmark format
                gold_hpo_terms = []
                for ann in doc_data.get("annotations", []):
                    hpo_id = ann.get("hpo_id", "")
                    label = ann.get("label", "")
                    status = ann.get("assertion_status", "affirmed")
                    assertion = ASSERTION_STATUS_MAP.get(status, "PRESENT")
                    evidence_spans = ann.get("evidence_spans", [])
                    gold_hpo_terms.append(
                        {
                            "id": hpo_id,
                            "label": label,
                            "assertion": assertion,
                            "evidence_spans": evidence_spans,
                        }
                    )

                documents.append(
                    {
                        "id": doc_data.get("doc_id", json_file.stem),
                        "text": doc_data.get("full_text", ""),
                        "gold_hpo_terms": gold_hpo_terms,
                        "source_dataset": dataset_name,
                    }
                )
                total_annotations += len(gold_hpo_terms)

        logger.info(
            f"Loaded {len(documents)} documents with {total_annotations} annotations "
            f"from PhenoBERT data"
        )

        return {
            "metadata": {
                "dataset_name": f"phenobert_{self.config.dataset}",
                "source": "phenobert",
                "total_documents": len(documents),
                "total_annotations": total_annotations,
            },
            "documents": documents,
        }

    def _parse_gold_terms(self, gold_hpo_terms: list[dict]) -> list[tuple[str, str]]:
        """Parse gold HPO terms into (id, assertion) tuples."""
        result = []
        for term in gold_hpo_terms:
            if isinstance(term, dict):
                hpo_id = term.get("id", term.get("hpo_id", ""))
                assertion = term.get("assertion", "PRESENT")
            elif isinstance(term, (list, tuple)) and len(term) >= 2:
                hpo_id, assertion = term[0], term[1]
            else:
                hpo_id = str(term)
                assertion = "PRESENT"
            result.append((hpo_id, assertion))
        return result

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

        # Pre-compute chunk positions in full text
        chunk_positions: dict[int, tuple[int, int]] = {}
        last_end = 0
        for chunk_info in extraction_details.get("processed_chunks", []):
            chunk_idx = chunk_info["chunk_idx"]
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

    def _save_results(
        self,
        results: list[ExtractionResult],
        metrics: CorpusMetrics,
        output_dir: Path,
        dataset_metadata: dict,
        detailed_results: list[dict[str, Any]] | None = None,
    ):
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
                            "model_name": self.config.model_name,
                            "language": self.config.language,
                            "averaging": self.config.averaging,
                            "include_assertions": self.config.include_assertions,
                            "relaxed_matching": self.config.relaxed_matching,
                            "chunk_retrieval_threshold": self.config.chunk_retrieval_threshold,
                            "min_confidence_for_aggregated": self.config.min_confidence_for_aggregated,
                        },
                        "dataset": dataset_metadata,
                    },
                    "results": serializable_results,
                    "corpus_metrics": {
                        "micro": metrics.micro,
                        "macro": metrics.macro,
                        "weighted": metrics.weighted,
                        "confidence_intervals": metrics.confidence_intervals,
                    },
                },
                f,
                indent=2,
            )

        # Save summary metrics
        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "micro_f1": metrics.micro.get("f1", 0),
                    "micro_precision": metrics.micro.get("precision", 0),
                    "micro_recall": metrics.micro.get("recall", 0),
                    "macro_f1": metrics.macro.get("f1", 0),
                    "macro_precision": metrics.macro.get("precision", 0),
                    "macro_recall": metrics.macro.get("recall", 0),
                    "weighted_f1": metrics.weighted.get("f1", 0),
                    "weighted_precision": metrics.weighted.get("precision", 0),
                    "weighted_recall": metrics.weighted.get("recall", 0),
                },
                f,
                indent=2,
            )

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
