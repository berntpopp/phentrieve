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
        from phentrieve.text_processing.hpo_extraction_orchestrator import (
            orchestrate_hpo_extraction,
        )

        self._lazy_init()
        assert self._pipeline is not None
        assert self._retriever is not None

        # Process text into chunks
        processed_chunks = self._pipeline.process(text)

        if not processed_chunks:
            return []

        text_chunks = [chunk["text"] for chunk in processed_chunks]
        assertion_statuses = [chunk["status"].value for chunk in processed_chunks]

        # Extract HPO terms
        aggregated_results, _ = orchestrate_hpo_extraction(
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

        # Convert to (hpo_id, assertion) tuples
        # Map assertion status to benchmark format
        assertion_map = {
            "affirmed": "PRESENT",
            "negated": "ABSENT",
            "uncertain": "UNCERTAIN",
            "normal": "PRESENT",  # normal findings are present
        }

        results = []
        for term in aggregated_results:
            hpo_id = term["id"]
            status = term.get("status", term.get("assertion_status", "affirmed"))
            assertion = assertion_map.get(status, "PRESENT")
            results.append((hpo_id, assertion))

        return results


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
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Load test data
        test_data = self._load_test_data(test_file)

        # Process each document
        results: list[ExtractionResult] = []
        total_docs = len(test_data["documents"])

        for idx, doc in enumerate(test_data["documents"]):
            logger.info(f"Processing document {idx + 1}/{total_docs}: {doc['id']}")

            # Extract HPO terms
            extracted = self.extractor.extract(doc["text"])

            # Parse gold standard
            gold = self._parse_gold_terms(doc["gold_hpo_terms"])

            results.append(
                ExtractionResult(
                    doc_id=doc["id"],
                    predicted=extracted,
                    gold=gold,
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

        # Save results
        self._save_results(results, metrics, output_dir, test_data.get("metadata", {}))

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
        # Map assertion statuses from PhenoBERT format to benchmark format
        assertion_map = {
            "affirmed": "PRESENT",
            "negated": "ABSENT",
            "uncertain": "UNCERTAIN",
        }

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
                    status = ann.get("assertion_status", "affirmed")
                    assertion = assertion_map.get(status, "PRESENT")
                    gold_hpo_terms.append({"id": hpo_id, "assertion": assertion})

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

    def _save_results(
        self,
        results: list[ExtractionResult],
        metrics: CorpusMetrics,
        output_dir: Path,
        dataset_metadata: dict,
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

        logger.info(f"Results saved to {output_dir}")
