"""
Mention-level HPO extraction orchestrator.

This module provides the main orchestrator for mention-based HPO
extraction, coordinating all stages of the pipeline:
- Document structure detection
- Mention discovery
- Assertion interpretation
- HPO candidate generation
- Candidate refinement
- Context propagation
- Mention grouping
- Document-level aggregation

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

from phentrieve.text_processing.document_structure import (
    DocumentStructureDetector,
)
from phentrieve.text_processing.mention import DocumentMentions, Mention, MentionGroup
from phentrieve.text_processing.mention_aggregator import (
    AggregationConfig,
    MentionAggregator,
)
from phentrieve.text_processing.mention_assertion import MentionAssertionDetector
from phentrieve.text_processing.mention_candidate_refiner import (
    CandidateRefinementConfig,
    MentionCandidateRefiner,
)
from phentrieve.text_processing.mention_context import (
    ContextPropagationConfig,
    MentionContextGraph,
)
from phentrieve.text_processing.mention_extractor import (
    MentionExtractionConfig,
    MentionExtractor,
)
from phentrieve.text_processing.mention_grouper import (
    MentionGrouper,
    MentionGroupingConfig,
)
from phentrieve.text_processing.mention_hpo_retriever import (
    MentionHPORetriever,
    MentionRetrievalConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MentionExtractionOrchestratorConfig:
    """
    Configuration for the mention-based extraction orchestrator.

    Attributes:
        language: Language code for NLP processing
        extraction_config: Mention extraction configuration
        retrieval_config: HPO retrieval configuration
        refinement_config: Candidate refinement configuration
        context_config: Context propagation configuration
        grouping_config: Mention grouping configuration
        aggregation_config: Document aggregation configuration
        enable_context_propagation: Whether to use context propagation
        enable_grouping: Whether to group mentions
        dataset_format: Target dataset for output format
    """

    language: str = "en"
    extraction_config: MentionExtractionConfig | None = None
    retrieval_config: MentionRetrievalConfig | None = None
    refinement_config: CandidateRefinementConfig | None = None
    context_config: ContextPropagationConfig | None = None
    grouping_config: MentionGroupingConfig | None = None
    aggregation_config: AggregationConfig | None = None
    enable_context_propagation: bool = True
    enable_grouping: bool = True
    dataset_format: str = "phenobert"


class MentionExtractionOrchestrator:
    """
    Orchestrator for mention-based HPO extraction.

    Coordinates all stages of the mention-level extraction pipeline
    to produce document-level HPO term sets compatible with existing
    benchmark evaluation.

    Example:
        >>> from phentrieve.retrieval.dense_retriever import DenseRetriever
        >>> retriever = DenseRetriever.from_model_name("all-MiniLM-L6-v2")
        >>> orchestrator = MentionExtractionOrchestrator(
        ...     retriever=retriever,
        ...     language="en",
        ... )
        >>> result = orchestrator.extract("Patient has seizures and headaches.")
        >>> print(result["benchmark_format"])
        [("HP:0001250", "PRESENT"), ("HP:0002315", "PRESENT")]
    """

    def __init__(
        self,
        retriever: Any,  # DenseRetriever
        config: MentionExtractionOrchestratorConfig | None = None,
        model: SentenceTransformer | None = None,
        cross_encoder: CrossEncoder | None = None,
        hpo_depth_map: dict[str, int] | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            retriever: DenseRetriever for HPO term lookup
            config: Orchestrator configuration
            model: SentenceTransformer for embeddings
            cross_encoder: CrossEncoder for reranking
            hpo_depth_map: Map of HPO ID to ontology depth
        """
        self.retriever = retriever
        self.config = config or MentionExtractionOrchestratorConfig()
        self.model = model
        self.cross_encoder = cross_encoder
        self.hpo_depth_map = hpo_depth_map or {}

        # Initialize components
        self._structure_detector: DocumentStructureDetector | None = None
        self._mention_extractor: MentionExtractor | None = None
        self._assertion_detector: MentionAssertionDetector | None = None
        self._hpo_retriever: MentionHPORetriever | None = None
        self._candidate_refiner: MentionCandidateRefiner | None = None
        self._grouper: MentionGrouper | None = None
        self._aggregator: MentionAggregator | None = None

    # Lazy initialization of components

    @property
    def structure_detector(self) -> DocumentStructureDetector:
        if self._structure_detector is None:
            self._structure_detector = DocumentStructureDetector(
                language=self.config.language
            )
        return self._structure_detector

    @property
    def mention_extractor(self) -> MentionExtractor:
        if self._mention_extractor is None:
            self._mention_extractor = MentionExtractor(
                language=self.config.language,
                config=self.config.extraction_config,
            )
        return self._mention_extractor

    @property
    def assertion_detector(self) -> MentionAssertionDetector:
        if self._assertion_detector is None:
            self._assertion_detector = MentionAssertionDetector(
                language=self.config.language
            )
        return self._assertion_detector

    @property
    def hpo_retriever(self) -> MentionHPORetriever:
        if self._hpo_retriever is None:
            self._hpo_retriever = MentionHPORetriever(
                retriever=self.retriever,
                config=self.config.retrieval_config,
                hpo_depth_map=self.hpo_depth_map,
            )
        return self._hpo_retriever

    @property
    def candidate_refiner(self) -> MentionCandidateRefiner:
        if self._candidate_refiner is None:
            self._candidate_refiner = MentionCandidateRefiner(
                cross_encoder=self.cross_encoder,
                config=self.config.refinement_config,
                hpo_depth_map=self.hpo_depth_map,
            )
        return self._candidate_refiner

    @property
    def grouper(self) -> MentionGrouper:
        if self._grouper is None:
            self._grouper = MentionGrouper(
                config=self.config.grouping_config,
                model=self.model,
            )
        return self._grouper

    @property
    def aggregator(self) -> MentionAggregator:
        if self._aggregator is None:
            agg_config = self.config.aggregation_config or AggregationConfig()
            agg_config.dataset_format = self.config.dataset_format
            self._aggregator = MentionAggregator(config=agg_config)
        return self._aggregator

    def extract(
        self,
        text: str,
        doc_id: str = "unknown",
        include_details: bool = False,
    ) -> dict[str, Any]:
        """
        Extract HPO terms from text using mention-level processing.

        Args:
            text: Document text
            doc_id: Document identifier
            include_details: Include detailed mention-level information

        Returns:
            Extraction results with benchmark-compatible format
        """
        logger.info(f"Starting mention-based extraction for {doc_id}")

        # Stage A: Document structure detection
        structure = self.structure_detector.analyze(text, doc_id=doc_id)
        logger.debug(
            f"Stage A complete: {structure.num_sentences} sentences, "
            f"{structure.num_sections} sections"
        )

        # Stage B: Mention discovery
        mentions = self.mention_extractor.extract(text, doc_structure=structure)
        logger.debug(f"Stage B complete: {len(mentions)} mentions discovered")

        if not mentions:
            logger.warning(f"No mentions found in document {doc_id}")
            return self._empty_result(doc_id)

        # Stage C: Assertion interpretation
        self.assertion_detector.detect_batch(mentions, text)
        logger.debug("Stage C complete: Assertions detected")

        # Stage D: HPO candidate generation
        self.hpo_retriever.retrieve_batch(mentions)
        logger.debug("Stage D complete: HPO candidates retrieved")

        # Stage E: Candidate refinement
        self.candidate_refiner.refine_batch(mentions)
        logger.debug("Stage E complete: Candidates refined")

        # Stage F: Context propagation (optional)
        if self.config.enable_context_propagation:
            context_graph = MentionContextGraph(config=self.config.context_config)
            context_graph.build_from_mentions(mentions, model=self.model)
            context_graph.propagate_context()
            logger.debug("Stage F complete: Context propagated")

        # Stage G: Mention grouping (optional)
        if self.config.enable_grouping:
            groups = self.grouper.group(mentions)
        else:
            # Each mention becomes its own group
            groups = self._mentions_to_singleton_groups(mentions)
        logger.debug(f"Stage G complete: {len(groups)} groups created")

        # Create DocumentMentions container
        doc_mentions = DocumentMentions(
            doc_id=doc_id,
            full_text=text,
            mentions=mentions,
            groups=groups,
            sentences=[(s.start_char, s.end_char) for s in structure.sentences],
            sections=[
                {
                    "type": sec.section_type,
                    "start": sec.start_char,
                    "end": sec.end_char,
                }
                for sec in structure.sections
            ],
        )

        # Stage H: Document-level aggregation
        if include_details:
            self.aggregator.config.include_details = True

        result = self.aggregator.aggregate(doc_mentions)
        logger.info(
            f"Extraction complete for {doc_id}: {result['num_terms']} terms extracted"
        )

        return result

    def extract_batch(
        self,
        documents: list[dict[str, Any]],
        include_details: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Extract HPO terms from multiple documents.

        Args:
            documents: List of {"doc_id": str, "text": str} dicts
            include_details: Include detailed information

        Returns:
            List of extraction results
        """
        results = []
        for doc in documents:
            doc_id = doc.get("doc_id", "unknown")
            text = doc.get("text", doc.get("full_text", ""))

            try:
                result = self.extract(
                    text, doc_id=doc_id, include_details=include_details
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Extraction failed for {doc_id}: {e}")
                results.append(self._empty_result(doc_id, error=str(e)))

        return results

    def _mentions_to_singleton_groups(
        self,
        mentions: list[Mention],
    ) -> list[MentionGroup]:
        """Convert each mention to its own group."""
        groups = []
        for mention in mentions:
            if not mention.hpo_candidates:
                continue

            group = MentionGroup(
                mentions=[mention],
                representative_mention=mention,
                ranked_hpo_explanations=mention.hpo_candidates[:3],
                final_hpo=mention.top_candidate,
                final_assertion=mention.assertion,
            )
            groups.append(group)

        return groups

    def _empty_result(
        self,
        doc_id: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Create an empty result for a document."""
        result = {
            "doc_id": doc_id,
            "num_terms": 0,
            "benchmark_format": [],
            "terms": [],
        }
        if error:
            result["error"] = error
        return result


def orchestrate_mention_extraction(
    text: str,
    retriever: Any,  # DenseRetriever
    language: str = "en",
    doc_id: str = "unknown",
    model: SentenceTransformer | None = None,
    cross_encoder: CrossEncoder | None = None,
    dataset_format: str = "phenobert",
    include_details: bool = False,
) -> dict[str, Any]:
    """
    Convenience function for mention-based HPO extraction.

    Args:
        text: Document text
        retriever: DenseRetriever instance
        language: Language code
        doc_id: Document identifier
        model: Optional SentenceTransformer
        cross_encoder: Optional CrossEncoder
        dataset_format: Target dataset format
        include_details: Include detailed output

    Returns:
        Extraction results
    """
    config = MentionExtractionOrchestratorConfig(
        language=language,
        dataset_format=dataset_format,
    )

    orchestrator = MentionExtractionOrchestrator(
        retriever=retriever,
        config=config,
        model=model,
        cross_encoder=cross_encoder,
    )

    return orchestrator.extract(text, doc_id=doc_id, include_details=include_details)


def extract_hpo_mentions(
    text: str,
    retriever: Any,
    language: str = "en",
) -> list[tuple[str, str]]:
    """
    Simple wrapper returning benchmark format.

    Args:
        text: Document text
        retriever: DenseRetriever instance
        language: Language code

    Returns:
        List of (hpo_id, assertion) tuples
    """
    result = orchestrate_mention_extraction(
        text=text,
        retriever=retriever,
        language=language,
    )
    benchmark_format: list[tuple[str, str]] = result["benchmark_format"]
    return benchmark_format
