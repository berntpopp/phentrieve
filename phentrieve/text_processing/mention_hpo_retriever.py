"""
Mention-level HPO candidate retrieval.

This module provides HPO term retrieval at the mention level,
generating high-recall candidate sets for each clinical finding
mention. The candidates are later refined and re-ranked.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from phentrieve.text_processing.mention import HPOCandidate, Mention

logger = logging.getLogger(__name__)

# Default retrieval configuration
DEFAULT_CANDIDATES_PER_MENTION = 15
DEFAULT_RETRIEVAL_THRESHOLD = 0.5  # Balanced threshold (0.3-0.6 range works best)
DEFAULT_BATCH_SIZE = 32


@dataclass
class MentionRetrievalConfig:
    """
    Configuration for mention-level HPO retrieval.

    Attributes:
        candidates_per_mention: Number of candidates to retrieve per mention
        retrieval_threshold: Minimum similarity threshold
        use_context: Whether to include context in retrieval query
        context_weight: Weight for context vs. mention text (0-1)
        batch_size: Batch size for embedding computation
    
    Note:
        Context-aware retrieval (use_context=True) significantly improves
        retrieval quality for short mentions. The context window provides
        additional semantic information that helps disambiguate ambiguous terms.
    """

    candidates_per_mention: int = DEFAULT_CANDIDATES_PER_MENTION
    retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD
    use_context: bool = True  # Enabled for better retrieval quality
    context_weight: float = 0.3
    batch_size: int = DEFAULT_BATCH_SIZE


class MentionHPORetriever:
    """
    Retrieve HPO candidates for clinical finding mentions.

    Uses dense retrieval to find potential HPO term matches for
    each mention. Supports batch processing for efficiency.

    Example:
        >>> from phentrieve.retrieval.dense_retriever import DenseRetriever
        >>> retriever = DenseRetriever.from_model_name("all-MiniLM-L6-v2")
        >>> mention_retriever = MentionHPORetriever(retriever=retriever)
        >>> mention = Mention(text="seizures", start_char=0, end_char=8)
        >>> mention_retriever.retrieve(mention)
        >>> len(mention.hpo_candidates)
        15
    """

    def __init__(
        self,
        retriever: Any,  # DenseRetriever
        config: MentionRetrievalConfig | None = None,
        hpo_depth_map: dict[str, int] | None = None,
    ):
        """
        Initialize the mention HPO retriever.

        Args:
            retriever: DenseRetriever instance for HPO term lookup
            config: Retrieval configuration
            hpo_depth_map: Optional map of HPO ID to ontology depth
        """
        self.retriever = retriever
        self.config = config or MentionRetrievalConfig()
        self.hpo_depth_map = hpo_depth_map or {}

    def retrieve(
        self,
        mention: Mention,
        update_mention: bool = True,
    ) -> list[HPOCandidate]:
        """
        Retrieve HPO candidates for a single mention.

        Args:
            mention: The mention to retrieve candidates for
            update_mention: If True, update mention's hpo_candidates

        Returns:
            List of HPOCandidate objects
        """
        # Build query text
        query = self._build_query(mention)

        # Query the retriever
        results = self.retriever.query(
            text=query,
            n_results=self.config.candidates_per_mention,
            include_similarities=True,
        )

        # Convert to HPOCandidate objects
        candidates = self._results_to_candidates(results)

        # Update mention if requested
        if update_mention:
            mention.hpo_candidates = candidates

        return candidates

    def retrieve_batch(
        self,
        mentions: list[Mention],
        update_mentions: bool = True,
    ) -> list[list[HPOCandidate]]:
        """
        Retrieve HPO candidates for multiple mentions in batch.

        This is more efficient than calling retrieve() for each mention
        as it batches the embedding computation.

        Args:
            mentions: List of mentions to process
            update_mentions: If True, update each mention's hpo_candidates

        Returns:
            List of candidate lists (one per mention)
        """
        if not mentions:
            return []

        logger.debug(f"Batch retrieving HPO candidates for {len(mentions)} mentions")

        # Build query texts
        queries = [self._build_query(m) for m in mentions]

        # Batch query the retriever
        all_results = self.retriever.query_batch(
            texts=queries,
            n_results=self.config.candidates_per_mention,
            include_similarities=True,
        )

        # Convert to candidates
        all_candidates: list[list[HPOCandidate]] = []
        for idx, results in enumerate(all_results):
            candidates = self._results_to_candidates(results)

            if update_mentions:
                mentions[idx].hpo_candidates = candidates

            all_candidates.append(candidates)

        return all_candidates

    def _build_query(self, mention: Mention) -> str:
        """
        Build the query text for a mention.

        Args:
            mention: The mention

        Returns:
            Query string
        """
        # Base query is the mention text
        query = mention.text

        # Optionally include context
        if self.config.use_context and mention.context_window:
            # Extract a short context snippet
            context = self._extract_short_context(mention)
            if context and context != mention.text:
                # Combine mention and context
                query = f"{mention.text} {context}"

        return query

    def _extract_short_context(self, mention: Mention) -> str:
        """
        Extract a short context snippet around the mention.

        Args:
            mention: The mention

        Returns:
            Short context string
        """
        if not mention.context_window:
            return ""

        # Limit context length
        max_context = 50
        context = mention.context_window

        # Try to find the mention in the context and get surrounding words
        mention_pos = context.find(mention.text)
        if mention_pos >= 0:
            # Get words before and after
            before = context[:mention_pos].strip().split()[-3:]
            after = context[mention_pos + len(mention.text) :].strip().split()[:3]
            context = " ".join(before + after)

        if len(context) > max_context:
            context = context[:max_context]

        return context.strip()

    def _results_to_candidates(
        self,
        results: dict[str, Any],
    ) -> list[HPOCandidate]:
        """
        Convert retriever results to HPOCandidate objects.

        Args:
            results: Results from the retriever

        Returns:
            List of HPOCandidate objects
        """
        candidates: list[HPOCandidate] = []

        if not results.get("metadatas") or not results["metadatas"][0]:
            return candidates

        metadatas = results["metadatas"][0]
        similarities = results.get("similarities", [[]])[0]

        for i, metadata in enumerate(metadatas):
            similarity = similarities[i] if i < len(similarities) else 0.0

            # Skip below threshold
            if similarity < self.config.retrieval_threshold:
                continue

            hpo_id = metadata.get("id") or metadata.get("hpo_id")
            label = metadata.get("label") or metadata.get("name")

            if not hpo_id or not label:
                continue

            # Get ontology depth
            depth = self.hpo_depth_map.get(hpo_id, 0)

            # Get synonyms if available
            synonyms = metadata.get("synonyms", [])
            if isinstance(synonyms, str):
                synonyms = [s.strip() for s in synonyms.split("|") if s.strip()]

            candidate = HPOCandidate(
                hpo_id=hpo_id,
                label=label,
                score=similarity,
                depth=depth,
                synonyms=synonyms,
                definition=metadata.get("definition"),
            )
            candidates.append(candidate)

        return candidates

    def compute_mention_embeddings(
        self,
        mentions: list[Mention],
        model: SentenceTransformer,
    ) -> np.ndarray:
        """
        Compute embeddings for mentions (for similarity-based grouping).

        Args:
            mentions: List of mentions
            model: SentenceTransformer model

        Returns:
            Array of embeddings (num_mentions x embedding_dim)
        """
        texts = [m.text for m in mentions]
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )

        # Update mentions with embeddings
        for i, mention in enumerate(mentions):
            mention.embedding = embeddings[i]

        return embeddings


def retrieve_hpo_for_mentions(
    mentions: list[Mention],
    retriever: Any,  # DenseRetriever
    config: MentionRetrievalConfig | None = None,
) -> list[list[HPOCandidate]]:
    """
    Convenience function to retrieve HPO candidates for mentions.

    Args:
        mentions: List of mentions
        retriever: DenseRetriever instance
        config: Retrieval configuration

    Returns:
        List of candidate lists
    """
    mention_retriever = MentionHPORetriever(retriever=retriever, config=config)
    return mention_retriever.retrieve_batch(mentions)
