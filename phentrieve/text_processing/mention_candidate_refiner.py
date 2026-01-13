"""
Mention candidate refinement and specificity control.

This module refines HPO candidates for mentions by:
1. Re-ranking with cross-encoder (if available)
2. Applying ontology-aware specificity scoring
3. Penalizing generic terms when specific alternatives exist

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from phentrieve.text_processing.mention import HPOCandidate, Mention

logger = logging.getLogger(__name__)

# Default refinement configuration
DEFAULT_GENERIC_PENALTY = 0.1
DEFAULT_MIN_SPECIFICITY_DEPTH = 3
DEFAULT_RERANK_TOP_K = 10


@dataclass
class CandidateRefinementConfig:
    """
    Configuration for candidate refinement.

    Attributes:
        enable_reranking: Whether to use cross-encoder reranking
        rerank_top_k: Number of top candidates to rerank
        enable_specificity_scoring: Apply ontology depth-based scoring
        generic_penalty: Penalty for generic (shallow) terms (0-1)
        min_specificity_depth: Minimum depth for non-generic terms
        specificity_weight: Weight for specificity in final score (0-1)
        prefer_specific_when_available: Boost specific terms over generic
    """

    enable_reranking: bool = True
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    enable_specificity_scoring: bool = True
    generic_penalty: float = DEFAULT_GENERIC_PENALTY
    min_specificity_depth: int = DEFAULT_MIN_SPECIFICITY_DEPTH
    specificity_weight: float = 0.2
    prefer_specific_when_available: bool = True


class MentionCandidateRefiner:
    """
    Refine HPO candidates for mentions.

    Applies re-ranking and specificity control to improve
    precision of HPO assignments.

    Example:
        >>> refiner = MentionCandidateRefiner()
        >>> mention = Mention(text="seizures", ...)
        >>> mention.hpo_candidates = [candidate1, candidate2, ...]
        >>> refiner.refine(mention)
        >>> mention.top_candidate  # Now reflects refined scores
    """

    def __init__(
        self,
        cross_encoder: CrossEncoder | None = None,
        config: CandidateRefinementConfig | None = None,
        hpo_depth_map: dict[str, int] | None = None,
        hpo_ancestor_map: dict[str, set[str]] | None = None,
    ):
        """
        Initialize the candidate refiner.

        Args:
            cross_encoder: Optional cross-encoder for reranking
            config: Refinement configuration
            hpo_depth_map: Map of HPO ID to ontology depth
            hpo_ancestor_map: Map of HPO ID to ancestor IDs
        """
        self.cross_encoder = cross_encoder
        self.config = config or CandidateRefinementConfig()
        self.hpo_depth_map = hpo_depth_map or {}
        self.hpo_ancestor_map = hpo_ancestor_map or {}

    def refine(self, mention: Mention) -> list[HPOCandidate]:
        """
        Refine candidates for a single mention.

        Args:
            mention: Mention with hpo_candidates

        Returns:
            Refined and reranked candidates
        """
        if not mention.hpo_candidates:
            return []

        candidates = mention.hpo_candidates.copy()

        # Apply cross-encoder reranking
        if self.config.enable_reranking and self.cross_encoder is not None:
            candidates = self._rerank_candidates(mention, candidates)

        # Apply specificity scoring
        if self.config.enable_specificity_scoring:
            candidates = self._apply_specificity_scoring(candidates)

        # Apply generic term penalty
        if self.config.prefer_specific_when_available:
            candidates = self._penalize_generic_terms(candidates)

        # Sort by refined score
        candidates.sort(key=lambda c: c.effective_score, reverse=True)

        # Update mention
        mention.hpo_candidates = candidates

        return candidates

    def refine_batch(self, mentions: list[Mention]) -> None:
        """
        Refine candidates for multiple mentions.

        More efficient than calling refine() repeatedly when
        cross-encoder reranking is enabled.

        Args:
            mentions: List of mentions to refine
        """
        if not mentions:
            return

        logger.debug(f"Refining candidates for {len(mentions)} mentions")

        # Batch reranking for efficiency
        if self.config.enable_reranking and self.cross_encoder is not None:
            self._batch_rerank(mentions)

        # Apply other refinements
        for mention in mentions:
            candidates = mention.hpo_candidates

            if self.config.enable_specificity_scoring:
                candidates = self._apply_specificity_scoring(candidates)

            if self.config.prefer_specific_when_available:
                candidates = self._penalize_generic_terms(candidates)

            candidates.sort(key=lambda c: c.effective_score, reverse=True)
            mention.hpo_candidates = candidates

    def _rerank_candidates(
        self,
        mention: Mention,
        candidates: list[HPOCandidate],
    ) -> list[HPOCandidate]:
        """
        Rerank candidates using cross-encoder.

        Args:
            mention: The mention
            candidates: Current candidates

        Returns:
            Reranked candidates
        """
        if not candidates or self.cross_encoder is None:
            return candidates

        # Only rerank top K
        top_k = min(self.config.rerank_top_k, len(candidates))
        to_rerank = candidates[:top_k]
        rest = candidates[top_k:]

        # Build query-candidate pairs
        query = mention.context_window or mention.text
        pairs = [(query, c.label) for c in to_rerank]

        try:
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

            # Update refined scores
            for i, candidate in enumerate(to_rerank):
                score = scores[i]
                if isinstance(score, (list, np.ndarray)) and len(score) > 1:
                    candidate.refined_score = float(score[0])
                else:
                    candidate.refined_score = float(score)

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            # Fall back to original scores
            for candidate in to_rerank:
                candidate.refined_score = candidate.score

        # Combine reranked and rest
        return to_rerank + rest

    def _batch_rerank(self, mentions: list[Mention]) -> None:
        """
        Batch rerank candidates across all mentions.

        Args:
            mentions: Mentions to rerank
        """
        if self.cross_encoder is None:
            return

        # Collect all pairs
        all_pairs: list[tuple[str, str]] = []
        pair_indices: list[tuple[int, int]] = []  # (mention_idx, candidate_idx)

        for m_idx, mention in enumerate(mentions):
            if not mention.hpo_candidates:
                continue

            query = mention.context_window or mention.text
            top_k = min(self.config.rerank_top_k, len(mention.hpo_candidates))

            for c_idx in range(top_k):
                candidate = mention.hpo_candidates[c_idx]
                all_pairs.append((query, candidate.label))
                pair_indices.append((m_idx, c_idx))

        if not all_pairs:
            return

        try:
            # Batch predict
            scores = self.cross_encoder.predict(all_pairs, show_progress_bar=False)

            # Assign scores back
            for i, (m_idx, c_idx) in enumerate(pair_indices):
                score = scores[i]
                if isinstance(score, (list, np.ndarray)) and len(score) > 1:
                    refined = float(score[0])
                else:
                    refined = float(score)

                mentions[m_idx].hpo_candidates[c_idx].refined_score = refined

        except Exception as e:
            logger.warning(f"Batch cross-encoder reranking failed: {e}")

    def _apply_specificity_scoring(
        self,
        candidates: list[HPOCandidate],
    ) -> list[HPOCandidate]:
        """
        Apply ontology depth-based specificity scoring.

        Args:
            candidates: Candidates to score

        Returns:
            Candidates with updated specificity scores
        """
        if not candidates:
            return candidates

        # Calculate max depth among candidates for normalization
        max_depth = max(c.depth for c in candidates) if candidates else 1
        max_depth = max(max_depth, 1)  # Avoid division by zero

        for candidate in candidates:
            # Get depth from map if not set
            if candidate.depth == 0 and candidate.hpo_id in self.hpo_depth_map:
                candidate.depth = self.hpo_depth_map[candidate.hpo_id]

            # Normalize depth to 0-1 specificity score
            specificity = candidate.depth / max_depth
            candidate.specificity_score = specificity

            # Mark as generic if too shallow
            if candidate.depth < self.config.min_specificity_depth:
                candidate.is_generic = True

        return candidates

    def _penalize_generic_terms(
        self,
        candidates: list[HPOCandidate],
    ) -> list[HPOCandidate]:
        """
        Penalize generic terms when specific alternatives exist.

        Only applies penalty if there are specific alternatives
        with reasonable scores.

        Args:
            candidates: Candidates to penalize

        Returns:
            Candidates with adjusted scores
        """
        if not candidates:
            return candidates

        # Check if we have any specific candidates with good scores
        specific_candidates = [c for c in candidates if not c.is_generic]
        if not specific_candidates:
            # No specific alternatives, don't penalize
            return candidates

        # Get threshold from best specific candidate
        best_specific_score = max(c.effective_score for c in specific_candidates)

        # Apply penalty to generic candidates
        for candidate in candidates:
            if candidate.is_generic:
                # Only penalize if there's a competitive specific alternative
                if candidate.effective_score <= best_specific_score * 1.2:
                    current_score = candidate.refined_score or candidate.score
                    penalty = self.config.generic_penalty
                    candidate.refined_score = current_score * (1 - penalty)

        return candidates

    def get_specificity_info(self, hpo_id: str) -> dict[str, Any]:
        """
        Get specificity information for an HPO term.

        Args:
            hpo_id: HPO identifier

        Returns:
            Dict with depth, is_generic, ancestors
        """
        depth = self.hpo_depth_map.get(hpo_id, 0)
        ancestors = self.hpo_ancestor_map.get(hpo_id, set())

        return {
            "hpo_id": hpo_id,
            "depth": depth,
            "is_generic": depth < self.config.min_specificity_depth,
            "num_ancestors": len(ancestors),
        }


def refine_mention_candidates(
    mentions: list[Mention],
    cross_encoder: CrossEncoder | None = None,
    config: CandidateRefinementConfig | None = None,
) -> None:
    """
    Convenience function to refine candidates for mentions.

    Args:
        mentions: List of mentions with candidates
        cross_encoder: Optional cross-encoder for reranking
        config: Refinement configuration
    """
    refiner = MentionCandidateRefiner(cross_encoder=cross_encoder, config=config)
    refiner.refine_batch(mentions)
