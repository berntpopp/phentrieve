"""
Mention-level assertion detection.

This module provides assertion detection at the mention level, using
the existing assertion detection infrastructure but scoped to individual
mentions rather than full chunks. This enables more precise assertion
assignment for clinical finding mentions.

See: plan/01-active/MENTION-LEVEL-HPO-EXTRACTION-PLAN.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language

from phentrieve.text_processing.assertion_detection import (
    AssertionDetector,
    AssertionStatus,
    CombinedAssertionDetector,
)
from phentrieve.text_processing.assertion_representation import (
    AssertionVector,
    affirmed_vector,
    negated_vector,
    normal_vector,
    uncertain_vector,
)
from phentrieve.text_processing.document_structure import detect_family_history_spans
from phentrieve.text_processing.mention import Mention

logger = logging.getLogger(__name__)


class MentionAssertionDetector:
    """
    Detect assertion status for individual mentions.

    Uses the existing assertion detection infrastructure (keyword-based
    and dependency-based) but applies it at the mention level with
    proper scoping.

    Example:
        >>> detector = MentionAssertionDetector(language="en")
        >>> mention = Mention(text="seizures", start_char=20, end_char=28)
        >>> context = "Patient denies any seizures or headaches."
        >>> detector.detect(mention, context)
        >>> mention.assertion.negation_score
        0.9
    """

    def __init__(
        self,
        language: str = "en",
        strategy: str = "combined",
        nlp: Language | None = None,
    ):
        """
        Initialize the mention assertion detector.

        Args:
            language: Language code
            strategy: Detection strategy ("keyword", "dependency", "combined")
            nlp: Optional pre-loaded spaCy model
        """
        self.language = language
        self.strategy = strategy
        self._nlp = nlp
        self._detector: AssertionDetector | None = None

    @property
    def detector(self) -> AssertionDetector:
        """Get the underlying assertion detector, initializing if needed."""
        if self._detector is None:
            self._detector = self._create_detector()
        return self._detector

    def _create_detector(self) -> AssertionDetector:
        """Create the appropriate assertion detector."""
        if self.strategy == "combined":
            return CombinedAssertionDetector(language=self.language)
        else:
            # Use combined detector as default
            return CombinedAssertionDetector(language=self.language)

    def detect(
        self,
        mention: Mention,
        context: str,
        update_mention: bool = True,
    ) -> AssertionVector:
        """
        Detect assertion status for a mention.

        Args:
            mention: The mention to analyze
            context: Surrounding text context (sentence or larger)
            update_mention: If True, update the mention's assertion attribute

        Returns:
            AssertionVector with detection results
        """
        # Get mention-local context
        local_context = self._get_local_context(mention, context)

        # Run assertion detection on local context
        status, confidence = self._detect_status(local_context, mention.text)

        # Convert to AssertionVector
        assertion = self._status_to_vector(status, confidence)

        # Check for family history context
        if self._is_family_history_context(mention, context):
            assertion = assertion.with_updates(family_history=True)

        # Update mention if requested
        if update_mention:
            mention.assertion = assertion

        return assertion

    def detect_batch(
        self,
        mentions: list[Mention],
        full_text: str,
    ) -> list[AssertionVector]:
        """
        Detect assertion status for multiple mentions.

        Args:
            mentions: List of mentions to analyze
            full_text: Full document text

        Returns:
            List of AssertionVector results (also updates mentions)
        """
        # Pre-compute family history spans for efficiency
        fh_spans = detect_family_history_spans(full_text, language=self.language)

        results: list[AssertionVector] = []
        for mention in mentions:
            # Get context around mention
            context = self._get_sentence_context(mention, full_text)

            # Check family history
            is_fh = self._is_in_family_history_spans(mention, fh_spans)

            # Detect assertion
            local_context = self._get_local_context(mention, context)
            status, confidence = self._detect_status(local_context, mention.text)
            assertion = self._status_to_vector(status, confidence)

            if is_fh:
                assertion = assertion.with_updates(family_history=True)

            mention.assertion = assertion
            results.append(assertion)

        return results

    def _get_local_context(self, mention: Mention, context: str) -> str:
        """
        Extract the local context around a mention for assertion detection.

        Uses the mention's position within the context to extract a
        window that includes potential negation triggers.

        Args:
            mention: The mention
            context: Surrounding text

        Returns:
            Local context string
        """
        # If context is short, use it all
        if len(context) <= 100:
            return context

        # Use the context window from the mention if available
        if mention.context_window:
            return mention.context_window

        # Otherwise, extract based on position
        # This is a fallback - the mention's context_window should be preferred
        return context

    def _get_sentence_context(self, mention: Mention, full_text: str) -> str:
        """
        Get the sentence containing the mention.

        Args:
            mention: The mention
            full_text: Full document text

        Returns:
            Sentence text containing the mention
        """
        # Simple sentence extraction using punctuation
        start = mention.start_char
        end = mention.end_char

        # Find sentence start
        sentence_start = max(0, start - 200)
        for i in range(start - 1, sentence_start - 1, -1):
            if i >= 0 and full_text[i] in ".!?\n":
                sentence_start = i + 1
                break

        # Find sentence end
        sentence_end = min(len(full_text), end + 200)
        for i in range(end, sentence_end):
            if i < len(full_text) and full_text[i] in ".!?\n":
                sentence_end = i + 1
                break

        return full_text[sentence_start:sentence_end].strip()

    def _detect_status(
        self,
        context: str,
        mention_text: str,
    ) -> tuple[AssertionStatus, float]:
        """
        Run the underlying assertion detector.

        Args:
            context: Context text
            mention_text: The mention text

        Returns:
            Tuple of (AssertionStatus, confidence)
        """
        try:
            result = self.detector.detect(context)

            # The detector returns tuple[AssertionStatus, dict[str, Any]]
            status: AssertionStatus
            if isinstance(result, tuple):
                status = result[0]
            elif hasattr(result, "value"):
                status = result  # type: ignore[assignment]
            else:
                status = AssertionStatus.AFFIRMED

            # Estimate confidence based on context analysis
            confidence = self._estimate_confidence(context, status)

            return status, confidence

        except Exception as e:
            logger.warning(f"Assertion detection failed: {e}")
            return AssertionStatus.AFFIRMED, 0.5

    def _estimate_confidence(
        self,
        context: str,
        status: AssertionStatus,
    ) -> float:
        """
        Estimate confidence in the assertion detection.

        Args:
            context: Context text
            status: Detected status

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Simple heuristic: stronger signals get higher confidence
        context_lower = context.lower()

        if status == AssertionStatus.NEGATED:
            # Strong negation signals
            strong_negations = [
                "no ",
                "not ",
                "denies ",
                "without ",
                "absent",
                "negative",
            ]
            if any(neg in context_lower for neg in strong_negations):
                return 0.9
            return 0.7

        elif status == AssertionStatus.UNCERTAIN:
            # Uncertainty signals
            uncertainty_signals = [
                "possible",
                "may",
                "might",
                "could",
                "suspected",
                "rule out",
            ]
            if any(sig in context_lower for sig in uncertainty_signals):
                return 0.85
            return 0.7

        elif status == AssertionStatus.NORMAL:
            # Normality signals
            normal_signals = ["normal", "unremarkable", "within normal", "wnl"]
            if any(sig in context_lower for sig in normal_signals):
                return 0.85
            return 0.7

        else:  # AFFIRMED
            # Default to high confidence for affirmed
            return 0.9

    def _status_to_vector(
        self,
        status: AssertionStatus,
        confidence: float,
    ) -> AssertionVector:
        """
        Convert AssertionStatus to AssertionVector.

        Args:
            status: Detected status
            confidence: Detection confidence

        Returns:
            AssertionVector
        """
        if status == AssertionStatus.NEGATED:
            return negated_vector(confidence)
        elif status == AssertionStatus.UNCERTAIN:
            return uncertain_vector(confidence)
        elif status == AssertionStatus.NORMAL:
            return normal_vector(confidence)
        else:
            return affirmed_vector(confidence)

    def _is_family_history_context(
        self,
        mention: Mention,
        context: str,
    ) -> bool:
        """
        Check if mention is in a family history context.

        Args:
            mention: The mention
            context: Context text

        Returns:
            True if in family history context
        """
        # Check section type
        if mention.section_type == "family_history":
            return True

        # Check context for family history patterns
        context_lower = context.lower()
        fh_patterns = [
            "family history",
            "familienanamnese",
            "mother has",
            "father has",
            "sibling has",
            "parent has",
            "family member",
            "runs in the family",
            "hereditary",
        ]
        return any(pattern in context_lower for pattern in fh_patterns)

    def _is_in_family_history_spans(
        self,
        mention: Mention,
        fh_spans: list[tuple[int, int]],
    ) -> bool:
        """
        Check if mention falls within pre-computed family history spans.

        Args:
            mention: The mention
            fh_spans: List of (start, end) family history spans

        Returns:
            True if mention is within a family history span
        """
        for start, end in fh_spans:
            if start <= mention.start_char < end:
                return True
        return False


def detect_mention_assertions(
    mentions: list[Mention],
    full_text: str,
    language: str = "en",
) -> list[AssertionVector]:
    """
    Convenience function to detect assertions for a list of mentions.

    Args:
        mentions: List of mentions
        full_text: Full document text
        language: Language code

    Returns:
        List of AssertionVector results
    """
    detector = MentionAssertionDetector(language=language)
    return detector.detect_batch(mentions, full_text)
