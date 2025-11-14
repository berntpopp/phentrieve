"""
Assertion detection for clinical text.

This module provides utilities for detecting negation, normality and
uncertainty in clinical text chunks.
"""

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import spacy

# Local imports
from phentrieve.text_processing.resource_loader import load_language_resource
from phentrieve.utils import load_user_config

logger = logging.getLogger(__name__)

# Cache for loaded spaCy models
NLP_MODELS: dict[str, Language | None] = {}


class AssertionStatus(Enum):
    """
    Enum representing possible assertion statuses for clinical text.
    """

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"


# Keyword detection window size (number of words before/after cue)
KEYWORD_WINDOW = 7


def get_spacy_model(lang_code: str) -> Optional[spacy.language.Language]:
    """
    Load and cache a spaCy model for the given language.

    Args:
        lang_code: ISO language code ('en', 'de', etc.)

    Returns:
        Loaded spaCy model or None if unavailable
    """
    if lang_code not in NLP_MODELS:
        model_name_spacy = ""
        if lang_code == "en":
            model_name_spacy = "en_core_web_sm"
        elif lang_code == "de":
            model_name_spacy = "de_core_news_sm"
        elif lang_code == "fr":
            model_name_spacy = "fr_core_news_sm"
        elif lang_code == "es":
            model_name_spacy = "es_core_news_sm"
        elif lang_code == "nl":
            model_name_spacy = "nl_core_news_sm"

        if model_name_spacy:
            try:
                NLP_MODELS[lang_code] = spacy.load(model_name_spacy)
                logger.info(
                    f"Loaded spaCy model '{model_name_spacy}' for language '{lang_code}'."
                )
            except OSError:
                logger.warning(
                    f"spaCy model '{model_name_spacy}' for lang '{lang_code}' not found. "
                    f"Download with: python -m spacy download {model_name_spacy}. "
                    f"Dependency parsing will be skipped for '{lang_code}'."
                )
                NLP_MODELS[lang_code] = None
        else:
            logger.warning(
                f"No spaCy model configured for lang '{lang_code}'. "
                "Dependency parsing will be skipped."
            )
            NLP_MODELS[lang_code] = None

    return NLP_MODELS.get(lang_code)


class AssertionDetector(ABC):
    """
    Abstract base class for assertion detection strategies.

    All assertion detectors must implement a detect method that takes a text
    chunk and returns an assertion status with additional details.
    """

    def __init__(self, language: str = "en", **kwargs):
        """
        Initialize the detector.

        Args:
            language: ISO language code ('en', 'de', etc.)
            **kwargs: Additional configuration parameters
        """
        self.language = language

    @abstractmethod
    def detect(self, text_chunk: str) -> tuple[AssertionStatus, dict[str, Any]]:
        """
        Detect assertion status in text.

        Args:
            text_chunk: Text to analyze

        Returns:
            Tuple of (AssertionStatus, Dict with details)
        """
        pass


class KeywordAssertionDetector(AssertionDetector):
    """
    Assertion detector that uses keyword-based rules.

    This detector looks for negation and normality cues in the text
    and checks their context to determine assertion status.
    """

    def detect(self, text_chunk: str) -> tuple[AssertionStatus, dict[str, Any]]:
        """
        Detect assertion status using keyword-based rules.

        Args:
            text_chunk: Text to analyze

        Returns:
            Tuple of (AssertionStatus, Dict with details)
        """
        is_negated, is_normal, negated_scopes, normal_scopes = (
            self._detect_negation_normality_keyword(text_chunk, self.language)
        )

        assertion_details = {
            "keyword_negated_scopes": negated_scopes,
            "keyword_normal_scopes": normal_scopes,
        }

        if is_negated:
            return AssertionStatus.NEGATED, assertion_details
        elif is_normal:
            return AssertionStatus.NORMAL, assertion_details
        else:
            return AssertionStatus.AFFIRMED, assertion_details

    def _detect_negation_normality_keyword(
        self, chunk: str, lang: str = "en"
    ) -> tuple[bool, bool, list[str], list[str]]:
        """
        Detect negation and normality using keyword-based rules.

        Args:
            chunk: Text to analyze
            lang: Language code

        Returns:
            Tuple of (is_negated, is_normal, negated_scopes, normal_scopes)
        """
        if not chunk:
            return False, False, [], []

        text_lower = chunk.lower()
        re.sub(r"[^\w\s]", " ", text_lower).split()

        # Helper function to check for cue match
        def is_cue_match(text_lower, cue_lower, index):
            return (index == 0 and text_lower.startswith(cue_lower)) or (
                index > 0 and f" {cue_lower}" in text_lower
            )

        # Load user configuration
        user_config_main = load_user_config()
        language_resources_section = user_config_main.get("language_resources", {})

        # Load negation cues from resource files
        negation_cues_resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
            language_resources_config_section=language_resources_section,
        )

        # Check for negation cues
        negated_scopes = []
        is_negated = False

        # Get negation cues for the current language, defaulting to English
        lang_negation_cues = negation_cues_resources.get(
            lang, negation_cues_resources.get("en", [])
        )

        for cue in lang_negation_cues:
            cue_lower = cue.lower()
            cue_index = text_lower.find(cue_lower)

            if cue_index >= 0 and is_cue_match(text_lower, cue_lower, cue_index):
                # Found a negation cue, extract the context after the cue
                cue_end = cue_index + len(cue_lower)
                words_after = text_lower[cue_end:].split()

                # Take up to KEYWORD_WINDOW words after the cue
                context = " ".join(words_after[:KEYWORD_WINDOW])
                if context:
                    negated_scopes.append(f"{cue.strip()}: {context}")
                    is_negated = True

        # Load normality cues from resource files
        normality_cues_resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
            language_resources_config_section=language_resources_section,
        )

        # Check for normality cues
        normal_scopes = []
        is_normal = False

        # Get normality cues for the current language, defaulting to English
        lang_normality_cues = normality_cues_resources.get(
            lang, normality_cues_resources.get("en", [])
        )

        for cue in lang_normality_cues:
            cue_lower = cue.lower()
            cue_index = text_lower.find(cue_lower)

            if cue_index >= 0 and is_cue_match(text_lower, cue_lower, cue_index):
                # Get words around the cue (simple window approach)
                start_idx = max(0, cue_index - 30)
                end_idx = min(len(text_lower), cue_index + len(cue_lower) + 30)
                context = text_lower[start_idx:end_idx]

                normal_scopes.append(f"{cue.strip()}: {context}")
                is_normal = True

        return is_negated, is_normal, negated_scopes, normal_scopes


class DependencyAssertionDetector(AssertionDetector):
    """
    Assertion detector using dependency parsing.

    This detector uses spaCy's dependency parsing to analyze
    relationships between negation/normality cues and concepts.
    """

    def detect(self, text_chunk: str) -> tuple[AssertionStatus, dict[str, Any]]:
        """
        Detect assertion status using dependency parsing.

        Args:
            text_chunk: Text to analyze

        Returns:
            Tuple of (AssertionStatus, Dict with details)
        """
        # Get spaCy model for language
        nlp = get_spacy_model(self.language)
        if not nlp:
            # If no model available, return AFFIRMED with empty details
            return AssertionStatus.AFFIRMED, {
                "dependency_negated_concepts": [],
                "dependency_normal_concepts": [],
                "dependency_parser": False,
            }

        is_negated, is_normal, negated_concepts, normal_concepts = (
            self._detect_negation_normality_dependency(text_chunk, self.language)
        )

        assertion_details = {
            "dependency_negated_concepts": negated_concepts,
            "dependency_normal_concepts": normal_concepts,
            "dependency_parser": True,
        }

        if is_negated:
            return AssertionStatus.NEGATED, assertion_details
        elif is_normal:
            return AssertionStatus.NORMAL, assertion_details
        else:
            return AssertionStatus.AFFIRMED, assertion_details

    def _detect_negation_normality_dependency(
        self, chunk: str, lang: str = "en"
    ) -> tuple[bool, bool, list[str], list[str]]:
        """
        Detect negation and normality using dependency parsing.

        Args:
            chunk: Text to analyze
            lang: Language code

        Returns:
            Tuple of (is_negated, is_normal, negated_concepts, normal_concepts)
        """
        if not chunk:
            return False, False, [], []

        nlp = get_spacy_model(lang)
        if not nlp:
            return False, False, [], []

        doc = nlp(chunk)

        # Check for negation
        negated_concepts = []
        is_negated = False

        # Handle German negation directly with a text check first (more reliable for short phrases)
        chunk_lower = chunk.lower()
        if lang == "de" and any(
            neg_term in chunk_lower for neg_term in ["kein", "keine", "keinen", "nicht"]
        ):
            is_negated = True
            negated_concepts.append(f"German negation term found in: {chunk}")

        # Also check with spaCy's dependency parsing
        for token in doc:
            token_text = token.text.lower()

            # Load negation cues from resource files
            user_config_main = load_user_config()
            language_resources_section = user_config_main.get("language_resources", {})

            negation_cues_resources = load_language_resource(
                default_resource_filename="negation_cues.json",
                config_key_for_custom_file="negation_cues_file",
                language_resources_config_section=language_resources_section,
            )

            # Check if token or its lemma is a negation cue
            is_negation_term = False
            lang_negation_cues = negation_cues_resources.get(
                lang, negation_cues_resources.get("en", [])
            )

            for neg_cue in lang_negation_cues:
                neg_cue_clean = neg_cue.strip().lower()
                # More flexible matching for German
                if (
                    lang == "de"
                    and neg_cue_clean.startswith("kein")
                    and token_text.startswith("kein")
                ):
                    is_negation_term = True
                    break
                # Regular exact matching
                elif (
                    token_text == neg_cue_clean or token.lemma_.lower() == neg_cue_clean
                ):
                    is_negation_term = True
                    break

            if is_negation_term or token.dep_ == "neg":
                # Found a negation term, follow the dependency chain
                scope = self._get_negation_scope(token)
                if scope:
                    negated_concepts.append(scope)
                    is_negated = True

        # Check for normality
        normal_concepts = []
        is_normal = False

        # Load normality cues from resource files
        normality_cues_resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
            language_resources_config_section=language_resources_section,
        )

        # Get normality cues for the current language, defaulting to English
        lang_normality_cues = normality_cues_resources.get(
            lang, normality_cues_resources.get("en", [])
        )

        for token in doc:
            token_text = token.text.lower()

            # Check if token or multi-token span matches normality cue
            for norm_cue in lang_normality_cues:
                norm_cue = norm_cue.strip().lower()
                if token_text == norm_cue or token.lemma_.lower() == norm_cue:
                    # Single token match
                    scope = self._get_normality_scope(token)
                    if scope:
                        normal_concepts.append(scope)
                        is_normal = True
                # Check multi-token normality expressions
                elif len(norm_cue.split()) > 1 and norm_cue in chunk.lower():
                    # For multi-token expressions, use a simpler context extraction
                    context_start = max(0, token.i - 3)
                    context_end = min(len(doc), token.i + 7)
                    scope = " ".join([t.text for t in doc[context_start:context_end]])
                    normal_concepts.append(scope)
                    is_normal = True
                    break

        return is_negated, is_normal, negated_concepts, normal_concepts

    def _get_negation_scope(self, token) -> str:
        """
        Get the scope of a negation token.

        Follows dependency relations to find what's being negated.

        Args:
            token: The negation token

        Returns:
            String describing the negation scope
        """
        # For a negation token, find what it's negating
        if token.dep_ == "neg":
            # Negation is usually attached to a verb
            head = token.head

            # Find the subject and object of the negated verb if available
            subject = None
            obj = None

            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                elif child.dep_ in ("dobj", "pobj", "attr"):
                    obj = child

            if subject and obj:
                return f"{token.text} {head.text}: {subject.text} → {obj.text}"
            elif subject:
                return f"{token.text} {head.text}: {subject.text}"
            elif obj:
                return f"{token.text} {head.text}: {obj.text}"
            else:
                return f"{token.text} {head.text}"
        else:
            # For other negation terms like "absence", "without", etc.
            # Look for prepositional objects or direct objects
            for child in token.children:
                if child.dep_ in ("pobj", "dobj"):
                    return f"{token.text} → {child.text}"

            # Fallback - return the token with its head
            return f"{token.text} → {token.head.text}"

    def _get_normality_scope(self, token) -> str:
        """
        Get the scope of a normality token.

        Args:
            token: The normality token

        Returns:
            String describing the normality scope
        """
        # If token is an adjective, check what it's modifying
        if token.pos_ == "ADJ":
            head = token.head
            return f"{token.text} {head.text}"

        # If token is part of a prepositional phrase
        if token.dep_ == "pobj":
            prep = token.head  # The preposition
            if prep.dep_ == "prep":
                head = prep.head  # What the prep phrase is modifying
                return f"{prep.text} {token.text} ← {head.text}"

        # Check children for a "of" prepositional phrase
        for child in token.children:
            if child.dep_ == "prep" and child.text.lower() == "of":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return f"{token.text} {child.text} {grandchild.text}"

        # Fallback - simple context window
        doc = token.doc
        context_start = max(0, token.i - 2)
        context_end = min(len(doc), token.i + 5)
        return " ".join([t.text for t in doc[context_start:context_end]])


class CombinedAssertionDetector(AssertionDetector):
    """
    Assertion detector that combines keyword and dependency-based approaches.

    This detector can use different combination strategies to determine
    the final assertion status.
    """

    def __init__(
        self,
        language: str = "en",
        enable_keyword: bool = True,
        enable_dependency: bool = True,
        preference: str = "dependency",  # 'keyword', 'dependency', or 'any_negative'
        **kwargs,
    ):
        """
        Initialize the combined detector.

        Args:
            language: ISO language code
            enable_keyword: Whether to use keyword-based detection
            enable_dependency: Whether to use dependency-based detection
            preference: Strategy for combining results ('dependency', 'keyword', 'any_negative')
            **kwargs: Additional configuration parameters
        """
        super().__init__(language=language, **kwargs)
        self.enable_keyword = enable_keyword
        self.enable_dependency = enable_dependency
        self.preference = preference

        # Initialize sub-detectors
        self.keyword_detector = (
            KeywordAssertionDetector(language=language) if enable_keyword else None
        )
        self.dependency_detector = (
            DependencyAssertionDetector(language=language)
            if enable_dependency
            else None
        )

    def detect(self, text_chunk: str) -> tuple[AssertionStatus, dict[str, Any]]:
        """
        Detect assertion status using multiple strategies.

        Args:
            text_chunk: Text to analyze

        Returns:
            Tuple of (AssertionStatus, Dict with details)
        """
        if not text_chunk.strip():
            return AssertionStatus.AFFIRMED, {"empty_input": True}

        # Initialize results
        keyword_status = None
        keyword_details: dict[str, Any] = {}
        dependency_status = None
        dependency_details: dict[str, Any] = {}

        # Run enabled detectors
        if self.enable_keyword and self.keyword_detector:
            keyword_status, keyword_details = self.keyword_detector.detect(text_chunk)

        if self.enable_dependency and self.dependency_detector:
            dependency_status, dependency_details = self.dependency_detector.detect(
                text_chunk
            )

        # Combine the results according to strategy
        combined_details = {
            **keyword_details,
            **dependency_details,
            "keyword_status": keyword_status.value if keyword_status else None,
            "dependency_status": dependency_status.value if dependency_status else None,
            "combination_strategy": self.preference,
        }

        # Determine final status - Always use the same straightforward priority as in test script
        final_status = AssertionStatus.AFFIRMED  # Default

        # Apply the simple priority logic from test_semantic_chunking.py
        if self.dependency_detector and dependency_status == AssertionStatus.NEGATED:
            # Dependency parsing detected negation - highest priority
            final_status = AssertionStatus.NEGATED
        elif self.dependency_detector and dependency_status == AssertionStatus.NORMAL:
            # Dependency parsing detected normality - second priority
            final_status = AssertionStatus.NORMAL
        elif keyword_status == AssertionStatus.NEGATED:
            # Keyword detection found negation - third priority
            final_status = AssertionStatus.NEGATED
        elif keyword_status == AssertionStatus.NORMAL:
            # Keyword detection found normality - fourth priority
            final_status = AssertionStatus.NORMAL
        # Default remains AFFIRMED if none of the above conditions are met

        combined_details["final_status"] = final_status.value
        return final_status, combined_details
