"""
Assertion detection for clinical text.

This module provides utilities for detecting negation, normality and
uncertainty in clinical text chunks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import spacy
from spacy.language import Language

# Local imports
from phentrieve.text_processing.resource_loader import load_language_resource
from phentrieve.utils import load_user_config
from phentrieve.utils import sanitize_log_value as _sanitize

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


class Direction(Enum):
    """
    Direction in which a ConText rule operates from the trigger phrase.

    Based on medspaCy ConText algorithm:
    - FORWARD: Rule applies to text AFTER the trigger (e.g., "no [fever]")
    - BACKWARD: Rule applies to text BEFORE the trigger (e.g., "[fever] is absent")
    - BIDIRECTIONAL: Rule applies to text both before and after (e.g., "neither [A] nor [B]")
    - TERMINATE: Ends the scope of previous rules (e.g., conjunctions like "but", "however")
    - PSEUDO: Marks false positives (e.g., "not only", "no increase" - NOT negations)
    """

    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    TERMINATE = "TERMINATE"
    PSEUDO = "PSEUDO"


class TriggerCategory(Enum):
    """
    Category of ConText trigger phrase.

    Based on medspaCy ConText categories:
    - NEGATED_EXISTENCE: Negates existence of finding (e.g., "no", "denies", "absent")
    - POSSIBLE_EXISTENCE: Indicates uncertainty (e.g., "possible", "may", "rule out")
    - HISTORICAL: Indicates past condition (e.g., "history of", "previous")
    - HYPOTHETICAL: Indicates non-actual condition (e.g., "if", "should", "in case of")
    - FAMILY: Indicates family member has condition (e.g., "family history", "mother has")
    - TERMINATE: Ends scope of previous triggers (e.g., "but", "however", "although")
    - PSEUDO: False positive trigger (e.g., "not only", "no increase")
    """

    NEGATED_EXISTENCE = "NEGATED_EXISTENCE"
    POSSIBLE_EXISTENCE = "POSSIBLE_EXISTENCE"
    HISTORICAL = "HISTORICAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    FAMILY = "FAMILY"
    TERMINATE = "TERMINATE"
    PSEUDO = "PSEUDO"


@dataclass(frozen=True)
class ConTextRule:
    """
    Represents a ConText rule for assertion detection.

    ConText rules define trigger phrases with their semantic category and
    directionality for identifying negation, uncertainty, temporality, and
    experiencer in clinical text.

    Attributes:
        literal: The trigger phrase text (e.g., "no", "kein", "ausgeschlossen")
        category: Semantic category (NEGATED_EXISTENCE, POSSIBLE_EXISTENCE, etc.)
        direction: Direction the rule operates (FORWARD, BACKWARD, BIDIRECTIONAL)
        metadata: Optional dictionary for source tracking, notes, etc.

    Example:
        >>> rule = ConTextRule(
        ...     literal="Ausschluss",
        ...     category=TriggerCategory.NEGATED_EXISTENCE,
        ...     direction=Direction.FORWARD,
        ...     metadata={"source": "NegEx-DE"}
        ... )
    """

    literal: str
    category: TriggerCategory
    direction: Direction
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate rule after initialization.

        Note: The isinstance checks are defensive guards against issues during
        JSON deserialization (where strings are converted to enums) and dynamic
        attribute modification. While dataclass type annotations provide static
        hints, they don't enforce types at runtime, so these checks catch
        deserialization errors early with clear messages.
        """
        if not self.literal or not self.literal.strip():
            raise ValueError("ConTextRule literal cannot be empty")
        if not isinstance(self.category, TriggerCategory):
            raise ValueError(f"Invalid category: {self.category}")
        if not isinstance(self.direction, Direction):
            raise ValueError(f"Invalid direction: {self.direction}")


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
                    "Loaded spaCy model '%s' for language '%s'.",
                    _sanitize(model_name_spacy),
                    _sanitize(lang_code),
                )
            except OSError:
                logger.warning(
                    "spaCy model '%s' for lang '%s' not found. "
                    "Download with: python -m spacy download %s. "
                    "Dependency parsing will be skipped for '%s'.",
                    _sanitize(model_name_spacy),
                    _sanitize(lang_code),
                    _sanitize(model_name_spacy),
                    _sanitize(lang_code),
                )
                NLP_MODELS[lang_code] = None
        else:
            logger.warning(
                "No spaCy model configured for lang '%s'. "
                "Dependency parsing will be skipped.",
                _sanitize(lang_code),
            )
            NLP_MODELS[lang_code] = None

    return NLP_MODELS.get(lang_code)


def _is_cue_match(text_lower: str, cue_lower: str, index: int) -> bool:
    """
    Check if a cue matches at the given position with word boundary awareness.

    Ensures that the cue is a complete word/phrase and not part of a larger word.
    For example, "no" should NOT match inside "Normal".

    Args:
        text_lower: Lowercased text to search in
        cue_lower: Lowercased cue to search for
        index: Position where cue was found

    Returns:
        True if cue matches with proper word boundaries
    """
    # Check character before the cue (must be start of string or non-alphanumeric)
    if index > 0:
        char_before = text_lower[index - 1]
        if char_before.isalnum():
            return False

    # Check character after the cue (must be end of string or non-alphanumeric)
    cue_end = index + len(cue_lower)
    if cue_end < len(text_lower):
        char_after = text_lower[cue_end]
        if char_after.isalnum():
            return False

    return True


def parse_context_rules(context_data: dict[str, Any]) -> list[ConTextRule]:
    """
    Parse ConText rules from medspaCy-style JSON format.

    Expected JSON format:
    {
      "context_rules": [
        {
          "literal": "no",
          "category": "NEGATED_EXISTENCE",
          "direction": "FORWARD",
          "metadata": {"source": "medspaCy", "language": "en"}
        },
        ...
      ]
    }

    Args:
        context_data: Dictionary loaded from JSON file

    Returns:
        List of ConTextRule objects

    Raises:
        ValueError: If JSON format is invalid or required fields are missing
        KeyError: If required fields are missing from rule definitions
    """
    if "context_rules" not in context_data:
        raise ValueError("JSON must contain 'context_rules' key")

    rules = []
    for idx, rule_dict in enumerate(context_data["context_rules"]):
        try:
            # Required fields
            literal = rule_dict["literal"]
            category_str = rule_dict["category"]
            direction_str = rule_dict["direction"]

            # Optional metadata
            metadata = rule_dict.get("metadata", None)

            # Convert string to enum
            try:
                category = TriggerCategory[category_str]
            except KeyError:
                raise ValueError(
                    f"Invalid category '{category_str}' at rule {idx}. "
                    f"Valid categories: {[c.name for c in TriggerCategory]}"
                )

            try:
                direction = Direction[direction_str]
            except KeyError:
                raise ValueError(
                    f"Invalid direction '{direction_str}' at rule {idx}. "
                    f"Valid directions: {[d.name for d in Direction]}"
                )

            # Create rule (validation happens in __post_init__)
            rule = ConTextRule(
                literal=literal,
                category=category,
                direction=direction,
                metadata=metadata,
            )
            rules.append(rule)

        except KeyError as e:
            raise KeyError(
                f"Missing required field {e} in rule at index {idx}: {rule_dict}"
            )
        except ValueError as e:
            raise ValueError(f"Error parsing rule at index {idx}: {e}")

    logger.info("Parsed %s ConText rules from JSON", len(rules))
    return rules


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

    def __init__(self, language: str = "en", **kwargs):
        """
        Initialize the KeywordAssertionDetector.

        Args:
            language: ISO language code ('en', 'de', etc.)
            **kwargs: Additional configuration parameters
        """
        super().__init__(language, **kwargs)
        # Cache ConText rules per language (dict of lang -> rules)
        self._context_rules_cache: dict[str, list[ConTextRule] | None] = {}
        # Pre-load rules for the default language
        self._context_rules_cache[language] = self._load_context_rules(language)

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

    def _load_context_rules(self, lang: str) -> list[ConTextRule] | None:
        """
        Load ConText rules for the given language.

        Tries to load ConText rules from context_rules_{lang}.json first,
        then falls back to English if not available.

        Args:
            lang: Language code (e.g., 'de', 'en', 'es', 'fr', 'nl')

        Returns:
            List of ConTextRule objects or None if no ConText rules available
        """
        import importlib.resources
        import json

        # Package path for default language resources
        resources_package = "phentrieve.text_processing.default_lang_resources"

        # Try to load ConText rules for the requested language
        context_filename = f"context_rules_{lang}.json"
        try:
            # Get a path to the bundled default JSON file
            resource_path = importlib.resources.files(resources_package).joinpath(
                context_filename
            )

            # Read the ConText rules file
            with resource_path.open("r", encoding="utf-8") as f:
                context_data = json.load(f)

            if context_data and "context_rules" in context_data:
                rules = parse_context_rules(context_data)
                logger.info(
                    "Loaded %s ConText rules for language '%s'",
                    len(rules),
                    _sanitize(lang),
                )
                return rules
        except (FileNotFoundError, AttributeError):
            logger.debug(
                "ConText rules file '%s' not found for '%s'",
                _sanitize(context_filename),
                _sanitize(lang),
            )

        # Fall back to English ConText rules if available
        if lang != "en":
            try:
                resource_path = importlib.resources.files(resources_package).joinpath(
                    "context_rules_en.json"
                )

                with resource_path.open("r", encoding="utf-8") as f:
                    context_data = json.load(f)

                if context_data and "context_rules" in context_data:
                    rules = parse_context_rules(context_data)
                    logger.info(
                        "Loaded %s English ConText rules as fallback for '%s'",
                        len(rules),
                        _sanitize(lang),
                    )
                    return rules
            except (FileNotFoundError, AttributeError):
                logger.debug("English ConText rules not found either")

        logger.debug("No ConText rules available for language '%s'", _sanitize(lang))
        return None

    def _detect_negation_normality_keyword(
        self, chunk: str, lang: str = "en"
    ) -> tuple[bool, bool, list[str], list[str]]:
        """
        Detect negation and normality using ConText-aware keyword rules.

        This method uses ConText rules with direction awareness (FORWARD, BACKWARD,
        BIDIRECTIONAL) to detect negation and normality.

        Args:
            chunk: Text to analyze
            lang: Language code (e.g., 'de', 'en', 'es', 'fr', 'nl')

        Returns:
            Tuple of (is_negated, is_normal, negated_scopes, normal_scopes)
        """
        if not chunk:
            return False, False, [], []

        text_lower = chunk.lower()

        # Get ConText rules for the requested language (with caching)
        if lang not in self._context_rules_cache:
            # Load and cache rules for this language
            self._context_rules_cache[lang] = self._load_context_rules(lang)

        context_rules = self._context_rules_cache[lang]

        if not context_rules:
            logger.warning(
                "No ConText rules found for language '%s', keyword detection disabled",
                _sanitize(lang),
            )
            return False, False, [], []

        # Use ConText rules with direction awareness
        return self._detect_with_context_rules(text_lower, context_rules, chunk, lang)

    def _detect_with_context_rules(
        self,
        text_lower: str,
        rules: list[ConTextRule],
        original_text: str,
        lang: str = "en",
    ) -> tuple[bool, bool, list[str], list[str]]:
        """
        Detect negation/normality using ConText rules with direction awareness.

        Uses ConText rules for negation detection and legacy normality_cues.json
        for normality detection (since normality is not part of ConText standard).

        Args:
            text_lower: Lowercased text to analyze
            rules: List of ConTextRule objects
            original_text: Original text (for scope attribution)
            lang: Language code for normality detection fallback

        Returns:
            Tuple of (is_negated, is_normal, negated_scopes, normal_scopes)
        """
        negated_scopes: list[str] = []
        normal_scopes: list[str] = []
        is_negated = False
        is_normal = False

        # First pass: Check for PSEUDO rules (false positives)
        # Track text spans matched by PSEUDO rules to prevent shorter overlapping matches
        pseudo_matches = set()
        pseudo_spans: list[tuple[int, int]] = []
        for rule in rules:
            if rule.direction == Direction.PSEUDO:
                cue_lower = rule.literal.lower()
                cue_index = text_lower.find(cue_lower)
                if cue_index >= 0 and _is_cue_match(text_lower, cue_lower, cue_index):
                    cue_end = cue_index + len(cue_lower)
                    pseudo_matches.add(cue_lower)
                    pseudo_spans.append((cue_index, cue_end))
                    logger.debug(
                        "PSEUDO rule matched at %s-%s: '%s' - skipping",
                        cue_index,
                        cue_end,
                        _sanitize(rule.literal),
                    )

        # Second pass: Find all TERMINATE trigger positions (scope boundaries)
        terminate_positions: list[tuple[int, int]] = []
        for rule in rules:
            if rule.direction == Direction.TERMINATE:
                cue_lower = rule.literal.lower()
                cue_index = text_lower.find(cue_lower)
                if cue_index >= 0 and _is_cue_match(text_lower, cue_lower, cue_index):
                    cue_end = cue_index + len(cue_lower)
                    terminate_positions.append((cue_index, cue_end))
                    logger.debug(
                        "TERMINATE rule matched at %s-%s: '%s'",
                        cue_index,
                        cue_end,
                        _sanitize(rule.literal),
                    )

        # Third pass: Process NEGATED_EXISTENCE and other categories
        for rule in rules:
            cue_lower = rule.literal.lower()

            # Skip if this cue was matched by a PSEUDO rule
            if cue_lower in pseudo_matches:
                continue

            # Skip PSEUDO and TERMINATE rules in this pass
            if rule.direction in (Direction.PSEUDO, Direction.TERMINATE):
                continue

            cue_index = text_lower.find(cue_lower)
            if cue_index < 0 or not _is_cue_match(text_lower, cue_lower, cue_index):
                continue

            # Check if this cue overlaps with any PSEUDO span
            # Note: This is O(P × N) where P = pseudo spans, N = negation rules.
            # For typical clinical text, P is 0-3 and N is 1-5 (<15 comparisons/chunk).
            # Early exit on first overlap keeps average case fast. An interval tree
            # would be overkill for this domain's typical data characteristics.
            cue_end = cue_index + len(cue_lower)
            overlaps_pseudo = False
            for pseudo_start, pseudo_end in pseudo_spans:
                # Check for any overlap: cue starts before pseudo ends AND cue ends after pseudo starts
                if cue_index < pseudo_end and cue_end > pseudo_start:
                    overlaps_pseudo = True
                    logger.debug(
                        "Skipping '%s' at %s-%s - overlaps with PSEUDO span %s-%s",
                        _sanitize(rule.literal),
                        cue_index,
                        cue_end,
                        pseudo_start,
                        pseudo_end,
                    )
                    break

            if overlaps_pseudo:
                continue

            # Found a match - extract scope based on direction with TERMINATE boundaries
            scope_text = self._extract_scope(
                text_lower, cue_index, cue_lower, rule.direction, terminate_positions
            )

            if scope_text:
                # Categorize by TriggerCategory
                if rule.category == TriggerCategory.NEGATED_EXISTENCE:
                    negated_scopes.append(f"{rule.literal}: {scope_text}")
                    is_negated = True
                # Note: Normality detection not part of ConText standard
                # Add support for other categories (POSSIBLE_EXISTENCE, etc.) in future

        # ConText doesn't define "normality" - use legacy normality detection
        is_normal, normal_scopes = self._detect_normality_legacy(text_lower, lang)

        return is_negated, is_normal, negated_scopes, normal_scopes

    def _extract_scope(
        self,
        text_lower: str,
        cue_index: int,
        cue_lower: str,
        direction: Direction,
        terminate_positions: list[tuple[int, int]] | None = None,
    ) -> str:
        """
        Extract scope text based on ConText rule direction with TERMINATE boundaries.

        TERMINATE rules act as scope boundaries that limit the extent of negation
        or other assertion modifiers. For example, in "no fever but has cough",
        the "but" TERMINATE rule prevents negation from affecting "has cough".

        Args:
            text_lower: Lowercased text
            cue_index: Position where cue was found
            cue_lower: Lowercased cue text
            direction: Direction to extract scope (FORWARD, BACKWARD, BIDIRECTIONAL)
            terminate_positions: List of (start, end) positions of TERMINATE triggers

        Returns:
            Extracted scope text limited by TERMINATE boundaries
        """
        cue_end = cue_index + len(cue_lower)
        terminate_positions = terminate_positions or []

        if direction == Direction.FORWARD:
            # Extract KEYWORD_WINDOW words AFTER the cue, stop at TERMINATE
            scope_end = len(text_lower)

            # Find first TERMINATE trigger after cue
            for term_start, term_end in terminate_positions:
                if term_start > cue_end and term_start < scope_end:
                    scope_end = term_start

            # Extract text up to scope boundary
            scope_text = text_lower[cue_end:scope_end]
            words_after = scope_text.split()
            return " ".join(words_after[:KEYWORD_WINDOW])

        elif direction == Direction.BACKWARD:
            # Extract KEYWORD_WINDOW words BEFORE the cue, stop at TERMINATE
            scope_start = 0

            # Find last TERMINATE trigger before cue
            for term_start, term_end in terminate_positions:
                if term_end < cue_index and term_end > scope_start:
                    scope_start = term_end

            # Extract text from scope boundary
            scope_text = text_lower[scope_start:cue_index]
            words_before = scope_text.split()
            # Take last KEYWORD_WINDOW words (closest to cue)
            relevant_words = words_before[-KEYWORD_WINDOW:] if words_before else []
            return " ".join(relevant_words)

        elif direction == Direction.BIDIRECTIONAL:
            # Extract words both before and after with TERMINATE boundaries
            # Find boundaries for backward scope
            scope_start = 0
            for term_start, term_end in terminate_positions:
                if term_end < cue_index and term_end > scope_start:
                    scope_start = term_end

            # Find boundaries for forward scope
            scope_end = len(text_lower)
            for term_start, term_end in terminate_positions:
                if term_start > cue_end and term_start < scope_end:
                    scope_end = term_start

            # Extract both scopes
            before_text = text_lower[scope_start:cue_index]
            after_text = text_lower[cue_end:scope_end]

            words_before = before_text.split()
            words_after = after_text.split()

            before_scope = (
                " ".join(words_before[-KEYWORD_WINDOW:]) if words_before else ""
            )
            after_scope = " ".join(words_after[:KEYWORD_WINDOW])

            # Combine with cue in the middle
            if before_scope and after_scope:
                return f"{before_scope} [{cue_lower}] {after_scope}"
            elif before_scope:
                return before_scope
            else:
                return after_scope

        return ""

    def _detect_normality_legacy(
        self, text_lower: str, lang: str
    ) -> tuple[bool, list[str]]:
        """
        Legacy normality detection using normality_cues.json.

        This is kept separate since "normality" is not part of the ConText standard
        but is a Phentrieve-specific feature.

        Args:
            text_lower: Lowercased text to analyze
            lang: Language code

        Returns:
            Tuple of (is_normal, normal_scopes)
        """
        # Load user configuration
        user_config_main = load_user_config()
        language_resources_section = user_config_main.get("language_resources", {})

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

            if cue_index >= 0 and _is_cue_match(text_lower, cue_lower, cue_index):
                # Get words around the cue (simple window approach)
                start_idx = max(0, cue_index - 30)
                end_idx = min(len(text_lower), cue_index + len(cue_lower) + 30)
                context = text_lower[start_idx:end_idx]

                normal_scopes.append(f"{cue.strip()}: {context}")
                is_normal = True

        return is_normal, normal_scopes


class DependencyAssertionDetector(AssertionDetector):
    """
    Assertion detector using dependency parsing.

    This detector uses spaCy's dependency parsing to analyze
    relationships between negation/normality cues and concepts.
    """

    def __init__(self, language: str = "en", **kwargs):
        """
        Initialize the DependencyAssertionDetector.

        Args:
            language: ISO language code ('en', 'de', etc.)
            **kwargs: Additional configuration parameters
        """
        super().__init__(language, **kwargs)
        # Check if spaCy model is available
        nlp = get_spacy_model(language)
        if nlp is None:
            raise RuntimeError(
                f"spaCy model for language '{language}' is not available. "
                "Please install the required spaCy model (e.g., 'python -m spacy download en_core_web_sm' for English)."
            )
        # Cache negation and normality cues per language
        self._negation_cues_cache: dict[str, list[str]] = {}
        self._normality_cues_cache: dict[str, list[str]] = {}
        # Pre-load cues for the default language
        self._negation_cues_cache[language] = self._load_negation_cues(language)
        self._normality_cues_cache[language] = self._load_normality_cues(language)

    def _load_negation_cues(self, lang: str) -> list[str]:
        """
        Load negation cues from ConText rules for the given language.

        Args:
            lang: Language code (e.g., 'de', 'en', 'es', 'fr', 'nl')

        Returns:
            List of negation cue strings (lowercase)
        """
        import importlib.resources
        import json

        resources_package = "phentrieve.text_processing.default_lang_resources"
        context_filename = f"context_rules_{lang}.json"

        try:
            resource_path = importlib.resources.files(resources_package).joinpath(
                context_filename
            )
            with resource_path.open("r", encoding="utf-8") as f:
                context_data = json.load(f)

            # Extract negation cue literals from NEGATED_EXISTENCE rules
            negation_cues = [
                rule["literal"].strip().lower()
                for rule in context_data.get("context_rules", [])
                if rule.get("category") == "NEGATED_EXISTENCE"
                and rule.get("direction") != "PSEUDO"
            ]
            logger.info(
                "Loaded %s negation cues from ConText rules for '%s'",
                len(negation_cues),
                _sanitize(lang),
            )
            return negation_cues
        except (FileNotFoundError, AttributeError, KeyError):
            # Fall back to English if language not found
            if lang != "en":
                try:
                    resource_path = importlib.resources.files(
                        resources_package
                    ).joinpath("context_rules_en.json")
                    with resource_path.open("r", encoding="utf-8") as f:
                        context_data = json.load(f)

                    negation_cues = [
                        rule["literal"].strip().lower()
                        for rule in context_data.get("context_rules", [])
                        if rule.get("category") == "NEGATED_EXISTENCE"
                        and rule.get("direction") != "PSEUDO"
                    ]
                    logger.info(
                        "Loaded %s English negation cues as fallback for '%s'",
                        len(negation_cues),
                        _sanitize(lang),
                    )
                    return negation_cues
                except (FileNotFoundError, AttributeError, KeyError):
                    pass

            logger.warning(
                "No ConText rules found for lang '%s', returning empty negation cues",
                _sanitize(lang),
            )
            return []

    def _load_normality_cues(self, lang: str) -> list[str]:
        """
        Load normality cues for the given language.

        Args:
            lang: Language code (e.g., 'de', 'en', 'es', 'fr', 'nl')

        Returns:
            List of normality cue strings (lowercase)
        """
        user_config_main = load_user_config()
        language_resources_section = user_config_main.get("language_resources", {})

        normality_cues_resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
            language_resources_config_section=language_resources_section,
        )
        return normality_cues_resources.get(
            lang, normality_cues_resources.get("en", [])
        )

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

        Uses ConText rules for negation detection combined with spaCy dependency parsing.

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

        # Get negation and normality cues for the requested language (with caching)
        if lang not in self._negation_cues_cache:
            # Load and cache cues for this language
            self._negation_cues_cache[lang] = self._load_negation_cues(lang)
            self._normality_cues_cache[lang] = self._load_normality_cues(lang)

        lang_negation_cues = self._negation_cues_cache[lang]
        lang_normality_cues = self._normality_cues_cache[lang]

        if not lang_negation_cues:
            logger.warning(
                "No negation cues available for lang '%s', dependency detection disabled",
                _sanitize(lang),
            )
            return False, False, [], []

        # Check for negation
        negated_concepts = []
        is_negated = False

        # Handle German negation directly with a text check first (more reliable for short phrases)
        chunk_lower = chunk.lower()
        # Use most common negation cues (first 5) for quick check - data-driven from ConText rules
        quick_check_cues = lang_negation_cues[:5]
        if lang == "de" and any(
            neg_term in chunk_lower for neg_term in quick_check_cues
        ):
            is_negated = True
            negated_concepts.append(
                f"German negation term found in: {_sanitize(chunk)}"
            )

        # Also check with spaCy's dependency parsing
        for token in doc:
            token_text = token.text.lower()

            # Check if token or its lemma is a negation cue
            is_negation_term = False

            for neg_cue in lang_negation_cues:
                # More flexible matching for German
                if (
                    lang == "de"
                    and neg_cue.startswith("kein")
                    and token_text.startswith("kein")
                ):
                    is_negation_term = True
                    break
                # Regular exact matching
                elif token_text == neg_cue or token.lemma_.lower() == neg_cue:
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

        # Resources already loaded at top of function (performance fix)
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
        self.dependency_detector = None
        if enable_dependency:
            try:
                self.dependency_detector = DependencyAssertionDetector(
                    language=language
                )
            except RuntimeError:
                logger.warning(
                    "Dependency-based assertion detection disabled for language '%s' due to missing spaCy model.",
                    _sanitize(language),
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
            "dependency_parser": self.dependency_detector is not None,
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
