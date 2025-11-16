"""
Language resource loader for Phentrieve.

This module provides utilities for loading language-specific resources from
JSON files. It supports both bundled default resources and user-provided custom resources.
"""

import copy
import importlib.resources
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Package path to the default language resources
DEFAULT_RESOURCES_PACKAGE_PATH = "phentrieve.text_processing.default_lang_resources"

# Cache for loaded resources to avoid repeatedly loading the same resources
# Key: tuple(resource_filename, custom_file_path)
# Value: loaded resource dictionary
_RESOURCE_CACHE: dict[str, dict[str, list[str]]] = {}


def load_language_resource(
    default_resource_filename: str,
    config_key_for_custom_file: str,
    language_resources_config_section: Optional[dict[str, Any]] = None,
) -> dict[str, list[str]]:
    """
    Load a language resource from JSON files, with user overrides if provided.

    This function loads language resources from the following sources in order:
    1. Default bundled JSON file (always loaded if available)
    2. User-provided custom JSON file (if specified in phentrieve.yaml config)

    The user's custom resources take precedence over default resources for any
    language they specifically define. For languages not defined in the user's
    custom file, the default values are preserved.

    Results are cached to avoid repeatedly loading the same resources.

    Args:
        default_resource_filename: Filename of the default resource JSON file
            (e.g., "coordinating_conjunctions.json")
        config_key_for_custom_file: Key in phentrieve.yaml's language_resources section
            that points to the user's custom file (e.g., "coordinating_conjunctions_file")
        language_resources_config_section: The language_resources section from phentrieve.yaml
            If None, only default resources will be loaded.

    Returns:
        Dictionary mapping language codes to lists of resource strings.
        Example: {"en": ["word1", "word2"], "de": ["wort1", "wort2"]}
    """
    # Generate a cache key based on inputs
    custom_file_path_str = None
    if language_resources_config_section is not None:
        custom_file_path_str = language_resources_config_section.get(
            config_key_for_custom_file
        )

    cache_key = f"{default_resource_filename}:{custom_file_path_str}"

    # Check if the resource is already in the cache
    if cache_key in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[cache_key]

    # If not in cache, load the resource
    # Initialize with empty dictionary (fallback if everything fails)
    effective_resources: dict[str, list[str]] = {}

    # 1. Load default resources from bundled package
    try:
        # Get a path to the bundled default JSON file
        default_resource_path = importlib.resources.files(
            DEFAULT_RESOURCES_PACKAGE_PATH
        ).joinpath(default_resource_filename)

        # Read the default resource file
        with default_resource_path.open("r", encoding="utf-8") as f:
            default_resources = json.load(f)

        # Create a deep copy to avoid modifying the original
        effective_resources = copy.deepcopy(default_resources)
        logger.info(
            f"Loaded default language resource from {default_resource_filename} "
            f"with {len(default_resources)} languages"
        )
    except Exception as e:
        logger.error(
            f"Error loading default resource from {default_resource_filename}: {str(e)}"
        )
        logger.warning(
            "Will proceed with empty default language resource dictionary. "
            "This may cause unexpected behavior!"
        )

    # 2. Check for and load user-provided custom resources if available
    if custom_file_path_str:
        custom_file_path = Path(custom_file_path_str)

        if custom_file_path.exists() and custom_file_path.is_file():
            try:
                with open(custom_file_path, encoding="utf-8") as f:
                    custom_resources = json.load(f)

                logger.info(
                    f"Loaded custom language resource from {custom_file_path} "
                    f"with {len(custom_resources)} languages"
                )

                # Merge custom resources with defaults (custom takes precedence)
                for lang_key, word_list in custom_resources.items():
                    # Ensure lang_key is lowercase
                    lang_key = lang_key.lower()

                    # Convert all items in the list to lowercase
                    word_list_lower = [str(item).lower() for item in word_list]

                    # Replace the default list with the custom list for this language
                    effective_resources[lang_key] = word_list_lower

                    logger.debug(
                        f"Custom resource for language '{lang_key}' overrides default "
                        f"with {len(word_list_lower)} items"
                    )

            except Exception as e:
                logger.error(
                    f"Error loading custom resource from {custom_file_path}: {str(e)}"
                )
                logger.warning(
                    f"Will continue using default resources for {default_resource_filename}"
                )
        else:
            logger.warning(
                "Custom language resource file specified in config "
                f"({custom_file_path}) does not exist or is not a file"
            )

    # 3. Ensure all strings in all language lists are lowercase
    for lang, words in effective_resources.items():
        effective_resources[lang] = [str(word).lower() for word in words]

    # Store in cache before returning
    _RESOURCE_CACHE[cache_key] = effective_resources

    return effective_resources
