"""Unit tests for language resource loader (pytest style)."""

import json
import pytest
from pathlib import Path

from phentrieve.text_processing.resource_loader import (
    _RESOURCE_CACHE,
    load_language_resource,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_resource_cache():
    """Clear resource cache before and after each test."""
    _RESOURCE_CACHE.clear()
    yield
    _RESOURCE_CACHE.clear()


class TestResourceLoader:
    """Test cases for language resource loader."""

    def test_load_default_resources(self):
        """Test loading default language resources."""
        resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Check resources for multiple languages
        assert "en" in resources
        assert "de" in resources

        # Check expected structure
        assert isinstance(resources["en"], list)
        assert len(resources["en"]) > 0

        # All items should be lowercase strings
        for item in resources["en"]:
            assert item.lower() == item

    def test_resource_caching(self):
        """Test that resources are cached and only loaded once."""
        # First load - should read from file
        resource1 = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Check cache has an entry
        assert len(_RESOURCE_CACHE) == 1

        # Second load - should use cache
        resource2 = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Verify resources are identical (same object reference if cached)
        assert resource1 is resource2

        # Cache should still have just one entry
        assert len(_RESOURCE_CACHE) == 1

    def test_custom_resource_override(self, tmp_path):
        """Test that custom resources override default ones."""
        # Create custom resource file
        custom_file = tmp_path / "custom_negation_cues.json"
        custom_data = {
            "en": ["custom_no", "custom_not", "custom_without"],
            "fr": ["custom_ne_pas", "custom_sans"],
        }

        with open(custom_file, "w", encoding="utf-8") as f:
            json.dump(custom_data, f)

        # Create mock config section
        config_section = {"negation_cues_file": str(custom_file)}

        # Load resources with custom override
        resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
            language_resources_config_section=config_section,
        )

        # Check custom resources were used
        assert resources["en"] == ["custom_no", "custom_not", "custom_without"]
        assert resources["fr"] == ["custom_ne_pas", "custom_sans"]

        # Languages not in custom file still use defaults
        assert "de" in resources
        assert len(resources["de"]) > 0

    def test_multiple_different_resources(self):
        """Test loading multiple different resources."""
        # Load negation cues
        negation_cues = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Load normality cues
        normality_cues = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
        )

        # Check cache has two entries
        assert len(_RESOURCE_CACHE) == 2

        # Resources should be different
        assert negation_cues is not normality_cues

    def test_nonexistent_custom_file(self):
        """Test behavior with nonexistent custom file."""
        # Config with nonexistent file
        config_section = {"negation_cues_file": "/path/to/nonexistent/file.json"}

        # Load resources
        resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
            language_resources_config_section=config_section,
        )

        # Should fall back to default resources
        assert "en" in resources
        assert len(resources["en"]) > 0
