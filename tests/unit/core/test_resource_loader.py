"""Unit tests for language resource loader (pytest style)."""

import json

import pytest

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
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
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
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
        )

        # Check cache has an entry
        assert len(_RESOURCE_CACHE) == 1

        # Second load - should use cache
        resource2 = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
        )

        # Verify resources are identical (same object reference if cached)
        assert resource1 is resource2

        # Cache should still have just one entry
        assert len(_RESOURCE_CACHE) == 1

    def test_custom_resource_override(self, tmp_path):
        """Test that custom resources override default ones."""
        # Create custom resource file
        custom_file = tmp_path / "custom_normality_cues.json"
        custom_data = {
            "en": ["custom_normal", "custom_healthy", "custom_stable"],
            "fr": ["custom_normal", "custom_sain"],
        }

        with open(custom_file, "w", encoding="utf-8") as f:
            json.dump(custom_data, f)

        # Create mock config section
        config_section = {"normality_cues_file": str(custom_file)}

        # Load resources with custom override
        resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
            language_resources_config_section=config_section,
        )

        # Check custom resources were used
        assert resources["en"] == ["custom_normal", "custom_healthy", "custom_stable"]
        assert resources["fr"] == ["custom_normal", "custom_sain"]

        # Languages not in custom file still use defaults
        assert "de" in resources
        assert len(resources["de"]) > 0

    def test_multiple_different_resources(self):
        """Test loading multiple different resources."""
        # Load normality cues
        normality_cues = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
        )

        # Load coordinating conjunctions
        conjunctions = load_language_resource(
            default_resource_filename="coordinating_conjunctions.json",
            config_key_for_custom_file="coordinating_conjunctions_file",
        )

        # Check cache has two entries
        assert len(_RESOURCE_CACHE) == 2

        # Resources should be different
        assert normality_cues is not conjunctions

    def test_nonexistent_custom_file(self):
        """Test behavior with nonexistent custom file."""
        # Config with nonexistent file
        config_section = {"normality_cues_file": "/path/to/nonexistent/file.json"}

        # Load resources
        resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
            language_resources_config_section=config_section,
        )

        # Should fall back to default resources
        assert "en" in resources
        assert len(resources["en"]) > 0

    def test_invalid_default_resource_file(self, mocker):
        """Test error handling when default resource file cannot be loaded."""
        # Mock importlib.resources.files to raise an exception
        mock_files = mocker.patch("importlib.resources.files")
        mock_files.side_effect = Exception("Failed to load default resource")

        # Load resources - should handle error and return empty dict
        resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
        )

        # Should return empty dict when default loading fails
        assert resources == {}

    def test_corrupt_custom_resource_file(self, tmp_path):
        """Test error handling when custom resource file has invalid JSON."""
        # Create corrupt JSON file
        custom_file = tmp_path / "corrupt_normality_cues.json"
        with open(custom_file, "w", encoding="utf-8") as f:
            f.write("{invalid json content!@#$")

        # Create config section
        config_section = {"normality_cues_file": str(custom_file)}

        # Load resources - should handle error and use defaults
        resources = load_language_resource(
            default_resource_filename="normality_cues.json",
            config_key_for_custom_file="normality_cues_file",
            language_resources_config_section=config_section,
        )

        # Should still have default resources even though custom file is corrupt
        assert "en" in resources
        assert "de" in resources
        assert len(resources["en"]) > 0
