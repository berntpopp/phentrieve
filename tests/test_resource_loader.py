"""
Tests for the language resource loader functionality.

This module tests the resource loading and caching mechanisms implemented
in the resource_loader module.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path

from phentrieve.text_processing.resource_loader import (
    load_language_resource,
    _RESOURCE_CACHE,
)


class TestResourceLoader(unittest.TestCase):
    """Test cases for the language resource loader."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear the cache before each test
        _RESOURCE_CACHE.clear()

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after tests."""
        # Clear the cache after each test
        _RESOURCE_CACHE.clear()

        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_default_resources(self):
        """Test loading default language resources."""
        # Load negation cues
        resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Check that we got resources for multiple languages
        self.assertIn("en", resources)
        self.assertIn("de", resources)

        # Check that the resources have the expected structure
        self.assertIsInstance(resources["en"], list)
        self.assertGreater(len(resources["en"]), 0)

        # All items should be lowercase strings
        for item in resources["en"]:
            self.assertEqual(item.lower(), item)

    def test_resource_caching(self):
        """Test that resources are cached and only loaded once."""
        # First load - should read from file
        resource1 = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Check that the cache has an entry
        self.assertEqual(len(_RESOURCE_CACHE), 1)

        # Second load - should use cache
        resource2 = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
        )

        # Verify resources are identical
        self.assertIs(
            resource1, resource2
        )  # Should be the same object reference if cached

        # Cache should still have just one entry
        self.assertEqual(len(_RESOURCE_CACHE), 1)

    def test_custom_resource_override(self):
        """Test that custom resources override default ones."""
        # Create a custom resource file
        custom_file = self.temp_path / "custom_negation_cues.json"
        custom_data = {
            "en": ["custom_no", "custom_not", "custom_without"],
            "fr": ["custom_ne_pas", "custom_sans"],
        }

        with open(custom_file, "w", encoding="utf-8") as f:
            json.dump(custom_data, f)

        # Create a mock config section
        config_section = {"negation_cues_file": str(custom_file)}

        # Load resources with custom override
        resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
            language_resources_config_section=config_section,
        )

        # Check that custom resources were used
        self.assertEqual(resources["en"], ["custom_no", "custom_not", "custom_without"])
        self.assertEqual(resources["fr"], ["custom_ne_pas", "custom_sans"])

        # Check that languages not in custom file still use defaults
        self.assertIn("de", resources)
        self.assertGreater(len(resources["de"]), 0)

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
        self.assertEqual(len(_RESOURCE_CACHE), 2)

        # Resources should be different
        self.assertIsNot(negation_cues, normality_cues)

    def test_nonexistent_custom_file(self):
        """Test behavior with nonexistent custom file."""
        # Create a config section with nonexistent file
        config_section = {"negation_cues_file": "/path/to/nonexistent/file.json"}

        # Load resources with nonexistent custom file
        resources = load_language_resource(
            default_resource_filename="negation_cues.json",
            config_key_for_custom_file="negation_cues_file",
            language_resources_config_section=config_section,
        )

        # Should fall back to default resources
        self.assertIn("en", resources)
        self.assertGreater(len(resources["en"]), 0)


if __name__ == "__main__":
    unittest.main()
