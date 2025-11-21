"""
Unit tests for HPO term details enrichment.

Tests cover:
- Pure function behavior (no input mutation)
- Database integration
- Error handling (missing database, missing terms)
- Connection reuse pattern
- Edge cases (empty results, all missing terms)
"""

import json
import sqlite3

import pytest

from phentrieve.data_processing.hpo_database import HPODatabase
from phentrieve.retrieval.details_enrichment import (
    enrich_results_with_details,
    get_shared_database,
)

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture
def temp_db_with_terms(tmp_path):
    """Create temporary database with sample terms."""
    db_path = tmp_path / "hpo_data.db"
    db = HPODatabase(db_path)
    db.initialize_schema()

    # Insert sample terms
    terms = [
        {
            "id": "HP:0001250",
            "label": "Seizure",
            "definition": "A seizure is an intermittent abnormality of nervous system physiology",
            "synonyms": json.dumps(["Seizures", "Epileptic seizure"]),
            "comments": json.dumps(["Common symptom"]),
        },
        {
            "id": "HP:0002119",
            "label": "Ventriculomegaly",
            "definition": "An increase in size of the ventricular system",
            "synonyms": json.dumps(["Ventricular dilatation", "Enlarged ventricles"]),
            "comments": json.dumps([]),
        },
        {
            "id": "HP:0000001",
            "label": "All",
            "definition": "",  # Empty definition
            "synonyms": json.dumps([]),  # Empty synonyms
            "comments": json.dumps([]),
        },
    ]

    db.bulk_insert_terms(terms)
    db.close()

    return db_path


class TestEnrichmentPureFunction:
    """Test that enrichment behaves as a pure function (no side effects)."""

    def test_enrich_returns_new_objects(self, temp_db_with_terms, tmp_path):
        """Ensure enrichment doesn't mutate input (pure function)."""
        original = [{"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}]
        original_copy = [dict(item) for item in original]  # Deep copy for comparison

        enriched = enrich_results_with_details(
            original, data_dir_override=str(tmp_path)
        )

        # Original unchanged
        assert original == original_copy
        assert "definition" not in original[0]
        assert "synonyms" not in original[0]

        # New object created
        assert enriched is not original
        assert "definition" in enriched[0]
        assert "synonyms" in enriched[0]

    def test_enrich_preserves_original_fields(self, temp_db_with_terms, tmp_path):
        """Test that all original fields are preserved in enriched results."""
        original = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": 0.95,
                "cross_encoder_score": 0.87,
                "original_rank": 1,
                "custom_field": "test",
            }
        ]

        enriched = enrich_results_with_details(
            original, data_dir_override=str(tmp_path)
        )

        # All original fields preserved
        assert enriched[0]["hpo_id"] == "HP:0001250"
        assert enriched[0]["label"] == "Seizure"
        assert enriched[0]["similarity"] == 0.95
        assert enriched[0]["cross_encoder_score"] == 0.87
        assert enriched[0]["original_rank"] == 1
        assert enriched[0]["custom_field"] == "test"

        # New fields added
        assert "definition" in enriched[0]
        assert "synonyms" in enriched[0]


class TestEnrichmentSuccess:
    """Test successful enrichment scenarios."""

    def test_enrich_empty_results(self):
        """Test empty input returns empty output."""
        result = enrich_results_with_details([])
        assert result == []

    def test_enrich_with_valid_terms(self, temp_db_with_terms, tmp_path):
        """Test successful enrichment with valid HPO terms."""
        results = [{"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert len(enriched) == 1
        assert (
            enriched[0]["definition"]
            == "A seizure is an intermittent abnormality of nervous system physiology"
        )
        assert enriched[0]["synonyms"] == ["Seizures", "Epileptic seizure"]
        assert enriched[0]["similarity"] == 0.95  # Original preserved

    def test_enrich_multiple_terms(self, temp_db_with_terms, tmp_path):
        """Test enrichment with multiple terms."""
        results = [
            {"hpo_id": "HP:0001250", "label": "Seizure"},
            {"hpo_id": "HP:0002119", "label": "Ventriculomegaly"},
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert len(enriched) == 2

        # First term
        assert enriched[0]["hpo_id"] == "HP:0001250"
        assert "seizure" in enriched[0]["definition"].lower()
        assert len(enriched[0]["synonyms"]) == 2

        # Second term
        assert enriched[1]["hpo_id"] == "HP:0002119"
        assert "ventricular" in enriched[1]["definition"].lower()
        assert len(enriched[1]["synonyms"]) == 2

    def test_enrich_converts_empty_to_none(self, temp_db_with_terms, tmp_path):
        """Test that empty strings/lists are converted to None for API clarity."""
        results = [
            {"hpo_id": "HP:0000001", "label": "All"}  # Has empty definition/synonyms
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert enriched[0]["definition"] is None  # Empty string → None
        assert enriched[0]["synonyms"] is None  # Empty list → None


class TestEnrichmentErrorHandling:
    """Test error handling and edge cases."""

    def test_enrich_without_database(self, tmp_path):
        """Test graceful handling when database doesn't exist."""
        results = [{"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95}]

        # Should not raise, return results with None details
        enriched = enrich_results_with_details(
            results,
            data_dir_override=str(tmp_path),  # No database here
        )

        assert len(enriched) == 1
        assert enriched[0]["definition"] is None
        assert enriched[0]["synonyms"] is None
        assert enriched[0]["similarity"] == 0.95  # Original preserved

    def test_enrich_with_missing_term(self, temp_db_with_terms, tmp_path):
        """Test enrichment when term not in database."""
        results = [
            {"hpo_id": "HP:9999999", "label": "Unknown"}  # Not in database
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert enriched[0]["definition"] is None
        assert enriched[0]["synonyms"] is None

    def test_enrich_mix_of_found_and_missing(self, temp_db_with_terms, tmp_path):
        """Test enrichment with mix of found and missing terms."""
        results = [
            {"hpo_id": "HP:0001250", "label": "Seizure"},  # Exists
            {"hpo_id": "HP:9999999", "label": "Unknown"},  # Missing
            {"hpo_id": "HP:0002119", "label": "Ventriculomegaly"},  # Exists
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert len(enriched) == 3

        # First (found)
        assert enriched[0]["definition"] is not None
        assert enriched[0]["synonyms"] is not None

        # Second (missing)
        assert enriched[1]["definition"] is None
        assert enriched[1]["synonyms"] is None

        # Third (found)
        assert enriched[2]["definition"] is not None
        assert enriched[2]["synonyms"] is not None

    def test_enrich_propagates_database_errors(self, tmp_path, monkeypatch):
        """Test that database errors propagate (fail fast)."""

        def mock_get_connection_error(*args):
            raise sqlite3.OperationalError("Database locked")

        # Ensure database exists first
        db_path = tmp_path / "hpo_data.db"
        db = HPODatabase(db_path)
        db.initialize_schema()
        db.close()

        # Clear cache to ensure fresh connection attempt
        get_shared_database.cache_clear()

        # Now mock the get_connection to raise error
        monkeypatch.setattr(HPODatabase, "get_connection", mock_get_connection_error)

        results = [{"hpo_id": "HP:0001250", "label": "Seizure"}]

        # Should raise (not swallow error)
        with pytest.raises(sqlite3.OperationalError, match="Database locked"):
            enrich_results_with_details(results, data_dir_override=str(tmp_path))


class TestConnectionReuse:
    """Test database connection reuse pattern."""

    def test_connection_reuse(self, temp_db_with_terms, tmp_path):
        """Verify database connection is reused across calls."""
        # Clear cache before test
        get_shared_database.cache_clear()

        db_path_str = str(temp_db_with_terms)

        # First call creates connection
        db1 = get_shared_database(db_path_str)

        # Second call reuses connection
        db2 = get_shared_database(db_path_str)

        assert db1 is db2  # Same object (cached)

        # Clean up for other tests
        get_shared_database.cache_clear()

    def test_enrichment_uses_shared_connection(self, temp_db_with_terms, tmp_path):
        """Test that enrichment function uses shared connection."""
        # Clear cache
        get_shared_database.cache_clear()

        results = [{"hpo_id": "HP:0001250", "label": "Seizure"}]

        # First enrichment
        enriched1 = enrich_results_with_details(
            results, data_dir_override=str(tmp_path)
        )

        # Second enrichment (should reuse connection)
        enriched2 = enrich_results_with_details(
            results, data_dir_override=str(tmp_path)
        )

        assert enriched1[0]["definition"] == enriched2[0]["definition"]

        # Clean up
        get_shared_database.cache_clear()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_enrich_with_none_values(self, temp_db_with_terms, tmp_path):
        """Test handling of None values in input."""
        # This shouldn't happen in practice, but test defensive programming
        results = [
            {
                "hpo_id": "HP:0001250",
                "label": "Seizure",
                "similarity": None,  # None value
            }
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert enriched[0]["similarity"] is None  # Preserved
        assert enriched[0]["definition"] is not None  # Still enriched

    def test_enrich_large_batch(self, temp_db_with_terms, tmp_path):
        """Test enrichment with larger batch of results."""
        # Create results with same term repeated (simulates real use case)
        results = [
            {"hpo_id": "HP:0001250", "label": "Seizure", "similarity": 0.95 - i * 0.01}
            for i in range(20)
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        assert len(enriched) == 20

        # All should have same definition (same term)
        definitions = {r["definition"] for r in enriched}
        assert len(definitions) == 1  # All the same

        # But different similarities (preserved)
        similarities = [r["similarity"] for r in enriched]
        assert len(set(similarities)) == 20  # All different

    def test_enrich_preserves_result_order(self, temp_db_with_terms, tmp_path):
        """Test that result order is preserved."""
        results = [
            {"hpo_id": "HP:0002119", "label": "Ventriculomegaly"},
            {"hpo_id": "HP:0001250", "label": "Seizure"},
            {"hpo_id": "HP:0000001", "label": "All"},
        ]

        enriched = enrich_results_with_details(results, data_dir_override=str(tmp_path))

        # Order preserved
        assert enriched[0]["hpo_id"] == "HP:0002119"
        assert enriched[1]["hpo_id"] == "HP:0001250"
        assert enriched[2]["hpo_id"] == "HP:0000001"


# Cleanup fixture to ensure cache is cleared after all tests
@pytest.fixture(autouse=True, scope="module")
def cleanup_cache():
    """Clear connection cache after all tests in this module."""
    yield
    get_shared_database.cache_clear()
