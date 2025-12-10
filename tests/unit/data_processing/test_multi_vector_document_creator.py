"""Unit tests for multi-vector document creator module.

Tests the creation of per-component documents for multi-vector
HPO term indexing.
"""

import pytest

from phentrieve.data_processing.multi_vector_document_creator import (
    create_multi_vector_documents,
    get_component_stats,
)


class TestCreateMultiVectorDocuments:
    """Test multi-vector document creation."""

    @pytest.fixture
    def sample_hpo_terms(self):
        """Sample HPO terms for testing."""
        return [
            {
                "id": "HP:0001250",
                "label": "Seizure",
                "synonyms": ["Fits", "Convulsions"],
                "definition": "A seizure is a transient occurrence.",
            },
            {
                "id": "HP:0001252",
                "label": "Hypotonia",
                "synonyms": ["Decreased muscle tone"],
                "definition": None,  # No definition
            },
            {
                "id": "HP:0001253",
                "label": "Spasticity",
                "synonyms": [],  # No synonyms
                "definition": "Velocity-dependent increase in muscle tone.",
            },
        ]

    def test_creates_documents_for_all_components(self, sample_hpo_terms):
        """Test creates separate documents for label, synonyms, and definition."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # Expected documents:
        # HP:0001250: label + 2 synonyms + definition = 4
        # HP:0001252: label + 1 synonym + no definition = 2
        # HP:0001253: label + no synonyms + definition = 2
        # Total: 8

        assert len(documents) == 8
        assert len(metadatas) == 8
        assert len(ids) == 8

    def test_document_ids_follow_format(self, sample_hpo_terms):
        """Test document IDs follow expected format."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # Check ID formats
        assert "HP:0001250__label__0" in ids
        assert "HP:0001250__synonym__0" in ids
        assert "HP:0001250__synonym__1" in ids
        assert "HP:0001250__definition__0" in ids

    def test_metadata_contains_hpo_id_and_component(self, sample_hpo_terms):
        """Test metadata includes hpo_id and component type."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # Find label metadata for first term
        label_idx = ids.index("HP:0001250__label__0")
        label_meta = metadatas[label_idx]

        assert label_meta["hpo_id"] == "HP:0001250"
        assert label_meta["component"] == "label"
        assert label_meta["label"] == "Seizure"

    def test_synonym_metadata_includes_synonym_text(self, sample_hpo_terms):
        """Test synonym metadata includes the synonym text."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # Find first synonym metadata
        syn_idx = ids.index("HP:0001250__synonym__0")
        syn_meta = metadatas[syn_idx]

        assert syn_meta["component"] == "synonym"
        assert syn_meta["synonym_text"] == "Fits"

    def test_documents_contain_correct_text(self, sample_hpo_terms):
        """Test document text is the component content."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # Check label document
        label_idx = ids.index("HP:0001250__label__0")
        assert documents[label_idx] == "Seizure"

        # Check synonym document
        syn_idx = ids.index("HP:0001250__synonym__0")
        assert documents[syn_idx] == "Fits"

        # Check definition document
        def_idx = ids.index("HP:0001250__definition__0")
        assert documents[def_idx] == "A seizure is a transient occurrence."

    def test_include_label_only(self, sample_hpo_terms):
        """Test creating documents with only labels."""
        documents, metadatas, ids = create_multi_vector_documents(
            sample_hpo_terms,
            include_label=True,
            include_synonyms=False,
            include_definition=False,
        )

        assert len(documents) == 3  # One label per term
        assert all(m["component"] == "label" for m in metadatas)

    def test_include_synonyms_only(self, sample_hpo_terms):
        """Test creating documents with only synonyms."""
        documents, metadatas, ids = create_multi_vector_documents(
            sample_hpo_terms,
            include_label=False,
            include_synonyms=True,
            include_definition=False,
        )

        # HP:0001250: 2 synonyms, HP:0001252: 1 synonym, HP:0001253: 0 synonyms
        assert len(documents) == 3
        assert all(m["component"] == "synonym" for m in metadatas)

    def test_include_definition_only(self, sample_hpo_terms):
        """Test creating documents with only definitions."""
        documents, metadatas, ids = create_multi_vector_documents(
            sample_hpo_terms,
            include_label=False,
            include_synonyms=False,
            include_definition=True,
        )

        # HP:0001250 and HP:0001253 have definitions, HP:0001252 doesn't
        assert len(documents) == 2
        assert all(m["component"] == "definition" for m in metadatas)

    def test_handles_empty_synonyms_list(self, sample_hpo_terms):
        """Test handles terms with empty synonyms list."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # HP:0001253 has empty synonyms, shouldn't create synonym docs
        spasticity_synonym_ids = [i for i in ids if "HP:0001253__synonym" in i]
        assert len(spasticity_synonym_ids) == 0

    def test_handles_none_definition(self, sample_hpo_terms):
        """Test handles terms with None definition."""
        documents, metadatas, ids = create_multi_vector_documents(sample_hpo_terms)

        # HP:0001252 has no definition, shouldn't create definition doc
        hypotonia_def_ids = [i for i in ids if "HP:0001252__definition" in i]
        assert len(hypotonia_def_ids) == 0

    def test_empty_input_returns_empty_lists(self):
        """Test empty input returns empty lists."""
        documents, metadatas, ids = create_multi_vector_documents([])
        assert documents == []
        assert metadatas == []
        assert ids == []


class TestGetComponentStats:
    """Test component statistics calculation."""

    @pytest.fixture
    def sample_hpo_terms(self):
        """Sample HPO terms for testing."""
        return [
            {
                "id": "HP:0001",
                "label": "Term 1",
                "synonyms": ["syn1", "syn2"],
                "definition": "Definition 1",
            },
            {
                "id": "HP:0002",
                "label": "Term 2",
                "synonyms": ["syn3"],
                "definition": None,
            },
            {
                "id": "HP:0003",
                "label": "Term 3",
                "synonyms": [],
                "definition": "Definition 3",
            },
        ]

    def test_counts_total_terms(self, sample_hpo_terms):
        """Test counts total terms correctly."""
        stats = get_component_stats(sample_hpo_terms)
        assert stats["total_terms"] == 3

    def test_counts_labels(self, sample_hpo_terms):
        """Test labels count equals terms count."""
        stats = get_component_stats(sample_hpo_terms)
        assert stats["total_labels"] == 3

    def test_counts_synonyms(self, sample_hpo_terms):
        """Test counts total synonyms."""
        stats = get_component_stats(sample_hpo_terms)
        # Term 1: 2, Term 2: 1, Term 3: 0 = 3 total
        assert stats["total_synonyms"] == 3

    def test_counts_definitions(self, sample_hpo_terms):
        """Test counts terms with definitions."""
        stats = get_component_stats(sample_hpo_terms)
        # Term 1 and Term 3 have definitions
        assert stats["total_definitions"] == 2

    def test_estimates_total_documents(self, sample_hpo_terms):
        """Test estimates total documents for multi-vector index."""
        stats = get_component_stats(sample_hpo_terms)
        # 3 labels + 3 synonyms + 2 definitions = 8
        assert stats["estimated_documents"] == 8

    def test_empty_input(self):
        """Test handles empty input."""
        stats = get_component_stats([])
        assert stats["total_terms"] == 0
        assert stats["estimated_documents"] == 0
