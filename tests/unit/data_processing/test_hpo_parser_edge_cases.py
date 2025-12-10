"""
Edge case tests for HPO parser schema resilience (Issue #23) and
obsolete term filtering (Issue #133).

Tests defensive programming patterns for handling malformed HPO JSON data:
- Missing required fields (graphs, nodes, edges)
- Malformed data structures (wrong types, empty arrays)
- Missing optional metadata (definitions, synonyms, comments)
- Edge field variants ('sub' vs 'subj')
- Obsolete term detection and filtering (Issue #133)
"""

import json
import logging

from phentrieve.data_processing.hpo_parser import (
    _extract_term_data_for_db,
    _parse_hpo_json_to_graphs,
    get_replacement_term,
    is_obsolete_term,
)


class TestParseHpoJsonEdgeCases:
    """Edge cases for _parse_hpo_json_to_graphs()."""

    def test_missing_graphs_field(self, caplog):
        """Test that missing 'graphs' field returns None tuple."""
        invalid_json = {"meta": "some data"}  # No 'graphs'!

        with caplog.at_level(logging.ERROR):
            result = _parse_hpo_json_to_graphs(invalid_json)

        # Returns 5-tuple with obsolete_count (Issue #133)
        assert result == (None, None, None, None, 0)
        assert "graphs' field is missing" in caplog.text
        assert "Available top-level keys" in caplog.text

    def test_graphs_wrong_type(self, caplog):
        """Test that graphs as non-list returns None tuple."""
        invalid_json = {"graphs": "not_a_list"}  # Wrong type!

        with caplog.at_level(logging.ERROR):
            result = _parse_hpo_json_to_graphs(invalid_json)

        # Returns 5-tuple with obsolete_count (Issue #133)
        assert result == (None, None, None, None, 0)
        assert "must be a list" in caplog.text
        assert "str" in caplog.text  # Shows actual type

    def test_missing_nodes_field(self, caplog):
        """Test handling of missing nodes field (not critical, warns)."""
        hpo_data = {"graphs": [{}]}  # No 'nodes' field

        with caplog.at_level(logging.WARNING):
            nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert nodes == {}
        assert p2c == {}
        assert c2p == {}
        assert ids == set()
        assert obsolete_count == 0
        assert "No nodes found" in caplog.text

    def test_missing_edges_field(self, caplog):
        """Test handling of missing edges field (not critical, warns)."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {"id": "HP:0001250", "lbl": "Seizure"},
                    ]
                    # No 'edges' field
                }
            ]
        }

        with caplog.at_level(logging.WARNING):
            nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(nodes) == 2
        assert len(p2c) == 0  # No edges processed
        assert len(c2p) == 0
        assert len(ids) == 2
        assert obsolete_count == 0
        assert "No edges found" in caplog.text

    def test_node_missing_id_field(self, caplog):
        """Test that node without 'id' is logged and skipped."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"lbl": "Missing ID"}  # No 'id'!
                    ]
                }
            ]
        }

        with caplog.at_level(logging.WARNING):
            nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(nodes) == 0  # Node skipped
        assert obsolete_count == 0
        assert "without required 'id' field" in caplog.text
        assert "Missing ID" in caplog.text  # Label shown for debugging

    def test_edge_with_sub_variant(self):
        """Test that edges with 'sub' field (instead of 'subj') work."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {"id": "HP:0001250", "lbl": "Seizure"},
                    ],
                    "edges": [
                        # Using 'sub' instead of 'subj'
                        {"sub": "HP:0001250", "pred": "is_a", "obj": "HP:0000001"}
                    ],
                }
            ]
        }

        nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(nodes) == 2
        assert "HP:0000001" in p2c
        assert "HP:0001250" in c2p
        assert c2p["HP:0001250"] == ["HP:0000001"]
        assert obsolete_count == 0

    def test_edge_with_subj_variant(self):
        """Test that edges with 'subj' field work."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {"id": "HP:0001250", "lbl": "Seizure"},
                    ],
                    "edges": [
                        # Using 'subj' (standard variant)
                        {"subj": "HP:0001250", "pred": "is_a", "obj": "HP:0000001"}
                    ],
                }
            ]
        }

        nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(nodes) == 2
        assert c2p["HP:0001250"] == ["HP:0000001"]
        assert obsolete_count == 0

    def test_edge_missing_subject_and_object(self, caplog):
        """Test that edge with missing sub/subj and obj is logged and skipped."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [{"id": "HP:0000001", "lbl": "All"}],
                    "edges": [
                        {"pred": "is_a"}  # Missing both subj and obj!
                    ],
                }
            ]
        }

        with caplog.at_level(logging.WARNING):
            nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(p2c) == 0  # Edge skipped
        assert obsolete_count == 0
        assert "missing subject or object" in caplog.text
        assert "subj/sub: None" in caplog.text
        assert "obj: None" in caplog.text


class TestExtractTermDataEdgeCases:
    """Edge cases for _extract_term_data_for_db()."""

    def test_missing_all_metadata(self, caplog):
        """Test term extraction when all metadata is missing."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "Seizure",
                # No 'meta' field at all
            }
        }

        with caplog.at_level(logging.INFO):
            result = _extract_term_data_for_db(node_data)

        assert len(result) == 1
        term = result[0]
        assert term["id"] == "HP:0001250"
        assert term["label"] == "Seizure"
        assert term["definition"] == ""  # Default
        assert term["synonyms"] == "[]"  # Empty list
        assert term["comments"] == "[]"  # Empty list

        # Check statistics logged
        assert "HPO Parsing Summary" in caplog.text
        assert "Missing definitions: 1" in caplog.text
        assert "Missing synonyms: 1" in caplog.text
        assert "Missing comments: 1" in caplog.text

    def test_partial_metadata_definition_only(self):
        """Test term extraction with only definition (no synonyms/comments)."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "Seizure",
                "meta": {
                    "definition": {"val": "A seizure is...", "xrefs": []},
                    # synonyms and comments missing
                },
            }
        }

        result = _extract_term_data_for_db(node_data)

        assert len(result) == 1
        term = result[0]
        assert term["definition"] == "A seizure is..."
        assert term["synonyms"] == "[]"
        assert term["comments"] == "[]"

    def test_malformed_synonyms_not_a_list(self, caplog):
        """Test handling of malformed synonym data (wrong type)."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "Seizure",
                "meta": {
                    "synonyms": "not_a_list"  # Wrong type!
                },
            }
        }

        with caplog.at_level(logging.DEBUG):
            result = _extract_term_data_for_db(node_data)

        assert len(result) == 1
        term = result[0]
        assert term["synonyms"] == "[]"  # Handled gracefully

    def test_malformed_comments_not_a_list(self):
        """Test handling of malformed comment data (wrong type)."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "Seizure",
                "meta": {
                    "comments": {"not": "a_list"}  # Wrong type!
                },
            }
        }

        result = _extract_term_data_for_db(node_data)

        assert len(result) == 1
        term = result[0]
        assert term["comments"] == "[]"  # Handled gracefully

    def test_empty_label_uses_id_fallback(self, caplog):
        """Test that empty label falls back to using term ID."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "",  # Empty!
            }
        }

        with caplog.at_level(logging.WARNING):
            result = _extract_term_data_for_db(node_data)

        assert len(result) == 1
        term = result[0]
        assert term["label"] == "HP:0001250"  # Fallback to ID
        assert "empty label - using ID as fallback" in caplog.text

    def test_synonyms_with_mixed_valid_invalid(self):
        """Test synonym extraction with mix of valid and invalid entries."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "Seizure",
                "meta": {
                    "synonyms": [
                        {"val": "Valid synonym 1"},
                        {"no_val_field": True},  # Missing 'val'
                        {"val": "Valid synonym 2"},
                        "not_a_dict",  # Wrong type
                        {"val": ""},  # Empty val
                    ]
                },
            }
        }

        result = _extract_term_data_for_db(node_data)

        term = result[0]
        # Should only extract valid synonyms
        synonyms = json.loads(term["synonyms"])  # Parse JSON string safely
        assert synonyms == ["Valid synonym 1", "Valid synonym 2"]

    def test_comments_filters_empty_and_non_strings(self):
        """Test comment extraction filters empty and non-string values."""
        node_data = {
            "HP:0001250": {
                "id": "http://purl.obolibrary.org/obo/HP_0001250",
                "lbl": "Seizure",
                "meta": {
                    "comments": [
                        "Valid comment 1",
                        "",  # Empty
                        "Valid comment 2",
                        None,  # None
                        123,  # Not a string
                        "Valid comment 3",
                    ]
                },
            }
        }

        result = _extract_term_data_for_db(node_data)

        term = result[0]
        comments = json.loads(term["comments"])  # Parse JSON string safely
        # Should only keep valid non-empty strings
        assert comments == ["Valid comment 1", "Valid comment 2", "Valid comment 3"]


# =============================================================================
# Obsolete Term Detection Tests (Issue #133)
# =============================================================================


class TestIsObsoleteTerm:
    """Tests for is_obsolete_term() function (Issue #133)."""

    def test_detects_deprecated_flag_true(self):
        """Test detection of meta.deprecated=true."""
        node = {
            "id": "HP:0000057",
            "lbl": "obsolete Clitoromegaly",
            "meta": {"deprecated": True},
        }
        assert is_obsolete_term(node) is True

    def test_detects_obsolete_label_prefix(self):
        """Test detection of 'obsolete ' prefix in label."""
        node = {
            "id": "HP:0000057",
            "lbl": "obsolete Clitoromegaly",
            "meta": {},  # No deprecated flag
        }
        assert is_obsolete_term(node) is True

    def test_detects_obsolete_label_prefix_case_insensitive(self):
        """Test that 'OBSOLETE ' prefix is detected case-insensitively."""
        node = {
            "id": "HP:0000057",
            "lbl": "OBSOLETE Clitoromegaly",
            "meta": {},
        }
        assert is_obsolete_term(node) is True

    def test_non_obsolete_term(self):
        """Test that non-obsolete terms return False."""
        node = {
            "id": "HP:0001250",
            "lbl": "Seizure",
            "meta": {"definition": {"val": "A seizure is..."}},
        }
        assert is_obsolete_term(node) is False

    def test_deprecated_false(self):
        """Test that deprecated=false returns False."""
        node = {
            "id": "HP:0001250",
            "lbl": "Seizure",
            "meta": {"deprecated": False},
        }
        assert is_obsolete_term(node) is False

    def test_missing_meta_field(self):
        """Test handling of missing meta field."""
        node = {
            "id": "HP:0001250",
            "lbl": "Seizure",
            # No 'meta' field
        }
        assert is_obsolete_term(node) is False

    def test_empty_label(self):
        """Test handling of empty label."""
        node = {
            "id": "HP:0001250",
            "lbl": "",
            "meta": {},
        }
        assert is_obsolete_term(node) is False

    def test_none_label(self):
        """Test handling of None label."""
        node = {
            "id": "HP:0001250",
            "lbl": None,
            "meta": {},
        }
        assert is_obsolete_term(node) is False


class TestGetReplacementTerm:
    """Tests for get_replacement_term() function (Issue #133)."""

    def test_extracts_replacement_term(self):
        """Test extraction of IAO_0100001 (term replaced by) annotation."""
        node = {
            "id": "HP:0000057",
            "lbl": "obsolete Clitoromegaly",
            "meta": {
                "deprecated": True,
                "basicPropertyValues": [
                    {
                        "pred": "http://purl.obolibrary.org/obo/IAO_0100001",
                        "val": "HP:0008665",
                    }
                ],
            },
        }
        assert get_replacement_term(node) == "HP:0008665"

    def test_normalizes_full_uri(self):
        """Test normalization of full URI to HP:XXXXXXX format."""
        node = {
            "meta": {
                "basicPropertyValues": [
                    {
                        "pred": "http://purl.obolibrary.org/obo/IAO_0100001",
                        "val": "http://purl.obolibrary.org/obo/HP_0008665",
                    }
                ]
            }
        }
        assert get_replacement_term(node) == "HP:0008665"

    def test_no_replacement_term(self):
        """Test when no replacement term is specified."""
        node = {
            "meta": {
                "deprecated": True,
                # No basicPropertyValues with IAO_0100001
            }
        }
        assert get_replacement_term(node) is None

    def test_empty_basic_property_values(self):
        """Test empty basicPropertyValues array."""
        node = {"meta": {"basicPropertyValues": []}}
        assert get_replacement_term(node) is None

    def test_missing_meta(self):
        """Test handling of missing meta field."""
        node = {"id": "HP:0001250", "lbl": "Seizure"}
        assert get_replacement_term(node) is None

    def test_malformed_basic_property_values(self):
        """Test handling of malformed basicPropertyValues."""
        node = {
            "meta": {
                "basicPropertyValues": [
                    "not_a_dict",  # Wrong type
                    {"pred": "IAO_0100001"},  # Missing val
                    {"val": "HP:0001234"},  # Missing pred
                ]
            }
        }
        assert get_replacement_term(node) is None


class TestObsoleteTermFiltering:
    """Integration tests for obsolete term filtering in _parse_hpo_json_to_graphs (Issue #133)."""

    def test_filters_obsolete_terms_by_default(self):
        """Test that obsolete terms are filtered by default."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {"id": "HP:0001250", "lbl": "Seizure"},
                        {
                            "id": "HP:0000057",
                            "lbl": "obsolete Clitoromegaly",
                            "meta": {"deprecated": True},
                        },
                    ],
                    "edges": [
                        {"sub": "HP:0001250", "pred": "is_a", "obj": "HP:0000001"}
                    ],
                }
            ]
        }

        nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(nodes) == 2  # Obsolete term filtered
        assert "HP:0000057" not in nodes
        assert "HP:0000057" not in ids
        assert obsolete_count == 1

    def test_includes_obsolete_terms_when_flag_set(self):
        """Test that obsolete terms are included when include_obsolete=True."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {"id": "HP:0001250", "lbl": "Seizure"},
                        {
                            "id": "HP:0000057",
                            "lbl": "obsolete Clitoromegaly",
                            "meta": {"deprecated": True},
                        },
                    ],
                    "edges": [
                        {"sub": "HP:0001250", "pred": "is_a", "obj": "HP:0000001"}
                    ],
                }
            ]
        }

        nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(
            hpo_data, include_obsolete=True
        )

        assert len(nodes) == 3  # All terms included
        assert "HP:0000057" in nodes
        assert "HP:0000057" in ids
        assert obsolete_count == 0  # Not filtered

    def test_filters_multiple_obsolete_terms(self):
        """Test filtering of multiple obsolete terms."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {"id": "HP:0001250", "lbl": "Seizure"},
                        {
                            "id": "HP:0000057",
                            "lbl": "obsolete Term1",
                            "meta": {"deprecated": True},
                        },
                        {"id": "HP:0000284", "lbl": "obsolete Term2"},
                        {
                            "id": "HP:0000361",
                            "lbl": "OBSOLETE Term3",  # Case-insensitive
                        },
                    ],
                    "edges": [],
                }
            ]
        }

        nodes, p2c, c2p, ids, obsolete_count = _parse_hpo_json_to_graphs(hpo_data)

        assert len(nodes) == 2  # Only non-obsolete terms
        assert obsolete_count == 3

    def test_logs_filtered_obsolete_terms(self, caplog):
        """Test that filtered obsolete terms are logged at debug level."""
        hpo_data = {
            "graphs": [
                {
                    "nodes": [
                        {"id": "HP:0000001", "lbl": "All"},
                        {
                            "id": "HP:0000057",
                            "lbl": "obsolete Clitoromegaly",
                            "meta": {
                                "deprecated": True,
                                "basicPropertyValues": [
                                    {
                                        "pred": "http://purl.obolibrary.org/obo/IAO_0100001",
                                        "val": "HP:0008665",
                                    }
                                ],
                            },
                        },
                    ],
                    "edges": [],
                }
            ]
        }

        with caplog.at_level(logging.DEBUG):
            _parse_hpo_json_to_graphs(hpo_data)

        # Check that obsolete term filtering was logged
        assert "Filtering obsolete term" in caplog.text
        assert "HP:0000057" in caplog.text
        assert "replaced by HP:0008665" in caplog.text
