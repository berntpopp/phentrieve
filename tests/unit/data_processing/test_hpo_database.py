"""
Unit tests for HPODatabase helper class.

Tests cover:
- Schema initialization
- Bulk inserts (terms, graph metadata)
- Data retrieval (all terms, graph data, label map)
- Transaction management
- Error handling
- Context manager protocol
"""

import json
import sqlite3

import pytest

from phentrieve.data_processing.hpo_database import HPODatabase

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


@pytest.fixture
def temp_db():
    """Create temporary in-memory database for testing."""
    db = HPODatabase(":memory:")
    db.initialize_schema()
    yield db
    db.close()


@pytest.fixture
def temp_file_db(tmp_path):
    """Create temporary file-based database for testing."""
    db_path = tmp_path / "test_hpo.db"
    db = HPODatabase(db_path)
    db.initialize_schema()
    yield db
    db.close()


@pytest.fixture
def sample_terms():
    """Sample HPO terms for testing."""
    return [
        {
            "id": "HP:0000001",
            "label": "All",
            "definition": "Root of all terms",
            "synonyms": json.dumps(["Everything"]),
            "comments": json.dumps(["Root term"]),
        },
        {
            "id": "HP:0000118",
            "label": "Phenotypic abnormality",
            "definition": "A phenotypic abnormality",
            "synonyms": json.dumps(["Organ abnormality"]),
            "comments": json.dumps([]),
        },
        {
            "id": "HP:0001250",
            "label": "Seizure",
            "definition": "A seizure is an intermittent abnormality",
            "synonyms": json.dumps(["Seizures", "Epileptic seizure"]),
            "comments": json.dumps(["Common symptom"]),
        },
    ]


@pytest.fixture
def sample_graph_metadata():
    """Sample graph metadata for testing."""
    return [
        {
            "term_id": "HP:0000001",
            "depth": 0,
            "ancestors": json.dumps(["HP:0000001"]),
        },
        {
            "term_id": "HP:0000118",
            "depth": 1,
            "ancestors": json.dumps(["HP:0000001", "HP:0000118"]),
        },
        {
            "term_id": "HP:0001250",
            "depth": 5,
            "ancestors": json.dumps(
                ["HP:0000001", "HP:0000118", "HP:0000707", "HP:0012638", "HP:0001250"]
            ),
        },
    ]


class TestSchemaInitialization:
    """Test database schema initialization."""

    def test_initialize_schema_creates_tables(self, temp_db):
        """Test that schema initialization creates required tables."""
        conn = temp_db.get_connection()

        # Check tables exist
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('hpo_terms', 'hpo_graph_metadata', 'generation_metadata')
            """
        )
        tables = {row[0] for row in cursor}

        assert "hpo_terms" in tables
        assert "hpo_graph_metadata" in tables
        assert "generation_metadata" in tables

    def test_initialize_schema_creates_indexes(self, temp_db):
        """Test that schema initialization creates required indexes."""
        conn = temp_db.get_connection()

        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN ('idx_hpo_terms_label', 'idx_hpo_graph_depth')
            """
        )
        indexes = {row[0] for row in cursor}

        assert "idx_hpo_terms_label" in indexes
        assert "idx_hpo_graph_depth" in indexes

    def test_initialize_schema_sets_pragmas(self, temp_file_db):
        """Test that schema initialization sets performance PRAGMAs."""
        conn = temp_file_db.get_connection()

        # Check journal mode (should be WAL for file-based DB)
        # Note: In-memory DBs use 'memory' mode, which is expected
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode.lower() == "wal"

        # Check foreign keys enabled
        fk_enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk_enabled == 1

    def test_initialize_schema_idempotent(self, temp_db):
        """Test that calling initialize_schema multiple times is safe."""
        # Should not raise error
        temp_db.initialize_schema()
        temp_db.initialize_schema()

        # Should still have correct structure
        term_count = temp_db.get_term_count()
        assert term_count == 0  # No duplicate initialization side effects

    def test_get_schema_version(self, temp_db):
        """Test retrieving schema version."""
        version = temp_db.get_schema_version()
        assert version == 1


class TestBulkInserts:
    """Test bulk insert operations."""

    def test_bulk_insert_terms_success(self, temp_db, sample_terms):
        """Test successful bulk insert of HPO terms."""
        count = temp_db.bulk_insert_terms(sample_terms)

        assert count == 3
        assert temp_db.get_term_count() == 3

    def test_bulk_insert_terms_empty_list(self, temp_db):
        """Test bulk insert with empty list returns 0."""
        count = temp_db.bulk_insert_terms([])
        assert count == 0

    def test_bulk_insert_terms_transaction_rollback(self, temp_db, sample_terms):
        """Test that failed insert rolls back entire transaction."""
        # Insert valid data first
        temp_db.bulk_insert_terms(sample_terms[:2])

        # Try to insert duplicate (should fail on PK constraint)
        invalid_data = [sample_terms[0]]  # Duplicate ID

        with pytest.raises(sqlite3.IntegrityError):
            temp_db.bulk_insert_terms(invalid_data)

        # Original data should still be there
        assert temp_db.get_term_count() == 2

    def test_bulk_insert_graph_metadata_success(
        self, temp_db, sample_terms, sample_graph_metadata
    ):
        """Test successful bulk insert of graph metadata."""
        # Must insert terms first (FK constraint)
        temp_db.bulk_insert_terms(sample_terms)

        count = temp_db.bulk_insert_graph_metadata(sample_graph_metadata)
        assert count == 3

        # Verify data
        ancestors, depths = temp_db.load_graph_data()
        assert len(ancestors) == 3
        assert len(depths) == 3

    def test_bulk_insert_graph_metadata_empty_list(self, temp_db):
        """Test bulk insert graph metadata with empty list."""
        count = temp_db.bulk_insert_graph_metadata([])
        assert count == 0

    def test_bulk_insert_graph_metadata_fk_constraint(
        self, temp_db, sample_graph_metadata
    ):
        """Test that FK constraint is enforced (term must exist)."""
        # Try to insert graph data without terms (FK violation)
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.bulk_insert_graph_metadata(sample_graph_metadata)


class TestDataRetrieval:
    """Test data retrieval operations."""

    def test_load_all_terms_empty(self, temp_db):
        """Test loading terms from empty database."""
        terms = temp_db.load_all_terms()
        assert terms == []

    def test_load_all_terms_deserializes_json(self, temp_db, sample_terms):
        """Test that load_all_terms deserializes JSON fields."""
        temp_db.bulk_insert_terms(sample_terms)

        terms = temp_db.load_all_terms()
        assert len(terms) == 3

        # Check first term
        term = terms[0]
        assert term["id"] == "HP:0000001"
        assert term["label"] == "All"
        assert term["definition"] == "Root of all terms"

        # Check JSON deserialization
        assert isinstance(term["synonyms"], list)
        assert term["synonyms"] == ["Everything"]

        assert isinstance(term["comments"], list)
        assert term["comments"] == ["Root term"]

    def test_load_all_terms_ordered_by_id(self, temp_db, sample_terms):
        """Test that terms are returned ordered by ID."""
        # Insert in random order
        shuffled = [sample_terms[2], sample_terms[0], sample_terms[1]]
        temp_db.bulk_insert_terms(shuffled)

        terms = temp_db.load_all_terms()

        # Should be ordered
        ids = [t["id"] for t in terms]
        assert ids == sorted(ids)

    def test_load_graph_data_empty(self, temp_db):
        """Test loading graph data from empty database."""
        ancestors, depths = temp_db.load_graph_data()
        assert ancestors == {}
        assert depths == {}

    def test_load_graph_data_correct_structure(
        self, temp_db, sample_terms, sample_graph_metadata
    ):
        """Test that load_graph_data returns correct data structures."""
        temp_db.bulk_insert_terms(sample_terms)
        temp_db.bulk_insert_graph_metadata(sample_graph_metadata)

        ancestors, depths = temp_db.load_graph_data()

        # Check structure
        assert len(ancestors) == 3
        assert len(depths) == 3

        # Check root term
        assert depths["HP:0000001"] == 0
        assert ancestors["HP:0000001"] == {"HP:0000001"}

        # Check nested term
        assert depths["HP:0001250"] == 5
        assert "HP:0000001" in ancestors["HP:0001250"]
        assert "HP:0000118" in ancestors["HP:0001250"]
        assert len(ancestors["HP:0001250"]) == 5

    def test_get_label_map_empty(self, temp_db):
        """Test label map from empty database."""
        labels = temp_db.get_label_map()
        assert labels == {}

    def test_get_label_map_correct(self, temp_db, sample_terms):
        """Test that get_label_map returns correct ID->label mapping."""
        temp_db.bulk_insert_terms(sample_terms)

        labels = temp_db.get_label_map()

        assert len(labels) == 3
        assert labels["HP:0000001"] == "All"
        assert labels["HP:0000118"] == "Phenotypic abnormality"
        assert labels["HP:0001250"] == "Seizure"

    def test_get_term_count(self, temp_db, sample_terms):
        """Test get_term_count method."""
        assert temp_db.get_term_count() == 0

        temp_db.bulk_insert_terms(sample_terms)
        assert temp_db.get_term_count() == 3


class TestTransactionManagement:
    """Test transaction context manager."""

    def test_transaction_commits_on_success(self, temp_db):
        """Test that transaction commits on successful completion."""
        with temp_db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO hpo_terms (id, label, definition, synonyms, comments)
                VALUES ('HP:0000001', 'All', 'Root', '[]', '[]')
                """
            )

        # Verify committed
        assert temp_db.get_term_count() == 1

    def test_transaction_rolls_back_on_error(self, temp_db):
        """Test that transaction rolls back on exception."""
        # Insert initial data
        with temp_db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO hpo_terms (id, label, definition, synonyms, comments)
                VALUES ('HP:0000001', 'All', 'Root', '[]', '[]')
                """
            )

        assert temp_db.get_term_count() == 1

        # Try to insert duplicate (should fail and rollback)
        try:
            with temp_db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO hpo_terms (id, label, definition, synonyms, comments)
                    VALUES ('HP:0000001', 'Duplicate', 'Dup', '[]', '[]')
                    """
                )
        except sqlite3.IntegrityError:
            pass  # Expected

        # Should still have only 1 term
        assert temp_db.get_term_count() == 1


class TestContextManager:
    """Test context manager protocol (__enter__/__exit__)."""

    def test_context_manager_closes_connection(self, tmp_path):
        """Test that context manager closes connection on exit."""
        db_path = tmp_path / "test.db"

        with HPODatabase(db_path) as db:
            db.initialize_schema()
            # Connection should be open
            assert db._conn is not None

        # Connection should be closed after context
        assert db._conn is None

    def test_context_manager_closes_on_exception(self, tmp_path):
        """Test that context manager closes connection even on exception."""
        db_path = tmp_path / "test.db"

        try:
            with HPODatabase(db_path) as db:
                db.initialize_schema()
                raise ValueError("Test error")
        except ValueError:
            pass

        # Connection should be closed
        assert db._conn is None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_load_terms_handles_null_json(self, temp_db):
        """Test that NULL JSON fields are handled correctly."""
        # Insert term with NULL synonyms/comments
        with temp_db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO hpo_terms (id, label, definition, synonyms, comments)
                VALUES ('HP:0000001', 'All', 'Root', NULL, NULL)
                """
            )

        terms = temp_db.load_all_terms()
        assert len(terms) == 1

        # Should deserialize to empty lists
        assert terms[0]["synonyms"] == []
        assert terms[0]["comments"] == []

    def test_close_idempotent(self, temp_db):
        """Test that calling close() multiple times is safe."""
        temp_db.close()
        temp_db.close()  # Should not raise error

    def test_connection_reuse_after_close(self, temp_file_db):
        """Test that new connection is created after close."""
        first_conn = temp_file_db.get_connection()
        temp_file_db.close()

        second_conn = temp_file_db.get_connection()
        assert second_conn is not first_conn


class TestOptimization:
    """Test database optimization methods."""

    def test_optimize_runs_analyze(self, temp_db, sample_terms):
        """Test that optimize() runs ANALYZE command."""
        temp_db.bulk_insert_terms(sample_terms)

        # Should not raise error
        temp_db.optimize()

        # Verify ANALYZE was run by checking sqlite_stat1 table exists
        conn = temp_db.get_connection()
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='sqlite_stat1'
            """
        )
        result = cursor.fetchone()

        # After ANALYZE, stat table should exist
        assert result is not None


class TestFileBasedDatabase:
    """Test file-based database operations."""

    def test_create_file_database(self, tmp_path):
        """Test creating file-based database."""
        db_path = tmp_path / "hpo.db"
        db = HPODatabase(db_path)
        db.initialize_schema()

        # Verify file created
        assert db_path.exists()

        db.close()

    def test_persistence_across_connections(self, tmp_path, sample_terms):
        """Test that data persists across connection closures."""
        db_path = tmp_path / "hpo.db"

        # Insert data
        db1 = HPODatabase(db_path)
        db1.initialize_schema()
        db1.bulk_insert_terms(sample_terms)
        db1.close()

        # Reopen and verify
        db2 = HPODatabase(db_path)
        terms = db2.load_all_terms()
        assert len(terms) == 3
        db2.close()

    def test_path_object_and_string_both_work(self, tmp_path, sample_terms):
        """Test that both Path objects and strings work for db_path."""
        # Test with Path object
        db_path = tmp_path / "hpo1.db"
        db1 = HPODatabase(db_path)
        db1.initialize_schema()
        db1.bulk_insert_terms(sample_terms[:1])
        assert db1.get_term_count() == 1
        db1.close()

        # Test with string
        db_path_str = str(tmp_path / "hpo2.db")
        db2 = HPODatabase(db_path_str)
        db2.initialize_schema()
        db2.bulk_insert_terms(sample_terms[:2])
        assert db2.get_term_count() == 2
        db2.close()
