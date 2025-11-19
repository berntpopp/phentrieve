"""
Simple database helper for HPO data storage.

This module provides a lightweight wrapper around SQLite for HPO term storage.
Following KISS principle: minimal abstraction, maximum clarity.

NOT a Repository pattern or ORM - just utility functions for common operations.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Schema SQL with performance optimizations
SCHEMA_SQL = """
-- Schema version tracking
PRAGMA user_version = 1;

-- Performance optimizations for read-heavy workload
PRAGMA journal_mode = WAL;        -- Write-Ahead Logging for concurrent reads
PRAGMA synchronous = NORMAL;      -- Balance safety and performance
PRAGMA cache_size = -64000;       -- 64 MB cache (negative = KB)
PRAGMA temp_store = MEMORY;       -- Store temp tables in RAM
PRAGMA mmap_size = 30000000000;   -- Memory-mapped I/O (30 GB limit)

-- Core terms table
CREATE TABLE IF NOT EXISTS hpo_terms (
    id TEXT PRIMARY KEY,              -- HP:0000123
    label TEXT NOT NULL,
    definition TEXT,
    synonyms TEXT,                    -- JSON array: ["syn1", "syn2"]
    comments TEXT,                    -- JSON array: ["comment1"]
    created_at TEXT DEFAULT (datetime('now'))
) WITHOUT ROWID;  -- Optimization: use TEXT PK directly, saves ~20% storage

-- Graph metadata (ancestors, depth)
CREATE TABLE IF NOT EXISTS hpo_graph_metadata (
    term_id TEXT PRIMARY KEY,
    depth INTEGER NOT NULL,          -- Distance from root HP:0000001
    ancestors TEXT NOT NULL,         -- JSON array: ["HP:0000001", "HP:0000118"]
    FOREIGN KEY (term_id) REFERENCES hpo_terms(id) ON DELETE CASCADE
) WITHOUT ROWID;

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_hpo_terms_label ON hpo_terms(label);
CREATE INDEX IF NOT EXISTS idx_hpo_graph_depth ON hpo_graph_metadata(depth);

-- Metadata table for tracking data source
CREATE TABLE IF NOT EXISTS generation_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


class HPODatabase:
    """
    Lightweight database helper for HPO data.

    This is NOT a Repository pattern! Just a thin wrapper for common operations.
    Keeps things simple and explicit.

    Usage:
        db = HPODatabase(Path("hpo_data.db"))
        db.initialize_schema()

        # Insert data
        terms = [{"id": "HP:0000001", "label": "All", ...}]
        db.bulk_insert_terms(terms)

        # Query data
        all_terms = db.load_all_terms()
        ancestors, depths = db.load_graph_data()

        db.close()
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize database helper.

        Args:
            db_path: Path to SQLite database file (or ":memory:" for in-memory)
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._conn: sqlite3.Connection | None = None

    def get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection (lazy initialization).

        Returns:
            Active SQLite connection with Row factory enabled
        """
        if self._conn is None:
            # Convert Path to string for sqlite3.connect
            db_path_str = (
                str(self.db_path) if isinstance(self.db_path, Path) else self.db_path
            )

            self._conn = sqlite3.connect(
                db_path_str,
                check_same_thread=False,  # Allow multi-threading
            )
            self._conn.row_factory = sqlite3.Row  # Enable dict-like row access
            self._conn.execute("PRAGMA foreign_keys = ON")  # Enforce FK constraints

        return self._conn

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Automatically commits on success, rolls back on exception.

        Usage:
            with db.transaction() as conn:
                conn.execute("INSERT ...")
                # Auto-commit on success, auto-rollback on exception
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize_schema(self) -> None:
        """
        Create database schema if it doesn't exist.

        Idempotent: Safe to call multiple times.
        """
        conn = self.get_connection()
        conn.executescript(SCHEMA_SQL)

        # Insert initial metadata (if not exists)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO generation_metadata (key, value)
                VALUES ('schema_version', '1')
            """)
            conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Could not initialize metadata: {e}")

        logger.info(f"Initialized database schema at: {self.db_path}")

    def bulk_insert_terms(self, terms: list[dict[str, Any]]) -> int:
        """
        Bulk insert HPO terms with transaction.

        Args:
            terms: List of term dicts with keys: id, label, definition, synonyms, comments
                  Note: synonyms and comments should be JSON strings

        Returns:
            Number of terms inserted

        Raises:
            sqlite3.Error: On database errors
        """
        if not terms:
            return 0

        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO hpo_terms (id, label, definition, synonyms, comments)
                VALUES (:id, :label, :definition, :synonyms, :comments)
                """,
                terms,
            )

        logger.debug(f"Inserted {len(terms)} HPO terms")
        return len(terms)

    def bulk_insert_graph_metadata(self, metadata: list[dict[str, Any]]) -> int:
        """
        Bulk insert graph metadata with transaction.

        Args:
            metadata: List of metadata dicts with keys: term_id, depth, ancestors
                     Note: ancestors should be a JSON string

        Returns:
            Number of metadata records inserted

        Raises:
            sqlite3.Error: On database errors
        """
        if not metadata:
            return 0

        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO hpo_graph_metadata (term_id, depth, ancestors)
                VALUES (:term_id, :depth, :ancestors)
                """,
                metadata,
            )

        logger.debug(f"Inserted {len(metadata)} graph metadata records")
        return len(metadata)

    def load_all_terms(self) -> list[dict[str, Any]]:
        """
        Load all HPO terms from database.

        Returns:
            List of term dicts with deserialized synonyms and comments
        """
        conn = self.get_connection()
        cursor = conn.execute(
            """
            SELECT id, label, definition, synonyms, comments
            FROM hpo_terms
            ORDER BY id
            """
        )

        terms = []
        for row in cursor:
            term = dict(row)
            # Deserialize JSON fields
            term["synonyms"] = json.loads(term["synonyms"] or "[]")
            term["comments"] = json.loads(term["comments"] or "[]")
            terms.append(term)

        logger.debug(f"Loaded {len(terms)} HPO terms from database")
        return terms

    def load_graph_data(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        """
        Load graph metadata (ancestors and depths).

        Returns:
            Tuple of (ancestors_map, depths_map)
            - ancestors_map: {term_id: set of ancestor IDs}
            - depths_map: {term_id: depth from root}
        """
        conn = self.get_connection()
        cursor = conn.execute(
            """
            SELECT term_id, depth, ancestors
            FROM hpo_graph_metadata
            """
        )

        ancestors_map: dict[str, set[str]] = {}
        depths_map: dict[str, int] = {}

        for row in cursor:
            term_id = row["term_id"]
            depths_map[term_id] = row["depth"]
            # Deserialize JSON array to set
            ancestors_map[term_id] = set(json.loads(row["ancestors"]))

        logger.debug(f"Loaded graph data for {len(ancestors_map)} terms")
        return ancestors_map, depths_map

    def get_label_map(self) -> dict[str, str]:
        """
        Get efficient ID -> label mapping.

        Optimized for CLI label lookup (avoids loading full term data).

        Returns:
            Dictionary mapping {term_id: label}
        """
        conn = self.get_connection()
        cursor = conn.execute("SELECT id, label FROM hpo_terms")

        label_map = {row["id"]: row["label"] for row in cursor}

        logger.debug(f"Loaded {len(label_map)} labels")
        return label_map

    def optimize(self) -> None:
        """
        Optimize database: run ANALYZE for query planner statistics.

        Should be called after bulk inserts to update query planner stats.
        Can improve query performance by 10-30%.
        """
        conn = self.get_connection()
        conn.execute("ANALYZE")
        conn.commit()
        logger.info("Database optimized with ANALYZE")

    def get_schema_version(self) -> int:
        """
        Get current schema version.

        Returns:
            Schema version number (from PRAGMA user_version)
        """
        conn = self.get_connection()
        result = conn.execute("PRAGMA user_version").fetchone()
        return result[0] if result else 0

    def get_term_count(self) -> int:
        """
        Get total number of HPO terms in database.

        Returns:
            Count of terms
        """
        conn = self.get_connection()
        result = conn.execute("SELECT COUNT(*) FROM hpo_terms").fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    @property
    def is_closed(self) -> bool:
        """
        Check if database connection is closed.

        Returns:
            True if connection is closed (None), False if open
        """
        return self._conn is None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: always close connection."""
        self.close()
        return False  # Don't suppress exceptions
