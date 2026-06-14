"""SQLite-backed user store.

Mirrors the raw-``sqlite3`` style of :mod:`api.llm_quota` (WAL, per-call
connections, idempotent schema) rather than introducing an ORM. Holds three
tables: ``users``, single-use ``auth_tokens`` (email verification + password
reset), and rotating ``refresh_sessions``.

All timestamps are stored as ISO-8601 UTC strings, which compare correctly
lexicographically for expiry checks.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email_lower TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_verified INTEGER NOT NULL DEFAULT 0,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

CREATE_AUTH_TOKENS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    purpose TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_at TEXT,
    created_at TEXT NOT NULL
)
"""

CREATE_REFRESH_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS refresh_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    created_at TEXT NOT NULL
)
"""

__all__ = [
    "User",
    "UserStore",
    "EmailExistsError",
]


class EmailExistsError(Exception):
    """Raised when registering an email that already exists."""


@dataclass(frozen=True, slots=True)
class User:
    id: int
    email: str
    password_hash: str
    is_verified: bool
    failed_attempts: int
    locked_until: str | None
    created_at: str
    updated_at: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_email(email: str) -> str:
    """Return the canonical (trimmed, lower-cased) form used as the unique key."""
    return email.strip().lower()


class UserStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(CREATE_USERS_TABLE_SQL)
            connection.execute(CREATE_AUTH_TOKENS_TABLE_SQL)
            connection.execute(CREATE_REFRESH_SESSIONS_TABLE_SQL)
            connection.commit()

    # ------------------------------------------------------------------ users
    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=int(row["id"]),
            email=row["email_lower"],
            password_hash=row["password_hash"],
            is_verified=bool(row["is_verified"]),
            failed_attempts=int(row["failed_attempts"]),
            locked_until=row["locked_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_user(self, *, email: str, password_hash: str) -> User:
        email_lower = normalize_email(email)
        now = _now_iso()
        with closing(self._connect()) as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO users (
                        email_lower, password_hash, is_verified,
                        failed_attempts, locked_until, created_at, updated_at
                    ) VALUES (?, ?, 0, 0, NULL, ?, ?)
                    """,
                    (email_lower, password_hash, now, now),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise EmailExistsError(email_lower) from exc
            user_id = int(cursor.lastrowid or 0)
        created = self.get_by_id(user_id)
        assert created is not None  # noqa: S101 - just inserted
        return created

    def get_by_email(self, email: str) -> User | None:
        email_lower = normalize_email(email)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email_lower = ?", (email_lower,)
            ).fetchone()
        return self._row_to_user(row) if row is not None else None

    def get_by_id(self, user_id: int) -> User | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row is not None else None

    def mark_verified(self, user_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE users SET is_verified = 1, updated_at = ? WHERE id = ?",
                (_now_iso(), user_id),
            )
            connection.commit()

    def update_password(self, user_id: int, password_hash: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (password_hash, _now_iso(), user_id),
            )
            connection.commit()

    # --------------------------------------------------------------- lockout
    def record_failed_login(
        self, user_id: int, *, max_attempts: int, lockout_seconds: int
    ) -> None:
        now = datetime.now(UTC)
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT failed_attempts FROM users WHERE id = ?", (user_id,)
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return
                attempts = int(row["failed_attempts"]) + 1
                locked_until: str | None = None
                if attempts >= max_attempts:
                    locked_until = (
                        now + timedelta(seconds=lockout_seconds)
                    ).isoformat()
                connection.execute(
                    """
                    UPDATE users
                    SET failed_attempts = ?, locked_until = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (attempts, locked_until, now.isoformat(), user_id),
                )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def reset_failed_login(self, user_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE users
                SET failed_attempts = 0, locked_until = NULL, updated_at = ?
                WHERE id = ?
                """,
                (_now_iso(), user_id),
            )
            connection.commit()

    @staticmethod
    def is_locked(user: User | None) -> bool:
        if user is None or not user.locked_until:
            return False
        return user.locked_until > datetime.now(UTC).isoformat()

    # ---------------------------------------------------------- single-use tokens
    def put_token(
        self,
        token_hash: str,
        *,
        purpose: str,
        user_id: int,
        expires_at: datetime,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO auth_tokens (
                    token_hash, purpose, user_id, expires_at, consumed_at, created_at
                ) VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (
                    token_hash,
                    purpose,
                    user_id,
                    expires_at.isoformat(),
                    _now_iso(),
                ),
            )
            connection.commit()

    def consume_token(self, token_hash: str, *, purpose: str) -> int | None:
        """Atomically consume a valid, unexpired, single-use token.

        Returns the owning ``user_id`` or ``None`` if the token is missing,
        wrong-purpose, already consumed, or expired.
        """
        now = _now_iso()
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT user_id, expires_at, consumed_at
                    FROM auth_tokens
                    WHERE token_hash = ? AND purpose = ?
                    """,
                    (token_hash, purpose),
                ).fetchone()
                if (
                    row is None
                    or row["consumed_at"] is not None
                    or row["expires_at"] <= now
                ):
                    connection.rollback()
                    return None
                connection.execute(
                    "UPDATE auth_tokens SET consumed_at = ? WHERE token_hash = ?",
                    (now, token_hash),
                )
                connection.commit()
                return int(row["user_id"])
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    # ------------------------------------------------------- refresh sessions
    def create_refresh_session(
        self, token_hash: str, *, user_id: int, expires_at: datetime
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO refresh_sessions (
                    token_hash, user_id, expires_at, revoked_at, created_at
                ) VALUES (?, ?, ?, NULL, ?)
                """,
                (token_hash, user_id, expires_at.isoformat(), _now_iso()),
            )
            connection.commit()

    def get_active_refresh_user(self, token_hash: str) -> int | None:
        now = _now_iso()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT user_id FROM refresh_sessions
                WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
        return int(row["user_id"]) if row is not None else None

    def revoke_refresh_session(self, token_hash: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE refresh_sessions SET revoked_at = ? WHERE token_hash = ?",
                (_now_iso(), token_hash),
            )
            connection.commit()

    def revoke_all_for_user(self, user_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE refresh_sessions SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (_now_iso(), user_id),
            )
            connection.commit()
