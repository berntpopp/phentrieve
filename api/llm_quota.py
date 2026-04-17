from __future__ import annotations

import hashlib
import ipaddress
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CREATE_LLM_DAILY_QUOTA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_daily_quota (
    subject_key TEXT NOT NULL,
    usage_date_utc TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (subject_key, usage_date_utc)
)
"""

__all__ = [
    "CREATE_LLM_DAILY_QUOTA_TABLE_SQL",
    "DailyQuotaStore",
    "QuotaExceededError",
    "QuotaStatus",
    "hash_subject_key",
    "resolve_subject_ip",
]


@dataclass(frozen=True, slots=True)
class QuotaStatus:
    subject_key: str
    usage_date_utc: str
    quota_used: int
    quota_limit: int
    quota_remaining: int

    def to_detail(self) -> dict[str, Any]:
        return {
            "quota_used": self.quota_used,
            "quota_limit": self.quota_limit,
            "quota_remaining": self.quota_remaining,
            "usage_date_utc": self.usage_date_utc,
        }


class QuotaExceededError(Exception):
    def __init__(
        self,
        *,
        quota_used: int,
        quota_limit: int,
        quota_remaining: int,
        usage_date_utc: str,
        message: str | None = None,
    ) -> None:
        self.quota_used = quota_used
        self.quota_limit = quota_limit
        self.quota_remaining = quota_remaining
        self.usage_date_utc = usage_date_utc
        self.message = message or "LLM daily quota exhausted."
        super().__init__(self.message)

    def to_detail(self) -> dict[str, Any]:
        return {
            "error_message": self.message,
            "quota_used": self.quota_used,
            "quota_limit": self.quota_limit,
            "quota_remaining": self.quota_remaining,
            "usage_date_utc": self.usage_date_utc,
        }


def _utc_now_date_string() -> str:
    return datetime.now(UTC).date().isoformat()


def hash_subject_key(subject_ip: str) -> str:
    normalized_subject = subject_ip.strip()
    return hashlib.sha256(normalized_subject.encode("utf-8")).hexdigest()


def _parse_ip(
    value: str | None,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if value is None:
        return None
    try:
        return ipaddress.ip_address(value.strip())
    except ValueError:
        return None


def _parse_trusted_networks(
    trusted_proxy_cidrs: list[str],
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for cidr in trusted_proxy_cidrs:
        cidr_value = cidr.strip()
        if not cidr_value:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr_value, strict=False))
        except ValueError as exc:
            logger.warning(
                "Ignoring invalid trusted proxy CIDR entry %r: %s",
                cidr_value,
                exc,
            )
            continue
    return networks


def _ip_is_trusted(
    candidate: ipaddress.IPv4Address | ipaddress.IPv6Address,
    trusted_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    return any(candidate in network for network in trusted_networks)


def resolve_subject_ip(
    *,
    client_host: str | None,
    x_forwarded_for: str | None,
    trusted_proxy_cidrs: list[str],
) -> str | None:
    client_ip = _parse_ip(client_host)
    if client_ip is None:
        return None

    trusted_networks = _parse_trusted_networks(trusted_proxy_cidrs)
    if trusted_proxy_cidrs and not trusted_networks:
        return client_ip.compressed

    forwarded_for = [
        candidate.strip()
        for candidate in (x_forwarded_for or "").split(",")
        if candidate.strip()
    ]

    if not forwarded_for:
        if _ip_is_trusted(client_ip, trusted_networks):
            return None
        return client_ip.compressed

    if not _ip_is_trusted(client_ip, trusted_networks):
        return None

    parsed_forwarded = []
    for candidate in forwarded_for:
        candidate_ip = _parse_ip(candidate)
        if candidate_ip is None:
            return None
        parsed_forwarded.append(candidate_ip)

    if not parsed_forwarded:
        return None

    if any(
        not _ip_is_trusted(candidate_ip, trusted_networks)
        for candidate_ip in parsed_forwarded[1:]
    ):
        return None

    subject_ip = parsed_forwarded[0]
    if _ip_is_trusted(subject_ip, trusted_networks):
        return None
    return subject_ip.compressed


class DailyQuotaStore:
    def __init__(self, db_path: Path, daily_limit: int) -> None:
        self.db_path = Path(db_path)
        self.daily_limit = daily_limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(CREATE_LLM_DAILY_QUOTA_TABLE_SQL)

    def _read_success_count(
        self,
        *,
        subject_key: str,
        usage_date_utc: str,
    ) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT success_count
                FROM llm_daily_quota
                WHERE subject_key = ? AND usage_date_utc = ?
                """,
                (subject_key, usage_date_utc),
            ).fetchone()
        if row is None:
            return 0
        return int(row[0])

    def _build_status(
        self,
        *,
        subject_key: str,
        usage_date_utc: str,
        success_count: int,
    ) -> QuotaStatus:
        quota_limit = max(self.daily_limit, 0)
        quota_remaining = max(quota_limit - success_count, 0)
        return QuotaStatus(
            subject_key=subject_key,
            usage_date_utc=usage_date_utc,
            quota_used=success_count,
            quota_limit=quota_limit,
            quota_remaining=quota_remaining,
        )

    def get_status(
        self,
        *,
        subject_key: str,
        usage_date_utc: str,
    ) -> QuotaStatus:
        success_count = self._read_success_count(
            subject_key=subject_key,
            usage_date_utc=usage_date_utc,
        )
        return self._build_status(
            subject_key=subject_key,
            usage_date_utc=usage_date_utc,
            success_count=success_count,
        )

    def record_success(
        self,
        *,
        subject_key: str,
        usage_date_utc: str,
    ) -> QuotaStatus:
        updated_at = datetime.now(UTC).isoformat()
        quota_limit = max(self.daily_limit, 0)
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT success_count
                    FROM llm_daily_quota
                    WHERE subject_key = ? AND usage_date_utc = ?
                    """,
                    (subject_key, usage_date_utc),
                ).fetchone()
                current_success_count = int(row[0]) if row is not None else 0

                if current_success_count >= quota_limit:
                    raise QuotaExceededError(
                        quota_used=current_success_count,
                        quota_limit=quota_limit,
                        quota_remaining=0,
                        usage_date_utc=usage_date_utc,
                    )

                next_success_count = current_success_count + 1
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO llm_daily_quota (
                            subject_key,
                            usage_date_utc,
                            success_count,
                            updated_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            subject_key,
                            usage_date_utc,
                            next_success_count,
                            updated_at,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE llm_daily_quota
                        SET success_count = ?, updated_at = ?
                        WHERE subject_key = ? AND usage_date_utc = ?
                        """,
                        (
                            next_success_count,
                            updated_at,
                            subject_key,
                            usage_date_utc,
                        ),
                    )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

        success_count = next_success_count
        return self._build_status(
            subject_key=subject_key,
            usage_date_utc=usage_date_utc,
            success_count=success_count,
        )

    def get_today_status(self, *, subject_key: str) -> QuotaStatus:
        return self.get_status(
            subject_key=subject_key,
            usage_date_utc=_utc_now_date_string(),
        )

    def record_today_success(self, *, subject_key: str) -> QuotaStatus:
        return self.record_success(
            subject_key=subject_key,
            usage_date_utc=_utc_now_date_string(),
        )
