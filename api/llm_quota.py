from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import api.config as api_config

if TYPE_CHECKING:
    from starlette.requests import Request

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
    "QuotaStoreUnavailableError",
    "anonymous_subject_key",
    "hash_subject_key",
    "quota_enforced",
    "quota_reset_at_iso",
    "read_llm_quota",
    "release_llm_quota",
    "reserve_llm_quota",
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
            "quota_reset_at": quota_reset_at_iso(self.usage_date_utc),
        }


class QuotaStoreUnavailableError(RuntimeError):
    """The quota DB could not be read or written; callers answer 503."""


def quota_reset_at_iso(usage_date_utc: str) -> str:
    current_day = date.fromisoformat(usage_date_utc)
    next_day = current_day + timedelta(days=1)
    return datetime.combine(next_day, time(0, 0, tzinfo=UTC)).isoformat()


def _utc_now_date_string() -> str:
    return datetime.now(UTC).date().isoformat()


_PROCESS_HASH_KEY = secrets.token_bytes(32)
_warned_process_hash_key = False


def _subject_hash_key() -> bytes:
    global _warned_process_hash_key
    secret = (
        api_config.PHENTRIEVE_LLM_QUOTA_PEPPER or api_config.PHENTRIEVE_AUTH_JWT_SECRET
    )
    if secret:
        return secret.encode("utf-8")
    if not _warned_process_hash_key:
        _warned_process_hash_key = True
        logger.warning(
            "Neither PHENTRIEVE_LLM_QUOTA_PEPPER nor PHENTRIEVE_AUTH_JWT_SECRET is "
            "set; using a random per-process key for LLM quota subject hashes. "
            "Quota counts reset on restart and are not shared across workers."
        )
    return _PROCESS_HASH_KEY


def hash_subject_key(subject_ip: str) -> str:
    # Keyed HMAC: a plain SHA-256 of an IPv4 address is reversible by brute force.
    normalized_subject = subject_ip.strip()
    return hmac.new(
        _subject_hash_key(), normalized_subject.encode("utf-8"), hashlib.sha256
    ).hexdigest()


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
    for index, cidr in enumerate(trusted_proxy_cidrs):
        cidr_value = cidr.strip()
        if not cidr_value:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr_value, strict=False))
        except ValueError as exc:
            logger.warning(
                "Ignoring invalid trusted proxy CIDR entry at index %d (length=%d): %s",
                index,
                len(cidr_value),
                type(exc).__name__,
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

    def reserve(
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

    def release(self, *, subject_key: str, usage_date_utc: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE llm_daily_quota
                SET success_count = MAX(success_count - 1, 0), updated_at = ?
                WHERE subject_key = ? AND usage_date_utc = ?
                """,
                (datetime.now(UTC).isoformat(), subject_key, usage_date_utc),
            )
            connection.commit()


def quota_enforced() -> bool:
    """Whether LLM quota is enforced.

    Tri-state ``PHENTRIEVE_LLM_QUOTA_ENFORCE``: explicit ``true``/``false``
    override, otherwise the legacy behaviour (enforce only in production).
    """
    override = api_config.PHENTRIEVE_LLM_QUOTA_ENFORCE
    if override == "true":
        return True
    if override == "false":
        return False
    return api_config.PHENTRIEVE_ENV.strip().lower() == "production"


def anonymous_subject_key(request: Request) -> str | None:
    """Hashed trusted client IP of an HTTP request, or None if untrusted."""
    subject_ip = resolve_subject_ip(
        client_host=request.client.host if request.client else None,
        x_forwarded_for=request.headers.get("x-forwarded-for"),
        trusted_proxy_cidrs=[
            cidr.strip()
            for cidr in api_config.PHENTRIEVE_TRUSTED_PROXY_CIDRS.split(",")
            if cidr.strip()
        ],
    )
    return hash_subject_key(subject_ip) if subject_ip is not None else None


def _quota_store(daily_limit: int) -> DailyQuotaStore:
    return DailyQuotaStore(
        db_path=Path(api_config.PHENTRIEVE_LLM_QUOTA_DB_PATH),
        daily_limit=daily_limit,
    )


_STORE_UNAVAILABLE_MESSAGE = (
    "Unable to evaluate LLM quota state. Verify "
    "PHENTRIEVE_LLM_QUOTA_DB_PATH and filesystem permissions."
)


def read_llm_quota(*, subject_key: str, daily_limit: int) -> QuotaStatus:
    """Today's quota status without consuming it."""
    try:
        return _quota_store(daily_limit).get_status(
            subject_key=subject_key, usage_date_utc=_utc_now_date_string()
        )
    except Exception as exc:  # noqa: BLE001
        raise QuotaStoreUnavailableError(_STORE_UNAVAILABLE_MESSAGE) from exc


def reserve_llm_quota(*, subject_key: str, daily_limit: int) -> QuotaStatus:
    """Atomically take one unit of today's quota before the LLM call.

    Raises QuotaExceededError when exhausted, QuotaStoreUnavailableError when
    the store fails. The returned status already counts this reservation.
    """
    try:
        return _quota_store(daily_limit).reserve(
            subject_key=subject_key, usage_date_utc=_utc_now_date_string()
        )
    except QuotaExceededError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QuotaStoreUnavailableError(_STORE_UNAVAILABLE_MESSAGE) from exc


def release_llm_quota(reservation: QuotaStatus) -> None:
    """Give back a reservation whose LLM call failed before producing a result."""
    try:
        _quota_store(reservation.quota_limit).release(
            subject_key=reservation.subject_key,
            usage_date_utc=reservation.usage_date_utc,
        )
    except Exception:  # noqa: BLE001 - already on an error path; keep the original
        logger.exception("Failed to release LLM quota reservation")
