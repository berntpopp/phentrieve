import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from api.llm_quota import (
    DailyQuotaStore,
    QuotaExceededError,
    hash_subject_key,
    quota_reset_at_iso,
    resolve_subject_ip,
)


def test_daily_quota_store_counts_successful_requests_only(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    outcome = store.reserve(
        subject_key="hash1",
        usage_date_utc="2026-04-15",
    )

    assert outcome.quota_used == 1
    assert outcome.quota_remaining == 2


def test_daily_quota_store_persists_record_success_across_fresh_reads(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    store.reserve(subject_key="hash1", usage_date_utc="2026-04-15")

    refreshed = store.get_status(subject_key="hash1", usage_date_utc="2026-04-15")

    assert refreshed.quota_used == 1
    assert refreshed.quota_remaining == 2


def test_daily_quota_store_prevents_overrun_when_limit_is_reached(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=1)

    first = store.reserve(subject_key="hash1", usage_date_utc="2026-04-15")
    assert first.quota_used == 1

    with pytest.raises(QuotaExceededError):
        store.reserve(subject_key="hash1", usage_date_utc="2026-04-15")

    refreshed = store.get_status(subject_key="hash1", usage_date_utc="2026-04-15")
    assert refreshed.quota_used == 1
    assert refreshed.quota_remaining == 0


def test_resolve_subject_ip_requires_trusted_proxy_for_forwarded_ip():
    subject_ip = resolve_subject_ip(
        client_host="172.18.0.10",
        x_forwarded_for="203.0.113.5",
        trusted_proxy_cidrs=[],
    )

    assert subject_ip is None


def test_resolve_subject_ip_uses_forwarded_ip_from_trusted_proxy():
    subject_ip = resolve_subject_ip(
        client_host="172.18.0.10",
        x_forwarded_for="203.0.113.5",
        trusted_proxy_cidrs=["172.16.0.0/12"],
    )

    assert subject_ip == "203.0.113.5"


def test_resolve_subject_ip_ignores_invalid_trusted_proxy_cidrs(caplog):
    caplog.set_level("WARNING")
    invalid_cidr = "not-a-cidr"

    subject_ip = resolve_subject_ip(
        client_host="172.18.0.10",
        x_forwarded_for="203.0.113.5",
        trusted_proxy_cidrs=[invalid_cidr, "172.16.0.0/12"],
    )

    assert subject_ip == "203.0.113.5"
    assert "Ignoring invalid trusted proxy CIDR entry at index 0" in caplog.text
    assert "length=10" in caplog.text
    assert "ValueError" in caplog.text
    assert invalid_cidr not in caplog.text


def test_daily_quota_store_enables_sqlite_wal_mode(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    with store._connect() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert int(synchronous) == 1


def test_quota_reset_at_iso_returns_next_midnight_utc():
    assert quota_reset_at_iso("2026-04-22") == "2026-04-23T00:00:00+00:00"


def test_concurrent_reservations_admit_exactly_one_at_limit_one(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=1)

    def attempt(_: int) -> bool:
        try:
            store.reserve(subject_key="hash1", usage_date_utc="2026-04-15")
        except QuotaExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(8)))

    assert outcomes.count(True) == 1
    status = store.get_status(subject_key="hash1", usage_date_utc="2026-04-15")
    assert status.quota_used == 1


def test_release_refunds_a_reservation_and_floors_at_zero(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=1)
    store.reserve(subject_key="hash1", usage_date_utc="2026-04-15")

    store.release(subject_key="hash1", usage_date_utc="2026-04-15")
    store.release(subject_key="hash1", usage_date_utc="2026-04-15")

    assert (
        store.get_status(subject_key="hash1", usage_date_utc="2026-04-15").quota_used
        == 0
    )
    # The refunded unit can be reserved again.
    assert (
        store.reserve(subject_key="hash1", usage_date_utc="2026-04-15").quota_used == 1
    )


def test_hash_subject_key_is_keyed_hmac(monkeypatch):
    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_PEPPER", "pepper-a")
    first = hash_subject_key("203.0.113.5")

    assert first != hashlib.sha256(b"203.0.113.5").hexdigest()
    assert hash_subject_key(" 203.0.113.5 ") == first

    monkeypatch.setattr("api.config.PHENTRIEVE_LLM_QUOTA_PEPPER", "pepper-b")
    assert hash_subject_key("203.0.113.5") != first
