import pytest

from api.llm_quota import (
    DailyQuotaStore,
    QuotaExceededError,
    quota_reset_at_iso,
    resolve_subject_ip,
)


def test_daily_quota_store_counts_successful_requests_only(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    outcome = store.record_success(
        subject_key="hash1",
        usage_date_utc="2026-04-15",
    )

    assert outcome.quota_used == 1
    assert outcome.quota_remaining == 2


def test_daily_quota_store_persists_record_success_across_fresh_reads(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    store.record_success(subject_key="hash1", usage_date_utc="2026-04-15")

    refreshed = store.get_status(subject_key="hash1", usage_date_utc="2026-04-15")

    assert refreshed.quota_used == 1
    assert refreshed.quota_remaining == 2


def test_daily_quota_store_prevents_overrun_when_limit_is_reached(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=1)

    first = store.record_success(subject_key="hash1", usage_date_utc="2026-04-15")
    assert first.quota_used == 1

    with pytest.raises(QuotaExceededError):
        store.record_success(subject_key="hash1", usage_date_utc="2026-04-15")

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
