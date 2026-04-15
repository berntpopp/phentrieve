from api.llm_quota import DailyQuotaStore, resolve_subject_ip


def test_daily_quota_store_counts_successful_requests_only(tmp_path):
    store = DailyQuotaStore(tmp_path / "quota.db", daily_limit=3)

    outcome = store.record_success(
        subject_key="hash1",
        usage_date_utc="2026-04-15",
    )

    assert outcome.quota_used == 1
    assert outcome.quota_remaining == 2


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
