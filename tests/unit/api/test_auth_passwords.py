"""Tests for bcrypt password hashing helpers."""

from api.auth.passwords import hash_password, needs_rehash, verify_password


def test_hash_and_verify_roundtrip():
    h = hash_password("Sup3r$ecret")
    assert h != "Sup3r$ecret"
    assert verify_password("Sup3r$ecret", h) is True


def test_verify_rejects_wrong_password():
    h = hash_password("Sup3r$ecret")
    assert verify_password("nope", h) is False


def test_verify_handles_garbage_hash():
    assert verify_password("x", "not-a-hash") is False


def test_hashes_are_salted_and_unique():
    assert hash_password("same") != hash_password("same")


def test_needs_rehash_for_weak_or_garbage():
    assert needs_rehash("not-a-hash") is True
    # A real cost-12 hash should not need rehashing.
    assert needs_rehash(hash_password("whatever-secret")) is False
