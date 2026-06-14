"""Optional startup seeding of a pre-verified account for local testing.

Controlled entirely by config: when ``PHENTRIEVE_AUTH_SEED_EMAIL`` and
``PHENTRIEVE_AUTH_SEED_PASSWORD`` are both set (and auth is enabled), a verified
account with those credentials is ensured at startup. The seed credentials are
authoritative: an existing seed account is re-verified and its password is reset
to match config, so changing the env always takes effect. Leave the env vars
empty in production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import api.config as api_config
from api.auth.passwords import hash_password
from api.auth.store import EmailExistsError, UserStore

logger = logging.getLogger(__name__)

__all__ = ["seed_user", "seed_user_from_config"]


def seed_user(store: UserStore, *, email: str, password: str) -> None:
    """Idempotently ensure a verified account with the given credentials."""
    password_hash = hash_password(password)
    existing = store.get_by_email(email)
    if existing is None:
        try:
            created = store.create_user(email=email, password_hash=password_hash)
        except EmailExistsError:
            # Lost a race; fall through to the update path.
            existing = store.get_by_email(email)
            if existing is None:
                return
        else:
            store.mark_verified(created.id)
            return
    # Existing account: make the seed credentials authoritative.
    store.update_password(existing.id, password_hash)
    if not existing.is_verified:
        store.mark_verified(existing.id)
    store.reset_failed_login(existing.id)


def seed_user_from_config() -> None:
    """Seed the configured dev account if enabled. Never raises."""
    if not api_config.PHENTRIEVE_AUTH_ENABLED:
        return
    email = api_config.PHENTRIEVE_AUTH_SEED_EMAIL.strip()
    password = api_config.PHENTRIEVE_AUTH_SEED_PASSWORD
    if not email or not password:
        return
    try:
        store = UserStore(Path(api_config.PHENTRIEVE_AUTH_DB_PATH))
        seed_user(store, email=email, password=password)
        logger.info("Seeded verified dev account: %s", email)
    except Exception:  # noqa: BLE001 - seeding must never crash startup
        logger.exception("Failed to seed dev account")
