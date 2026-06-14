"""Password hashing helpers (bcrypt).

bcrypt is already a transitive dependency of the project. A work factor of 12
is a reasonable default for interactive logins. bcrypt silently truncates
inputs longer than 72 bytes, so callers must enforce a max password length
(see ``api.auth.schemas``).
"""

from __future__ import annotations

import bcrypt

_ROUNDS = 12


def hash_password(password: str) -> str:
    """Return a salted bcrypt hash for ``password``."""
    salt = bcrypt.gensalt(rounds=_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True iff ``password`` matches ``password_hash``.

    Never raises: a malformed stored hash returns False rather than erroring.
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """Return True if ``password_hash`` was produced with a weaker work factor."""
    try:
        cost = int(password_hash.split("$")[2])
    except (IndexError, ValueError):
        return True
    return cost < _ROUNDS
