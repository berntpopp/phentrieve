"""Authentication and account management for the Phentrieve API.

Opt-in (gated by ``PHENTRIEVE_AUTH_ENABLED``). Provides password hashing,
JWT access tokens, rotating refresh sessions, email delivery, a SQLite user
store, and the auth router. A verified, logged-in user receives a higher
daily full-text LLM quota than anonymous (IP-keyed) clients.
"""
