"""
API-specific configuration management.

This module handles loading and accessing configuration for the Phentrieve FastAPI backend.
It is completely independent from the CLI configuration in phentrieve.yaml.
"""

import functools
import logging
import os
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# Public API exports - these are imported by other modules
__all__ = [
    "API_PORT",
    "API_HOST",
    "API_RELOAD",
    "API_WORKERS",
    "LOG_LEVEL",
    "SBERT_LOAD_TIMEOUT",
    "ALLOWED_ORIGINS",
    "CORS_ALLOW_CREDENTIALS",
    "CORS_ALLOW_METHODS",
    "CORS_ALLOW_HEADERS",
    "DATA_ROOT_DIR",
    "PHENTRIEVE_ENV",
    "PHENTRIEVE_TRUSTED_PROXY_CIDRS",
    "PHENTRIEVE_LLM_DAILY_LIMIT",
    "PHENTRIEVE_LLM_QUOTA_DB_PATH",
    "PHENTRIEVE_PUBLIC_HOSTED_MODE",
    "PHENTRIEVE_REQUIRE_RESEARCH_ACK",
    "PHENTRIEVE_AUTH_ENABLED",
    "PHENTRIEVE_AUTH_JWT_SECRET",
    "PHENTRIEVE_AUTH_DB_PATH",
    "PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS",
    "PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS",
    "PHENTRIEVE_AUTH_COOKIE_SECURE",
    "PHENTRIEVE_AUTH_COOKIE_SAMESITE",
    "PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS",
    "PHENTRIEVE_AUTH_LOCKOUT_SECONDS",
    "PHENTRIEVE_AUTH_SEED_EMAIL",
    "PHENTRIEVE_AUTH_SEED_PASSWORD",
    "PHENTRIEVE_EMAIL_BACKEND",
    "PHENTRIEVE_EMAIL_FROM",
    "PHENTRIEVE_SMTP_HOST",
    "PHENTRIEVE_SMTP_PORT",
    "PHENTRIEVE_SMTP_USERNAME",
    "PHENTRIEVE_SMTP_PASSWORD",
    "PHENTRIEVE_SMTP_TLS",
    "PHENTRIEVE_PUBLIC_BASE_URL",
    "PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT",
    "PHENTRIEVE_LLM_QUOTA_ENFORCE",
    "get_api_config_value",
]

# Default fallback values for API configuration
_DEFAULT_SERVER_PORT = 8734
_DEFAULT_SERVER_HOST = "0.0.0.0"
_DEFAULT_SERVER_RELOAD = False  # Production default
_DEFAULT_SERVER_WORKERS = 1

_DEFAULT_LOG_LEVEL = "INFO"

_DEFAULT_SBERT_TIMEOUT = 60

_DEFAULT_CORS_ORIGINS = ["http://localhost:5734", "http://localhost:8734"]
_DEFAULT_CORS_CREDENTIALS = True
_DEFAULT_CORS_METHODS = ["*"]
_DEFAULT_CORS_HEADERS = ["*"]

_DEFAULT_DATA_ROOT_DIR = "../data"


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a conservative default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back to a default."""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s; using default %d", name, default)
        return default


@functools.lru_cache(maxsize=1)
def _load_api_yaml_config() -> dict[Any, Any]:
    """
    Load API configuration from api.yaml.

    Searches for api.yaml in the following locations (in order):
    1. api/api.yaml (relative to current working directory)
    2. Same directory as this module

    Returns:
        dict: Configuration dictionary from YAML, or empty dict if not found
    """
    # Try to import yaml, fall back to empty dict if not available
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Using default API configuration.")
        return {}

    # Search locations for api.yaml
    search_paths = [
        Path("api/api.yaml"),  # From project root
        Path(__file__).parent / "api.yaml",  # Same directory as this module
    ]

    for config_path in search_paths:
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        logger.warning(
                            f"API config file {config_path} is empty. Using defaults."
                        )
                        return {}
                    logger.info(f"Loaded API configuration from: {config_path}")
                    return cast(dict[Any, Any], config)
            except (yaml.YAMLError, OSError) as e:
                logger.warning(
                    f"Failed to load API config from {config_path}: {e}. Using defaults."
                )
                return {}

    logger.info("No api.yaml found. Using default API configuration.")
    return {}


def get_api_config_value(key: str, default: Any, nested_key: str | None = None) -> Any:
    """
    Get an API configuration value from YAML config with fallback to default.

    Args:
        key: Top-level key in YAML config
        default: Default value if key not found in config
        nested_key: Optional nested key for hierarchical configs

    Returns:
        Configuration value from YAML, or default if not found

    Examples:
        >>> get_api_config_value("server", 8734, "port")
        >>> get_api_config_value("logging", "INFO", "level")
    """
    config = _load_api_yaml_config()

    if key not in config:
        return default

    if nested_key is None:
        return config.get(key, default)

    # Handle nested keys
    if isinstance(config[key], dict):
        return config[key].get(nested_key, default)

    return default


# =============================================================================
# Public API Configuration Constants
# =============================================================================
# These constants are loaded from api.yaml if present, otherwise use
# the fallback values defined above.
# Environment variables take precedence over YAML config.

# Server settings
API_PORT: int = int(
    os.getenv("API_PORT", get_api_config_value("server", _DEFAULT_SERVER_PORT, "port"))
)
API_HOST: str = os.getenv(
    "API_HOST", get_api_config_value("server", _DEFAULT_SERVER_HOST, "host")
)
API_RELOAD: bool = (
    os.getenv("RELOAD", "").lower() == "true"
    if os.getenv("RELOAD")
    else get_api_config_value("server", _DEFAULT_SERVER_RELOAD, "reload")
)
API_WORKERS: int = int(
    os.getenv(
        "API_WORKERS",
        get_api_config_value("server", _DEFAULT_SERVER_WORKERS, "workers"),
    )
)

# Logging
LOG_LEVEL: str = os.getenv(
    "LOG_LEVEL", get_api_config_value("logging", _DEFAULT_LOG_LEVEL, "level")
).upper()

# Model loading timeouts (in seconds)
SBERT_LOAD_TIMEOUT: float = float(
    os.getenv(
        "PHENTRIEVE_SBERT_LOAD_TIMEOUT",
        get_api_config_value("model_loading", _DEFAULT_SBERT_TIMEOUT, "sbert_timeout"),
    )
)


# CORS settings
# Priority: ALLOWED_ORIGINS env var (complete override) > api.yaml > defaults
# CORS_EXTRA_ORIGINS env var appends to the resolved list (for easy production config)
def _resolve_allowed_origins() -> list[str]:
    """
    Resolve CORS allowed origins with support for additive configuration.

    Resolution order:
    1. If ALLOWED_ORIGINS env var is set: use it as complete override
    2. Otherwise: use api.yaml config or defaults
    3. If CORS_EXTRA_ORIGINS env var is set: append those origins (comma-separated)

    This allows production to simply set CORS_EXTRA_ORIGINS without losing defaults.
    """
    # Start with base origins (env override or config/defaults)
    if os.getenv("ALLOWED_ORIGINS"):
        base_origins = [
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        ]
    else:
        base_origins = get_api_config_value(
            "cors", _DEFAULT_CORS_ORIGINS, "allowed_origins"
        )

    # Append extra origins if provided (production use case)
    extra_origins_env = os.getenv("CORS_EXTRA_ORIGINS", "")
    if extra_origins_env:
        extra_origins = [
            origin.strip() for origin in extra_origins_env.split(",") if origin.strip()
        ]
        # Use set to deduplicate while preserving order
        seen = set(base_origins)
        for origin in extra_origins:
            if origin not in seen:
                base_origins.append(origin)
                seen.add(origin)

    return base_origins


ALLOWED_ORIGINS: list[str] = _resolve_allowed_origins()
CORS_ALLOW_CREDENTIALS: bool = get_api_config_value(
    "cors", _DEFAULT_CORS_CREDENTIALS, "allow_credentials"
)
CORS_ALLOW_METHODS: list[str] = get_api_config_value(
    "cors", _DEFAULT_CORS_METHODS, "allow_methods"
)
CORS_ALLOW_HEADERS: list[str] = get_api_config_value(
    "cors", _DEFAULT_CORS_HEADERS, "allow_headers"
)

# Data paths
DATA_ROOT_DIR: str = os.getenv(
    "PHENTRIEVE_DATA_ROOT_DIR",
    get_api_config_value("data", _DEFAULT_DATA_ROOT_DIR, "root_dir"),
)

PHENTRIEVE_ENV: str = os.getenv("PHENTRIEVE_ENV", "development")
PHENTRIEVE_TRUSTED_PROXY_CIDRS: str = os.getenv("PHENTRIEVE_TRUSTED_PROXY_CIDRS", "")
PHENTRIEVE_LLM_DAILY_LIMIT: int = int(os.getenv("PHENTRIEVE_LLM_DAILY_LIMIT", "5"))
PHENTRIEVE_LLM_QUOTA_DB_PATH: str = os.getenv(
    "PHENTRIEVE_LLM_QUOTA_DB_PATH",
    "../data/app/llm_quota.db",
)
PHENTRIEVE_PUBLIC_HOSTED_MODE: bool = _env_bool("PHENTRIEVE_PUBLIC_HOSTED_MODE", False)
PHENTRIEVE_REQUIRE_RESEARCH_ACK: bool = _env_bool(
    "PHENTRIEVE_REQUIRE_RESEARCH_ACK", False
)

# =============================================================================
# Authentication and accounts
# =============================================================================
# Auth is opt-in. When disabled, the API behaves exactly as before (anonymous,
# IP-keyed quota only). Enable locally for testing; configure secrets/SMTP for
# production.
PHENTRIEVE_AUTH_ENABLED: bool = _env_bool("PHENTRIEVE_AUTH_ENABLED", False)
PHENTRIEVE_AUTH_JWT_SECRET: str = os.getenv("PHENTRIEVE_AUTH_JWT_SECRET", "")
PHENTRIEVE_AUTH_DB_PATH: str = os.getenv(
    "PHENTRIEVE_AUTH_DB_PATH", "../data/app/users.db"
)
PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS: int = _env_int(
    "PHENTRIEVE_AUTH_ACCESS_TTL_SECONDS", 1800
)
PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS: int = _env_int(
    "PHENTRIEVE_AUTH_REFRESH_TTL_SECONDS", 1_209_600
)
PHENTRIEVE_AUTH_COOKIE_SECURE: bool = _env_bool("PHENTRIEVE_AUTH_COOKIE_SECURE", True)
PHENTRIEVE_AUTH_COOKIE_SAMESITE: str = os.getenv(
    "PHENTRIEVE_AUTH_COOKIE_SAMESITE", "lax"
).lower()
PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS: int = _env_int(
    "PHENTRIEVE_AUTH_MAX_FAILED_ATTEMPTS", 5
)
PHENTRIEVE_AUTH_LOCKOUT_SECONDS: int = _env_int("PHENTRIEVE_AUTH_LOCKOUT_SECONDS", 900)

# Optional dev convenience: seed a pre-verified account at startup so it can be
# used immediately for testing. Leave empty in production. Both must be set.
PHENTRIEVE_AUTH_SEED_EMAIL: str = os.getenv("PHENTRIEVE_AUTH_SEED_EMAIL", "")
PHENTRIEVE_AUTH_SEED_PASSWORD: str = os.getenv("PHENTRIEVE_AUTH_SEED_PASSWORD", "")

# Email delivery (console backend for local/dev/tests, smtp for production)
PHENTRIEVE_EMAIL_BACKEND: str = os.getenv("PHENTRIEVE_EMAIL_BACKEND", "console").lower()
PHENTRIEVE_EMAIL_FROM: str = os.getenv(
    "PHENTRIEVE_EMAIL_FROM", "noreply@phentrieve.org"
)
PHENTRIEVE_SMTP_HOST: str = os.getenv("PHENTRIEVE_SMTP_HOST", "")
PHENTRIEVE_SMTP_PORT: int = _env_int("PHENTRIEVE_SMTP_PORT", 587)
PHENTRIEVE_SMTP_USERNAME: str = os.getenv("PHENTRIEVE_SMTP_USERNAME", "")
PHENTRIEVE_SMTP_PASSWORD: str = os.getenv("PHENTRIEVE_SMTP_PASSWORD", "")
PHENTRIEVE_SMTP_TLS: str = os.getenv("PHENTRIEVE_SMTP_TLS", "starttls").lower()
PHENTRIEVE_PUBLIC_BASE_URL: str = os.getenv(
    "PHENTRIEVE_PUBLIC_BASE_URL", "http://localhost:5734"
)

# Quota: authenticated (verified) users get a higher daily LLM limit.
PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT: int = _env_int(
    "PHENTRIEVE_LLM_AUTHENTICATED_DAILY_LIMIT", 10
)
# Tri-state enforcement override:
#   "" (unset) -> enforce only when PHENTRIEVE_ENV == "production" (legacy)
#   "true"/"false" -> explicit on/off (useful for local testing)
PHENTRIEVE_LLM_QUOTA_ENFORCE: str = os.getenv(
    "PHENTRIEVE_LLM_QUOTA_ENFORCE", ""
).lower()
