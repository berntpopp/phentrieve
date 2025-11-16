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

# Default fallback values for API configuration
_DEFAULT_SERVER_PORT = 8734
_DEFAULT_SERVER_HOST = "0.0.0.0"
_DEFAULT_SERVER_RELOAD = False  # Production default
_DEFAULT_SERVER_WORKERS = 1

_DEFAULT_LOG_LEVEL = "INFO"

_DEFAULT_SBERT_TIMEOUT = 60
_DEFAULT_CROSS_ENCODER_TIMEOUT = 10

_DEFAULT_CORS_ORIGINS = ["http://localhost:5734", "http://localhost:8734"]
_DEFAULT_CORS_CREDENTIALS = True
_DEFAULT_CORS_METHODS = ["*"]
_DEFAULT_CORS_HEADERS = ["*"]

_DEFAULT_DATA_ROOT_DIR = "../data"


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
            except Exception as e:
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
CROSS_ENCODER_LOAD_TIMEOUT: float = float(
    os.getenv(
        "PHENTRIEVE_CROSS_ENCODER_LOAD_TIMEOUT",
        get_api_config_value(
            "model_loading", _DEFAULT_CROSS_ENCODER_TIMEOUT, "cross_encoder_timeout"
        ),
    )
)

# CORS settings
ALLOWED_ORIGINS: list[str] = (
    os.getenv("ALLOWED_ORIGINS", "").split(",")
    if os.getenv("ALLOWED_ORIGINS")
    else get_api_config_value("cors", _DEFAULT_CORS_ORIGINS, "allowed_origins")
)
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
