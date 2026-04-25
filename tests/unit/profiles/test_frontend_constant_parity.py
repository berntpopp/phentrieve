"""Cross-language parity check between frontend defaults and Python config."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DEFAULTS = REPO_ROOT / "frontend" / "src" / "constants" / "defaults.js"


# Map of frontend constant -> Python config attribute that should match.
# Whitelist: DEFAULT_NUM_RESULTS_PER_CHUNK is intentional UI divergence (3 vs backend 10).
PARITY_MAP: dict[str, str] = {
    "DEFAULT_NUM_RESULTS": "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD": "MIN_SIMILARITY_THRESHOLD",
    "DEFAULT_SPLIT_THRESHOLD": "DEFAULT_SPLITTING_THRESHOLD",
    "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD": "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD",
    "DEFAULT_AGGREGATED_TERM_CONFIDENCE": "DEFAULT_MIN_CONFIDENCE_AGGREGATED",
    "DEFAULT_WINDOW_SIZE": "DEFAULT_WINDOW_SIZE_TOKENS",
    "DEFAULT_STEP_SIZE": "DEFAULT_STEP_SIZE_TOKENS",
    "DEFAULT_MIN_SEGMENT_LENGTH": "DEFAULT_MIN_SEGMENT_LENGTH_WORDS",
}

# Intentional UI-specific divergences. Documented in docs/user-guide/configuration-profiles.md.
INTENTIONAL_DIVERGENCES: set[str] = {"DEFAULT_NUM_RESULTS_PER_CHUNK"}


def _extract_js_numeric_constants(text: str) -> dict[str, float]:
    """Parse `export const NAME = NUMBER;` lines."""
    pattern = re.compile(r"export\s+const\s+(\w+)\s*=\s*([\d.]+)\s*;")
    return {name: float(value) for name, value in pattern.findall(text)}


def test_frontend_defaults_file_exists():
    assert FRONTEND_DEFAULTS.exists(), f"{FRONTEND_DEFAULTS} not found"


@pytest.mark.parametrize("frontend_name,python_name", list(PARITY_MAP.items()))
def test_constant_parity(frontend_name: str, python_name: str):
    from phentrieve import config as phentrieve_config

    js_constants = _extract_js_numeric_constants(FRONTEND_DEFAULTS.read_text())
    assert frontend_name in js_constants, (
        f"Frontend constant {frontend_name} missing from {FRONTEND_DEFAULTS}. "
        f"Expected to match Python {python_name}."
    )
    python_value = getattr(phentrieve_config, python_name, None)
    assert python_value is not None, f"Python config has no attribute {python_name}"
    assert js_constants[frontend_name] == float(python_value), (
        f"Parity mismatch: frontend {frontend_name} = {js_constants[frontend_name]}, "
        f"Python {python_name} = {python_value}. "
        f"Update one to match the other."
    )


def test_intentional_divergences_documented():
    js_constants = _extract_js_numeric_constants(FRONTEND_DEFAULTS.read_text())
    for name in INTENTIONAL_DIVERGENCES:
        assert name in js_constants, (
            f"Whitelisted-divergence {name} not found in frontend; "
            f"remove from INTENTIONAL_DIVERGENCES if it was deleted."
        )
