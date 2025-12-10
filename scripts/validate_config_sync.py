#!/usr/bin/env python3
"""
Validate that configuration defaults are synchronized across Python and Frontend.

This script checks that:
1. Python config.py constants match expected values
2. Frontend QueryInterface.vue defaults match Python constants
3. API schema defaults match Python constants

Usage:
    python scripts/validate_config_sync.py
    make config-validate

Exit codes:
    0 - All configurations synchronized
    1 - Configuration mismatch detected
"""

import re
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Expected values (single source of truth for validation)
# These MUST match phentrieve/config.py constants
EXPECTED_CONFIG = {
    "window_size_tokens": 3,
    "step_size_tokens": 1,
    "splitting_threshold": 0.5,
    "min_segment_length_words": 2,
    "chunk_retrieval_threshold": 0.7,
    "min_confidence_aggregated": 0.75,
    "chunking_strategy": "sliding_window_punct_conj_cleaned",
}


def check_python_config() -> tuple[bool, list[str]]:
    """Verify phentrieve/config.py has correct constants."""
    errors: list[str] = []

    try:
        # Import from config.py
        sys.path.insert(0, str(PROJECT_ROOT))
        from phentrieve.config import (
            DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
            DEFAULT_CHUNKING_STRATEGY,
            DEFAULT_MIN_CONFIDENCE_AGGREGATED,
            DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
            DEFAULT_SPLITTING_THRESHOLD,
            DEFAULT_STEP_SIZE_TOKENS,
            DEFAULT_WINDOW_SIZE_TOKENS,
        )

        checks = [
            (
                "window_size_tokens",
                DEFAULT_WINDOW_SIZE_TOKENS,
                EXPECTED_CONFIG["window_size_tokens"],
            ),
            (
                "step_size_tokens",
                DEFAULT_STEP_SIZE_TOKENS,
                EXPECTED_CONFIG["step_size_tokens"],
            ),
            (
                "splitting_threshold",
                DEFAULT_SPLITTING_THRESHOLD,
                EXPECTED_CONFIG["splitting_threshold"],
            ),
            (
                "min_segment_length_words",
                DEFAULT_MIN_SEGMENT_LENGTH_WORDS,
                EXPECTED_CONFIG["min_segment_length_words"],
            ),
            (
                "chunk_retrieval_threshold",
                DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
                EXPECTED_CONFIG["chunk_retrieval_threshold"],
            ),
            (
                "min_confidence_aggregated",
                DEFAULT_MIN_CONFIDENCE_AGGREGATED,
                EXPECTED_CONFIG["min_confidence_aggregated"],
            ),
            (
                "chunking_strategy",
                DEFAULT_CHUNKING_STRATEGY,
                EXPECTED_CONFIG["chunking_strategy"],
            ),
        ]

        for name, actual, expected in checks:
            if actual != expected:
                errors.append(f"  {name}: expected {expected}, got {actual}")

    except ImportError as e:
        errors.append(f"  Import error: {e}")

    return len(errors) == 0, errors


def check_frontend_defaults() -> tuple[bool, list[str]]:
    """Verify frontend QueryInterface.vue has matching defaults."""
    errors: list[str] = []

    vue_file = PROJECT_ROOT / "frontend/src/components/QueryInterface.vue"
    if not vue_file.exists():
        errors.append("  QueryInterface.vue not found")
        return False, errors

    content = vue_file.read_text()

    # Extract data() section defaults using regex
    patterns = {
        "window_size_tokens": r"windowSize:\s*(\d+)",
        "step_size_tokens": r"stepSize:\s*(\d+)",
        "splitting_threshold": r"splitThreshold:\s*([\d.]+)",
        "min_segment_length_words": r"minSegmentLength:\s*(\d+)",
        "chunk_retrieval_threshold": r"chunkRetrievalThreshold:\s*([\d.]+)",
        "min_confidence_aggregated": r"aggregatedTermConfidence:\s*([\d.]+)",
        "chunking_strategy": r"chunkingStrategy:\s*['\"]([^'\"]+)['\"]",
    }

    for name, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            actual_str = match.group(1)
            expected = EXPECTED_CONFIG[name]

            # Convert to appropriate type for comparison
            if isinstance(expected, int):
                actual = int(actual_str)
            elif isinstance(expected, float):
                actual = float(actual_str)
            else:
                actual = actual_str

            if actual != expected:
                errors.append(f"  {name}: expected {expected}, got {actual}")
        else:
            errors.append(f"  {name}: pattern not found in Vue file")

    return len(errors) == 0, errors


def check_api_schema_defaults() -> tuple[bool, list[str]]:
    """Verify API schema has matching defaults."""
    errors: list[str] = []

    schema_file = PROJECT_ROOT / "api/schemas/text_processing_schemas.py"
    if not schema_file.exists():
        errors.append("  text_processing_schemas.py not found")
        return False, errors

    content = schema_file.read_text()

    # Check that schema imports from config.py (DRY pattern)
    required_imports = [
        "DEFAULT_WINDOW_SIZE_TOKENS",
        "DEFAULT_STEP_SIZE_TOKENS",
        "DEFAULT_SPLITTING_THRESHOLD",
        "DEFAULT_MIN_SEGMENT_LENGTH_WORDS",
        "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD",
        "DEFAULT_MIN_CONFIDENCE_AGGREGATED",
        "DEFAULT_CHUNKING_STRATEGY",
    ]

    for import_name in required_imports:
        if import_name not in content:
            errors.append(f"  Missing import: {import_name}")

    return len(errors) == 0, errors


def check_api_router_defaults() -> tuple[bool, list[str]]:
    """Verify API router uses config constants."""
    errors: list[str] = []

    router_file = PROJECT_ROOT / "api/routers/text_processing_router.py"
    if not router_file.exists():
        errors.append("  text_processing_router.py not found")
        return False, errors

    content = router_file.read_text()

    # Check that router imports from config.py
    required_imports = [
        "DEFAULT_WINDOW_SIZE_TOKENS",
        "DEFAULT_STEP_SIZE_TOKENS",
        "DEFAULT_SPLITTING_THRESHOLD",
        "DEFAULT_MIN_SEGMENT_LENGTH_WORDS",
        "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD",
        "DEFAULT_MIN_CONFIDENCE_AGGREGATED",
    ]

    for import_name in required_imports:
        if import_name not in content:
            errors.append(f"  Missing import: {import_name}")

    # Check for hardcoded values that should be constants
    hardcoded_patterns = [
        (r"chunk_retrieval_threshold\s*=\s*0\.\d+[^,\n]*,", "chunk_retrieval_threshold"),
        (r"min_confidence[^=]*=\s*0\.\d+[^,\n]*,", "min_confidence"),
    ]

    for pattern, name in hardcoded_patterns:
        if re.search(pattern, content):
            # Check it's not using the constant
            if f"DEFAULT_{name.upper()}" not in content:
                errors.append(f"  Hardcoded value found for {name}")

    return len(errors) == 0, errors


def main() -> int:
    """Run all configuration synchronization checks."""
    print("=" * 60)
    print("Configuration Synchronization Validator")
    print("=" * 60)
    print()

    all_passed = True
    checks = [
        ("Python config.py", check_python_config),
        ("Frontend QueryInterface.vue", check_frontend_defaults),
        ("API Schema", check_api_schema_defaults),
        ("API Router", check_api_router_defaults),
    ]

    for name, check_fn in checks:
        passed, errors = check_fn()
        status = "\u2713" if passed else "\u2717"
        print(f"{status} {name}")

        if errors:
            for error in errors:
                print(error)
            all_passed = False

    print()
    if all_passed:
        print("\u2713 All configuration sources are synchronized!")
        return 0
    else:
        print("\u2717 Configuration mismatch detected!")
        print()
        print("To fix:")
        print("  1. Update mismatched values to match EXPECTED_CONFIG")
        print("  2. Or update EXPECTED_CONFIG if intentional change")
        print()
        print("Expected values (single source of truth):")
        for key, value in EXPECTED_CONFIG.items():
            print(f"  {key}: {value}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
