# Config Resolver Refactoring - Implementation Plan

**Status**: 🔄 In Progress
**Created**: 2025-11-19
**Estimated Effort**: 3.5 hours
**Risk Level**: Low

## Executive Summary

Refactor duplicated chunking configuration resolution logic from CLI and API into a shared utility module. This addresses DRY violations while maintaining KISS principles and avoiding over-engineering.

**Current Problem**: ~230 lines of duplicated config resolution logic between CLI and API
**Proposed Solution**: Extract common logic to `phentrieve/text_processing/config_resolver.py` (~120 lines)
**Net Effect**: 40% reduction in code (230 → 155 lines), improved maintainability

## Motivation

### Current State Analysis

**Code Duplication** (DRY Violation):
- `phentrieve/cli/utils.py::resolve_chunking_pipeline_config()` - 170 lines
- `api/routers/text_processing_router.py::_get_chunking_config_for_api()` - 60 lines
- Both perform identical operations: strategy lookup, config file loading, parameter overrides

**Common Pattern**:
```python
# Both CLI and API do:
1. Load config from file (if provided) → CLI only
2. Look up predefined strategy by name
3. Apply parameter overrides (window_size, step_size, threshold, min_segment_length)
4. Return list[dict] configuration
```

### Why NOT Factory Pattern?

Rejected Factory Pattern (see `plan/01-active/PIPELINE-FACTORY-PATTERN-REVIEW.md`):
- ❌ Over-engineering: 450 lines (300 abstraction + 150 logic) vs. current 230 lines
- ❌ Violates KISS: Adds unnecessary abstraction layers
- ❌ Violates SRP: Factory would handle 6 different responsibilities
- ❌ Not appropriate: Only ONE pipeline type exists, no runtime type selection needed

### Why Shared Config Resolver?

✅ **Addresses DRY**: Consolidates duplicated strategy lookup logic
✅ **Maintains KISS**: Simple utility function, no complex abstractions
✅ **Follows SRP**: Single responsibility (config resolution only)
✅ **Transparent**: Config resolution remains explicit step before pipeline creation
✅ **Net Code Reduction**: 230 → 155 lines (40% reduction)

## Implementation Phases

### Phase 1: Create Shared Config Resolver (2 hours)

**Goal**: Extract common config resolution logic to new module

**Create**: `phentrieve/text_processing/config_resolver.py`

```python
"""Shared configuration resolution logic for text processing pipelines."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from phentrieve.config import (
    get_detailed_chunking_config,
    get_semantic_chunking_config,
    get_simple_chunking_config,
    get_sliding_window_cleaned_config,
    get_sliding_window_config_with_params,
    get_sliding_window_punct_cleaned_config,
    get_sliding_window_punct_conj_cleaned_config,
)

logger = logging.getLogger(__name__)


class ChunkingConfigError(Exception):
    """Raised when chunking configuration resolution fails."""
    pass


def resolve_chunking_config(
    strategy_name: Optional[str] = None,
    config_file: Optional[Path] = None,
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    threshold: Optional[float] = None,
    min_segment_length: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Resolve chunking configuration from multiple sources.

    Priority:
    1. Config file (if provided)
    2. Strategy name lookup
    3. Raise error if neither provided

    After resolving base config, applies parameter overrides if provided.

    Args:
        strategy_name: Name of predefined strategy (e.g., "simple", "detailed", "semantic")
        config_file: Path to YAML/JSON config file
        window_size: Override window size (tokens) for sliding window strategies
        step_size: Override step size (tokens) for sliding window strategies
        threshold: Override similarity threshold (0.0-1.0) for splitting
        min_segment_length: Override minimum segment length (words)

    Returns:
        List of chunking pipeline configuration dictionaries

    Raises:
        ChunkingConfigError: If neither strategy_name nor config_file provided,
                            or if config file doesn't exist or has invalid format

    Examples:
        >>> # Load from config file
        >>> config = resolve_chunking_config(config_file=Path("config.yaml"))

        >>> # Use predefined strategy
        >>> config = resolve_chunking_config(strategy_name="simple")

        >>> # Use strategy with parameter overrides
        >>> config = resolve_chunking_config(
        ...     strategy_name="detailed",
        ...     window_size=7,
        ...     threshold=0.6
        ... )
    """
    # Step 1: Resolve base configuration
    base_config = _resolve_base_config(strategy_name, config_file)

    # Step 2: Apply parameter overrides
    if any(
        param is not None
        for param in [window_size, step_size, threshold, min_segment_length]
    ):
        base_config = _apply_parameter_overrides(
            base_config,
            window_size=window_size,
            step_size=step_size,
            threshold=threshold,
            min_segment_length=min_segment_length,
        )

    return base_config


def _resolve_base_config(
    strategy_name: Optional[str],
    config_file: Optional[Path],
) -> list[dict[str, Any]]:
    """
    Resolve base configuration from file or strategy name.

    Args:
        strategy_name: Strategy name
        config_file: Config file path

    Returns:
        Base configuration list

    Raises:
        ChunkingConfigError: If config resolution fails
    """
    # Priority 1: Config file
    if config_file is not None:
        return _load_config_from_file(config_file)

    # Priority 2: Strategy name
    if strategy_name is not None:
        return _get_strategy_config(strategy_name)

    # Error: Neither provided
    raise ChunkingConfigError(
        "Must provide either strategy_name or config_file for chunking configuration"
    )


def _load_config_from_file(config_file: Path) -> list[dict[str, Any]]:
    """
    Load chunking configuration from YAML or JSON file.

    Args:
        config_file: Path to config file

    Returns:
        Configuration list from file

    Raises:
        ChunkingConfigError: If file doesn't exist or has invalid format
    """
    if not config_file.exists():
        raise ChunkingConfigError(f"Config file not found: {config_file}")

    suffix = config_file.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ChunkingConfigError(
            f"Unsupported config file format: {suffix}. Use .json, .yaml, or .yml"
        )

    try:
        with open(config_file, encoding="utf-8") as f:
            if suffix == ".json":
                config_data = json.load(f)
            else:  # .yaml or .yml
                config_data = yaml.safe_load(f)

        chunking_pipeline = config_data.get("chunking_pipeline")
        if chunking_pipeline is None:
            raise ChunkingConfigError(
                f"Config file missing 'chunking_pipeline' key: {config_file}"
            )

        return chunking_pipeline

    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ChunkingConfigError(f"Failed to parse config file {config_file}: {e}")


def _get_strategy_config(strategy_name: str) -> list[dict[str, Any]]:
    """
    Get predefined chunking strategy configuration by name.

    Args:
        strategy_name: Strategy name (e.g., "simple", "detailed", "semantic")

    Returns:
        Strategy configuration list

    Note:
        Returns default config for unknown strategy names (with warning).
    """
    strategy_name = strategy_name.lower()

    # Map strategy names to config functions
    strategy_map = {
        "simple": get_simple_chunking_config,
        "detailed": get_detailed_chunking_config,
        "semantic": get_semantic_chunking_config,
        "sliding_window": lambda: get_sliding_window_config_with_params(),
        "sliding_window_cleaned": get_sliding_window_cleaned_config,
        "sliding_window_punct_cleaned": get_sliding_window_punct_cleaned_config,
        "sliding_window_punct_conj_cleaned": get_sliding_window_punct_conj_cleaned_config,
    }

    if strategy_name in strategy_map:
        config_func = strategy_map[strategy_name]
        return list(config_func())
    else:
        logger.warning(
            f"Unknown strategy '{strategy_name}', using sliding_window_punct_conj_cleaned"
        )
        return list(get_sliding_window_punct_conj_cleaned_config())


def _apply_parameter_overrides(
    config: list[dict[str, Any]],
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    threshold: Optional[float] = None,
    min_segment_length: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Apply parameter overrides to sliding window components in config.

    Modifies config in-place by updating parameters for any components
    with type='sliding_window'.

    Args:
        config: Chunking pipeline configuration (modified in-place)
        window_size: Window size in tokens (if provided)
        step_size: Step size in tokens (if provided)
        threshold: Similarity threshold for splitting (if provided)
        min_segment_length: Minimum segment length in words (if provided)

    Returns:
        Modified configuration (same object as input)
    """
    # Build override dict with only provided parameters
    overrides = {}
    if window_size is not None:
        overrides["window_size_tokens"] = window_size
    if step_size is not None:
        overrides["step_size_tokens"] = step_size
    if threshold is not None:
        overrides["splitting_threshold"] = threshold
    if min_segment_length is not None:
        overrides["min_split_segment_length_words"] = min_segment_length

    # Apply overrides to all sliding_window components
    if overrides:
        for component in config:
            if component.get("type") == "sliding_window" and "config" in component:
                component["config"].update(overrides)

    return config
```

**Estimated Time**: 1.5 hours (implementation + docstrings)

---

### Phase 2: Refactor CLI (30 minutes)

**Goal**: Replace CLI-specific implementation with shared resolver

**Modify**: `phentrieve/cli/utils.py`

**Before** (170 lines):
```python
def resolve_chunking_pipeline_config(
    chunking_pipeline_config_file: Optional[Path],
    strategy_arg: str,
    window_size: int = 3,
    step_size: int = 1,
    threshold: float = 0.5,
    min_segment_length: int = 2,
) -> list[dict]:
    # 170 lines of config loading and strategy lookup...
    # Lots of if/elif chains
    # File loading logic
    # Parameter override logic
```

**After** (~25 lines):
```python
from phentrieve.text_processing.config_resolver import (
    ChunkingConfigError,
    resolve_chunking_config,
)

def resolve_chunking_pipeline_config(
    chunking_pipeline_config_file: Optional[Path],
    strategy_arg: str,
    window_size: int = 3,
    step_size: int = 1,
    threshold: float = 0.5,
    min_segment_length: int = 2,
) -> list[dict]:
    """
    Resolve the chunking pipeline configuration from a file or a strategy name.

    This is a CLI-specific wrapper around the shared config resolver.

    Args:
        chunking_pipeline_config_file: Optional path to a config file
        strategy_arg: Strategy name to use if no config file is provided
        window_size: Window size for sliding window chunker (tokens)
        step_size: Step size for sliding window chunker (tokens)
        threshold: Similarity threshold for sliding window chunker
        min_segment_length: Minimum segment length for sliding window chunker (words)

    Returns:
        List of chunker configurations

    Raises:
        typer.Exit: If config resolution fails
    """
    try:
        return resolve_chunking_config(
            strategy_name=strategy_arg,
            config_file=chunking_pipeline_config_file,
            window_size=window_size,
            step_size=step_size,
            threshold=threshold,
            min_segment_length=min_segment_length,
        )
    except ChunkingConfigError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
```

**Estimated Time**: 30 minutes (refactoring + testing)

---

### Phase 3: Refactor API (30 minutes)

**Goal**: Replace API-specific implementation with shared resolver

**Modify**: `api/routers/text_processing_router.py`

**Before** (~60 lines):
```python
def _apply_sliding_window_params(...):
    # 10 lines

def _get_chunking_config_for_api(request: TextProcessingRequest):
    # 60 lines of strategy lookup and parameter application
```

**After** (~15 lines):
```python
from phentrieve.text_processing.config_resolver import resolve_chunking_config

def _get_chunking_config_for_api(
    request: TextProcessingRequest,
) -> list[dict[str, Any]]:
    """
    Get chunking configuration based on request strategy and parameters.

    This is an API-specific wrapper around the shared config resolver.

    Args:
        request: Text processing request with strategy and parameters

    Returns:
        Chunking pipeline configuration list
    """
    strategy_name = (
        request.chunking_strategy.lower()
        if request.chunking_strategy
        else "sliding_window_punct_conj_cleaned"
    )

    # Extract parameters with defaults
    ws = request.window_size if request.window_size is not None else 7
    ss = request.step_size if request.step_size is not None else 1
    th = request.split_threshold if request.split_threshold is not None else 0.5
    msl = request.min_segment_length if request.min_segment_length is not None else 3

    logger.debug(
        f"API: Building config for '{strategy_name}': ws={ws}, ss={ss}, th={th}, msl={msl}"
    )

    # Use shared resolver
    return resolve_chunking_config(
        strategy_name=strategy_name,
        config_file=None,  # API doesn't support config files
        window_size=ws,
        step_size=ss,
        threshold=th,
        min_segment_length=msl,
    )
```

**Note**: Remove `_apply_sliding_window_params()` helper (no longer needed)

**Estimated Time**: 30 minutes (refactoring + testing)

---

### Phase 4: Write Comprehensive Tests (1 hour)

**Goal**: Test shared resolver with 100% coverage

**Create**: `tests/unit/text_processing/test_config_resolver.py`

**Test Coverage**:

```python
"""Tests for shared chunking configuration resolver."""

import json
from pathlib import Path

import pytest
import yaml

from phentrieve.text_processing.config_resolver import (
    ChunkingConfigError,
    resolve_chunking_config,
)

pytestmark = pytest.mark.unit


class TestConfigFileLoading:
    """Test configuration loading from files."""

    def test_load_valid_yaml_config(self, tmp_path):
        """Test loading valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "chunking_pipeline": [
                {"type": "simple_sentence", "config": {}}
            ]
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = resolve_chunking_config(config_file=config_file)
        assert len(result) == 1
        assert result[0]["type"] == "simple_sentence"

    def test_load_valid_json_config(self, tmp_path):
        """Test loading valid JSON config file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "chunking_pipeline": [
                {"type": "simple_sentence", "config": {}}
            ]
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = resolve_chunking_config(config_file=config_file)
        assert len(result) == 1
        assert result[0]["type"] == "simple_sentence"

    def test_config_file_not_found(self, tmp_path):
        """Test error when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(ChunkingConfigError, match="not found"):
            resolve_chunking_config(config_file=nonexistent)

    def test_unsupported_file_format(self, tmp_path):
        """Test error for unsupported file format."""
        bad_file = tmp_path / "config.txt"
        bad_file.touch()
        with pytest.raises(ChunkingConfigError, match="Unsupported config file format"):
            resolve_chunking_config(config_file=bad_file)

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error for invalid YAML syntax."""
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: syntax: [")
        with pytest.raises(ChunkingConfigError, match="Failed to parse"):
            resolve_chunking_config(config_file=config_file)

    def test_missing_chunking_pipeline_key(self, tmp_path):
        """Test error when config missing chunking_pipeline key."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"other_key": "value"}, f)
        with pytest.raises(ChunkingConfigError, match="missing 'chunking_pipeline'"):
            resolve_chunking_config(config_file=config_file)


class TestStrategyResolution:
    """Test resolution of predefined strategies."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple",
            "detailed",
            "semantic",
            "sliding_window",
            "sliding_window_cleaned",
            "sliding_window_punct_cleaned",
            "sliding_window_punct_conj_cleaned",
        ],
    )
    def test_known_strategies_resolve(self, strategy_name):
        """Test that all known strategies can be resolved."""
        config = resolve_chunking_config(strategy_name=strategy_name)
        assert isinstance(config, list)
        assert len(config) > 0

    def test_unknown_strategy_uses_default(self, caplog):
        """Test unknown strategy falls back to default with warning."""
        config = resolve_chunking_config(strategy_name="unknown_strategy")
        assert isinstance(config, list)
        assert any("Unknown strategy" in record.message for record in caplog.records)

    def test_strategy_name_case_insensitive(self):
        """Test strategy names are case-insensitive."""
        config_lower = resolve_chunking_config(strategy_name="simple")
        config_upper = resolve_chunking_config(strategy_name="SIMPLE")
        assert config_lower == config_upper


class TestParameterOverrides:
    """Test parameter override application."""

    def test_window_size_override(self):
        """Test window size override is applied."""
        config = resolve_chunking_config(
            strategy_name="sliding_window",
            window_size=10,
        )
        # Find sliding window component
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 10

    def test_all_parameters_override(self):
        """Test all parameters can be overridden together."""
        config = resolve_chunking_config(
            strategy_name="detailed",
            window_size=15,
            step_size=3,
            threshold=0.7,
            min_segment_length=5,
        )
        # Find sliding window component
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 15
        assert sw_component["config"]["step_size_tokens"] == 3
        assert sw_component["config"]["splitting_threshold"] == 0.7
        assert sw_component["config"]["min_split_segment_length_words"] == 5

    def test_partial_parameter_override(self):
        """Test that partial parameter overrides work."""
        config = resolve_chunking_config(
            strategy_name="semantic",
            window_size=12,  # Only override window_size
        )
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        assert sw_component["config"]["window_size_tokens"] == 12
        # Other params should have strategy defaults
        assert "step_size_tokens" in sw_component["config"]

    def test_no_overrides_preserves_defaults(self):
        """Test config without overrides uses strategy defaults."""
        config = resolve_chunking_config(strategy_name="detailed")
        sw_component = next(c for c in config if c["type"] == "sliding_window")
        # Should have default values
        assert "window_size_tokens" in sw_component["config"]


class TestErrorHandling:
    """Test error handling."""

    def test_neither_strategy_nor_file_raises_error(self):
        """Test error when neither strategy nor file provided."""
        with pytest.raises(ChunkingConfigError, match="Must provide either"):
            resolve_chunking_config()

    def test_both_strategy_and_file_prefers_file(self, tmp_path):
        """Test that file takes priority over strategy."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "chunking_pipeline": [
                {"type": "custom", "config": {}}
            ]
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = resolve_chunking_config(
            strategy_name="simple",
            config_file=config_file,
        )
        # Should use file (custom), not strategy (simple)
        assert result[0]["type"] == "custom"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_cli_style_usage(self):
        """Test usage pattern from CLI code."""
        # Simulates old CLI pattern
        config = resolve_chunking_config(
            strategy_name="detailed",
            window_size=3,
            step_size=1,
            threshold=0.5,
            min_segment_length=2,
        )
        assert isinstance(config, list)

    def test_api_style_usage(self):
        """Test usage pattern from API code."""
        # Simulates old API pattern
        config = resolve_chunking_config(
            strategy_name="sliding_window_punct_conj_cleaned",
            window_size=7,
            step_size=1,
            threshold=0.5,
            min_segment_length=3,
        )
        assert isinstance(config, list)
```

**Estimated Time**: 1 hour (write tests + achieve 100% coverage)

---

### Phase 5: Update CLI Tests (15 minutes)

**Goal**: Update existing CLI tests to work with refactored code

**Modify**: Tests that mock `resolve_chunking_pipeline_config`

**Pattern**: Tests should continue to work because function signature unchanged

**Verification**:
```bash
pytest tests/unit/cli/ -v
```

**Estimated Time**: 15 minutes (verify + fix if needed)

---

### Phase 6: Update API Tests (15 minutes)

**Goal**: Update existing API tests to work with refactored code

**Modify**: Tests that use `_get_chunking_config_for_api`

**Pattern**: Tests should continue to work because function behavior unchanged

**Verification**:
```bash
pytest tests/unit/api/ -v
pytest tests_new/e2e/test_api_workflow.py -v
```

**Estimated Time**: 15 minutes (verify + fix if needed)

---

### Phase 7: Documentation (15 minutes)

**Goal**: Update documentation to reflect new architecture

**Update**:
1. **CLAUDE.md**: Add section on config resolver pattern
2. **plan/STATUS.md**: Mark refactoring complete
3. **Docstrings**: Ensure all functions have comprehensive docs

**Estimated Time**: 15 minutes

---

## Success Criteria

### Functional Requirements

- ✅ All 425+ existing tests pass
- ✅ New config resolver module has 100% test coverage
- ✅ CLI config resolution behavior unchanged (backward compatible)
- ✅ API config resolution behavior unchanged (backward compatible)
- ✅ All predefined strategies continue to work
- ✅ Config file loading (YAML/JSON) continues to work
- ✅ Parameter overrides continue to work

### Code Quality Requirements

- ✅ 0 linting errors (`make check`)
- ✅ 0 type errors (`make typecheck-fast`)
- ✅ All docstrings complete with examples
- ✅ All error cases have clear error messages

### Architectural Requirements

- ✅ DRY: No duplicated config resolution logic
- ✅ KISS: Simple utility function, no over-engineering
- ✅ SRP: Config resolver has single responsibility
- ✅ Net code reduction: ~40% fewer lines of code
- ✅ Transparency: Config resolution remains explicit

### Performance Requirements

- ✅ No performance regression (config resolution is fast)
- ✅ Test suite runs in same time or faster

---

## Rollback Plan

If issues arise during or after implementation:

1. **Immediate**: Revert commit with `git revert`
2. **Verification**: Run full test suite to confirm rollback
3. **Analysis**: Investigate failure root cause
4. **Re-attempt**: Fix issues and re-apply with additional tests

The refactoring is low-risk because:
- Function signatures remain unchanged (backward compatible)
- Behavior is identical (same logic, just consolidated)
- Comprehensive test coverage catches regressions

---

## Timeline

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Create shared config resolver | 1.5h | 1.5h |
| 2 | Refactor CLI | 30min | 2.0h |
| 3 | Refactor API | 30min | 2.5h |
| 4 | Write comprehensive tests | 1h | 3.5h |
| 5 | Update CLI tests | 15min | 3.75h |
| 6 | Update API tests | 15min | 4.0h |
| 7 | Documentation | 15min | 4.25h |

**Total Estimated Time**: ~4.25 hours

---

## Implementation Checklist

### Pre-Implementation

- [x] Review Pipeline Factory Pattern review document
- [x] Read existing CLI and API implementations
- [x] Draft detailed implementation plan
- [ ] Create feature branch `refactor/config-resolver`

### Implementation

- [ ] Phase 1: Create `phentrieve/text_processing/config_resolver.py`
- [ ] Phase 2: Refactor `phentrieve/cli/utils.py`
- [ ] Phase 3: Refactor `api/routers/text_processing_router.py`
- [ ] Phase 4: Create `tests/unit/text_processing/test_config_resolver.py`
- [ ] Phase 5: Verify CLI tests pass
- [ ] Phase 6: Verify API tests pass
- [ ] Phase 7: Update documentation

### Verification

- [ ] Run `make check` (linting)
- [ ] Run `make typecheck-fast` (type checking)
- [ ] Run `make test` (all tests)
- [ ] Verify test coverage for new module (100%)
- [ ] Manual smoke test: CLI text processing command
- [ ] Manual smoke test: API text processing endpoint

### Finalization

- [ ] Create PR with comprehensive description
- [ ] Wait for CI to pass
- [ ] Merge to main

---

## Notes

**Design Decisions**:
- Chose utility function over class (KISS principle)
- Preserved all existing function signatures (backward compatibility)
- Used exception-based error handling (Pythonic)
- Applied strategy pattern for config function lookup (clean code)

**Alternative Approaches Considered**:
- ❌ Factory Pattern: Rejected (over-engineering)
- ❌ Singleton Service: Rejected (unnecessary state)
- ✅ Shared Utility: Accepted (simple, effective)

**Related Documents**:
- `plan/01-active/PIPELINE-FACTORY-PATTERN-REVIEW.md` - Initial review
- `plan/STATUS.md` - Project status tracking
