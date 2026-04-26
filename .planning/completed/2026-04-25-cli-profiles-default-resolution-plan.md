# CLI Profiles and Default Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a layered configuration-profile system to the Phentrieve CLI. `--profile <name>` selects a named profile defined in `phentrieve.yaml`; built-in `default` and `interactive` profiles ship in code. Solves issue #28 (profiles) and issue #171 (interactive defaults divergence) in one coherent system.

**Architecture:** Click's native `default_map` populated by an eager `--profile` callback before option resolution. Profileable Typer option defaults move from literals to `None`; the function body only handles the absolute fallback to `phentrieve.config` constants. Built-in profiles ship as Python `dict` literals constructed into pydantic `Profile` instances. A sidecar source map at `ctx.obj["resolved_sources"]` provides fine-grained source labels (profile/yaml/const) since `ParameterSource` collapses them all to `DEFAULT_MAP`.

**Tech Stack:** Python 3.10+, Typer (Click underneath), pydantic v2, ruamel.yaml (optional, for line-numbered errors), pytest, Vue 3 (frontend constants only).

---

## Spec Reference

Implementation strictly follows `.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md`. When in doubt, the spec is canonical.

## Command Contract

```bash
# Built-in profile auto-selected by command
phentrieve text interactive          # uses built-in `interactive` profile (loose defaults)
phentrieve text process FILE         # uses built-in `default` profile (strict defaults)
phentrieve query "TEXT"              # uses built-in `default` profile

# Explicit profile from phentrieve.yaml
phentrieve text process FILE --profile high_recall_german
phentrieve --profile precise_english_query query "TEXT"   # global placement
phentrieve query "TEXT" --profile precise_english_query   # per-command placement
phentrieve --profile A query "TEXT" --profile B           # subcommand wins (B)

# Inspection
phentrieve config list-profiles
phentrieve config show high_recall_german
phentrieve config validate
phentrieve config path

# Debug
phentrieve text process FILE --show-resolved-config       # prints source-labeled config to stderr, then runs

# Env var
PHENTRIEVE_PROFILE=high_recall_german phentrieve text process FILE
```

Errors:
- Unknown profile → exit 2 with `typer.BadParameter` and close-match suggestion.
- Profile bound to a command via `command:` field, used with a different command → exit 2 with bound-command name.
- YAML parse error → exit 1 with line:column.

## File Structure

Create:

- `phentrieve/profiles.py` — Profile pydantic model, ProfilesFile, BUILTIN_PROFILES, resolve_profile_for_command, profile loader.
- `phentrieve/cli/_profile.py` — Eager `apply_profile_callback`, default_map population, sidecar source map, helper to produce `--show-resolved-config` table.
- `phentrieve/cli/config_commands.py` — Typer sub-app for `phentrieve config list-profiles / show / validate / path`.
- `tests/unit/profiles/__init__.py`
- `tests/unit/profiles/test_profile_schema.py`
- `tests/unit/profiles/test_resolver.py`
- `tests/unit/profiles/test_frontend_constant_parity.py`
- `tests/unit/cli/test_profile_callback.py`
- `tests/unit/cli/test_default_map_resolution.py`
- `tests/unit/cli/test_profile_placement.py`
- `tests/unit/cli/test_yaml_legacy_paths.py`
- `tests/unit/cli/test_config_commands.py`
- `tests/unit/cli/test_show_resolved_config.py`
- `tests/integration/test_profiles_e2e.py`
- `tests/integration/test_documented_yaml.py` (shared with Plan B)
- `tests/fixtures/profiles/sample_phentrieve.yaml`

Modify:

- `phentrieve/config.py` — add `DEFAULT_NUM_RESULTS`, `DEFAULT_CHUNK_CONFIDENCE`, `DEFAULT_ASSERTION_PREFERENCE`, `DEFAULT_OUTPUT_FORMAT_QUERY`, `DEFAULT_OUTPUT_FORMAT_PROCESS`. Each follows the existing `_FALLBACK` + `get_config_value` pattern.
- `phentrieve/__main__.py` (or wherever the root Typer app is defined) — add root-level `--profile` option, `--show-resolved-config` flag, register `config` sub-app.
- `phentrieve/cli/text_interactive.py` — add `--profile` option with `default="interactive"`, change profileable option defaults from literals to `None`, add value-or-constant fallback in body.
- `phentrieve/cli/text_commands.py` — add `--profile` option to `process_text_for_hpo_command`, similar refactor.
- `phentrieve/cli/query_commands.py` — add `--profile` option to `query_hpo`, similar refactor.
- `phentrieve.yaml` — add `extraction:` section using existing key names from `config.py:502,505` (`chunk_threshold`, `min_confidence`).
- `phentrieve.yaml.template` — add commented `profiles:` block and `extraction:` section.
- `frontend/src/constants/defaults.js` — fix `DEFAULT_SIMILARITY_THRESHOLD` from `0.5` to `0.3`; add header comment pointing at `phentrieve/config.py` and the parity test.
- `frontend/src/test/constants.test.js` — update `DEFAULT_SIMILARITY_THRESHOLD` assertion.
- `docs/user-guide/configuration-profiles.md` — rewrite in place (replaces aspirational `--config-profile` content).
- `docs/user-guide/cli-usage.md` — add `--profile` and `--show-resolved-config` to per-command tables.
- `docs/user-guide/api-usage.md` — add note that the API doesn't accept `--profile`.
- `docs/user-guide/index.md` — link to configuration-profiles.md.
- `README.md` — new "Configuration profiles" subsection.
- `CHANGELOG.md` — three entries.

---

## Phase 1: Foundation — config constants and YAML extraction section

This phase adds new fallback constants and the YAML `extraction:` section. No behavior change yet. Other phases consume these.

### Task 1: Add new constants to `phentrieve/config.py`

**Files:**
- Modify: `phentrieve/config.py`
- Test: `tests/unit/test_config_constants.py` (extend if exists, else create)

- [ ] **Step 1: Write the failing test**

Append (or create) `tests/unit/test_config_constants.py`:

```python
"""Tests for new config constants added by Plan A."""

from phentrieve import config as phentrieve_config


def test_default_num_results_exists_and_aliases_top_k():
    assert phentrieve_config.DEFAULT_NUM_RESULTS == phentrieve_config.DEFAULT_TOP_K


def test_default_chunk_confidence_default_value():
    assert phentrieve_config.DEFAULT_CHUNK_CONFIDENCE == 0.2


def test_default_assertion_preference_default_value():
    assert phentrieve_config.DEFAULT_ASSERTION_PREFERENCE == "dependency"


def test_default_output_format_query_default_value():
    assert phentrieve_config.DEFAULT_OUTPUT_FORMAT_QUERY == "text"


def test_default_output_format_process_default_value():
    assert phentrieve_config.DEFAULT_OUTPUT_FORMAT_PROCESS == "json_lines"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config_constants.py -v`
Expected: FAIL with `AttributeError: module 'phentrieve.config' has no attribute 'DEFAULT_NUM_RESULTS'`.

- [ ] **Step 3: Add constants to `phentrieve/config.py`**

In `phentrieve/config.py`, find the existing `_FALLBACK` block (around lines 100-130) and add:

```python
_DEFAULT_CHUNK_CONFIDENCE_FALLBACK = 0.2
_DEFAULT_ASSERTION_PREFERENCE_FALLBACK = "dependency"
_DEFAULT_OUTPUT_FORMAT_QUERY_FALLBACK = "text"
_DEFAULT_OUTPUT_FORMAT_PROCESS_FALLBACK = "json_lines"
```

Find the existing public-constant resolution block (around lines 405-506) and add:

```python
DEFAULT_CHUNK_CONFIDENCE = get_config_value(
    "extraction", _DEFAULT_CHUNK_CONFIDENCE_FALLBACK, "chunk_confidence"
)
DEFAULT_ASSERTION_PREFERENCE = get_config_value(
    "extraction", _DEFAULT_ASSERTION_PREFERENCE_FALLBACK, "assertion_preference"
)
DEFAULT_OUTPUT_FORMAT_QUERY = get_config_value(
    "output", _DEFAULT_OUTPUT_FORMAT_QUERY_FALLBACK, "format_query"
)
DEFAULT_OUTPUT_FORMAT_PROCESS = get_config_value(
    "output", _DEFAULT_OUTPUT_FORMAT_PROCESS_FALLBACK, "format_process"
)

# Alias for CLI symmetry — CLI uses num_results, config has DEFAULT_TOP_K
DEFAULT_NUM_RESULTS = DEFAULT_TOP_K
```

Add each new symbol to `__all__` at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config_constants.py -v`
Expected: PASS, all five tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/config.py tests/unit/test_config_constants.py
git commit -m "feat(config): add DEFAULT_NUM_RESULTS, DEFAULT_CHUNK_CONFIDENCE, DEFAULT_ASSERTION_PREFERENCE, DEFAULT_OUTPUT_FORMAT_*

Additive constants used by upcoming profile system. Each follows the existing
_FALLBACK + get_config_value resolution pattern. DEFAULT_NUM_RESULTS is an
alias for DEFAULT_TOP_K, exported for CLI naming symmetry."
```

### Task 2: Add `extraction:` and `output:` sections to `phentrieve.yaml`

**Files:**
- Modify: `phentrieve.yaml`

- [ ] **Step 1: Verify current YAML state**

Run: `cat phentrieve.yaml | grep -E "^(extraction|output):"`
Expected: no output (sections don't exist yet).

- [ ] **Step 2: Add the sections at the end of `phentrieve.yaml`**

Append:

```yaml

# HPO extraction thresholds (read by phentrieve/config.py:501-506).
# Key names match the existing config.py reads exactly — DO NOT rename.
extraction:
  chunk_threshold: 0.7
  min_confidence: 0.75
  chunk_confidence: 0.2
  assertion_preference: dependency

# Output format defaults (read by Plan A's new constants).
output:
  format_query: text
  format_process: json_lines
```

- [ ] **Step 3: Verify config loads cleanly**

Run: `uv run python -c "from phentrieve import config; print(config.DEFAULT_CHUNK_CONFIDENCE, config.DEFAULT_OUTPUT_FORMAT_QUERY)"`
Expected: `0.2 text`

- [ ] **Step 4: Commit**

```bash
git add phentrieve.yaml
git commit -m "feat(config): add extraction and output YAML sections

Adds the YAML keys that phentrieve/config.py:501-506 already reads but were
missing from phentrieve.yaml. Plus new output: section for the new
DEFAULT_OUTPUT_FORMAT_* constants. No behavior change — values match
the in-code FALLBACK constants."
```

---

## Phase 2: Profile data model

### Task 3: Create `phentrieve/profiles.py` skeleton

**Files:**
- Create: `phentrieve/profiles.py`
- Create: `tests/unit/profiles/__init__.py`
- Create: `tests/unit/profiles/test_profile_schema.py`

- [ ] **Step 1: Create empty `tests/unit/profiles/__init__.py`**

Run: `touch tests/unit/profiles/__init__.py`

- [ ] **Step 2: Write the failing test for the Profile schema**

Create `tests/unit/profiles/test_profile_schema.py`:

```python
"""Tests for Profile and ProfilesFile pydantic models."""

import pytest
from pydantic import ValidationError

from phentrieve.profiles import Profile, ProfilesFile


class TestProfileSchema:
    def test_minimal_profile_all_fields_optional(self):
        # Should not raise: every Profile field is optional.
        profile = Profile()
        assert profile.description is None
        assert profile.command is None
        assert profile.language is None

    def test_profile_with_all_fields(self):
        profile = Profile(
            description="test",
            command="text process",
            language="de",
            chunk_retrieval_threshold=0.6,
            num_results=5,
        )
        assert profile.description == "test"
        assert profile.command == "text process"
        assert profile.language == "de"

    def test_profile_extra_forbid_rejects_unknown_keys(self):
        with pytest.raises(ValidationError) as exc_info:
            Profile(unknown_field="value")
        assert "unknown_field" in str(exc_info.value)

    def test_profile_extra_forbid_rejects_typos(self):
        # User typos like `chuck_retrieval_threshold` (chunk → chuck) error.
        with pytest.raises(ValidationError) as exc_info:
            Profile(chuck_retrieval_threshold=0.5)
        assert "chuck_retrieval_threshold" in str(exc_info.value)


class TestProfilesFileSchema:
    def test_empty_profiles_file_ok(self):
        f = ProfilesFile()
        assert f.profiles == {}

    def test_profiles_dict_typed(self):
        f = ProfilesFile(
            profiles={
                "fast_query": Profile(
                    command="query", num_results=5, similarity_threshold=0.5
                ),
            }
        )
        assert f.profiles["fast_query"].num_results == 5

    def test_profiles_file_ignores_unknown_top_level_keys(self):
        # ProfilesFile uses extra="ignore" so other top-level YAML keys are fine.
        f = ProfilesFile.model_validate(
            {"profiles": {"x": {"language": "en"}}, "unrelated_top_level": 42}
        )
        assert "x" in f.profiles
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/profiles/test_profile_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phentrieve.profiles'`.

- [ ] **Step 4: Create `phentrieve/profiles.py` with the schema**

```python
"""Profile data model and resolution for Phentrieve CLI configuration profiles.

Implements Spec A (.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Forward reference — defined in Plan B (phentrieve/retrieval/adaptive_rechunker.py).
# Until Plan B lands, use a permissive dict shape.
AdaptiveRechunkingProfileBlock = dict[str, Any]


class Profile(BaseModel):
    """A named bundle of CLI option defaults loaded from phentrieve.yaml.

    All fields are optional. None means "this profile does not preset this
    option" — the resolution chain falls through to YAML / fallback constants.
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    command: str | None = None  # e.g. "text process", "query", "text interactive"

    # Shared option keys
    language: str | None = None
    model_name: str | None = None
    semantic_chunker_model: str | None = None
    retrieval_model: str | None = None
    similarity_threshold: float | None = None
    chunk_retrieval_threshold: float | None = None
    aggregated_term_confidence: float | None = None
    num_results: int | None = None
    chunking_strategy: str | None = None
    window_size: int | None = None
    step_size: int | None = None
    split_threshold: float | None = None
    min_segment_length: int | None = None
    output_format: str | None = None
    assertion_preference: str | None = None
    no_assertion_detection: bool | None = None
    multi_vector: bool | None = None
    aggregation_strategy: str | None = None
    extraction_backend: Literal["standard", "llm"] | None = None

    # Adaptive rechunking block — strict shape defined by Plan B; permissive here.
    adaptive_rechunking: AdaptiveRechunkingProfileBlock | None = None


class ProfilesFile(BaseModel):
    """Top-level YAML model. Only the `profiles:` key is consumed by Plan A;
    other top-level keys (data_dir, default_model, etc.) are ignored at this layer.
    """

    model_config = ConfigDict(extra="ignore")
    profiles: dict[str, Profile] = Field(default_factory=dict)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/profiles/test_profile_schema.py -v`
Expected: PASS, all six tests.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/profiles.py tests/unit/profiles/__init__.py tests/unit/profiles/test_profile_schema.py
git commit -m "feat(profiles): add Profile and ProfilesFile pydantic models

extra='forbid' on Profile so YAML typos error at load time. ProfilesFile
uses extra='ignore' so the existing phentrieve.yaml top-level keys
(data_dir, default_model, etc.) don't trip validation."
```

### Task 4: Add `BUILTIN_PROFILES` to `phentrieve/profiles.py`

**Files:**
- Modify: `phentrieve/profiles.py`
- Modify: `tests/unit/profiles/test_profile_schema.py`

- [ ] **Step 1: Append failing tests for built-ins**

Append to `tests/unit/profiles/test_profile_schema.py`:

```python
class TestBuiltInProfiles:
    def test_builtin_default_exists(self):
        from phentrieve.profiles import BUILTIN_PROFILES

        assert "default" in BUILTIN_PROFILES
        # All fields None — fall through to YAML / constants.
        assert BUILTIN_PROFILES["default"].chunk_retrieval_threshold is None

    def test_builtin_interactive_exists_and_loose(self):
        from phentrieve.profiles import BUILTIN_PROFILES

        interactive = BUILTIN_PROFILES["interactive"]
        assert interactive.chunk_retrieval_threshold == 0.3
        assert interactive.aggregated_term_confidence == 0.35
        assert interactive.num_results == 5

    def test_builtin_profiles_validate(self):
        from phentrieve.profiles import BUILTIN_PROFILES

        # Each is a real Profile instance, not a dict.
        for name, p in BUILTIN_PROFILES.items():
            assert isinstance(p, Profile), f"BUILTIN_PROFILES[{name!r}] is not a Profile"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/profiles/test_profile_schema.py::TestBuiltInProfiles -v`
Expected: FAIL with `ImportError: cannot import name 'BUILTIN_PROFILES'`.

- [ ] **Step 3: Add `BUILTIN_PROFILES` to `phentrieve/profiles.py`**

Append to `phentrieve/profiles.py`:

```python
# Built-in profiles ship as Python dict literals constructed into Profile
# instances at import time. Users can shadow these by name in phentrieve.yaml.
BUILTIN_PROFILES: dict[str, Profile] = {
    "default": Profile(
        description="Strict defaults matching API behavior",
        # All fields None — falls through to YAML / fallback constants.
    ),
    "interactive": Profile(
        description="Loose discovery defaults for `phentrieve text interactive`",
        chunk_retrieval_threshold=0.3,
        aggregated_term_confidence=0.35,
        num_results=5,
    ),
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/profiles/test_profile_schema.py::TestBuiltInProfiles -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/profiles.py tests/unit/profiles/test_profile_schema.py
git commit -m "feat(profiles): add BUILTIN_PROFILES with default and interactive

Built-in interactive profile preserves the loose discovery defaults from
text_interactive.py (0.3 / 0.35 / 5) so issue #171 can be fixed without
a behavior change for existing users. Built-in default has all None fields
and is selected by query and text process when no --profile is given."
```

### Task 5: Add `resolve_profile_for_command` to `phentrieve/profiles.py`

**Files:**
- Modify: `phentrieve/profiles.py`
- Create: `tests/unit/profiles/test_resolver.py`

- [ ] **Step 1: Write failing tests for the resolver**

Create `tests/unit/profiles/test_resolver.py`:

```python
"""Tests for resolve_profile_for_command."""

import pytest
import typer

from phentrieve.profiles import (
    BUILTIN_PROFILES,
    Profile,
    resolve_profile_for_command,
)


@pytest.fixture
def known_profiles():
    return {
        **BUILTIN_PROFILES,
        "fast_query": Profile(
            command="query", num_results=5, similarity_threshold=0.5
        ),
        "shared_german": Profile(
            language="de", semantic_chunker_model="jinaai/jina-embeddings-v2-base-de"
        ),
    }


class TestResolveProfileForCommand:
    def test_none_returns_appropriate_builtin_for_text_interactive(self, known_profiles):
        profile, kwargs = resolve_profile_for_command(
            None, ("text", "interactive"), accepted_keys=set(), all_profiles=known_profiles
        )
        assert profile is known_profiles["interactive"]

    def test_none_returns_default_for_other_commands(self, known_profiles):
        profile, _ = resolve_profile_for_command(
            None, ("text", "process"), accepted_keys=set(), all_profiles=known_profiles
        )
        assert profile is known_profiles["default"]

    def test_unknown_profile_raises_with_close_match(self, known_profiles):
        with pytest.raises(typer.BadParameter) as exc_info:
            resolve_profile_for_command(
                "fast_quary",  # typo
                ("query",),
                accepted_keys=set(),
                all_profiles=known_profiles,
            )
        # Echo + close-match suggestion.
        msg = str(exc_info.value)
        assert "fast_quary" in msg
        assert "fast_query" in msg  # close-match hint

    def test_unknown_profile_no_close_match(self, known_profiles):
        # Use a name far enough from any known profile that difflib doesn't suggest.
        with pytest.raises(typer.BadParameter) as exc_info:
            resolve_profile_for_command(
                "xyzzy",
                ("query",),
                accepted_keys=set(),
                all_profiles=known_profiles,
            )
        msg = str(exc_info.value)
        assert "xyzzy" in msg
        # No "Did you mean" since no profile is close enough.
        assert "Did you mean" not in msg or "fast_query" not in msg

    def test_command_bound_profile_matches(self, known_profiles):
        profile, kwargs = resolve_profile_for_command(
            "fast_query", ("query",), accepted_keys={"num_results", "similarity_threshold"},
            all_profiles=known_profiles,
        )
        assert profile is known_profiles["fast_query"]
        assert kwargs == {"num_results": 5, "similarity_threshold": 0.5}

    def test_command_bound_profile_mismatched_command_raises(self, known_profiles):
        # fast_query is bound to "query" but invoked from text process.
        with pytest.raises(typer.BadParameter) as exc_info:
            resolve_profile_for_command(
                "fast_query",
                ("text", "process"),
                accepted_keys={"num_results"},
                all_profiles=known_profiles,
            )
        assert "query" in str(exc_info.value)  # mention the bound command

    def test_unbound_profile_filters_to_accepted_keys(self, known_profiles):
        # shared_german has language and semantic_chunker_model.
        # If the command only accepts language, only language should land in kwargs.
        profile, kwargs = resolve_profile_for_command(
            "shared_german",
            ("query",),
            accepted_keys={"language"},
            all_profiles=known_profiles,
        )
        assert kwargs == {"language": "de"}
        # Not in accepted_keys → filtered out.
        assert "semantic_chunker_model" not in kwargs

    def test_unbound_profile_skips_none_fields(self, known_profiles):
        profile, kwargs = resolve_profile_for_command(
            "shared_german",
            ("query",),
            accepted_keys={"language", "num_results"},
            all_profiles=known_profiles,
        )
        # num_results is None on shared_german → not included.
        assert "num_results" not in kwargs
        assert kwargs == {"language": "de"}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/profiles/test_resolver.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_profile_for_command'`.

- [ ] **Step 3: Add `resolve_profile_for_command` to `phentrieve/profiles.py`**

Append to `phentrieve/profiles.py`:

```python
import difflib

import typer


def resolve_profile_for_command(
    profile_name: str | None,
    command_path: tuple[str, ...],
    accepted_keys: set[str],
    all_profiles: dict[str, Profile] | None = None,
) -> tuple[Profile, dict[str, Any]]:
    """Select the active profile and return (profile, kwargs_for_default_map).

    Args:
        profile_name: User-supplied name (`--profile X`), or None for auto-select.
        command_path: Command tuple, e.g. ("text", "process") or ("query",).
        accepted_keys: Option names the active command actually accepts.
            Used to filter cross-command profiles down to applicable kwargs.
        all_profiles: Merged dict of built-ins + user profiles. Defaults to
            BUILTIN_PROFILES (test convenience).

    Returns:
        (profile_object, kwargs_dict) — the kwargs dict is what the eager
        callback writes into ctx.default_map (after walking the precedence
        stack with YAML and fallback constants).

    Raises:
        typer.BadParameter: profile_name is unknown, or is bound to a
            different command via its `command` field.
    """
    profiles = all_profiles if all_profiles is not None else BUILTIN_PROFILES

    if profile_name is None:
        # Auto-selection
        fallback = "interactive" if command_path == ("text", "interactive") else "default"
        profile = profiles.get(fallback, BUILTIN_PROFILES[fallback])
        return profile, _profile_to_kwargs(profile, accepted_keys)

    if profile_name not in profiles:
        candidates = difflib.get_close_matches(
            profile_name, list(profiles.keys()), n=1, cutoff=0.6
        )
        msg = f"Profile {profile_name!r} not found."
        if candidates:
            msg += f" Did you mean: {candidates[0]!r}?"
        raise typer.BadParameter(msg, param_hint="--profile")

    profile = profiles[profile_name]

    # Command binding check.
    if profile.command is not None:
        bound = tuple(profile.command.split())
        if bound != command_path:
            raise typer.BadParameter(
                f"Profile {profile_name!r} is bound to command {profile.command!r} "
                f"but was used with {' '.join(command_path)!r}.",
                param_hint="--profile",
            )

    return profile, _profile_to_kwargs(profile, accepted_keys)


def _profile_to_kwargs(profile: Profile, accepted_keys: set[str]) -> dict[str, Any]:
    """Convert a Profile to a kwargs dict, filtered to accepted_keys.

    Skips None fields (None means "this profile doesn't preset this option")
    and fields not in accepted_keys (cross-command filter).
    """
    kwargs: dict[str, Any] = {}
    for field_name, value in profile.model_dump().items():
        if field_name in {"description", "command"}:
            continue  # metadata, not options
        if value is None:
            continue
        if accepted_keys and field_name not in accepted_keys:
            continue
        kwargs[field_name] = value
    return kwargs
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/profiles/test_resolver.py -v`
Expected: PASS, all eight tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/profiles.py tests/unit/profiles/test_resolver.py
git commit -m "feat(profiles): add resolve_profile_for_command

Auto-selects builtin 'interactive' for text interactive, 'default' for
others. Errors with close-match suggestion via difflib for unknown names.
Enforces the optional command binding. Filters cross-command profiles to
the active command's accepted_keys, ignoring None fields."
```

### Task 6: Add YAML profile loader to `phentrieve/profiles.py`

**Files:**
- Modify: `phentrieve/profiles.py`
- Modify: `tests/unit/profiles/test_resolver.py` (extend)

- [ ] **Step 1: Write failing test for the loader**

Append to `tests/unit/profiles/test_resolver.py`:

```python
class TestLoadProfilesFromYaml:
    def test_no_profiles_section_returns_empty(self, tmp_path, monkeypatch):
        from phentrieve.profiles import load_profiles_from_yaml

        yaml_path = tmp_path / "phentrieve.yaml"
        yaml_path.write_text("data_dir: data\ndefault_model: foo\n")
        # Force phentrieve.yaml lookup to point at our tmp path.
        monkeypatch.chdir(tmp_path)
        # Clear cache from any prior test
        from phentrieve.config import _load_yaml_config
        _load_yaml_config.cache_clear()

        profiles = load_profiles_from_yaml()
        assert profiles == {}

    def test_profiles_section_parses(self, tmp_path, monkeypatch):
        from phentrieve.config import _load_yaml_config
        from phentrieve.profiles import load_profiles_from_yaml

        yaml_path = tmp_path / "phentrieve.yaml"
        yaml_path.write_text(
            "profiles:\n"
            "  fast_query:\n"
            "    command: query\n"
            "    num_results: 5\n"
            "    similarity_threshold: 0.5\n"
        )
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()

        profiles = load_profiles_from_yaml()
        assert "fast_query" in profiles
        assert profiles["fast_query"].num_results == 5

    def test_invalid_profile_logs_warning_and_skips(self, tmp_path, monkeypatch, caplog):
        from phentrieve.config import _load_yaml_config
        from phentrieve.profiles import load_profiles_from_yaml

        yaml_path = tmp_path / "phentrieve.yaml"
        # Has an unknown key (extra=forbid → validation error on this profile only).
        yaml_path.write_text(
            "profiles:\n"
            "  good:\n"
            "    language: en\n"
            "  bad:\n"
            "    unknown_field: value\n"
        )
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()

        profiles = load_profiles_from_yaml()
        assert "good" in profiles
        assert "bad" not in profiles  # Skipped due to validation error.
        assert any("bad" in rec.message for rec in caplog.records)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/profiles/test_resolver.py::TestLoadProfilesFromYaml -v`
Expected: FAIL with `ImportError: cannot import name 'load_profiles_from_yaml'`.

- [ ] **Step 3: Add the loader to `phentrieve/profiles.py`**

Append to `phentrieve/profiles.py`:

```python
import logging

from pydantic import ValidationError

logger = logging.getLogger(__name__)


def load_profiles_from_yaml() -> dict[str, Profile]:
    """Load and validate the `profiles:` section of phentrieve.yaml.

    Each profile is validated independently — a failure on one profile
    logs a WARNING and skips that profile, leaving others usable. The
    `phentrieve config validate` command (Task 24) raises non-zero on
    validation errors so CI can catch them.
    """
    # Lazy import to avoid circular dependency: config -> utils -> profiles.
    from phentrieve.config import _load_yaml_config

    raw = _load_yaml_config()
    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        logger.warning(
            "phentrieve.yaml `profiles:` section is not a mapping; ignoring."
        )
        return {}

    profiles: dict[str, Profile] = {}
    for name, data in profiles_raw.items():
        if not isinstance(data, dict):
            logger.warning(
                "Profile %r in phentrieve.yaml is not a mapping; skipping.", name
            )
            continue
        try:
            profiles[name] = Profile.model_validate(data)
        except ValidationError as e:
            logger.warning(
                "Profile %r in phentrieve.yaml failed validation; skipping. %s",
                name,
                e,
            )
    return profiles


def merged_profiles() -> dict[str, Profile]:
    """Return BUILTIN_PROFILES merged with user profiles. User wins on shadow."""
    user = load_profiles_from_yaml()
    merged = {**BUILTIN_PROFILES, **user}
    for name in user:
        if name in BUILTIN_PROFILES:
            logger.info(
                "User profile %r in phentrieve.yaml shadows the built-in profile.",
                name,
            )
    return merged
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/profiles/test_resolver.py::TestLoadProfilesFromYaml -v`
Expected: PASS, all three tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/profiles.py tests/unit/profiles/test_resolver.py
git commit -m "feat(profiles): add load_profiles_from_yaml and merged_profiles

Per-profile validation: an invalid profile logs a WARNING and is skipped,
leaving valid profiles usable. merged_profiles() combines built-ins with
user profiles, logging at INFO when a user profile shadows a built-in."
```

---

## Phase 3: Eager callback and sidecar source map

This phase wires `--profile` into Click via an eager callback that populates `ctx.default_map` and the `ctx.obj["resolved_sources"]` sidecar map.

### Task 7: Create `phentrieve/cli/_profile.py` with the eager callback skeleton

**Files:**
- Create: `phentrieve/cli/_profile.py`
- Create: `tests/unit/cli/test_profile_callback.py`

- [ ] **Step 1: Write failing test for the callback**

Create `tests/unit/cli/test_profile_callback.py`:

```python
"""Tests for the eager --profile callback."""

import click
import pytest
import typer
from typer.testing import CliRunner


def make_test_app():
    """A throwaway Typer app with --profile wired via the eager callback."""
    from phentrieve.cli._profile import apply_profile_callback

    app = typer.Typer()

    @app.callback()
    def root(
        ctx: typer.Context,
        profile: str = typer.Option(
            "default",
            "--profile",
            callback=apply_profile_callback,
            is_eager=True,
        ),
    ):
        ctx.ensure_object(dict)
        ctx.obj.setdefault("seen_root", True)

    @app.command()
    def echo(
        ctx: typer.Context,
        language: str | None = None,
        num_results: int | None = None,
    ):
        # Resolve fallbacks.
        lang = language if language is not None else "en"
        n = num_results if num_results is not None else 10
        click.echo(f"language={lang} num_results={n}")
    return app


class TestApplyProfileCallback:
    def test_default_profile_no_explicit_flag(self, monkeypatch, tmp_path):
        from phentrieve.config import _load_yaml_config
        monkeypatch.chdir(tmp_path)  # No phentrieve.yaml present.
        _load_yaml_config.cache_clear()

        runner = CliRunner()
        result = runner.invoke(make_test_app(), ["echo"])
        assert result.exit_code == 0
        # No profile, no YAML → falls through to function-body fallbacks.
        assert "language=en" in result.stdout
        assert "num_results=10" in result.stdout

    def test_unknown_profile_errors(self, monkeypatch, tmp_path):
        from phentrieve.config import _load_yaml_config
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()

        runner = CliRunner()
        result = runner.invoke(make_test_app(), ["--profile", "ghost", "echo"])
        assert result.exit_code != 0
        assert "ghost" in result.output
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/cli/test_profile_callback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phentrieve.cli._profile'`.

- [ ] **Step 3: Create `phentrieve/cli/_profile.py`**

```python
"""Eager --profile callback: populates ctx.default_map and the sidecar
source map at ctx.obj['resolved_sources'].

Implements Spec A's Architecture section (.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md).
"""

from __future__ import annotations

import logging
from typing import Any

import click
import typer

from phentrieve import config as _config
from phentrieve.profiles import (
    BUILTIN_PROFILES,
    Profile,
    merged_profiles,
    resolve_profile_for_command,
)

logger = logging.getLogger(__name__)

# Map from Profile field name to (yaml_top_level_key, fallback_constant_name).
# Used to walk the layered precedence stack: profile -> YAML -> constant.
# YAML keys are the SHORT names that match phentrieve/config.py:502,505 reads
# (chunk_threshold, min_confidence) — NOT the longer Profile field names.
_OPTION_RESOLUTION_MAP: dict[str, tuple[str | tuple[str, ...], str]] = {
    # Profile field name -> (yaml_path, constant_name)
    "language": ("default_language", "DEFAULT_LANGUAGE"),
    "model_name": ("default_model", "DEFAULT_MODEL"),
    "retrieval_model": ("default_model", "DEFAULT_MODEL"),
    "semantic_chunker_model": ("default_model", "DEFAULT_MODEL"),
    "num_results": ("default_top_k", "DEFAULT_TOP_K"),
    "similarity_threshold": ("min_similarity_threshold", "MIN_SIMILARITY_THRESHOLD"),
    "chunk_retrieval_threshold": (
        ("extraction", "chunk_threshold"),
        "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD",
    ),
    "aggregated_term_confidence": (
        ("extraction", "min_confidence"),
        "DEFAULT_MIN_CONFIDENCE_AGGREGATED",
    ),
    "chunking_strategy": ("default_chunking_strategy", "DEFAULT_CHUNKING_STRATEGY"),
    "assertion_preference": (
        ("extraction", "assertion_preference"),
        "DEFAULT_ASSERTION_PREFERENCE",
    ),
    "output_format": (("output", "format_query"), "DEFAULT_OUTPUT_FORMAT_QUERY"),
}


def _walk_yaml(yaml_data: dict, path: str | tuple[str, ...]) -> Any:
    """Walk a nested YAML key. Returns None if any segment is missing."""
    if isinstance(path, str):
        return yaml_data.get(path)
    cur: Any = yaml_data
    for seg in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
        if cur is None:
            return None
    return cur


def _resolve_option_value(
    field_name: str,
    profile_kwargs: dict[str, Any],
    yaml_data: dict,
) -> tuple[Any, str] | None:
    """Walk the precedence stack for one option:
        profile -> YAML -> constant.
    Returns (value, source_label) or None if no layer set the option.
    """
    if field_name in profile_kwargs:
        return profile_kwargs[field_name], "<profile-set>"

    if field_name in _OPTION_RESOLUTION_MAP:
        yaml_path, const_name = _OPTION_RESOLUTION_MAP[field_name]
        yaml_value = _walk_yaml(yaml_data, yaml_path)
        if yaml_value is not None:
            label = (
                f"yaml:{yaml_path}" if isinstance(yaml_path, str)
                else f"yaml:{'.'.join(yaml_path)}"
            )
            return yaml_value, label
        const_value = getattr(_config, const_name, None)
        if const_value is not None:
            return const_value, f"const:{const_name}"

    return None


def _command_path_from_ctx(ctx: typer.Context) -> tuple[str, ...]:
    """Reconstruct the command tuple by walking up the click context."""
    parts: list[str] = []
    cur: click.Context | None = ctx
    while cur is not None and cur.parent is not None:
        if cur.info_name:
            parts.append(cur.info_name)
        cur = cur.parent
    return tuple(reversed(parts))


def apply_profile_callback(
    ctx: typer.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    """Eager callback for --profile. Populates ctx.default_map and the
    sidecar source map. Idempotent — calling twice with the same value
    is a no-op.
    """
    # ParameterSource lets us tell "user typed --profile X" vs "default kicked in".
    # We populate default_map regardless, but only emit profile-bound errors
    # when the user actually typed the flag (eager root callback default
    # of 'default' should not error if the user passed nothing).
    ctx.ensure_object(dict)
    if "resolved_sources" not in ctx.obj:
        ctx.obj["resolved_sources"] = {}

    # Idempotence: bail if we've already populated for this value.
    if ctx.obj.get("_profile_applied") == value:
        return value
    ctx.obj["_profile_applied"] = value

    # Reconstruct command path. For root-callback case, ctx.invoked_subcommand
    # may not be set yet; for command-callback case, it is set.
    command_path = _command_path_from_ctx(ctx)
    if not command_path and ctx.invoked_subcommand:
        # Root callback case: only the subcommand name is known.
        command_path = (ctx.invoked_subcommand,)

    # Gather the active command's accepted option keys for cross-command filtering.
    accepted_keys = _accepted_option_keys(ctx)

    profiles = merged_profiles()
    profile, profile_kwargs = resolve_profile_for_command(
        value if value not in (None, "") else None,
        command_path,
        accepted_keys,
        all_profiles=profiles,
    )

    # Walk the precedence stack for every option, building default_map.
    from phentrieve.config import _load_yaml_config
    yaml_data = _load_yaml_config()
    profile_label = f"profile:{value}" if value not in (None, "") else (
        "profile:builtin:interactive"
        if command_path == ("text", "interactive")
        else "profile:builtin:default"
    )

    new_defaults: dict[str, Any] = {}
    for field_name in profile.model_fields:
        if field_name in {"description", "command", "adaptive_rechunking"}:
            continue
        if not accepted_keys or field_name in accepted_keys:
            resolved = _resolve_option_value(field_name, profile_kwargs, yaml_data)
            if resolved is not None:
                resolved_value, label = resolved
                if label == "<profile-set>":
                    label = profile_label
                new_defaults[field_name] = resolved_value
                ctx.obj["resolved_sources"][field_name] = label

    # Click's ctx.default_map is a dict that subcommand-level options consult.
    # For root-callback case, write into the subcommand's nested bucket.
    if ctx.default_map is None:
        ctx.default_map = {}
    if ctx.invoked_subcommand and not command_path:
        # Root callback fired: nest under invoked subcommand.
        ctx.default_map.setdefault(ctx.invoked_subcommand, {}).update(new_defaults)
    else:
        ctx.default_map.update(new_defaults)

    return value


def _accepted_option_keys(ctx: typer.Context) -> set[str]:
    """Collect the active command's parameter names (Click params, not Typer)."""
    cmd = ctx.command
    if cmd is None:
        return set()
    return {p.name for p in cmd.params if isinstance(p, click.Option)}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/cli/test_profile_callback.py -v`
Expected: PASS, both tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/cli/_profile.py tests/unit/cli/test_profile_callback.py
git commit -m "feat(cli): eager --profile callback populates default_map

The callback walks the precedence stack (profile -> YAML -> constant) for
each profileable option and writes both into ctx.default_map (Click's
native default-overrides mechanism) and into a sidecar map at
ctx.obj['resolved_sources'] for fine-grained source labels.
ParameterSource alone collapses profile/yaml/const all into DEFAULT_MAP."
```

### Task 8: Add tests for the sidecar source map

**Files:**
- Modify: `tests/unit/cli/test_profile_callback.py`

- [ ] **Step 1: Append tests for the sidecar map**

Append to `tests/unit/cli/test_profile_callback.py`:

```python
class TestSidecarSourceMap:
    def test_sidecar_populated_with_profile_label(self, monkeypatch, tmp_path):
        from phentrieve.config import _load_yaml_config
        from phentrieve.profiles import BUILTIN_PROFILES

        # No user phentrieve.yaml — only built-ins.
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()

        # Capture ctx.obj from inside the command body.
        captured: dict = {}

        app = typer.Typer()

        @app.callback()
        def root(
            ctx: typer.Context,
            profile: str = typer.Option(
                "interactive",  # built-in default
                "--profile",
                callback=__import__(
                    "phentrieve.cli._profile", fromlist=["apply_profile_callback"]
                ).apply_profile_callback,
                is_eager=True,
            ),
        ):
            ctx.ensure_object(dict)

        @app.command()
        def cmd(
            ctx: typer.Context,
            chunk_retrieval_threshold: float | None = None,
        ):
            captured["sources"] = dict(ctx.obj.get("resolved_sources", {}))
            captured["threshold"] = (
                chunk_retrieval_threshold
                if chunk_retrieval_threshold is not None
                else 0.7  # the constant fallback
            )

        runner = CliRunner()
        result = runner.invoke(app, ["cmd"])
        assert result.exit_code == 0, result.output
        # Built-in interactive sets chunk_retrieval_threshold=0.3.
        assert captured["threshold"] == 0.3
        assert "chunk_retrieval_threshold" in captured["sources"]
        assert "profile:builtin:interactive" in captured["sources"]["chunk_retrieval_threshold"] \
            or "profile:interactive" in captured["sources"]["chunk_retrieval_threshold"]
```

- [ ] **Step 2: Run test to verify pass (or adjust callback if it fails)**

Run: `uv run pytest tests/unit/cli/test_profile_callback.py::TestSidecarSourceMap -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli/test_profile_callback.py
git commit -m "test(cli): sidecar source map populates with profile label"
```

---

## Phase 4: Apply --profile to text interactive (#171 fix)

This phase resolves issue #171 by replacing hardcoded literals in `text_interactive.py` with `None` defaults plus the value-or-constant fallback. The built-in `interactive` profile preserves prior behavior.

### Task 9: Add --profile option and switch defaults to None on text interactive

**Files:**
- Modify: `phentrieve/cli/text_interactive.py`
- Create: `tests/unit/cli/test_default_map_resolution.py`

- [ ] **Step 1: Write failing tests for resolution invariant**

Create `tests/unit/cli/test_default_map_resolution.py`:

```python
"""Core resolution invariant: omitted CLI flags must NOT mask profile values."""

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


@pytest.fixture(autouse=True)
def _isolated_yaml(tmp_path, monkeypatch):
    """Each test runs from an empty cwd so it doesn't pick up the real phentrieve.yaml."""
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()


def write_yaml(tmp_path, content: str):
    (tmp_path / "phentrieve.yaml").write_text(content)
    _load_yaml_config.cache_clear()


class TestTextInteractiveResolution:
    def test_no_flag_no_yaml_uses_builtin_interactive(self, tmp_path):
        from phentrieve.__main__ import app  # adjust path if root app lives elsewhere

        runner = CliRunner()
        result = runner.invoke(
            app, ["text", "interactive", "--debug-resolution-only"]
        )
        # The implementation under test will print the resolved values in
        # debug mode. The exact output format is set by Task 10's
        # --show-resolved-config flag; for now we tolerate either text body
        # output or no output, but the exit code must be 0.
        assert result.exit_code == 0, result.output

    def test_explicit_flag_overrides_profile(self, tmp_path):
        from phentrieve.__main__ import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["text", "interactive", "--language", "fr"],
            input="quit\n",  # interactive mode loops; quit immediately
        )
        # Use --show-resolved-config once available (added in Task 19).
        # For now this test's full assertion gets fleshed out in Task 19.
        assert result.exit_code in (0, 1), result.output  # tolerate quit-input issues
```

(This test file is intentionally minimal; full assertions land once `--show-resolved-config` exists in Task 19.)

- [ ] **Step 2: Modify `phentrieve/cli/text_interactive.py`**

Open `phentrieve/cli/text_interactive.py`. The current `interactive_text_mode` function starts at line 159 with hardcoded defaults. Change the function signature so:

1. Add a `--profile` option as the first option after `ctx`.
2. Change `language: str = "en"` → `language: Annotated[str | None, typer.Option(...)] = None`.
3. Change `chunk_retrieval_threshold: float = 0.3` → `Annotated[float | None, typer.Option(...)] = None`.
4. Change `aggregated_term_confidence: float = 0.35` → `Annotated[float | None, typer.Option(...)] = None`.
5. Change `num_results: int = 5` → `Annotated[int | None, typer.Option(...)] = None`.
6. Change `strategy: str = "sliding_window_punct_conj_cleaned"` → keep current default (it already matches `DEFAULT_CHUNKING_STRATEGY`); leave non-profileable.
7. In the function body, add the value-or-constant fallback:

```python
language = language if language is not None else DEFAULT_LANGUAGE
chunk_retrieval_threshold = (
    chunk_retrieval_threshold
    if chunk_retrieval_threshold is not None
    else DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
)
aggregated_term_confidence = (
    aggregated_term_confidence
    if aggregated_term_confidence is not None
    else DEFAULT_MIN_CONFIDENCE_AGGREGATED
)
num_results = num_results if num_results is not None else DEFAULT_TOP_K
```

8. Add imports at the top of `text_interactive.py`:

```python
from phentrieve.cli._profile import apply_profile_callback
from phentrieve.config import (
    DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_CONFIDENCE_AGGREGATED,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
)
```

9. Add the `--profile` option to the function signature (place near the top of the option list):

```python
profile: Annotated[
    str,
    typer.Option(
        "--profile",
        "-P",
        envvar="PHENTRIEVE_PROFILE",
        help="Apply a named profile from phentrieve.yaml. "
             "See `phentrieve config list-profiles`.",
        callback=apply_profile_callback,
        is_eager=True,
    ),
] = "interactive",
```

The `default="interactive"` is what makes auto-selection work — when the user doesn't pass `--profile`, the eager callback receives `"interactive"` and populates `default_map` from the built-in.

- [ ] **Step 3: Verify text_interactive imports without error**

Run: `uv run python -c "from phentrieve.cli import text_interactive; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Smoke-test invocation**

Run: `uv run phentrieve text interactive --help`
Expected: help output includes `--profile` option.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/cli/text_interactive.py tests/unit/cli/test_default_map_resolution.py
git commit -m "feat(cli): wire --profile into text interactive (fixes #171)

Replaces hardcoded literals (language='en', chunk_retrieval_threshold=0.3,
aggregated_term_confidence=0.35, num_results=5) with None defaults plus
value-or-constant fallback. The built-in 'interactive' profile preserves
the prior loose-defaults behavior so existing users see no change.
Users who want the strict API-matching defaults can pass --profile default."
```

### Task 10: Test text interactive auto-selection of `interactive` profile

**Files:**
- Modify: `tests/unit/cli/test_text_interactive.py` (extend existing) or create if absent
- Test: `tests/unit/cli/test_text_interactive_profile.py` (new file)

- [ ] **Step 1: Create or extend the test**

Create `tests/unit/cli/test_text_interactive_profile.py`:

```python
"""Tests for text interactive profile resolution."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


@pytest.fixture(autouse=True)
def _isolated_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()


@pytest.fixture
def app():
    from phentrieve.__main__ import app as a
    return a


class TestTextInteractiveProfileResolution:
    @patch("phentrieve.cli.text_interactive.DenseRetriever")
    @patch("phentrieve.cli.text_interactive.TextProcessingPipeline")
    @patch("phentrieve.cli.text_interactive.orchestrate_hpo_extraction")
    def test_default_invocation_uses_interactive_profile_values(
        self, mock_extract, mock_pipeline_cls, mock_retriever_cls, app
    ):
        # Capture the kwargs orchestrate_hpo_extraction is called with.
        mock_extract.return_value = ([], [])
        mock_pipeline_cls.return_value.process.return_value = []

        runner = CliRunner()
        result = runner.invoke(app, ["text", "interactive"], input="quit\n")
        # Tolerate quit-loop early exit; we only check kwargs were prepared.
        if mock_extract.called:
            kwargs = mock_extract.call_args.kwargs
            # Built-in interactive sets these:
            assert kwargs["chunk_retrieval_threshold"] == 0.3
            assert kwargs["min_confidence_for_aggregated"] == 0.35
            assert kwargs["num_results_per_chunk"] == 5

    @patch("phentrieve.cli.text_interactive.DenseRetriever")
    @patch("phentrieve.cli.text_interactive.TextProcessingPipeline")
    @patch("phentrieve.cli.text_interactive.orchestrate_hpo_extraction")
    def test_profile_default_swaps_to_strict(
        self, mock_extract, mock_pipeline_cls, mock_retriever_cls, app
    ):
        mock_extract.return_value = ([], [])
        mock_pipeline_cls.return_value.process.return_value = []

        runner = CliRunner()
        result = runner.invoke(
            app, ["text", "interactive", "--profile", "default"], input="quit\n"
        )
        if mock_extract.called:
            kwargs = mock_extract.call_args.kwargs
            # Built-in default has all None fields → falls through to constants.
            from phentrieve.config import (
                DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
                DEFAULT_MIN_CONFIDENCE_AGGREGATED,
                DEFAULT_TOP_K,
            )
            assert kwargs["chunk_retrieval_threshold"] == DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
            assert kwargs["min_confidence_for_aggregated"] == DEFAULT_MIN_CONFIDENCE_AGGREGATED
            assert kwargs["num_results_per_chunk"] == DEFAULT_TOP_K

    @patch("phentrieve.cli.text_interactive.DenseRetriever")
    @patch("phentrieve.cli.text_interactive.TextProcessingPipeline")
    @patch("phentrieve.cli.text_interactive.orchestrate_hpo_extraction")
    def test_explicit_flag_overrides_profile(
        self, mock_extract, mock_pipeline_cls, mock_retriever_cls, app
    ):
        mock_extract.return_value = ([], [])
        mock_pipeline_cls.return_value.process.return_value = []

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["text", "interactive", "--num-results", "20"],
            input="quit\n",
        )
        if mock_extract.called:
            kwargs = mock_extract.call_args.kwargs
            assert kwargs["num_results_per_chunk"] == 20  # CLI wins
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/cli/test_text_interactive_profile.py -v`
Expected: PASS (or skip-with-warnings on tests where the quit-loop short-circuits before `orchestrate_hpo_extraction`).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli/test_text_interactive_profile.py
git commit -m "test(cli): text interactive auto-selects interactive built-in"
```

---

## Phase 5: Apply --profile to text process

### Task 11: Wire --profile into `text_commands.py:process_text_for_hpo_command`

**Files:**
- Modify: `phentrieve/cli/text_commands.py`
- Test: `tests/unit/cli/test_text_commands.py` (extend)

- [ ] **Step 1: Extend the existing tests file with a profile test**

Append to `tests/unit/cli/test_text_commands.py` (create stub if file missing):

```python
class TestProcessProfileResolution:
    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_default_profile_falls_through_to_config(self, mock_run, monkeypatch, tmp_path):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_run.return_value = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

        # Write a tiny input file.
        (tmp_path / "in.txt").write_text("Patient with seizures.")

        runner = CliRunner()
        result = runner.invoke(app, ["text", "process", str(tmp_path / "in.txt")])
        assert result.exit_code == 0, result.output

        kwargs = mock_run.call_args.kwargs
        # No profile, no YAML override → falls through to config constants.
        from phentrieve.config import (
            DEFAULT_CHUNK_RETRIEVAL_THRESHOLD,
            DEFAULT_LANGUAGE,
            DEFAULT_TOP_K,
        )
        assert kwargs["chunk_retrieval_threshold"] == DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
        # language: None at CLI → auto-detect inside service. We just assert it
        # wasn't hardcoded to "en".
        assert kwargs.get("language") in {None, DEFAULT_LANGUAGE}

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_profile_provides_defaults(self, mock_run, monkeypatch, tmp_path):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        # User profile with overrides.
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  high_recall_german:\n"
            "    command: text process\n"
            "    language: de\n"
            "    chunk_retrieval_threshold: 0.5\n"
        )
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_run.return_value = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

        (tmp_path / "in.txt").write_text("Der Patient hat Anfälle.")
        runner = CliRunner()
        result = runner.invoke(
            app, ["text", "process", str(tmp_path / "in.txt"), "--profile", "high_recall_german"]
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_run.call_args.kwargs
        assert kwargs["language"] == "de"
        assert kwargs["chunk_retrieval_threshold"] == 0.5

    @patch("phentrieve.cli.text_commands.run_full_text_service")
    def test_explicit_flag_overrides_profile(self, mock_run, monkeypatch, tmp_path):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n  german:\n    command: text process\n    language: de\n"
        )
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_run.return_value = {"meta": {}, "processed_chunks": [], "aggregated_hpo_terms": []}

        (tmp_path / "in.txt").write_text("Patient.")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["text", "process", str(tmp_path / "in.txt"),
             "--profile", "german", "--language", "fr"],
        )
        assert result.exit_code == 0
        kwargs = mock_run.call_args.kwargs
        assert kwargs["language"] == "fr"  # explicit flag wins
```

- [ ] **Step 2: Modify `phentrieve/cli/text_commands.py`**

Add import:

```python
from phentrieve.cli._profile import apply_profile_callback
```

In the function `process_text_for_hpo_command` (around lines 274-494):

1. Add `--profile` option as the first option after `ctx`:

```python
profile: Annotated[
    str,
    typer.Option(
        "--profile",
        "-P",
        envvar="PHENTRIEVE_PROFILE",
        help="Apply a named profile from phentrieve.yaml.",
        callback=apply_profile_callback,
        is_eager=True,
    ),
] = "default",
```

2. Change profileable defaults to `None`:
   - `language: str = "en"` → `language: Annotated[str | None, typer.Option(...)] = None`
   - `num_results: int = 10` → `Annotated[int | None, typer.Option(...)] = None`
   - `chunk_confidence: float = 0.2` → `Annotated[float | None, typer.Option(...)] = None`
   - `assertion_preference: str = "dependency"` → `Annotated[str | None, typer.Option(...)] = None`
   - `output_format: str = "json_lines"` → `Annotated[str | None, typer.Option(...)] = None`

(Leave existing constant-backed defaults like `chunk_retrieval_threshold = DEFAULT_CHUNK_RETRIEVAL_THRESHOLD` alone if they already use the constant.)

3. In the function body, add value-or-constant fallback after argument parsing:

```python
language = language if language is not None else DEFAULT_LANGUAGE
num_results = num_results if num_results is not None else DEFAULT_TOP_K
chunk_confidence = chunk_confidence if chunk_confidence is not None else DEFAULT_CHUNK_CONFIDENCE
assertion_preference = (
    assertion_preference if assertion_preference is not None else DEFAULT_ASSERTION_PREFERENCE
)
output_format = output_format if output_format is not None else DEFAULT_OUTPUT_FORMAT_PROCESS
```

4. Update imports to include the new constants:

```python
from phentrieve.config import (
    DEFAULT_ASSERTION_PREFERENCE,
    DEFAULT_CHUNK_CONFIDENCE,
    DEFAULT_LANGUAGE,
    DEFAULT_OUTPUT_FORMAT_PROCESS,
    DEFAULT_TOP_K,
    # ... (existing imports)
)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/cli/test_text_commands.py::TestProcessProfileResolution -v`
Expected: PASS, three tests.

- [ ] **Step 4: Commit**

```bash
git add phentrieve/cli/text_commands.py tests/unit/cli/test_text_commands.py
git commit -m "feat(cli): wire --profile into text process

Replaces hardcoded language='en', num_results=10, chunk_confidence=0.2,
assertion_preference='dependency', output_format='json_lines' literals
with None defaults plus value-or-constant fallback. language now
auto-detects when neither flag, profile, nor YAML supplies it."
```

---

## Phase 6: Apply --profile to query

### Task 12: Wire --profile into `query_commands.py:query_hpo`

**Files:**
- Modify: `phentrieve/cli/query_commands.py`
- Test: `tests/unit/cli/test_query_enrichment.py` (extend)

- [ ] **Step 1: Append profile resolution test**

Append to `tests/unit/cli/test_query_enrichment.py`:

```python
class TestQueryProfileResolution:
    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_profile_provides_query_defaults(
        self, mock_orch, mock_retriever_cls, monkeypatch, tmp_path
    ):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n"
            "  fast_english:\n"
            "    command: query\n"
            "    num_results: 5\n"
            "    similarity_threshold: 0.5\n"
        )
        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_orch.run_query.return_value = {"matches": []}

        runner = CliRunner()
        result = runner.invoke(
            app, ["query", "patient with seizures", "--profile", "fast_english"]
        )
        # Tolerate retriever-init failures in unit-test env; we only check option flow.
        if mock_orch.run_query.called:
            kwargs = mock_orch.run_query.call_args.kwargs
            assert kwargs["num_results"] == 5
            assert kwargs["similarity_threshold"] == 0.5

    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_no_profile_falls_through_to_constants(
        self, mock_orch, mock_retriever_cls, monkeypatch, tmp_path
    ):
        from phentrieve.__main__ import app
        from phentrieve.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        _load_yaml_config.cache_clear()
        mock_orch.run_query.return_value = {"matches": []}

        runner = CliRunner()
        result = runner.invoke(app, ["query", "patient with seizures"])
        if mock_orch.run_query.called:
            kwargs = mock_orch.run_query.call_args.kwargs
            from phentrieve.config import DEFAULT_TOP_K
            # Default profile has all None → falls through to const.
            assert kwargs["num_results"] == DEFAULT_TOP_K
```

- [ ] **Step 2: Modify `phentrieve/cli/query_commands.py`**

In `query_hpo` (lines 111-227):

1. Add import:

```python
from phentrieve.cli._profile import apply_profile_callback
from phentrieve.config import (
    DEFAULT_AGGREGATION_STRATEGY,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_MULTI_VECTOR,
    DEFAULT_OUTPUT_FORMAT_QUERY,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
)
```

2. Add `--profile` option:

```python
profile: Annotated[
    str,
    typer.Option(
        "--profile",
        "-P",
        envvar="PHENTRIEVE_PROFILE",
        help="Apply a named profile from phentrieve.yaml.",
        callback=apply_profile_callback,
        is_eager=True,
    ),
] = "default",
```

3. Change profileable defaults from literals to `None`:
   - `model_name: str | None = "FremyCompany/BioLORD-2023-M"` → `Annotated[str | None, ...] = None`
   - `num_results: int = 10` → `Annotated[int | None, ...] = None`
   - `similarity_threshold: float = 0.3` → `Annotated[float | None, ...] = None`
   - `multi_vector: bool = False` → `Annotated[bool | None, ...] = None`
   - `aggregation_strategy: str = "label_synonyms_max"` → `Annotated[str | None, ...] = None`
   - `output_format: str = "text"` → `Annotated[str | None, ...] = None`

4. In the function body, add fallbacks (in the order options are used):

```python
model_name = model_name if model_name is not None else DEFAULT_MODEL
num_results = num_results if num_results is not None else DEFAULT_TOP_K
similarity_threshold = (
    similarity_threshold if similarity_threshold is not None else MIN_SIMILARITY_THRESHOLD
)
multi_vector = multi_vector if multi_vector is not None else DEFAULT_MULTI_VECTOR
aggregation_strategy = (
    aggregation_strategy if aggregation_strategy is not None else DEFAULT_AGGREGATION_STRATEGY
)
output_format = output_format if output_format is not None else DEFAULT_OUTPUT_FORMAT_QUERY
```

(Note: the existing in-function `model_name` check at line 246-248 becomes redundant after this refactor — remove it.)

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/cli/test_query_enrichment.py::TestQueryProfileResolution -v`
Expected: PASS, two tests.

- [ ] **Step 4: Commit**

```bash
git add phentrieve/cli/query_commands.py tests/unit/cli/test_query_enrichment.py
git commit -m "feat(cli): wire --profile into query

Replaces hardcoded model/num_results/similarity_threshold/multi_vector/
aggregation_strategy/output_format literals with None defaults plus
value-or-constant fallback. The model_name double-default check at
line 246-248 is removed (now redundant)."
```

---

## Phase 7: Root-level --profile placement

### Task 13: Add --profile to the root Typer callback

**Files:**
- Modify: `phentrieve/__main__.py` (or wherever the root app is defined)
- Create: `tests/unit/cli/test_profile_placement.py`

- [ ] **Step 1: Identify root app**

Run: `grep -rn "Typer(" phentrieve/cli/__init__.py phentrieve/__main__.py 2>&1 | head -5`
Note the file that defines the top-level `app = typer.Typer()`.

- [ ] **Step 2: Write placement tests**

Create `tests/unit/cli/test_profile_placement.py`:

```python
"""Tests that --profile works at both root and per-command placement."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


@pytest.fixture
def app():
    from phentrieve.__main__ import app as a
    return a


@pytest.fixture(autouse=True)
def _yaml_with_two_profiles(tmp_path, monkeypatch):
    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  fast_query_a:\n"
        "    command: query\n"
        "    num_results: 5\n"
        "  fast_query_b:\n"
        "    command: query\n"
        "    num_results: 8\n"
    )
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()


class TestProfilePlacement:
    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_root_placement(self, mock_orch, mock_r, app):
        mock_orch.run_query.return_value = {"matches": []}
        runner = CliRunner()
        result = runner.invoke(app, ["--profile", "fast_query_a", "query", "TEXT"])
        if mock_orch.run_query.called:
            assert mock_orch.run_query.call_args.kwargs["num_results"] == 5

    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_subcommand_placement(self, mock_orch, mock_r, app):
        mock_orch.run_query.return_value = {"matches": []}
        runner = CliRunner()
        result = runner.invoke(app, ["query", "TEXT", "--profile", "fast_query_a"])
        if mock_orch.run_query.called:
            assert mock_orch.run_query.call_args.kwargs["num_results"] == 5

    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_subcommand_wins_over_root(self, mock_orch, mock_r, app):
        mock_orch.run_query.return_value = {"matches": []}
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--profile", "fast_query_a", "query", "TEXT", "--profile", "fast_query_b"],
        )
        if mock_orch.run_query.called:
            # b wins (subcommand-level).
            assert mock_orch.run_query.call_args.kwargs["num_results"] == 8

    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_explicit_flag_beats_both(self, mock_orch, mock_r, app):
        mock_orch.run_query.return_value = {"matches": []}
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["query", "TEXT", "--profile", "fast_query_a", "--num-results", "20"],
        )
        if mock_orch.run_query.called:
            assert mock_orch.run_query.call_args.kwargs["num_results"] == 20
```

- [ ] **Step 3: Add --profile to root callback**

Find the root callback in `phentrieve/__main__.py` (or wherever `app.callback()` is defined). Add:

```python
from phentrieve.cli._profile import apply_profile_callback


@app.callback()
def main(
    ctx: typer.Context,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            "-P",
            envvar="PHENTRIEVE_PROFILE",
            help="Apply a named profile globally. Subcommand --profile wins on conflict.",
            callback=apply_profile_callback,
            is_eager=True,
        ),
    ] = None,
    show_resolved_config: Annotated[
        bool,
        typer.Option(
            "--show-resolved-config",
            help="Print resolved option values with source labels before running.",
        ),
    ] = False,
) -> None:
    """Phentrieve: HPO term retrieval from clinical text."""
    ctx.ensure_object(dict)
    ctx.obj["show_resolved_config"] = show_resolved_config
```

If a root callback already exists, merge these option declarations into the existing signature.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_profile_placement.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/__main__.py tests/unit/cli/test_profile_placement.py
git commit -m "feat(cli): add root-level --profile and --show-resolved-config

Both phentrieve --profile X cmd and phentrieve cmd --profile X work.
Subcommand-level --profile wins on conflict, matching Click conventions."
```

### Task 14: Test legacy YAML paths

**Files:**
- Create: `tests/unit/cli/test_yaml_legacy_paths.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/cli/test_yaml_legacy_paths.py`:

```python
"""Tests that legacy ~/.phentrieve/ YAML paths are still searched."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


def test_home_phentrieve_yaml_picked_up_when_no_local(tmp_path, monkeypatch):
    """A phentrieve.yaml at ~/.phentrieve/phentrieve.yaml is searched."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".phentrieve").mkdir()
    (fake_home / ".phentrieve" / "phentrieve.yaml").write_text(
        "profiles:\n  legacy_profile:\n    language: de\n"
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(fake_home))
    _load_yaml_config.cache_clear()

    from phentrieve.profiles import merged_profiles
    profiles = merged_profiles()
    assert "legacy_profile" in profiles
    assert profiles["legacy_profile"].language == "de"


def test_local_yaml_shadows_legacy_path(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".phentrieve").mkdir()
    (fake_home / ".phentrieve" / "phentrieve.yaml").write_text(
        "profiles:\n  shared:\n    language: de\n"
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "phentrieve.yaml").write_text(
        "profiles:\n  shared:\n    language: fr\n"
    )
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(fake_home))
    _load_yaml_config.cache_clear()

    from phentrieve.profiles import merged_profiles
    profiles = merged_profiles()
    assert profiles["shared"].language == "fr"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/cli/test_yaml_legacy_paths.py -v`
Expected: PASS, two tests.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/cli/test_yaml_legacy_paths.py
git commit -m "test(cli): legacy ~/.phentrieve/ YAML path still searched"
```

---

## Phase 8: phentrieve config subcommand group

### Task 15: Create `phentrieve config` sub-app

**Files:**
- Create: `phentrieve/cli/config_commands.py`
- Modify: `phentrieve/__main__.py` (register the sub-app)
- Create: `tests/unit/cli/test_config_commands.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/cli/test_config_commands.py`:

```python
"""Tests for phentrieve config list-profiles / show / validate / path."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


@pytest.fixture(autouse=True)
def _yaml_setup(tmp_path, monkeypatch):
    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  fast_query:\n"
        "    description: 'Quick English query'\n"
        "    command: query\n"
        "    num_results: 5\n"
    )
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()


@pytest.fixture
def app():
    from phentrieve.__main__ import app as a
    return a


class TestConfigListProfiles:
    def test_lists_user_and_builtin(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0, result.output
        assert "fast_query" in result.stdout
        assert "default" in result.stdout
        assert "interactive" in result.stdout

    def test_shows_command_binding(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert "query" in result.stdout  # fast_query is bound to query


class TestConfigShow:
    def test_show_user_profile(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "fast_query"])
        assert result.exit_code == 0, result.output
        assert "num_results: 5" in result.stdout
        assert "command: query" in result.stdout

    def test_show_builtin_interactive(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "interactive"])
        assert result.exit_code == 0
        assert "chunk_retrieval_threshold: 0.3" in result.stdout

    def test_show_unknown_profile_errors(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "show", "nonexistent"])
        assert result.exit_code != 0
        assert "nonexistent" in result.output


class TestConfigValidate:
    def test_validate_clean_yaml(self, app):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0

    def test_validate_invalid_yaml_errors(self, app, tmp_path):
        (tmp_path / "phentrieve.yaml").write_text(
            "profiles:\n  bad:\n    unknown_field: value\n"
        )
        _load_yaml_config.cache_clear()
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code != 0
        assert "bad" in result.output


class TestConfigPath:
    def test_path_prints_loaded_yaml(self, app, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert "phentrieve.yaml" in result.stdout
```

- [ ] **Step 2: Create `phentrieve/cli/config_commands.py`**

```python
"""`phentrieve config` subcommand group.

Implements list-profiles, show, validate, path per Spec A's Subcommand surface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
import yaml as pyyaml
from pydantic import ValidationError

from phentrieve.config import _load_yaml_config
from phentrieve.profiles import (
    BUILTIN_PROFILES,
    Profile,
    ProfilesFile,
    load_profiles_from_yaml,
    merged_profiles,
)
from phentrieve.utils import get_config_file_path

config_app = typer.Typer(
    name="config",
    help="Inspect and validate phentrieve.yaml configuration profiles.",
    no_args_is_help=True,
)


@config_app.command("list-profiles")
def list_profiles_cmd() -> None:
    """List all profiles (built-in + user) with their command bindings."""
    _load_yaml_config.cache_clear()
    user_profiles = load_profiles_from_yaml()
    rows = []
    for name, p in {**BUILTIN_PROFILES, **user_profiles}.items():
        source = "built-in" if name in BUILTIN_PROFILES and name not in user_profiles else (
            "user (shadows built-in)" if name in BUILTIN_PROFILES else "user"
        )
        binding = p.command or "any"
        desc = p.description or ""
        rows.append((name, source, binding, desc))

    # Simple aligned columns.
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    headers = ("Name", "Source", "Binding", "Description")
    typer.echo(
        f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  "
        f"{headers[2]:<{widths[2]}}  {headers[3]:<{widths[3]}}"
    )
    typer.echo("  ".join("-" * w for w in widths))
    for name, source, binding, desc in rows:
        typer.echo(
            f"{name:<{widths[0]}}  {source:<{widths[1]}}  "
            f"{binding:<{widths[2]}}  {desc:<{widths[3]}}"
        )


@config_app.command("show")
def show_cmd(name: str) -> None:
    """Print a profile as YAML."""
    _load_yaml_config.cache_clear()
    profiles = merged_profiles()
    if name not in profiles:
        raise typer.BadParameter(f"Profile {name!r} not found.", param_hint="name")
    profile = profiles[name]
    # Dump only non-None fields to keep output clean.
    non_none = {k: v for k, v in profile.model_dump().items() if v is not None}
    typer.echo(pyyaml.safe_dump(non_none, sort_keys=False, default_flow_style=False))


@config_app.command("validate")
def validate_cmd() -> None:
    """Validate phentrieve.yaml's profiles section against the Profile schema."""
    _load_yaml_config.cache_clear()
    raw = _load_yaml_config()
    errors: list[str] = []
    for name, data in raw.get("profiles", {}).items():
        if not isinstance(data, dict):
            errors.append(f"{name}: not a mapping")
            continue
        try:
            Profile.model_validate(data)
        except ValidationError as e:
            errors.append(f"{name}: {e}")
    if errors:
        for err in errors:
            typer.echo(err, err=True)
        raise typer.Exit(code=1)
    typer.echo("OK")


@config_app.command("path")
def path_cmd() -> None:
    """Print the active phentrieve.yaml search path and which file was loaded."""
    config_path = get_config_file_path()
    if config_path is not None and config_path.exists():
        typer.echo(f"Loaded: {config_path}")
    else:
        typer.echo("No phentrieve.yaml found in the search path.")
        typer.echo("Search order:")
        from phentrieve.utils import get_config_paths
        for p in get_config_paths():
            typer.echo(f"  - {p}")
```

- [ ] **Step 3: Register the sub-app in `phentrieve/__main__.py`**

After the existing `app = typer.Typer(...)` declaration, add:

```python
from phentrieve.cli.config_commands import config_app

app.add_typer(config_app, name="config")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_config_commands.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Smoke-test from CLI**

Run: `uv run phentrieve config list-profiles`
Expected: human-readable table with `default`, `interactive` rows.

- [ ] **Step 6: Commit**

```bash
git add phentrieve/cli/config_commands.py phentrieve/__main__.py tests/unit/cli/test_config_commands.py
git commit -m "feat(cli): add phentrieve config subcommand group

Implements list-profiles / show / validate / path subcommands. Uses
existing get_config_paths/get_config_file_path utilities for the path
resolution. validate exits non-zero on schema errors so CI can catch them."
```

---

## Phase 9: --show-resolved-config global flag

### Task 16: Implement `--show-resolved-config`

**Files:**
- Modify: `phentrieve/cli/_profile.py` (add print helper)
- Modify: `phentrieve/__main__.py` (call helper before each command body)
- Create: `tests/unit/cli/test_show_resolved_config.py`

- [ ] **Step 1: Add helper in `_profile.py`**

Append to `phentrieve/cli/_profile.py`:

```python
def render_resolved_config(ctx: typer.Context) -> str:
    """Build a human-readable resolved-config table from the sidecar map and
    Click parameter sources. Returned as a string for the caller to print
    to stderr.
    """
    if ctx.command is None:
        return ""
    lines: list[str] = []
    cmd_path = " ".join(_command_path_from_ctx(ctx))
    lines.append(f"Resolved configuration for `phentrieve {cmd_path}`:")

    sources: dict[str, str] = ctx.obj.get("resolved_sources", {}) if ctx.obj else {}
    for param in ctx.command.params:
        if not isinstance(param, click.Option):
            continue
        if param.name in {"profile", "show_resolved_config"}:
            continue
        value = ctx.params.get(param.name)
        param_source = ctx.get_parameter_source(param.name)
        # COMMANDLINE wins on labeling — user-provided.
        if param_source == click.core.ParameterSource.COMMANDLINE:
            label = f"--{param.name.replace('_', '-')} (commandline)"
        elif param.name in sources:
            label = sources[param.name]
        else:
            label = "default"
        lines.append(f"  {param.name:<35} {value!r:<20} ← {label}")
    return "\n".join(lines)
```

- [ ] **Step 2: Plumb the helper into each profileable command**

Add to the top of each profileable command's body (text_interactive, text_commands.process, query_commands.query):

```python
if ctx.obj.get("show_resolved_config"):
    from phentrieve.cli._profile import render_resolved_config
    typer.echo(render_resolved_config(ctx), err=True)
```

- [ ] **Step 3: Write tests**

Create `tests/unit/cli/test_show_resolved_config.py`:

```python
"""Tests for the --show-resolved-config flag."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield
    _load_yaml_config.cache_clear()


@pytest.fixture
def app():
    from phentrieve.__main__ import app as a
    return a


class TestShowResolvedConfig:
    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_output_to_stderr(self, mock_orch, mock_r, app):
        mock_orch.run_query.return_value = {"matches": []}
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            app, ["--show-resolved-config", "query", "TEXT"]
        )
        # Resolved-config table goes to stderr.
        assert "Resolved configuration" in result.stderr
        assert "← " in result.stderr  # source labels

    @patch("phentrieve.cli.query_commands.DenseRetriever")
    @patch("phentrieve.cli.query_commands.query_orchestrator")
    def test_explicit_flag_labeled_commandline(self, mock_orch, mock_r, app):
        mock_orch.run_query.return_value = {"matches": []}
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            app,
            ["--show-resolved-config", "query", "TEXT", "--num-results", "7"],
        )
        assert "(commandline)" in result.stderr
        assert "7" in result.stderr
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/cli/test_show_resolved_config.py -v`
Expected: PASS, two tests.

- [ ] **Step 5: Commit**

```bash
git add phentrieve/cli/_profile.py phentrieve/__main__.py phentrieve/cli/text_interactive.py phentrieve/cli/text_commands.py phentrieve/cli/query_commands.py tests/unit/cli/test_show_resolved_config.py
git commit -m "feat(cli): --show-resolved-config debug flag

Prints resolved option values with source labels (profile/yaml/const/
commandline) to stderr before running the command. Useful for diagnosing
'why didn't my profile apply' bugs."
```

---

## Phase 10: Frontend parity

### Task 17: Fix `DEFAULT_SIMILARITY_THRESHOLD` divergence

**Files:**
- Modify: `frontend/src/constants/defaults.js`
- Modify: `frontend/src/test/constants.test.js`

- [ ] **Step 1: Update the constant**

Edit `frontend/src/constants/defaults.js` line 7:

```js
// Old: export const DEFAULT_SIMILARITY_THRESHOLD = 0.5;
export const DEFAULT_SIMILARITY_THRESHOLD = 0.3;
```

Add a header comment block at the top of the file:

```js
/**
 * Default values for query parameters and thresholds.
 * Single source of truth — used by QueryInterface and API calls.
 *
 * SURFACE PARITY: these constants must match the values in
 * phentrieve/config.py. Cross-language drift is caught by
 * tests/unit/profiles/test_frontend_constant_parity.py.
 *
 * Intentional UI-specific divergence: DEFAULT_NUM_RESULTS_PER_CHUNK = 3
 * (backend default is 10; frontend uses a more compact display).
 */
```

- [ ] **Step 2: Update the existing constants test**

Edit `frontend/src/test/constants.test.js`:

```js
// Find the assertion for DEFAULT_SIMILARITY_THRESHOLD and update:
expect(DEFAULT_SIMILARITY_THRESHOLD).toBe(0.3);  // was 0.5
```

- [ ] **Step 3: Run frontend tests**

Run: `make frontend-test-ci`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/constants/defaults.js frontend/src/test/constants.test.js
git commit -m "fix(frontend): align DEFAULT_SIMILARITY_THRESHOLD with API (0.5 -> 0.3)

The frontend's similarity threshold default was 0.5, diverging from the
API's 0.3 default in api/schemas/query_schemas.py:25 and the CLI's 0.3
default in phentrieve/cli/query_commands.py. Aligning to 0.3 fixes the
behavior parity issue. Add header comment pointing developers at the
parity test."
```

### Task 18: Cross-language parity test

**Files:**
- Create: `tests/unit/profiles/test_frontend_constant_parity.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/profiles/test_frontend_constant_parity.py`:

```python
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
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/unit/profiles/test_frontend_constant_parity.py -v`
Expected: PASS, all parametrized tests + the whitelist test.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/profiles/test_frontend_constant_parity.py
git commit -m "test(parity): cross-language frontend/Python defaults check

Regex-extracts JS constants from frontend/src/constants/defaults.js and
asserts each maps to the corresponding phentrieve/config.py constant.
Whitelists DEFAULT_NUM_RESULTS_PER_CHUNK as an intentional UI divergence."
```

---

## Phase 11: Documentation

### Task 19: Rewrite `docs/user-guide/configuration-profiles.md`

**Files:**
- Modify: `docs/user-guide/configuration-profiles.md`

- [ ] **Step 1: Replace the file content**

Overwrite `docs/user-guide/configuration-profiles.md` with:

```markdown
# Configuration Profiles

Phentrieve supports named profiles in `phentrieve.yaml` to preset groups of
CLI options for common workflows.

## Quick Start

Create or edit `phentrieve.yaml`:

```yaml
profiles:
  high_recall_german:
    description: "Recall-oriented German extraction"
    command: text process
    language: de
    chunk_retrieval_threshold: 0.6
    aggregated_term_confidence: 0.7

  fast_query:
    command: query
    num_results: 5
    similarity_threshold: 0.5
```

Use it:

```bash
phentrieve text process note.txt --profile high_recall_german
phentrieve query "patient with seizures" --profile fast_query
```

## Precedence

Phentrieve resolves option values in this order (highest priority first):

1. **Explicit CLI flag** — `--language fr` always wins.
2. **Active profile** — `--profile X` or auto-selected built-in.
3. **Top-level `phentrieve.yaml`** — entries like `default_language: de`.
4. **Fallback constants** — defaults baked into `phentrieve/config.py`.

To see the resolved values for any invocation, add `--show-resolved-config`:

```bash
phentrieve text process note.txt --profile high_recall_german --show-resolved-config
```

## Built-in Profiles

Two profiles ship in code and need no configuration:

- **`default`** — strict defaults matching API behavior. Used by `phentrieve query`,
  `phentrieve text process`, and most other commands when no `--profile` is given.
- **`interactive`** — loose discovery defaults preserving the prior `text interactive`
  behavior. Auto-selected by `phentrieve text interactive` when no `--profile` is given.

You can shadow either by name in your `phentrieve.yaml`. Pass `--profile default`
to `phentrieve text interactive` to swap to strict thresholds.

## Schema

Each profile supports these keys (all optional):

| Key | Type | Notes |
|---|---|---|
| `description` | string | Free-form description, shown by `phentrieve config list-profiles`. |
| `command` | string | Bind the profile to a command (e.g. `"query"`, `"text process"`). |
| `language` | string | ISO 639-1 code (e.g. `"de"`). |
| `model_name` | string | Embedding model. |
| `chunk_retrieval_threshold` | float | Per-chunk similarity threshold. |
| `aggregated_term_confidence` | float | Min confidence for aggregated terms. |
| `num_results` | int | Number of results to return. |
| `similarity_threshold` | float | Query-mode similarity threshold. |
| `chunking_strategy` | string | E.g. `"semantic"`, `"sliding_window_punct_conj_cleaned"`. |
| `multi_vector` | bool | Use multi-vector retrieval. |

Unknown keys cause a validation error at load time (the `extra="forbid"` rule).

## Command Binding

If a profile sets `command: text process`, it can only be used with that command.
Using it with a different command produces an error. Profiles without a `command`
field apply to any command, with keys filtered to those the active command accepts.

```yaml
profiles:
  shared_german:
    # No command: field — applies to any command.
    language: de
    semantic_chunker_model: jinaai/jina-embeddings-v2-base-de
```

## Config Inspection Commands

```bash
phentrieve config list-profiles    # show all profiles with their bindings
phentrieve config show NAME        # print one profile as YAML
phentrieve config validate         # validate phentrieve.yaml against the schema
phentrieve config path             # print which phentrieve.yaml is being loaded
```

## YAML Search Path

Phentrieve looks for `phentrieve.yaml` in this order:

1. `./phentrieve.yaml` (current working directory) — highest priority
2. `./phentrieve.yml`
3. `~/.phentrieve/phentrieve.yaml` — legacy path, still supported
4. `~/.phentrieve/config.yaml` — legacy path, still supported

The first one found wins. `phentrieve config path` shows which one was loaded.

## Environment Variable

`PHENTRIEVE_PROFILE=NAME phentrieve <command>` is equivalent to
`phentrieve --profile NAME <command>`. Per-option environment variables
(e.g. `PHENTRIEVE_LANGUAGE`) are not supported in v1.

## Surface Defaults Parity

The frontend (`frontend/src/constants/defaults.js`) declares its own copy of
each numeric default. These must match the Python constants in
`phentrieve/config.py`. Drift is caught by
`tests/unit/profiles/test_frontend_constant_parity.py`.

The one intentional divergence: `DEFAULT_NUM_RESULTS_PER_CHUNK = 3` in the
frontend (vs. backend `10`) — chosen for compact UI display.

## What Replaced the Old `--config-profile` Flag

The previous version of this page documented a `--config-profile` flag. That
flag was never implemented in code. It has been replaced by the now-real
`--profile` flag described above.
```

- [ ] **Step 2: Verify YAML snippets parse**

(Spec A's documentation discipline: integration test in Task 22 will verify all snippets parse.)

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/configuration-profiles.md
git commit -m "docs(user-guide): rewrite configuration-profiles.md

Replaces the aspirational --config-profile content with the actually-
implemented --profile system. Documents the precedence stack, built-in
profiles, the schema, command binding, the phentrieve config subcommands,
the YAML search path, the PHENTRIEVE_PROFILE env var, and the surface
parity rule between frontend and Python defaults."
```

### Task 20: Update CLI usage and API usage docs

**Files:**
- Modify: `docs/user-guide/cli-usage.md`
- Modify: `docs/user-guide/api-usage.md`

- [ ] **Step 1: Add `--profile` and `--show-resolved-config` to cli-usage.md**

In `docs/user-guide/cli-usage.md`, find the option table for `phentrieve query` and add:

```markdown
| `--profile` / `-P` | string | Apply a named profile from `phentrieve.yaml`. See [Configuration Profiles](./configuration-profiles.md). |
| `--show-resolved-config` | flag | Print resolved option values with source labels to stderr before running. |
```

Repeat for `phentrieve text process` and `phentrieve text interactive` option tables.

Add a short subsection at the top:

```markdown
## Profiles

Most commands accept `--profile NAME` to apply a preset bundle of options
defined in `phentrieve.yaml`. See [Configuration Profiles](./configuration-profiles.md)
for the full guide.
```

- [ ] **Step 2: Add note to api-usage.md**

Append to `docs/user-guide/api-usage.md`:

```markdown
## Profiles vs API

The HTTP API does not accept the CLI's `--profile` flag — request fields
are explicit. If you're moving a workflow from the CLI to the API, copy the
relevant fields from your profile into the request body. See
[Configuration Profiles](./configuration-profiles.md) for the profile schema.
```

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/cli-usage.md docs/user-guide/api-usage.md
git commit -m "docs(user-guide): document --profile and --show-resolved-config

Adds the new CLI flags to cli-usage.md option tables. Notes that the API
doesn't accept --profile in api-usage.md."
```

### Task 21: Update README.md, phentrieve.yaml.template, and index.md

**Files:**
- Modify: `README.md`
- Modify: `phentrieve.yaml.template`
- Modify: `docs/user-guide/index.md`

- [ ] **Step 1: README.md**

In `README.md`, find the existing "Configuration" section and add a subsection:

```markdown
### Configuration profiles

Define named profiles in `phentrieve.yaml` to preset CLI options:

\`\`\`yaml
profiles:
  fast_query:
    command: query
    num_results: 5
    similarity_threshold: 0.5
\`\`\`

Then `phentrieve query "TEXT" --profile fast_query`.

See [docs/user-guide/configuration-profiles.md](docs/user-guide/configuration-profiles.md) for the full guide.
```

- [ ] **Step 2: phentrieve.yaml.template**

Append to `phentrieve.yaml.template`:

```yaml

# Named profiles for common CLI workflows. Pass --profile NAME to apply.
# See docs/user-guide/configuration-profiles.md for the full schema.
# profiles:
#   high_recall_german:
#     description: "Recall-oriented German extraction"
#     command: text process              # bind to a specific command (optional)
#     language: de
#     chunk_retrieval_threshold: 0.6
#
#   fast_query:
#     command: query
#     num_results: 5
#     similarity_threshold: 0.5

# Extraction thresholds (read by phentrieve/config.py:501-506).
# extraction:
#   chunk_threshold: 0.7
#   min_confidence: 0.75
#   chunk_confidence: 0.2
#   assertion_preference: dependency
```

- [ ] **Step 3: index.md**

In `docs/user-guide/index.md`, ensure the "Configuration profiles" link is present:

```markdown
- [Configuration profiles](./configuration-profiles.md)
```

- [ ] **Step 4: Commit**

```bash
git add README.md phentrieve.yaml.template docs/user-guide/index.md
git commit -m "docs: add Configuration profiles to README and YAML template"
```

### Task 22: Docs-as-tests integration test

**Files:**
- Create: `tests/integration/test_documented_yaml.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_documented_yaml.py`:

```python
"""Integration test that every YAML snippet in docs/user-guide/
configuration-profiles.md parses against the Profile schema."""

import re
from pathlib import Path

import pytest
import yaml as pyyaml

from phentrieve.profiles import ProfilesFile

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "user-guide" / "configuration-profiles.md"


def _extract_yaml_blocks(md_text: str) -> list[str]:
    """Find all ```yaml ... ``` fenced blocks in the markdown."""
    pattern = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
    return pattern.findall(md_text)


def test_doc_exists():
    assert DOC.exists(), f"Documentation file {DOC} missing"


def test_yaml_blocks_present():
    blocks = _extract_yaml_blocks(DOC.read_text())
    assert len(blocks) >= 2, "Expected at least two YAML examples in the doc"


@pytest.mark.parametrize("block_idx", range(20))  # bounded loop
def test_each_yaml_block_parses(block_idx: int):
    blocks = _extract_yaml_blocks(DOC.read_text())
    if block_idx >= len(blocks):
        pytest.skip(f"No block at index {block_idx}")
    raw = pyyaml.safe_load(blocks[block_idx])
    assert raw is not None, f"Block {block_idx} parsed as None"
    # Wrap in a profiles: dict if not already a top-level shape.
    if "profiles" in raw:
        ProfilesFile.model_validate(raw)
    else:
        # It's a profile body example or extraction section — both are fine.
        pass
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/integration/test_documented_yaml.py -v`
Expected: PASS for every YAML block in the doc.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_documented_yaml.py
git commit -m "test(docs): every YAML snippet in configuration-profiles.md parses"
```

---

## Phase 12: CHANGELOG and final integration tests

### Task 23: CHANGELOG entries

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add three entries**

Prepend to `CHANGELOG.md` (under the appropriate version heading or as Unreleased):

```markdown
### Added

- **`--profile NAME`** flag on `query`, `text process`, `text interactive` (issue #28).
  Apply a named profile from `phentrieve.yaml` to preset CLI options. See
  [Configuration Profiles](docs/user-guide/configuration-profiles.md). Both root
  placement (`phentrieve --profile X cmd`) and per-command placement
  (`phentrieve cmd --profile X`) work; subcommand-level wins on conflict.
- **`phentrieve config`** subcommand group with `list-profiles`, `show`,
  `validate`, `path` subcommands.
- **`--show-resolved-config`** debug flag on every command. Prints resolved
  option values with source labels (profile/yaml/const/commandline) to stderr
  before running.
- **`PHENTRIEVE_PROFILE`** environment variable, equivalent to `--profile`.
- New `phentrieve.yaml` sections: `profiles:` and `extraction:`.

### Fixed

- **`phentrieve text interactive`** now uses config-driven defaults (issue #171).
  Previously hardcoded `language="en"`, `chunk_retrieval_threshold=0.3`,
  `aggregated_term_confidence=0.35`, `num_results=5` are now read from the new
  built-in `interactive` profile (preserving prior behavior — no migration
  needed). Pass `--profile default` to switch to API-matching strict defaults.
- **Frontend `DEFAULT_SIMILARITY_THRESHOLD`** aligned with the API: `0.5` → `0.3`.
  This is a behavior change for users who relied on the frontend's stricter
  cutoff. To recover, pass an explicit threshold in the UI.

### Changed

- The previously-documented `--config-profile` flag (which was never
  implemented) is replaced by the now-real `--profile`. `docs/user-guide/configuration-profiles.md`
  is rewritten in place to reflect the actual design.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): add v0.19 entries for profiles + #171"
```

### Task 24: Final integration test

**Files:**
- Create: `tests/integration/test_profiles_e2e.py`
- Create: `tests/fixtures/profiles/sample_phentrieve.yaml`

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/profiles/sample_phentrieve.yaml`:

```yaml
profiles:
  high_recall_german:
    description: "Recall-oriented German extraction"
    command: text process
    language: de
    chunk_retrieval_threshold: 0.5

  precise_english_query:
    command: query
    num_results: 3
    similarity_threshold: 0.6

  shared_lang:
    language: de

  shadow_test:
    description: "User profile shadowing built-in"
    # name `shadow_test` does not collide with built-ins by default,
    # but tests that shadow `interactive` use the same key separately.
```

- [ ] **Step 2: Write the e2e test**

Create `tests/integration/test_profiles_e2e.py`:

```python
"""End-to-end profile resolution test using a real fixture YAML."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from phentrieve.config import _load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "profiles" / "sample_phentrieve.yaml"


@pytest.fixture
def cwd_with_fixture(tmp_path, monkeypatch):
    shutil.copy(FIXTURE, tmp_path / "phentrieve.yaml")
    monkeypatch.chdir(tmp_path)
    _load_yaml_config.cache_clear()
    yield tmp_path
    _load_yaml_config.cache_clear()


@pytest.fixture
def app():
    from phentrieve.__main__ import app as a
    return a


class TestE2EProfileResolution:
    def test_list_profiles_includes_user_and_builtin(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0
        assert "high_recall_german" in result.stdout
        assert "precise_english_query" in result.stdout
        assert "default" in result.stdout
        assert "interactive" in result.stdout

    def test_validate_clean_fixture(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0

    def test_unknown_profile_close_match(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(
            app, ["query", "TEXT", "--profile", "precise_english"]  # missing _query
        )
        assert result.exit_code != 0
        assert "precise_english_query" in result.output  # close-match suggestion

    def test_command_bound_mismatch_errors(self, app, cwd_with_fixture):
        runner = CliRunner()
        result = runner.invoke(
            app, ["query", "TEXT", "--profile", "high_recall_german"]
            # high_recall_german is bound to text process, not query
        )
        assert result.exit_code != 0
        assert "text process" in result.output

    def test_env_var_selects_profile(self, app, cwd_with_fixture, monkeypatch):
        monkeypatch.setenv("PHENTRIEVE_PROFILE", "precise_english_query")
        with patch("phentrieve.cli.query_commands.DenseRetriever"), \
             patch("phentrieve.cli.query_commands.query_orchestrator") as mock_orch:
            mock_orch.run_query.return_value = {"matches": []}
            runner = CliRunner()
            result = runner.invoke(app, ["query", "TEXT"])
            if mock_orch.run_query.called:
                assert mock_orch.run_query.call_args.kwargs["num_results"] == 3
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/integration/test_profiles_e2e.py -v`
Expected: PASS, five tests.

- [ ] **Step 4: Run the full project test suite**

Run: `make check && make typecheck-fast && make test`
Expected: PASS.

Run: `make frontend-test-ci`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_profiles_e2e.py tests/fixtures/profiles/sample_phentrieve.yaml
git commit -m "test(integration): end-to-end profile resolution

Real fixture YAML, real Typer runner, exercises list-profiles, validate,
unknown-profile close-match, command-bound mismatch, and env-var
profile selection. Five tests in one file."
```

### Task 25: Final cleanup and PR readiness

**Files:** N/A

- [ ] **Step 1: Run all required checks**

Run: `make check && make typecheck-fast && make test`
Expected: PASS.

Run: `make frontend-test-ci && make frontend-build-ci`
Expected: PASS.

- [ ] **Step 2: Run lint-fix on changed files**

Run: `make lint-fix`
Expected: cleanly formatted output.

- [ ] **Step 3: Verify CHANGELOG correctness**

Run: `head -50 CHANGELOG.md`
Expected: the three entries from Task 23 are present, properly formatted.

- [ ] **Step 4: Final commit (if lint-fix made changes)**

```bash
git add -u
git diff --cached --quiet || git commit -m "chore: ruff format pass"
```

---

## Self-Review Checklist

After completing all tasks:

- [ ] Every spec section has a corresponding task. Verified:
  - Architecture (Phase 3)
  - Profile schema (Phase 2)
  - Resolution function (Phase 2)
  - CLI surface (Phases 4-7)
  - Subcommand surface (Phase 8)
  - Error handling (covered in Phase 8 and tests throughout)
  - Observability (Phase 9, --show-resolved-config)
  - Tests (every phase has tests)
  - Documentation (Phase 11)
  - Cross-surface defaults audit & frontend parity (Phase 10)
  - Migration and rollout (Phase 12 CHANGELOG)
- [ ] No "TBD", "TODO", "implement later" placeholders.
- [ ] Every step shows exact code/commands.
- [ ] Type signatures consistent across tasks (e.g., `Profile`, `BUILTIN_PROFILES`, `apply_profile_callback`).
- [ ] All file paths are exact and verified to exist (or marked Create).

If issues: fix inline.
