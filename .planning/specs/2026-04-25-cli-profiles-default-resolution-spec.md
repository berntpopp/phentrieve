# Spec: CLI Profiles and Default Resolution

- **Date**: 2026-04-25
- **Status**: Draft (awaiting user review)
- **Issues addressed**: #28 (CLI profiles), #171 (interactive defaults divergence)
- **Companion spec**: `2026-04-25-adaptive-rechunking-spec.md`

## Goal

Replace the current ad hoc, per-command default resolution with a single layered configuration system. Profiles defined in `phentrieve.yaml` preset groups of CLI options for common workflows. Built-in profiles ship in code so the default behavior of `phentrieve text interactive` becomes deterministic and consistent with the rest of the CLI.

## Reconciling existing `configuration-profiles.md`

`docs/user-guide/configuration-profiles.md` already exists and documents an aspirational `--config-profile` flag. The flag is **not implemented in code** — `grep -rn "config-profile" phentrieve/ api/` returns zero hits. The doc describes a system that was never built. No users could depend on a flag that doesn't exist, so this spec fully supersedes that documentation without a deprecation alias.

Distinct from the unimplemented flag, two things from that doc *are* real and continue to work unchanged:

- The `~/.phentrieve/config.yaml` and `~/.phentrieve/phentrieve.yaml` paths *are* searched today by `phentrieve.utils.get_config_paths()` (`utils.py:152-183`). They remain supported as legacy YAML locations alongside `./phentrieve.yaml`. The rewrite documents this as the YAML search path, not as a separate "profile config" file.
- The `--config-file` flag on `phentrieve text process` (`text_commands.py:293`) loads a chunking-pipeline JSON config and is unrelated to `--profile`. Preserved unchanged.

The existing doc file is rewritten in place. CHANGELOG calls out that the previously-documented `--config-profile` flag has been replaced by `--profile` (now actually implemented).

## Non-goals

- Profile support for `phentrieve text chunk`, `benchmark run`, `index *`, `data *`, `similarity *`, `mcp *` (deferred — see Future work).
- Migrating environment-variable handling (only the new `PHENTRIEVE_PROFILE` env var is added).
- A `phentrieve config calibrate-thresholds` subcommand (deferred — see Future work).
- A JSON Schema file for `phentrieve.yaml` (deferred).

## Architecture

A new module `phentrieve/profiles.py` owns the profile data model and resolution logic. A new module `phentrieve/cli/_profile.py` owns the Typer integration glue. Together they implement a four-level precedence stack:

```
explicit CLI flag  >  active profile  >  top-level YAML  >  fallback constants
```

(Per-option environment variables like `PHENTRIEVE_LANGUAGE` are **not** supported in v1. The only env var added is `PHENTRIEVE_PROFILE`, which selects the active profile. Per-option env vars are deferred — see Future work.)

### Resolution mechanism — `default_map` populated at the right time

Click resolves option defaults **before** running the command function body. So `default_map` must be populated before Click's option resolution for the active subcommand happens — not inside the subcommand body. The earlier draft's "lazy auto-selection inside the command body" is incorrect; this section corrects it.

The mechanism has four rules that must hold together:

1. **Profileable Typer options have `None` as their default**, not a literal value. Today `phentrieve text interactive` declares `language: str = "en"`, which means Typer always passes `"en"` to the function whether or not the user typed `--language`. We change these to `None`. Fallback values move into the layered `default_map` chain.
2. **Each profile-aware subcommand declares its own `--profile` option** with an `is_eager=True` callback and a built-in default name (`default` for most commands; `interactive` for `phentrieve text interactive`). Because the option is eager, its callback fires before Click resolves the subcommand's other options. The default name is what makes auto-selection work — when the user doesn't pass `--profile`, the eager callback receives the built-in name and populates `default_map` for that profile. When the user passes `--profile X`, the callback receives `X` and populates from that.
3. **The root callback also has `--profile`** with the same callback. This makes `phentrieve --profile X text process` work: the root callback fires first, writes the profile into `default_map`, and the subcommand's per-command default of `"default"` / `"interactive"` is then ignored because Click's nested `default_map` already contains the resolved values. (Click reads `default_map` for non-explicit options regardless of where the value originated.) Net effect: subcommand-level `--profile` wins over root-level; root-level wins over no flag at all; no flag uses the per-command built-in default.
4. **The eager callback walks the precedence stack** for each profileable option name: explicit profile (if `--profile` was passed) → top-level YAML → fallback constant. It writes the highest-priority non-`None` value into `default_map[option_name]` (or nested under `default_map[subcommand_name][option_name]` for subcommand-scoped resolution). Click's parameter resolution then uses these as the option's effective default. Explicit CLI flags still win because Click consults `default_map` only when the user didn't pass the flag.

**There is no `resolve_option` helper, and no "lazy auto-selection inside the command body".** Both were broken patterns in earlier drafts. Function bodies just use the value Click resolved. The eager `--profile` callback is the single point that populates `default_map`.

For options whose only fallback is a `phentrieve.config` constant (no profile or YAML path), the `default_map` chain ends with that constant. Constants stay the canonical source for unprofileable defaults like model paths.

### Source labeling — sidecar map plus `ParameterSource`

`ctx.get_parameter_source(name)` returns one of `COMMANDLINE`, `ENVIRONMENT`, `DEFAULT_MAP`, `DEFAULT` — useful but **not sufficient**. Once profile, YAML, and constant values are merged into `default_map`, they all report as `DEFAULT_MAP`. To distinguish them, the eager callback maintains a **sidecar source map** in `ctx.obj["resolved_sources"]` — a dict `{option_name: source_label}` populated as the callback walks the stack:

```python
ctx.obj["resolved_sources"] = {
    "language": "profile:high_recall_german",
    "chunk_retrieval_threshold": "profile:builtin:interactive",
    "num_results": "yaml:default_top_k",
    "model_name": "const:DEFAULT_MODEL",
}
```

`--show-resolved-config` uses this sidecar map for fine-grained labels, and `ParameterSource` for the user-vs-injected distinction (`COMMANDLINE` overrides any sidecar entry — the user typed it). Tests assert both signals.

### `--profile` placement

The flag is registered **both** on the root Typer callback and on each profile-aware subcommand (`query`, `text process`, `text interactive`). Both forms work; subcommand-level wins on conflict. The eager callback is the same function and is idempotent — calling it twice with the same effective value is a no-op.

### Built-in profiles

Two built-in profiles ship in code as Python `dict` literals constructed into `Profile` instances at import time:
- `default` — all fields `None` (falls through to YAML → fallback constants).
- `interactive` — loose discovery defaults preserving prior `text interactive` behavior: `chunk_retrieval_threshold=0.3`, `aggregated_term_confidence=0.35`, `num_results=5`.

User profiles in `phentrieve.yaml` can shadow built-ins by name. Shadowing is logged at INFO on first use per session.

A new `phentrieve config` subcommand group hosts inspection commands. A global `--show-resolved-config` flag, available on every command, prints the merged option set with source labels before executing the command (still executes — not a dry-run).

## Profile schema

Pydantic model in `phentrieve/profiles.py` with `extra="forbid"` so YAML typos error at load time, not at use site:

```python
class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    command: str | None = None      # e.g. "text process", "query", "text interactive"

    # Shared option keys, all optional
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

    # Adaptive rechunking block — imported from
    # phentrieve.retrieval.adaptive_rechunker (defined in Spec B)
    adaptive_rechunking: AdaptiveRechunkingProfileBlock | None = None

class ProfilesFile(BaseModel):
    model_config = ConfigDict(extra="ignore")  # other top-level YAML keys are fine
    profiles: dict[str, Profile] = Field(default_factory=dict)
```

YAML structure (additive — existing `phentrieve.yaml` is unchanged):

```yaml
profiles:
  high_recall_german:
    description: "Recall-oriented German extraction with semantic chunking"
    command: text process
    language: de
    chunking_strategy: semantic
    semantic_chunker_model: jinaai/jina-embeddings-v2-base-de
    chunk_retrieval_threshold: 0.6
    aggregated_term_confidence: 0.7

  precise_english_query:
    command: query
    similarity_threshold: 0.5
    num_results: 5

  shared_german:
    # No `command:` — applies to any command, with key filtering
    language: de
    semantic_chunker_model: jinaai/jina-embeddings-v2-base-de
```

Profile binding: `command` is optional. If present, the profile applies only to that command (mismatch raises). If absent, the profile applies to any command and silently filters keys to those the active command accepts.

## Resolution function

```python
def resolve_profile_for_command(
    profile_name: str | None,
    command_path: tuple[str, ...],   # e.g. ("text", "process")
    accepted_keys: set[str],         # discovered from the command's Click params
) -> tuple[Profile, dict[str, Any]]:
    """Returns (profile_object, applicable_kwargs_for_default_map)."""
```

Behaviour:
- If `profile_name is None`: returns the appropriate built-in (`default` for most, `interactive` for `text interactive`).
- If `profile_name` not found: `typer.BadParameter` with the requested name echoed and `difflib.get_close_matches(name, available, n=1, cutoff=0.6)` for "Did you mean: …?". Exit code 2.
- If `profile.command` is set and doesn't match `command_path`: `typer.BadParameter` with the bound command name.
- If `profile.command is None`: silently filter to `accepted_keys`.

Built-in shadowing: a user profile in `phentrieve.yaml` named `interactive` or `default` shadows the built-in. Logged at INFO on first use.

## CLI surface

Commands gaining `--profile` and `--show-resolved-config` in v1:

| Command | Why | Change |
|---|---|---|
| `phentrieve query` | #28 explicit | Add `--profile`. Replace hardcoded `num_results=10`, `similarity_threshold=0.3`, `multi_vector=False`, `aggregation_strategy=...` with `DEFAULT_*` constants. Wire `get_active_profile()` for `default` fallback. |
| `phentrieve text process` | #28 explicit | Add `--profile`. Fix `language="en"` → `DEFAULT_LANGUAGE`, `num_results=10` → `DEFAULT_TOP_K`, `chunk_confidence=0.2` → new `DEFAULT_CHUNK_CONFIDENCE` constant, `assertion_preference="dependency"` → `DEFAULT_ASSERTION_PREFERENCE` constant. |
| `phentrieve text interactive` | #171 | Add `--profile`. Replace all hardcoded literals with config constants. Auto-select built-in `interactive` profile when `--profile` not given. |

Commands not getting `--profile` in v1: `text chunk`, `benchmark run`, `index *`, `data *`, `similarity *`, `mcp *`.

New constants in `phentrieve/config.py`, each with `_FALLBACK` and `get_config_value(...)` resolver:

```python
DEFAULT_NUM_RESULTS              # alias of DEFAULT_TOP_K, exported for CLI symmetry
DEFAULT_CHUNK_CONFIDENCE         # 0.2 (current literal in text process)
DEFAULT_ASSERTION_PREFERENCE     # "dependency"
DEFAULT_OUTPUT_FORMAT_QUERY      # "text"
DEFAULT_OUTPUT_FORMAT_PROCESS    # "json_lines"
```

YAML additions (additive). **Note**: this spec preserves the YAML key names that `phentrieve/config.py:502,505` already reads — `chunk_threshold` and `min_confidence`, *not* `chunk_retrieval_threshold` / `min_confidence_aggregated`. Renaming would silently break any user `phentrieve.yaml` that already uses the existing keys. The implementation can document both names as accepted aliases (with the new long names preferred for clarity), but v1 sticks with the existing keys to avoid migration risk.

```yaml
# New top-level section that backs the existing config.py reads at lines 501-506.
# Key names match phentrieve/config.py:502,505 exactly.
extraction:
  chunk_threshold: 0.7        # → DEFAULT_CHUNK_RETRIEVAL_THRESHOLD (existing)
  min_confidence: 0.75        # → DEFAULT_MIN_CONFIDENCE_AGGREGATED (existing)
  chunk_confidence: 0.2       # → new DEFAULT_CHUNK_CONFIDENCE constant
  assertion_preference: dependency   # → new DEFAULT_ASSERTION_PREFERENCE constant
```

The Profile schema field names use the longer, self-explanatory CLI-aligned names (`chunk_retrieval_threshold`, `aggregated_term_confidence`) since profiles are user-authored and benefit from clarity. The `extraction:` top-level section uses the historical short keys for backward-compat. The implementation maps between the two at load time.

`text interactive` resolution shape (post-refactor) — note **no auto-selection helper inside the function body**:

```python
@app.command("interactive")
def interactive_text_mode(
    ctx: typer.Context,
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            envvar="PHENTRIEVE_PROFILE",
            callback=apply_profile_callback,
            is_eager=True,
            help="Apply a named profile from phentrieve.yaml.",
        ),
    ] = "interactive",                                                  # built-in default for this command
    language: Annotated[str | None, typer.Option(...)] = None,           # was "en"
    chunk_retrieval_threshold: Annotated[float | None, typer.Option(...)] = None,  # was 0.3
    aggregated_term_confidence: Annotated[float | None, typer.Option(...)] = None, # was 0.35
    num_results: Annotated[int | None, typer.Option(...)] = None,        # was 5
    # ...
) -> None:
    # By the time the function body runs, the eager --profile callback has
    # already populated ctx.default_map and Click has already resolved the
    # other options. None here means "not set anywhere in the precedence
    # stack" — fall back to the config constant.
    language = language if language is not None else DEFAULT_LANGUAGE
    chunk_retrieval_threshold = (
        chunk_retrieval_threshold
        if chunk_retrieval_threshold is not None
        else DEFAULT_CHUNK_RETRIEVAL_THRESHOLD
    )
    # ... etc
```

The single `value if value is not None else CONSTANT` pattern at the end is the only fallback the function body needs. Profile values arrive via Click's `default_map`; the constant catches the case where neither flag, profile, nor YAML supplied a value. **No `resolve_option` helper, no triple-fallback, no body-level auto-selection.** Auto-selection happens via the per-command `--profile` default name being `"interactive"` (or `"default"` for other commands), which feeds into the eager callback before option resolution.

## Subcommand surface

`phentrieve config` (new sub-app):
- `phentrieve config list-profiles` — table with name, source (built-in / user), command binding, key options, description.
- `phentrieve config show <name>` — full resolved Profile printed as YAML.
- `phentrieve config validate` — runs `ProfilesFile.model_validate(yaml_data)` and reports errors with line:column if `ruamel.yaml` is available, plain message otherwise.
- `phentrieve config path` — prints the active YAML search path and which file was actually loaded.

Cache invalidation: `_load_yaml_config.cache_clear()` is called at the start of `phentrieve config validate` and `list-profiles` so they read fresh state. Other commands keep cached read.

## Error handling

| Condition | Response | Exit |
|---|---|---|
| `--profile` name not found | `typer.BadParameter` with close-match suggestion | 2 |
| Profile is command-bound, command mismatches | `typer.BadParameter` echoing bound command | 2 |
| YAML parse error | `typer.Exit(code=1)` with line:column | 1 |
| Pydantic validation error in `phentrieve.yaml` | At load: WARNING log, skip invalid profile, other profiles still load. `phentrieve config validate` raises non-zero. | 0 (or 1 from validate) |
| Built-in shadowing | INFO log first use per session | 0 |
| Empty YAML / no `profiles:` block | Silent, falls back to built-ins | 0 |

## Observability

`--show-resolved-config` flag (root-level, available on every command). Prints to stderr before command body runs:

```
Resolved configuration for `phentrieve text interactive`:
  language                       de        ← profile:high_recall_german
  chunk_retrieval_threshold      0.6       ← profile:high_recall_german
  aggregated_term_confidence     0.35      ← profile:builtin:interactive
  num_results                    10        ← --num-results (commandline)
  semantic_chunker_model         BioLORD…  ← yaml:default_model
  output_format                  text      ← const:DEFAULT_OUTPUT_FORMAT
```

`logging.getLogger("phentrieve.profiles")` logs profile resolution at DEBUG. Existing `--debug` propagates.

No new metrics or telemetry.

## Tests

Coverage target: 100% for new modules, no drop on touched files. Tests must specifically prove that **omitted CLI args do not mask profile/YAML values** — this is the most likely failure mode of the resolution mechanism.

| Layer | Test file | What it covers |
|---|---|---|
| Unit | `tests/unit/profiles/test_profile_schema.py` | `extra="forbid"` rejects unknown keys; required-field omission allowed; `command` validation; built-in profile construction |
| Unit | `tests/unit/profiles/test_resolver.py` | Bound profile + matching command; bound profile + mismatched command (raises); cross-command profile + key filtering; unknown profile + close-match suggestion |
| Unit | `tests/unit/cli/test_profile_callback.py` | Eager callback populates `default_map` with merged precedence stack; missing profile errors with exit code 2 + close-match hint; `PHENTRIEVE_PROFILE` env var; per-subcommand `default_map` keys land in nested buckets |
| Unit | `tests/unit/cli/test_default_map_resolution.py` | **Core resolution invariant**: with a profile setting `language: de` and no `--language` flag, the function body sees `de`, not the Typer-default literal. With `--language fr`, function sees `fr` regardless of profile. With YAML `default_language: it` and no profile / no flag, function sees `it`. With nothing set anywhere, function sees the `phentrieve.config` constant. Each path is asserted via two signals: (a) `ctx.get_parameter_source(name)` returning `COMMANDLINE` vs `DEFAULT_MAP` for the user-vs-injected distinction, and (b) `ctx.obj["resolved_sources"][name]` returning the fine-grained label (`profile:foo`, `yaml:default_language`, `const:DEFAULT_LANGUAGE`). The sidecar source map is what makes profile/yaml/const distinguishable; `ParameterSource` alone collapses them all to `DEFAULT_MAP`. |
| Unit | `tests/unit/cli/test_show_resolved_config.py` | `--show-resolved-config` output for a typical invocation contains source labels matching `ctx.obj["resolved_sources"]` plus user-passed flags labeled as `← --flag (commandline)`. Asserts the output is written to stderr (not stdout, which is reserved for command output). |
| Unit | `tests/unit/cli/test_profile_placement.py` | Both `phentrieve --profile X text process` and `phentrieve text process --profile X` resolve identically. With `--profile A` at root and `--profile B` at the subcommand, the subcommand-level value (`B`) wins. With env var `PHENTRIEVE_PROFILE=Z` and an explicit flag, the explicit flag wins regardless of placement. |
| Unit | `tests/unit/cli/test_yaml_legacy_paths.py` | A `phentrieve.yaml` at `~/.phentrieve/phentrieve.yaml` (legacy path supported by `utils.get_config_paths`) is found and used when no `./phentrieve.yaml` exists; `./phentrieve.yaml` shadows the legacy path. |
| Unit | `tests/unit/cli/test_text_interactive.py` (extend) | Default invocation auto-selects `interactive`; `--profile default` swaps to strict; explicit flag beats both |
| Unit | `tests/unit/cli/test_query_enrichment.py` (extend) | `--profile precise_english_query` from fixture YAML applies; explicit `--num-results 3` overrides profile's `num_results: 5` |
| Unit | `tests/unit/cli/test_text_commands.py` (extend) | Same shape as query test; covers `text process`. `language` auto-detect kicks in when YAML doesn't set it. |
| Unit | `tests/unit/cli/test_config_commands.py` | `phentrieve config list-profiles` lists built-ins + user profiles + binding info; `show <name>` prints YAML; `validate` rejects bad YAML; `path` prints loaded file |
| Unit | `tests/unit/profiles/test_frontend_constant_parity.py` | Reads `frontend/src/constants/defaults.js` (regex-extract numeric literals) and asserts each maps to the correct `phentrieve/config.py` constant. Whitelists `DEFAULT_NUM_RESULTS_PER_CHUNK = 3` with rationale comment. Asserts post-fix `DEFAULT_SIMILARITY_THRESHOLD = 0.3`. |
| Integration | `tests/integration/test_profiles_e2e.py` | Real fixture `phentrieve.yaml` in tmp dir, real Typer runner, full precedence stack including `PHENTRIEVE_PROFILE` env var and `--show-resolved-config` |

Fixtures: `tests/fixtures/profiles/sample_phentrieve.yaml` covering bound, cross-command, and shadowing-builtin shapes. Pytest fixture `cli_with_yaml(tmp_path, yaml_content)` writes a YAML file in a temp cwd, clears `_load_yaml_config.cache_clear()`, and returns a `CliRunner`.

**Note: `test_resolve_option.py` from the previous draft is removed.** No `resolve_option` helper exists anymore — Click's `default_map` is the resolution mechanism. The replacement test is `test_default_map_resolution.py` above.

## Documentation

Existing files updated (all paths verified against the repo as of 2026-04-25):

- `README.md` — new "Configuration profiles" subsection under the existing "Configuration" heading; quick-start example linking to `docs/user-guide/configuration-profiles.md`.
- `docs/user-guide/configuration-profiles.md` — **rewritten in place** to replace the aspirational `--config-profile` content. New content: precedence stack diagram, full YAML schema table, all built-in profiles documented, three worked examples (high-recall German extraction, fast English query, shared cross-command profile), the `phentrieve config list-profiles` / `show` / `validate` / `path` subcommands, the `--show-resolved-config` debug flag.
- `docs/user-guide/cli-usage.md` — add `--profile` and `--show-resolved-config` to the per-command option tables for `query`, `text process`, `text interactive`. Add a one-paragraph cross-reference to `configuration-profiles.md`.
- `docs/user-guide/api-usage.md` — note that the API does not accept `--profile` (it's a CLI-only concern); request fields remain explicit. Cross-reference for users moving between CLI and API.
- `docs/user-guide/index.md` — add link to `configuration-profiles.md` if not already present.
- `phentrieve.yaml.template` — add commented `profiles:` block and `extraction:` section matching the user-guide examples; per-knob comment lines explain `command:` field and override behavior.
- `CHANGELOG.md` — three entries: profile system addition; `text interactive` defaults fix (#171); replacement of the previously-documented `--config-profile` flag by the now-implemented `--profile`.
- Each modified CLI command's Typer `help=` text — mentions `--profile` and `--show-resolved-config` briefly.

New files:
- `tests/fixtures/profiles/sample_phentrieve.yaml`.

Not created:
- Migration guide. Not needed — `interactive` built-in profile preserves prior behavior; the previously-documented `--config-profile` was never implemented.
- JSON Schema for `phentrieve.yaml`. Deferred.
- Tutorial / blog-style guide. The user-guide page covers the practical case.
- Standalone `profiles.md` — the existing `configuration-profiles.md` is the canonical page.

Documentation discipline: every YAML snippet in `docs/user-guide/configuration-profiles.md` is loaded by `tests/integration/test_documented_yaml.py` and asserted to parse + resolve cleanly. Docs cannot drift from working code.

## Cross-surface defaults audit

There are **four** surfaces today that declare defaults independently: `phentrieve/config.py`, the CLI (`phentrieve/cli/...`), the API (`api/schemas/...` and `api/routers/...`), and the frontend (`frontend/src/constants/defaults.js`). Each has drifted at least slightly. Verified against the tree as of 2026-04-25:

| Constant | `phentrieve/config.py` | `api/schemas/` | CLI | `frontend/src/constants/defaults.js` | Status |
|---|---|---|---|---|---|
| `chunk_retrieval_threshold` | `DEFAULT_CHUNK_RETRIEVAL_THRESHOLD = 0.7` | `0.7` (uses constant) | `0.7` (uses constant in `text process`) | `0.7` | aligned |
| `aggregated_term_confidence` | `DEFAULT_MIN_CONFIDENCE_AGGREGATED = 0.75` | `0.75` (uses constant) | `0.75` (uses constant) | `0.75` | aligned |
| `num_results` (top-level) | `DEFAULT_TOP_K = 10` | `10` (literal in `query_schemas.py:22`) | `10` (literal in `query_commands.py`) | `10` | aligned by value, **literal vs constant in API/CLI is the bug Spec A fixes** |
| `num_results_per_chunk` | (no constant) | `10` (literal in `text_processing_schemas.py:85`) | `10` (literal in `text process`) | `3` | **frontend intentional UI divergence** |
| `similarity_threshold` (query) | (no constant; CLI/API literal) | `0.3` (literal in `query_schemas.py:25`) | `0.3` (literal in `query_commands.py`) | `0.5` (`DEFAULT_SIMILARITY_THRESHOLD`) | **frontend bug — diverges from CLI/API** |
| `language` | `DEFAULT_LANGUAGE = "en"` | resolves auto-detect when null | `"en"` literal in 3 places | (n/a — UI provides language picker) | **CLI literal is the bug Spec A fixes** |
| `chunk_confidence` (text process) | (no constant) | (n/a) | `0.2` literal | (n/a) | **CLI literal — Spec A adds `DEFAULT_CHUNK_CONFIDENCE`** |
| `assertion_preference` | (lives inside `DEFAULT_ASSERTION_CONFIG` dict) | resolves from dict | `"dependency"` literal | (n/a) | **CLI literal — Spec A adds `DEFAULT_ASSERTION_PREFERENCE`** |

Conclusions:
- **`num_results_per_chunk = 3` in the frontend is an intentional UI choice** (compact display). Spec A adds a backend constant `DEFAULT_NUM_RESULTS_PER_CHUNK = 10` and explicitly whitelists the frontend's `3` as UX-specific in the parity test.
- **Frontend `DEFAULT_SIMILARITY_THRESHOLD = 0.5` vs CLI/API `0.3` is a bug**, not an intentional UX choice. Spec A treats this as an alignment fix: the frontend constant moves to `0.3` to match the API's `query_schemas.py:25`. This is a behavior change for users who relied on the frontend's 0.5 cutoff; documented in CHANGELOG.
- **API and CLI literals (`10`, `0.3`)** that happen to equal the corresponding `phentrieve/config.py` constants are still the wrong source-of-truth pattern. Spec A replaces them with constant references.

This spec does not introduce profile support in the frontend (frontend has no concept of named profiles in v1; the user always specifies options through the UI directly). However, it does:

1. **Add a cross-surface parity test** (`tests/unit/profiles/test_frontend_constant_parity.py`) that reads `frontend/src/constants/defaults.js` via regex-extraction and asserts each constant matches its Python counterpart. The intentional `DEFAULT_NUM_RESULTS_PER_CHUNK = 3` divergence is whitelisted with an inline rationale comment.
2. **Add a constant-source comment block** at the top of `frontend/src/constants/defaults.js` pointing developers at the parity test and at `phentrieve/config.py` as the upstream source.
3. **Document the parity rule** in a short "Surface defaults parity" subsection of `docs/user-guide/configuration-profiles.md` so frontend contributors know the rule.
4. **Fix the `DEFAULT_SIMILARITY_THRESHOLD` frontend divergence** by changing the constant from `0.5` to `0.3`. CHANGELOG entry: "Frontend query similarity threshold default aligned to API value (0.5 → 0.3)."

Profile support in the frontend (profile picker dropdown, profile-aware default reset) is deferred — see Future work.

## Migration and rollout

- Behavior change for users of `text interactive`: none. Built-in `interactive` profile preserves existing 0.3 / 0.35 / 5 defaults.
- Behavior change for users of `text process`: `language` now auto-detects when YAML/profile leave it unset (matches API). Other defaults unchanged.
- Behavior change for users of `query`: none (hardcoded → constant with same value).
- New env var: `PHENTRIEVE_PROFILE`.
- No data migration. No YAML migration — existing files keep working; new sections (`extraction:`, `profiles:`) are additive.
- Legacy YAML paths (`~/.phentrieve/phentrieve.yaml`, `~/.phentrieve/config.yaml`) continue to be searched by `phentrieve.utils.get_config_paths()`. Profiles defined there work the same as in `./phentrieve.yaml`. Documented in the rewritten `configuration-profiles.md` as the YAML search path.
- **Frontend behavior change**: `frontend/src/constants/defaults.js` `DEFAULT_SIMILARITY_THRESHOLD` aligned from `0.5` to `0.3` to match API/CLI. CHANGELOG entry calls this out separately.

## Future work

- **Profile support for additional commands**: `text chunk`, `benchmark run`, `index *`, `data *`, `similarity *`, `mcp *`. Each command's option shape is different; needs separate review per command. → GitHub issue.
- **`phentrieve config calibrate-thresholds` subcommand**: runs current model against a fixture set, suggests `quality_threshold` / `margin_threshold` values. Useful when users switch retrieval models. → GitHub issue (shared with Spec B).
- **Frontend profile support**: profile picker dropdown in the advanced-options UI, profile-aware default reset, optional saved active profile in browser storage. Deferred — backend profile system must land first; UX work is non-trivial. → GitHub issue.
- **JSON Schema for `phentrieve.yaml`**: auto-generated from pydantic for editor autocomplete. Low effort, YAGNI for v1. → Note here (no issue).
- **Profile inheritance / `extends:`**: Docker-compose-style profile composition. YAGNI for v1; revisit if profiles grow. → Note here.
- **Saved active profile** (`phentrieve config use <name>`): kubectl-style "default profile" written to YAML. Useful for users who have one workflow they always run. → Note here, revisit after measuring use.

## Open questions

None at spec-write time. All architecture and schema decisions are locked.
