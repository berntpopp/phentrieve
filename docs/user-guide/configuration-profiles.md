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

1. **Explicit CLI flag** - `--language fr` always wins.
2. **Active profile** - `--profile X` or auto-selected built-in.
3. **Top-level `phentrieve.yaml`** - entries like `default_language: de`.
4. **Fallback constants** - defaults baked into `phentrieve/config.py`.

To see the resolved values for any invocation, add `--show-resolved-config`:

```bash
phentrieve text process note.txt --profile high_recall_german --show-resolved-config
```

## Built-in Profiles

Two profiles ship in code and need no configuration:

- **`default`** - strict defaults matching API behavior. Used by `phentrieve query`,
  `phentrieve text process`, and most other commands when no `--profile` is given.
- **`interactive`** - loose discovery defaults preserving the prior `text interactive`
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
    # No command: field - applies to any command.
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

1. `./phentrieve.yaml` (current working directory) - highest priority
2. `./phentrieve.yml`
3. `~/.phentrieve/phentrieve.yaml` - legacy path, still supported
4. `~/.phentrieve/config.yaml` - legacy path, still supported

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
frontend (vs. backend `10`) - chosen for compact UI display.

## What Replaced the Old `--config-profile` Flag

The previous version of this page documented a `--config-profile` flag. That
flag was never implemented in code. It has been replaced by the now-real
`--profile` flag described above.
