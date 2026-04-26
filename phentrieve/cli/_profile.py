"""Eager --profile callback for the Phentrieve CLI.

Populates ``ctx.default_map`` with the merged precedence stack
(profile -> top-level YAML -> fallback constants) and maintains a
sidecar source map at ``ctx.obj['resolved_sources']`` so callers can
distinguish profile- vs YAML- vs constant-sourced defaults
(``ParameterSource`` alone collapses all three to ``DEFAULT_MAP``).

Implements Spec A's Architecture section:
``.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md``.
"""

from __future__ import annotations

import logging
from typing import Any

import click
import typer

from phentrieve import config as _config
from phentrieve.profiles import (
    Profile,
    merged_profiles,
    resolve_profile_for_command,
)

logger = logging.getLogger(__name__)


# Map from Profile field name to (yaml_path, fallback_constant_name).
# Used to walk the layered precedence stack: profile -> YAML -> constant.
# YAML keys for the extraction-block options use the SHORT names that
# phentrieve/config.py:514-525 already reads (chunk_threshold, min_confidence,
# assertion_preference) - NOT the longer Profile field names.
_OPTION_RESOLUTION_MAP: dict[str, tuple[str | tuple[str, ...], str]] = {
    "language": ("default_language", "DEFAULT_LANGUAGE"),
    "model_name": ("default_model", "DEFAULT_MODEL"),
    "retrieval_model": ("default_model", "DEFAULT_MODEL"),
    "semantic_chunker_model": ("default_model", "DEFAULT_MODEL"),
    "num_results": ("default_top_k", "DEFAULT_TOP_K"),
    "similarity_threshold": (
        "min_similarity_threshold",
        "MIN_SIMILARITY_THRESHOLD",
    ),
    "chunk_retrieval_threshold": (
        ("extraction", "chunk_threshold"),
        "DEFAULT_CHUNK_RETRIEVAL_THRESHOLD",
    ),
    "aggregated_term_confidence": (
        ("extraction", "min_confidence"),
        "DEFAULT_MIN_CONFIDENCE_AGGREGATED",
    ),
    "chunking_strategy": (
        "default_chunking_strategy",
        "DEFAULT_CHUNKING_STRATEGY",
    ),
    "assertion_preference": (
        ("extraction", "assertion_preference"),
        "DEFAULT_ASSERTION_PREFERENCE",
    ),
    "output_format": (
        ("output", "format_query"),
        "DEFAULT_OUTPUT_FORMAT_QUERY",
    ),
}


# Per-command overrides for fields whose YAML key / fallback constant
# depends on the active subcommand (e.g. ``output_format`` resolves to
# ``output.format_query`` for ``query`` but ``output.format_process`` for
# ``text process``). Lookup is keyed by the leaf command name extracted
# from ``command_path``.
_OPTION_COMMAND_OVERRIDES: dict[str, dict[str, tuple[str | tuple[str, ...], str]]] = {
    "output_format": {
        "process": (
            ("output", "format_process"),
            "DEFAULT_OUTPUT_FORMAT_PROCESS",
        ),
    },
}


def _walk_yaml(yaml_data: dict, path: str | tuple[str, ...]) -> Any:
    """Walk a nested YAML key path. Returns None if any segment is missing."""
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
    command_path: tuple[str, ...] = (),
) -> tuple[Any, str] | None:
    """Walk the precedence stack for a single option:
    profile -> YAML -> fallback constant.

    Returns ``(value, source_label)`` where source_label is one of:
    - ``"<profile-set>"`` (caller substitutes the named-profile label)
    - ``"yaml:<key>"`` for YAML-sourced
    - ``"const:<NAME>"`` for fallback constant

    Returns None if no layer in the stack supplies a value for this option.

    ``command_path`` lets command-specific overrides in
    ``_OPTION_COMMAND_OVERRIDES`` take precedence over the global mapping
    (e.g. ``output_format`` reads ``output.format_process`` for the
    ``text process`` subcommand instead of ``output.format_query``).
    """
    if field_name in profile_kwargs:
        return profile_kwargs[field_name], "<profile-set>"

    overrides = _OPTION_COMMAND_OVERRIDES.get(field_name, {})
    leaf = command_path[-1] if command_path else ""
    resolution = overrides.get(leaf) or _OPTION_RESOLUTION_MAP.get(field_name)
    if resolution is not None:
        yaml_path, const_name = resolution
        yaml_value = _walk_yaml(yaml_data, yaml_path)
        if yaml_value is not None:
            label = (
                f"yaml:{yaml_path}"
                if isinstance(yaml_path, str)
                else f"yaml:{'.'.join(yaml_path)}"
            )
            return yaml_value, label
        const_value = getattr(_config, const_name, None)
        if const_value is not None:
            return const_value, f"const:{const_name}"

    return None


def _command_path_from_ctx(ctx: click.Context) -> tuple[str, ...]:
    """Reconstruct the command tuple by walking up the click context.

    Returns the chain of subcommand info_names from outermost to innermost,
    excluding the root program. For a root-callback invocation this returns
    an empty tuple - callers should fall back to ``ctx.invoked_subcommand``.
    """
    parts: list[str] = []
    cur: click.Context | None = ctx
    while cur is not None and cur.parent is not None:
        if cur.info_name:
            parts.append(cur.info_name)
        cur = cur.parent
    return tuple(reversed(parts))


def _accepted_option_keys(ctx: click.Context) -> set[str]:
    """Collect the active command's option parameter names."""
    cmd = ctx.command
    if cmd is None:
        return set()
    return {p.name for p in cmd.params if isinstance(p, click.Option) and p.name}


def _profile_label(value: str | None, command_path: tuple[str, ...]) -> str:
    """Build the source label for profile-set values."""
    if value not in (None, ""):
        return f"profile:{value}"
    if command_path == ("text", "interactive"):
        return "profile:builtin:interactive"
    return "profile:builtin:default"


def apply_profile_callback(
    ctx: typer.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    """Eager Click/Typer callback for ``--profile``.

    Walks the precedence stack (explicit profile -> top-level YAML ->
    fallback constants) for each profileable Profile field and writes the
    resolved values into ``ctx.default_map`` (or a nested bucket keyed by
    the invoked subcommand when called from the root callback). Maintains
    a fine-grained sidecar source map at ``ctx.obj['resolved_sources']``.

    Idempotent: invoking with the same value twice is a no-op. When the
    root and a subcommand both register ``--profile``, the subcommand
    invocation overwrites the root's default_map entries because the eager
    callback runs at the subcommand level after the root.
    """
    ctx.ensure_object(dict)
    if "resolved_sources" not in ctx.obj:
        ctx.obj["resolved_sources"] = {}

    # Idempotence guard: bail if already populated for this exact value.
    if ctx.obj.get("_profile_applied") == value:
        return value

    # Placement precedence: when the root callback has already applied an
    # explicit --profile (root --profile X), and this subcommand-level
    # callback is firing with the option's built-in default (i.e. no
    # subcommand-level --profile and no PHENTRIEVA_PROFILE env), do not
    # overwrite the root's resolved defaults. The subcommand value only
    # wins when the user (or env) supplied it explicitly.
    parent_obj = ctx.parent.obj if ctx.parent is not None else None
    parent_applied = (
        parent_obj.get("_profile_applied") if isinstance(parent_obj, dict) else None
    )
    if parent_applied not in (None, ""):
        try:
            from click.core import ParameterSource

            source = ctx.get_parameter_source(param.name) if param.name else None
        except Exception:  # noqa: BLE001 - defensive: older click variants
            source = None
        if source is not None and source == ParameterSource.DEFAULT:
            # Inherit the parent's resolved sources/defaults; bail.
            ctx.obj["_profile_applied"] = parent_applied
            return value

    ctx.obj["_profile_applied"] = value

    # Reconstruct command path. For the root-callback case where the
    # subcommand context hasn't been entered yet, ctx.invoked_subcommand
    # gives the next leg.
    command_path = _command_path_from_ctx(ctx)
    if not command_path and ctx.invoked_subcommand:
        command_path = (ctx.invoked_subcommand,)

    raw_accepted_keys = _accepted_option_keys(ctx)

    # Root callback fires before Click selects the subcommand, so the
    # active command's params (raw_accepted_keys) reflect only the root's
    # options. Don't filter Profile fields against that - the subcommand
    # is what we actually want to pre-populate. Pass an empty set to
    # resolve_profile_for_command so it doesn't filter either, and skip
    # the per-field accepted-keys gate below for the root case.
    is_root = ctx.parent is None
    accepted_keys = set() if is_root else raw_accepted_keys

    profiles = merged_profiles()
    # At the root callback Click has not yet selected the subcommand, so the
    # command-binding check in resolve_profile_for_command would always reject
    # bound profiles like `command: query`. Skip the binding check for the
    # root case; the per-subcommand eager callback re-runs at the subcommand
    # level and re-validates the binding then.
    profile, profile_kwargs = resolve_profile_for_command(
        value if value not in (None, "") else None,
        command_path,
        accepted_keys,
        all_profiles=profiles,
        skip_binding_check=is_root and not command_path,
    )

    # Lazy import - phentrieve.config has its own initialization side effects
    # we don't want at module import time.
    from phentrieve.config import _load_yaml_config

    yaml_data = _load_yaml_config()
    profile_label = _profile_label(value, command_path)

    new_defaults: dict[str, Any] = {}
    for field_name in Profile.model_fields:
        if field_name in {"description", "command", "adaptive_rechunking"}:
            continue
        if accepted_keys and field_name not in accepted_keys:
            continue
        resolved = _resolve_option_value(
            field_name, profile_kwargs, yaml_data, command_path=command_path
        )
        if resolved is None:
            continue
        resolved_value, label = resolved
        if label == "<profile-set>":
            label = profile_label
        new_defaults[field_name] = resolved_value
        ctx.obj["resolved_sources"][field_name] = label

    if ctx.default_map is None:
        ctx.default_map = {}
    if is_root:
        # Root-callback case. Click hasn't parsed the subcommand token yet
        # (ctx.invoked_subcommand is still None), so we don't yet know which
        # subcommand will run. Click resolves a child's default_map as
        # parent.default_map.get(child_info_name); writing flat keys won't
        # propagate. Pre-populate a nested bucket for every known immediate
        # subcommand so whichever one Click selects sees its defaults.
        # Group sub-groups (e.g. "text") get the same treatment recursively
        # via their own eager --profile registration in later phases; here
        # we cover the leaf subcommands one level down. We also write flat
        # so a non-group caller still sees the values.
        ctx.default_map.update(new_defaults)
        if isinstance(ctx.command, click.Group):
            for sub_name in ctx.command.list_commands(ctx):
                ctx.default_map.setdefault(sub_name, {}).update(new_defaults)
    else:
        # Subcommand-callback case: command_path is correct; write flat into
        # this command's own default_map.
        ctx.default_map.update(new_defaults)

    logger.debug(
        "apply_profile_callback: command_path=%s value=%r defaults=%s",
        command_path,
        value,
        new_defaults,
    )
    return value
