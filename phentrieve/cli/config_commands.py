"""`phentrieve config` subcommand group.

Implements list-profiles, show, validate, and path subcommands per Spec A's
"Subcommand surface" section. These commands let users inspect which profiles
are defined, view the resolved YAML for a given profile, validate the
profiles section against the Profile schema, and discover which
phentrieve.yaml file is in effect.
"""

from __future__ import annotations

import typer
import yaml as pyyaml
from pydantic import ValidationError

from phentrieve.config import _load_yaml_config
from phentrieve.profiles import (
    BUILTIN_PROFILES,
    Profile,
    load_profiles_from_yaml,
    merged_profiles,
)
from phentrieve.utils import (
    get_config_file_path,
    get_config_paths,
    load_user_config,
)

config_app = typer.Typer(
    name="config",
    help="Inspect and validate phentrieve.yaml configuration profiles.",
    no_args_is_help=True,
)


def _clear_config_caches() -> None:
    """Drop on-disk YAML caches so subsequent reads see fresh content."""
    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


def _summarize_options(profile: Profile) -> str:
    """Return a short comma-separated list of preset option keys (excluding
    description/command/None values). Caps at three entries with an ellipsis.
    """
    keys = [
        k
        for k, v in profile.model_dump().items()
        if v is not None and k not in {"description", "command"}
    ]
    if not keys:
        return ""
    if len(keys) <= 3:
        return ", ".join(keys)
    return ", ".join(keys[:3]) + ", ..."


@config_app.command("list-profiles")
def list_profiles_cmd() -> None:
    """List all profiles (built-in + user) with command bindings and key options."""
    _clear_config_caches()
    user_profiles = load_profiles_from_yaml()
    rows: list[tuple[str, str, str, str, str]] = []
    seen: set[str] = set()
    for name, profile in BUILTIN_PROFILES.items():
        seen.add(name)
        if name in user_profiles:
            source = "user (shadows built-in)"
            effective = user_profiles[name]
        else:
            source = "built-in"
            effective = profile
        binding = effective.command or "any"
        keys = _summarize_options(effective)
        desc = effective.description or ""
        rows.append((name, source, binding, keys, desc))

    for name, profile in user_profiles.items():
        if name in seen:
            continue
        binding = profile.command or "any"
        keys = _summarize_options(profile)
        desc = profile.description or ""
        rows.append((name, "user", binding, keys, desc))

    headers = ("Name", "Source", "Binding", "Key options", "Description")
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    typer.echo("  ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers))))
    typer.echo("  ".join("-" * w for w in widths))
    for row in rows:
        typer.echo("  ".join(f"{row[i]:<{widths[i]}}" for i in range(len(headers))))


@config_app.command("show")
def show_cmd(name: str = typer.Argument(..., help="Profile name to show.")) -> None:
    """Print the fully resolved profile as YAML."""
    profiles = merged_profiles()
    if name not in profiles:
        raise typer.BadParameter(f"Profile {name!r} not found.", param_hint="name")
    profile = profiles[name]
    # Drop None fields so the output reflects only what the profile presets.
    non_none = {k: v for k, v in profile.model_dump().items() if v is not None}
    typer.echo(
        pyyaml.safe_dump(non_none, sort_keys=False, default_flow_style=False).rstrip()
    )


def _format_validation_error(name: str, err: ValidationError) -> str:
    """Compact, single-line-per-error message for `validate`."""
    parts: list[str] = [f"{name}:"]
    for detail in err.errors():
        loc = ".".join(str(x) for x in detail.get("loc", ()))
        msg = detail.get("msg", "invalid value")
        parts.append(f"  {loc}: {msg}" if loc else f"  {msg}")
    return "\n".join(parts)


@config_app.command("validate")
def validate_cmd() -> None:
    """Validate phentrieve.yaml's `profiles:` section against the Profile schema.

    Exits non-zero with a per-profile error summary if any entry is invalid;
    otherwise prints OK.
    """
    _clear_config_caches()
    raw = _load_yaml_config()
    profiles_raw = raw.get("profiles", {})
    errors: list[str] = []

    if not isinstance(profiles_raw, dict):
        typer.echo("phentrieve.yaml `profiles:` section is not a mapping.", err=True)
        raise typer.Exit(code=1)

    for name, data in profiles_raw.items():
        if not isinstance(data, dict):
            errors.append(f"{name}: not a mapping")
            continue
        try:
            Profile.model_validate(data)
        except ValidationError as e:
            errors.append(_format_validation_error(name, e))

    if errors:
        for line in errors:
            typer.echo(line, err=True)
        raise typer.Exit(code=1)

    typer.echo("OK")


@config_app.command("path")
def path_cmd() -> None:
    """Print the active phentrieve.yaml search path and which file was loaded."""
    config_path = get_config_file_path()
    if config_path is not None and config_path.exists():
        typer.echo(f"Loaded: {config_path}")
        typer.echo("Search order:")
        for p in get_config_paths():
            marker = " (loaded)" if p == config_path else ""
            typer.echo(f"  - {p}{marker}")
    else:
        typer.echo("No phentrieve.yaml found in the search path.")
        typer.echo("Search order:")
        for p in get_config_paths():
            typer.echo(f"  - {p}")
