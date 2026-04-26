"""Profile data model and resolution for Phentrieve CLI configuration profiles.

Implements Spec A (.planning/specs/2026-04-25-cli-profiles-default-resolution-spec.md).
"""

from __future__ import annotations

import difflib
from typing import Any, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field

# Forward reference - defined in Plan B (phentrieve/retrieval/adaptive_rechunker.py).
# Until Plan B lands, use a permissive dict shape.
AdaptiveRechunkingProfileBlock = dict[str, Any]


class Profile(BaseModel):
    """A named bundle of CLI option defaults loaded from phentrieve.yaml.

    All fields are optional. None means "this profile does not preset this
    option" - the resolution chain falls through to YAML / fallback constants.
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

    # Adaptive rechunking block - strict shape defined by Plan B; permissive here.
    adaptive_rechunking: AdaptiveRechunkingProfileBlock | None = None


class ProfilesFile(BaseModel):
    """Top-level YAML model.

    Only the `profiles:` key is consumed by Plan A; other top-level keys
    (data_dir, default_model, etc.) are ignored at this layer.
    """

    model_config = ConfigDict(extra="ignore")
    profiles: dict[str, Profile] = Field(default_factory=dict)


# Built-in profiles ship as Python dict literals constructed into Profile
# instances at import time. Users can shadow these by name in phentrieve.yaml.
BUILTIN_PROFILES: dict[str, Profile] = {
    "default": Profile(
        description="Strict defaults matching API behavior",
        # All fields None - falls through to YAML / fallback constants.
    ),
    "interactive": Profile(
        description="Loose discovery defaults for `phentrieve text interactive`",
        chunk_retrieval_threshold=0.3,
        aggregated_term_confidence=0.35,
        num_results=5,
    ),
}


def _profile_to_kwargs(profile: Profile, accepted_keys: set[str]) -> dict[str, Any]:
    """Convert a Profile to a kwargs dict, filtered to accepted_keys.

    Skips None fields (None means "this profile doesn't preset this option")
    and fields not in accepted_keys (cross-command filter). An empty
    accepted_keys set is treated as "no filter" (the auto-select callers
    that don't care about the kwargs dict pass set()).
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
        (profile_object, kwargs_dict) - the kwargs dict is what the eager
        callback writes into ctx.default_map (after walking the precedence
        stack with YAML and fallback constants).

    Raises:
        typer.BadParameter: profile_name is unknown, or is bound to a
            different command via its `command` field.
    """
    profiles = all_profiles if all_profiles is not None else BUILTIN_PROFILES

    if profile_name is None:
        # Auto-selection
        fallback = (
            "interactive" if command_path == ("text", "interactive") else "default"
        )
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
                f"Profile {profile_name!r} is bound to command "
                f"{profile.command!r} but was used with "
                f"{' '.join(command_path)!r}.",
                param_hint="--profile",
            )

    return profile, _profile_to_kwargs(profile, accepted_keys)
