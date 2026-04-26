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
        "fast_query": Profile(command="query", num_results=5, similarity_threshold=0.5),
        "shared_german": Profile(
            language="de", semantic_chunker_model="jinaai/jina-embeddings-v2-base-de"
        ),
    }


class TestResolveProfileForCommand:
    def test_none_returns_appropriate_builtin_for_text_interactive(
        self, known_profiles
    ):
        profile, kwargs = resolve_profile_for_command(
            None,
            ("text", "interactive"),
            accepted_keys=set(),
            all_profiles=known_profiles,
        )
        assert profile is known_profiles["interactive"]

    def test_none_returns_default_for_other_commands(self, known_profiles):
        profile, _ = resolve_profile_for_command(
            None,
            ("text", "process"),
            accepted_keys=set(),
            all_profiles=known_profiles,
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
            "fast_query",
            ("query",),
            accepted_keys={"num_results", "similarity_threshold"},
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
        # Not in accepted_keys -> filtered out.
        assert "semantic_chunker_model" not in kwargs

    def test_unbound_profile_skips_none_fields(self, known_profiles):
        profile, kwargs = resolve_profile_for_command(
            "shared_german",
            ("query",),
            accepted_keys={"language", "num_results"},
            all_profiles=known_profiles,
        )
        # num_results is None on shared_german -> not included.
        assert "num_results" not in kwargs
        assert kwargs == {"language": "de"}
