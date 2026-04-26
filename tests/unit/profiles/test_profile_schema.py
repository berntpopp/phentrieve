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
        # User typos like `chuck_retrieval_threshold` (chunk -> chuck) error.
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


class TestBuiltInProfiles:
    def test_builtin_default_exists(self):
        from phentrieve.profiles import BUILTIN_PROFILES

        assert "default" in BUILTIN_PROFILES
        # All fields None - fall through to YAML / constants.
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
            assert isinstance(p, Profile), (
                f"BUILTIN_PROFILES[{name!r}] is not a Profile"
            )
