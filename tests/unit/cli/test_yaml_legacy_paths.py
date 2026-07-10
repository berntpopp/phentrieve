"""Tests that legacy ``~/.phentrieve/`` YAML paths are still searched.

Spec A Phase 7 Task 14: ``phentrieve.utils.get_config_paths()`` returns the
legacy ``~/.phentrieve/phentrieve.yaml`` location after the local
``./phentrieve.yaml``. A profile defined in the legacy file is loaded by
``phentrieve.profiles.merged_profiles()`` when no local YAML exists; a local
``./phentrieve.yaml`` shadows the legacy entry.
"""

from __future__ import annotations


def _clear_yaml_cache() -> None:
    from phentrieve.config import _load_yaml_config
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


def test_get_config_paths_includes_legacy_home(tmp_path, monkeypatch):
    """``get_config_paths()`` lists ``~/.phentrieve/phentrieve.yaml``."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.chdir(tmp_path)

    from phentrieve.utils import get_config_paths

    paths = get_config_paths()
    legacy = fake_home / ".phentrieve" / "phentrieve.yaml"
    assert legacy in paths
    # Local cwd path should appear before the legacy location.
    local = (tmp_path / "phentrieve.yaml").resolve()
    legacy_resolved = legacy.resolve()
    resolved_paths = [p.resolve() for p in paths]
    assert resolved_paths.index(local) < resolved_paths.index(legacy_resolved)


def test_home_phentrieve_yaml_picked_up_when_no_local(tmp_path, monkeypatch):
    """A ``phentrieve.yaml`` at ``~/.phentrieve/`` is searched when no local."""
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
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    _clear_yaml_cache()

    from phentrieve.profiles import merged_profiles

    profiles = merged_profiles()
    assert "legacy_profile" in profiles
    assert profiles["legacy_profile"].language == "de"
    _clear_yaml_cache()


def test_local_yaml_shadows_legacy_path(tmp_path, monkeypatch):
    """``./phentrieve.yaml`` shadows ``~/.phentrieve/phentrieve.yaml``."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".phentrieve").mkdir()
    (fake_home / ".phentrieve" / "phentrieve.yaml").write_text(
        "profiles:\n  shared:\n    language: de\n"
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "phentrieve.yaml").write_text("profiles:\n  shared:\n    language: fr\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    _clear_yaml_cache()

    from phentrieve.profiles import merged_profiles

    profiles = merged_profiles()
    assert profiles["shared"].language == "fr"
    _clear_yaml_cache()
