"""Policy checks for dependency vulnerability remediations."""

import json
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _uv_packages() -> dict[str, Version]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {
        package["name"]: Version(package["version"])
        for package in lock["package"]
        if isinstance(package.get("name"), str)
    }


def _frontend_packages() -> dict[str, dict[str, object]]:
    package_lock = json.loads(
        (REPO_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8")
    )
    return package_lock["packages"]


def test_chromadb_stays_outside_current_unpatched_vulnerable_range() -> None:
    """GHSA-f4j7-r4q5-qw2c affects chromadb >=1.0.0, <=1.5.9."""
    dependencies = _pyproject()["project"]["dependencies"]
    packages = _uv_packages()

    assert any(
        "chromadb" in dependency and "<1.0.0" in dependency
        for dependency in dependencies
    )
    assert packages["chromadb"] < Version("1.0.0")


def test_esbuild_lockfile_uses_patched_version_if_present() -> None:
    """GHSA-gv7w-rqvm-qjhr and GHSA-g7r4-m6w7-qqqr are fixed in esbuild 0.28.1."""
    packages = _frontend_packages()
    esbuild = packages.get("node_modules/esbuild")

    if esbuild is not None:
        assert Version(esbuild["version"]) >= Version("0.28.1")

    for package_path, package in packages.items():
        if package_path.startswith("node_modules/@esbuild/"):
            assert Version(package["version"]) >= Version("0.28.1")


def test_security_workflow_no_longer_ignores_chromadb_vulnerability() -> None:
    security_workflow = (
        REPO_ROOT / ".github" / "workflows" / "security.yml"
    ).read_text(encoding="utf-8")

    assert "CVE-2026-45829" not in security_workflow
    assert "GHSA-f4j7-r4q5-qw2c" not in security_workflow


def test_chromadb_posthog_transitive_dependency_uses_compatible_api() -> None:
    """ChromaDB 0.6.x emits telemetry errors with PostHog 6+ APIs."""
    packages = _uv_packages()

    assert Version("2.4.0") <= packages["posthog"] < Version("6.0.0")


def test_torch_vulnerability_exception_is_explicitly_limited_to_no_patch() -> None:
    """Torch remains necessary for SentenceTransformers until a patched release exists."""
    security_workflow = (
        REPO_ROOT / ".github" / "workflows" / "security.yml"
    ).read_text(encoding="utf-8")

    assert "CVE-2025-3000" in security_workflow
    assert "no patched release" in security_workflow
