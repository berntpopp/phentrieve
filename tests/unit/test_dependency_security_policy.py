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


def test_chromadb_pinned_to_1x_for_bundle_compatibility() -> None:
    """ChromaDB 1.x is required to read the published multi-vector data bundles.

    GHSA-f4j7-r4q5-qw2c affects chromadb >=1.0.0,<=1.5.9 and 1.5.9 is the latest
    available release (no patched 1.x yet), so the advisory is an accepted,
    documented risk (see .github/workflows/security.yml) rather than downgrading
    to 0.6.x, which cannot read the bundles (KeyError '_type').
    """
    dependencies = _pyproject()["project"]["dependencies"]
    packages = _uv_packages()

    assert any(
        "chromadb" in dependency and ">=1.5.9" in dependency
        for dependency in dependencies
    )
    assert Version("1.5.9") <= packages["chromadb"] < Version("2.0.0")


def test_esbuild_lockfile_uses_patched_version_if_present() -> None:
    """GHSA-gv7w-rqvm-qjhr and GHSA-g7r4-m6w7-qqqr are fixed in esbuild 0.28.1."""
    packages = _frontend_packages()
    esbuild = packages.get("node_modules/esbuild")

    if esbuild is not None:
        assert Version(esbuild["version"]) >= Version("0.28.1")

    for package_path, package in packages.items():
        if package_path.startswith("node_modules/@esbuild/"):
            assert Version(package["version"]) >= Version("0.28.1")


def test_security_workflow_documents_accepted_chromadb_vulnerability() -> None:
    """chromadb 1.5.9 is required for data-bundle compatibility and has no patched
    release, so GHSA-f4j7-r4q5-qw2c is an explicit, documented pip-audit ignore."""
    security_workflow = (
        REPO_ROOT / ".github" / "workflows" / "security.yml"
    ).read_text(encoding="utf-8")

    assert "--ignore-vuln GHSA-f4j7-r4q5-qw2c" in security_workflow


def test_dependency_review_gate_allowlists_accepted_chromadb_vulnerability() -> None:
    """The ci.yml Dependency Review gate must allowlist the same accepted advisory
    as the pip-audit ignore in security.yml, so chromadb 1.5.9 (required for
    data-bundle compatibility) does not fail it on GHSA-f4j7-r4q5-qw2c."""
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "allowed_advisories" in ci_workflow
    assert "GHSA-f4j7-r4q5-qw2c" in ci_workflow


def test_transformers_uses_the_patched_5x_major() -> None:
    """Transformers 5.x carries the fixes for the remote-code-execution advisories.

    The upgrade was blocked by the JinaBert model, whose pinned remote implementation
    imports ``transformers.pytorch_utils.find_pruneable_heads_and_indices`` -- a symbol
    removed in Transformers 5.x. Dropping that model unblocked the patched major, so the
    matching pip-audit and Dependency Review exceptions were removed with it.
    """
    security_workflow = (
        REPO_ROOT / ".github" / "workflows" / "security.yml"
    ).read_text(encoding="utf-8")
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    dependencies = _pyproject()["project"]["dependencies"]
    packages = _uv_packages()

    assert any(
        "transformers>=5.5.0,<6.0.0" in dependency for dependency in dependencies
    )
    # GHSA-29pf-2h5f-8g72 is fixed in 5.3.0 and GHSA-fgcw-684q-jj6r in 5.5.0.
    assert Version("5.5.0") <= packages["transformers"] < Version("6.0.0")

    for advisory in (
        "PYSEC-2025-217",
        "PYSEC-2026-2288",
        "PYSEC-2026-2289",
        "PYSEC-2026-2290",
    ):
        assert f"--ignore-vuln {advisory}" not in security_workflow
    for advisory in ("GHSA-29pf-2h5f-8g72", "GHSA-fgcw-684q-jj6r"):
        assert advisory not in ci_workflow


def test_jinabert_model_is_not_reintroduced() -> None:
    """JinaBert's pinned remote code cannot load under Transformers 5.x.

    Re-adding the model would force a downgrade to the vulnerable 4.x major, so the
    release contract and the bundle slug map must both stay free of it.
    """
    contract = (
        REPO_ROOT / "phentrieve" / "data_processing" / "release_contract.py"
    ).read_text(encoding="utf-8")
    manifest = (
        REPO_ROOT / "phentrieve" / "data_processing" / "bundle_manifest.py"
    ).read_text(encoding="utf-8")

    assert "jina" not in contract.lower()
    assert "jina" not in manifest.lower()


def test_mcp_uses_the_patched_transport_release() -> None:
    """GHSA-hvrp-rf83-w775 / GHSA-jpw9-pfvf-9f58 are fixed in 1.27.2 and
    GHSA-vj7q-gjh5-988w in 1.28.1."""
    optional = _pyproject()["project"]["optional-dependencies"]
    packages = _uv_packages()

    assert any("mcp[cli]>=1.28.1" in dependency for dependency in optional["mcp"])
    assert packages["mcp"] >= Version("1.28.1")


def test_transitive_security_floors_are_pinned_in_constraints() -> None:
    """Advisory floors for transitive dependencies must survive a future `uv lock`."""
    constraints = _pyproject()["tool"]["uv"]["constraint-dependencies"]
    packages = _uv_packages()

    expected = {
        "cryptography": Version("50.0.0"),  # GHSA-g6cj-pr64-35w5
        "pyasn1": Version("0.6.4"),  # GHSA-m4p7-r5rc-7g4j and friends
        "pymdown-extensions": Version("11.0.1"),  # GHSA-gm37-52c6-37mw
    }
    for name, floor in expected.items():
        assert any(
            constraint.startswith(f"{name}>={floor}") for constraint in constraints
        ), f"missing constraint floor for {name}"
        assert packages[name] >= floor


def test_chromadb_posthog_transitive_dependency_uses_compatible_api() -> None:
    """ChromaDB 1.5.9 no longer pulls posthog; if a future resolution does, keep
    it below the PostHog 6 capture() signature break (constrained in pyproject)."""
    packages = _uv_packages()

    posthog = packages.get("posthog")
    if posthog is not None:
        assert Version("2.4.0") <= posthog < Version("6.0.0")


def test_torch_vulnerability_exception_is_explicitly_limited_to_no_patch() -> None:
    """Torch remains necessary for SentenceTransformers until a patched release exists."""
    security_workflow = (
        REPO_ROOT / ".github" / "workflows" / "security.yml"
    ).read_text(encoding="utf-8")

    assert "CVE-2025-3000" in security_workflow
    assert "no patched release" in security_workflow


def test_setuptools_uses_the_patched_pysec_2026_3447_release() -> None:
    """The pip-audit requirement set must not resolve vulnerable setuptools 81."""
    dev_dependencies = _pyproject()["dependency-groups"]["dev"]
    packages = _uv_packages()

    assert any(
        dependency.startswith("setuptools>=83.0.0") for dependency in dev_dependencies
    )
    assert packages["setuptools"] >= Version("83.0.0")
