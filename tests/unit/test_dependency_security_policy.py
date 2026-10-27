"""Policy checks for dependency vulnerability remediations."""

import json
import re
import tomllib
from collections.abc import Iterable
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


_SETUPTOOLS_FLOOR = re.compile(r"setuptools>=(\d+(?:\.\d+)*)")


def _setuptools_floor(requirements: Iterable[str]) -> Version:
    """Lowest setuptools version accepted by the single pin in *requirements*."""
    floors = [
        Version(match.group(1))
        for requirement in requirements
        if (match := _SETUPTOOLS_FLOOR.search(requirement))
    ]
    assert len(floors) == 1, f"expected exactly one setuptools pin, found {floors}"
    return floors[0]


def _dockerfile_setuptools_floor() -> Version:
    dockerfile = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    return _setuptools_floor(
        line for line in dockerfile.splitlines() if "pip install" in line
    )


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
    GHSA-vj7q-gjh5-988w in 1.28.1, GHSA-84m7-p3x7-pcfv in 1.30.0."""
    optional = _pyproject()["project"]["optional-dependencies"]
    packages = _uv_packages()

    assert any("mcp[cli]>=1.30.0" in dependency for dependency in optional["mcp"])
    assert any("fastmcp>=3.4.10" in dependency for dependency in optional["mcp"])
    assert packages["mcp"] >= Version("1.30.0")
    assert packages["fastmcp"] >= Version("3.4.10")  # GHSA-7xgj-8392-5q4r


def test_transitive_security_floors_are_pinned_in_constraints() -> None:
    """Advisory floors for transitive dependencies must survive a future `uv lock`."""
    constraints = _pyproject()["tool"]["uv"]["constraint-dependencies"]
    packages = _uv_packages()

    expected = {
        "cryptography": Version("50.0.0"),  # GHSA-g6cj-pr64-35w5
        "pyasn1": Version("0.6.4"),  # GHSA-m4p7-r5rc-7g4j and friends
        "pymdown-extensions": Version("11.0.1"),  # GHSA-gm37-52c6-37mw
        "anyio": Version("4.14.2"),  # CVE-2026-63374, CVE-2026-64847
        "pyjwt": Version("2.15.0"),  # GHSA-ffc3-869f-jxw9, GHSA-42vr-xj54-vc7v
        "oauthlib": Version("4.0.0"),  # GHSA-xpv3-w29h-x7cv, GHSA-hj66-6f7g-4r5v
        "virtualenv": Version("21.14.9"),  # GHSA-c978-59m8-c77v and friends
        "pillow": Version("13.0.0"),  # GHSA-5x8w-8jj5-q52h and friends
        "filelock": Version("4.2.1"),  # GHSA-v4x8-3wr5-3wq8 and friends
        "fsspec": Version("2026.6.0"),  # GHSA-27vj-qcqg-25rc
        "opentelemetry-api": Version("1.45.1"),  # GHSA-xcq7-p3pf-p88h
        "certifi": Version("2026.10.19"),  # PYSEC-2026-4320
        "authlib": Version("1.9.1"),  # CVE-2026-96760, CVE-2026-114614..114620
        "dnspython": Version("2.9.0"),  # CVE-2026-114605
        "msgpack": Version("1.2.3"),  # CVE-2026-114613
    }
    for name, floor in expected.items():
        assert any(
            constraint.startswith(f"{name}>={floor}") for constraint in constraints
        ), f"missing constraint floor for {name}"
        assert packages[name] >= floor


def test_urllib3_uses_the_patched_2_8_release() -> None:
    """GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g are fixed in 2.8.0."""
    dependencies = _pyproject()["project"]["dependencies"]

    assert any(dependency.startswith("urllib3>=2.8.0") for dependency in dependencies)
    assert _uv_packages()["urllib3"] >= Version("2.8.0")


def test_chromadb_posthog_transitive_dependency_uses_compatible_api() -> None:
    """ChromaDB 1.5.9 no longer pulls posthog; if a future resolution does, keep
    it below the PostHog 6 capture() signature break (constrained in pyproject)."""
    packages = _uv_packages()

    posthog = packages.get("posthog")
    if posthog is not None:
        assert Version("2.4.0") <= posthog < Version("6.0.0")


def _security_workflow_ignores() -> set[str]:
    security_workflow = (
        REPO_ROOT / ".github" / "workflows" / "security.yml"
    ).read_text(encoding="utf-8")
    return set(re.findall(r"--ignore-vuln (\S+)", security_workflow))


def test_torch_cve_2025_3000_is_no_longer_ignored() -> None:
    """CVE-2025-3000 affects torch <= 2.12.0; the locked torch is past it (#413)."""
    trivyignore = (REPO_ROOT / ".trivyignore").read_text(encoding="utf-8").split()

    assert _uv_packages()["torch"] > Version("2.12.0")
    assert "CVE-2025-3000" not in _security_workflow_ignores()
    assert "CVE-2025-3000" not in trivyignore


def test_local_pip_audit_matches_the_ci_ignore_list() -> None:
    """`make security-python` must report the same advisories CI does (#413)."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^PIP_AUDIT_IGNORES := (.*)$", makefile, re.MULTILINE)
    assert match, "Makefile must define PIP_AUDIT_IGNORES"

    assert set(match.group(1).split()) == _security_workflow_ignores()
    assert "uv run pip-audit --strict" not in makefile


def test_setuptools_uses_the_patched_pysec_2026_3447_release() -> None:
    """The pip-audit requirement set must not resolve vulnerable setuptools 81."""
    dev_dependencies = _pyproject()["dependency-groups"]["dev"]
    packages = _uv_packages()

    assert any(
        dependency.startswith("setuptools>=84.1.0") for dependency in dev_dependencies
    )
    assert packages["setuptools"] >= Version("84.1.0")  # GHSA-xmj3-9gf7-h52w


def test_setuptools_floors_stay_aligned_across_build_backend_and_api_image() -> None:
    """Every setuptools floor must match the patched dev-group floor.

    CVE-2026-59890 (PYSEC-2026-3447 / GHSA-h35f-9h28-mq5c) is fixed in 83.0.0, and a
    floor below that is not merely cosmetic. ``pip install --upgrade "setuptools>=..."``
    in api/Dockerfile resolves to whatever release is newest when BuildKit materialises
    that layer, and the registry build cache (``cache-from: type=registry,...:buildcache``
    in docker-publish.yml) then replays the layer unchanged for months. A non-binding
    floor therefore freezes whichever setuptools happened to be current into the
    published image. Keeping the three floors identical makes the constraint binding,
    and makes any future bump edit the Dockerfile line, which busts the cached layer.
    """
    patched = Version("84.1.0")
    dev_floor = _setuptools_floor(_pyproject()["dependency-groups"]["dev"])
    build_backend_floor = _setuptools_floor(_pyproject()["build-system"]["requires"])
    api_image_floor = _dockerfile_setuptools_floor()

    assert dev_floor >= patched, f"dev group floor {dev_floor} predates {patched}"
    assert build_backend_floor == dev_floor, (
        f"pyproject build-system floor {build_backend_floor} has drifted from the "
        f"dev group floor {dev_floor}"
    )
    assert api_image_floor == dev_floor, (
        f"api/Dockerfile floor {api_image_floor} has drifted from the dev group "
        f"floor {dev_floor}"
    )
