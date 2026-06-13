"""Policy checks for the frontend build toolchain."""

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def _makefile_text() -> str:
    return (REPO_ROOT / "Makefile").read_text(encoding="utf-8")


def _make_target_recipe(target: str) -> str:
    makefile = _makefile_text()
    match = re.search(
        rf"^{re.escape(target)}:.*\n(?P<recipe>(?:\t.*\n)+)",
        makefile,
        re.MULTILINE,
    )
    assert match is not None, f"Makefile target {target!r} not found"
    return match.group("recipe")


def _read_frontend_package_json() -> dict[str, object]:
    return json.loads((FRONTEND_DIR / "package.json").read_text(encoding="utf-8"))


def _read_frontend_package_lock() -> dict[str, object]:
    return json.loads((FRONTEND_DIR / "package-lock.json").read_text(encoding="utf-8"))


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    return tuple(int(part) for part in (*parts, "0", "0")[:3])


def _docker_node_version() -> tuple[int, int, int] | None:
    dockerfile = (FRONTEND_DIR / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"^ARG NODE_VERSION=([0-9]+(?:\.[0-9]+){0,2})-", dockerfile, re.M)
    if match is None:
        return None
    return _parse_version(match.group(1))


def test_frontend_package_specs_target_vite_8_and_vue_router_5_1() -> None:
    package_json = _read_frontend_package_json()
    dependencies = package_json["dependencies"]
    dev_dependencies = package_json["devDependencies"]

    assert dependencies["vue-router"].startswith("^5.1.")
    assert dev_dependencies["vite"].startswith("^8.")


def test_frontend_lockfile_resolves_vite_8_and_vue_router_5_1() -> None:
    package_lock = _read_frontend_package_lock()
    packages = package_lock["packages"]

    assert packages[""]["dependencies"]["vue-router"].startswith("^5.1.")
    assert packages[""]["devDependencies"]["vite"].startswith("^8.")
    assert _parse_version(packages["node_modules/vue-router"]["version"]) >= (5, 1, 0)
    assert _parse_version(packages["node_modules/vite"]["version"]) >= (8, 0, 0)
    assert _parse_version(packages["node_modules/vite"]["version"]) < (9, 0, 0)


def test_frontend_docker_node_version_satisfies_vite_8_engine_floor() -> None:
    node_version = _docker_node_version()

    assert node_version is not None
    assert node_version >= (22, 12, 0) or (20, 19, 0) <= node_version < (21, 0, 0)


def test_dependabot_no_longer_ignores_vue_router_minor_updates() -> None:
    dependabot_config = (REPO_ROOT / ".github/dependabot.yml").read_text(
        encoding="utf-8"
    )

    assert 'dependency-name: "vue-router"' not in dependabot_config


def test_github_actions_node_versions_satisfy_vite_8_engine_floor() -> None:
    workflow_text = "\n".join(
        workflow.read_text(encoding="utf-8")
        for workflow in sorted(WORKFLOW_DIR.glob("*.yml"))
    )

    assert "node-version: '20.19'" in workflow_text
    assert "node-version: '20'" not in workflow_text


def test_vite_8_config_does_not_use_object_form_manual_chunks() -> None:
    vite_config = (FRONTEND_DIR / "vite.config.js").read_text(encoding="utf-8")

    assert "manualChunks: {" not in vite_config
    assert "advancedChunks" not in vite_config
    assert "codeSplitting" in vite_config


def test_local_frontend_ci_reuses_existing_dependencies_by_default() -> None:
    makefile = _makefile_text()
    ci_frontend = _make_target_recipe("ci-frontend")
    ci_frontend_clean = _make_target_recipe("ci-frontend-clean")

    assert "frontend-deps" in ci_frontend
    assert "npm ci" not in ci_frontend
    assert "ci-frontend-clean" in makefile
    assert "frontend-install-ci" in ci_frontend_clean
