#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage:
  scripts/ci-python-compat.sh [PYTHON_VERSION ...]

Examples:
  scripts/ci-python-compat.sh
  scripts/ci-python-compat.sh 3.13
  scripts/ci-python-compat.sh 3.12 3.13

Runs the same Python compatibility command used by GitHub Actions:
  uv sync --python <version> --locked --all-extras --dev
  uv run --python <version> pytest tests/ -q -m "not slow and not e2e" --no-cov

When no versions are provided, the script checks the CI matrix versions.
USAGE
  exit 0
fi

versions=("$@")
if [[ "${#versions[@]}" -eq 0 ]]; then
  versions=(3.12 3.13)
fi

for python_version in "${versions[@]}"; do
  echo "==> Python compatibility: ${python_version}"
  uv sync --python "${python_version}" --locked --all-extras --dev
  uv run --python "${python_version}" pytest tests/ -q -m "not slow and not e2e" --no-cov
done
