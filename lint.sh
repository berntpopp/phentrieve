#!/bin/bash
# Helper script for linting Python code with Black in WSL environment

# Set explicit path references to avoid WSL path resolution issues
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR" || exit 1

echo "Running Black formatter on Python code..."

# Run black with explicit path arguments to avoid project root detection issues
black --config "$SCRIPT_DIR/pyproject.toml" ./phentrieve ./api 

# Exit with Black's status code
exit $?
