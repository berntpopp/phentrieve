"""Add lazy imports to test files."""

import re
import sys


def add_lazy_imports(filepath, import_statement):
    """Add lazy import as first line after docstring in each test method."""
    with open(filepath) as f:
        content = f.read()

    # Pattern: Find test method definitions followed by docstring
    # Add the import after the closing docstring
    pattern = r'(    def test_[^(]+\([^)]+\):\s+"""[^"]*""")\s*\n'
    replacement = rf"\1\n        {import_statement}\n\n"

    new_content = re.sub(pattern, replacement, content)

    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"Added lazy imports to {filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_lazy_imports.py <filepath> <import_statement>")
        sys.exit(1)

    filepath = sys.argv[1]
    import_statement = sys.argv[2]
    add_lazy_imports(filepath, import_statement)
