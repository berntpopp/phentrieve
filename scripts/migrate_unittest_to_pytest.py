#!/usr/bin/env python3
"""Script to semi-automate migration from unittest to pytest style.

This script performs basic conversions:
- Removes unittest.TestCase inheritance
- Converts self.assert* to assert statements
- Converts setUp/tearDown to fixtures
- Adds pytest markers
"""

import re
import sys
from pathlib import Path


def convert_assertions(content: str) -> str:
    """Convert unittest assertions to pytest assertions."""
    conversions = [
        # assertEqual
        (r"self\.assertEqual\(([^,]+),\s*([^,]+)\)", r"assert \1 == \2"),
        (r"self\.assertEqual\(([^,]+),\s*([^,]+),\s*([^)]+)\)", r"assert \1 == \2, \3"),

        # assertNotEqual
        (r"self\.assertNotEqual\(([^,]+),\s*([^,]+)\)", r"assert \1 != \2"),

        # assertTrue/assertFalse
        (r"self\.assertTrue\(([^)]+)\)", r"assert \1"),
        (r"self\.assertFalse\(([^)]+)\)", r"assert not \1"),

        # assertIn/assertNotIn
        (r"self\.assertIn\(([^,]+),\s*([^)]+)\)", r"assert \1 in \2"),
        (r"self\.assertNotIn\(([^,]+),\s*([^)]+)\)", r"assert \1 not in \2"),

        # assertIsNone/assertIsNotNone
        (r"self\.assertIsNone\(([^)]+)\)", r"assert \1 is None"),
        (r"self\.assertIsNotNone\(([^)]+)\)", r"assert \1 is not None"),

        # assertIs/assertIsNot
        (r"self\.assertIs\(([^,]+),\s*([^)]+)\)", r"assert \1 is \2"),
        (r"self\.assertIsNot\(([^,]+),\s*([^)]+)\)", r"assert \1 is not \2"),

        # assertGreater/assertLess
        (r"self\.assertGreater\(([^,]+),\s*([^)]+)\)", r"assert \1 > \2"),
        (r"self\.assertLess\(([^,]+),\s*([^)]+)\)", r"assert \1 < \2"),
        (r"self\.assertGreaterEqual\(([^,]+),\s*([^)]+)\)", r"assert \1 >= \2"),
        (r"self\.assertLessEqual\(([^,]+),\s*([^)]+)\)", r"assert \1 <= \2"),

        # assertIsInstance
        (r"self\.assertIsInstance\(([^,]+),\s*([^)]+)\)", r"assert isinstance(\1, \2)"),

        # assertRaises - needs to be handled differently, keep as is for now
    ]

    for pattern, replacement in conversions:
        content = re.sub(pattern, replacement, content)

    return content


def convert_class_definition(content: str) -> str:
    """Remove unittest.TestCase inheritance."""
    # Replace unittest.TestCase with object or just remove inheritance
    content = re.sub(
        r"class\s+(\w+)\(unittest\.TestCase\):",
        r"class \1:",
        content
    )
    return content


def convert_imports(content: str) -> str:
    """Update imports for pytest."""
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        # Remove unittest import if present
        if 'import unittest' in line and 'from unittest.mock' not in line:
            # Add pytest import instead
            new_lines.append('import pytest')
            continue

        # Keep unittest.mock imports
        if 'from unittest.mock' in line or 'from unittest import mock' in line:
            new_lines.append(line)
            continue

        new_lines.append(line)

    return '\n'.join(new_lines)


def add_pytest_marker(content: str, marker: str = "unit") -> str:
    """Add pytest marker after imports."""
    lines = content.split('\n')

    # Find the last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i

    # Insert marker after imports
    marker_line = f"\n\n# Mark all tests in this file as {marker} tests\npytestmark = pytest.mark.{marker}\n"

    lines.insert(last_import_idx + 1, marker_line)

    return '\n'.join(lines)


def convert_setup_teardown(content: str) -> str:
    """Convert setUp/tearDown to pytest fixtures (basic conversion)."""
    # This is a simple placeholder - manual review needed
    # Just add a comment for now
    if 'def setUp(self):' in content or 'def tearDown(self):' in content:
        content = "# TODO: Convert setUp/tearDown to pytest fixtures\n" + content

    return content


def migrate_file(input_path: Path, output_path: Path, marker: str = "unit"):
    """Migrate a single test file."""
    print(f"Migrating {input_path} -> {output_path}")

    # Read original file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply conversions
    content = convert_imports(content)
    content = convert_class_definition(content)
    content = convert_assertions(content)
    content = add_pytest_marker(content, marker)
    content = convert_setup_teardown(content)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Migrated successfully")


def main():
    """Main migration script."""
    if len(sys.argv) < 3:
        print("Usage: python migrate_unittest_to_pytest.py <input_file> <output_file> [marker]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    marker = sys.argv[3] if len(sys.argv) > 3 else "unit"

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    migrate_file(input_path, output_path, marker)


if __name__ == "__main__":
    main()
