"""Add lazy imports to test_similarity_commands.py."""

import re


def add_lazy_imports(filepath):
    """Add lazy import as first line after docstring in each test method."""
    with open(filepath) as f:
        content = f.read()

    # Pattern: Find test method definitions followed by docstring
    # Add the imports after the closing docstring
    pattern = r'(    def test_[^(]+\([^)]+\):\s+"""[^"]*""")\s*\n'

    # Different imports for different test classes
    def replacement_func(match):
        method_def = match.group(1)
        # Check if it's a cache test or similarity test
        if "_cache" in method_def.lower():
            imports = """
        from phentrieve.cli.similarity_commands import _ensure_cli_hpo_label_cache

"""
        else:
            imports = """
        from phentrieve.cli.similarity_commands import hpo_similarity_cli
        from phentrieve.evaluation.metrics import SimilarityFormula

"""
        return method_def + imports

    new_content = re.sub(pattern, replacement_func, content)

    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"Added lazy imports to {filepath}")


if __name__ == "__main__":
    add_lazy_imports("test_similarity_commands.py")
