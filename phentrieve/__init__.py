"""
Phentrieve - A modular Python package for retrieving HPO terms.

This package provides RAG capabilities for mapping clinical text
in multiple languages to Human Phenotype Ontology terms.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phentrieve")
except PackageNotFoundError:
    # Package not installed, read from pyproject.toml
    from pathlib import Path

    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        __version__ = data["project"]["version"]

__all__ = ["__version__"]
