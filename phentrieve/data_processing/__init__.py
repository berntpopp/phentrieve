"""
Data processing modules for the Phentrieve package.

This package contains modules for loading, parsing, and processing HPO data
and test cases.
"""

from phentrieve.data_processing.document_creator import (
    create_hpo_documents,
    load_hpo_terms,
)
from phentrieve.data_processing.multi_vector_document_creator import (
    create_multi_vector_documents,
    get_component_stats,
)

__all__ = [
    "create_hpo_documents",
    "create_multi_vector_documents",
    "get_component_stats",
    "load_hpo_terms",
]
