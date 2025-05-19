"""
Evaluation modules for the Phentrieve package.

This package contains modules for evaluating the performance of HPO term
retrieval using various metrics like MRR, Hit@K, and ontology similarity.
"""

from .full_text_loader import load_full_text_annotations

__all__ = [
    "load_full_text_annotations",
    # Other functions/classes from this package will be added here
]
