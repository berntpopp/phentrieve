"""
Phentrieve Reasoning Package

This package provides graph-based reasoning capabilities for HPO extraction:

- HPO ontology consistency checking
- Hybrid inference combining text and ontology evidence
- Method comparison utilities for A/B testing

All modules are ADDITIVE extensions that preserve backward compatibility
with existing extraction methods.
"""

from phentrieve.reasoning.hpo_consistency import (
    ConsistencyCheckResult,
    ConsistencyConfig,
    ConsistencyViolation,
    HPOConsistencyChecker,
    ViolationSeverity,
    ViolationType,
    check_hpo_consistency,
)
from phentrieve.reasoning.hybrid_inference import (
    EvidenceSource,
    FinalAssertion,
    HybridInferenceConfig,
    HybridInferenceEngine,
    HybridInferenceResult,
    MethodComparisonResult,
    compare_methods,
    create_inference_engine,
)

__all__ = [
    # HPO Consistency
    "HPOConsistencyChecker",
    "ConsistencyViolation",
    "ConsistencyCheckResult",
    "ConsistencyConfig",
    "ViolationType",
    "ViolationSeverity",
    "check_hpo_consistency",
    # Hybrid Inference
    "HybridInferenceEngine",
    "HybridInferenceConfig",
    "HybridInferenceResult",
    "FinalAssertion",
    "EvidenceSource",
    "MethodComparisonResult",
    "create_inference_engine",
    "compare_methods",
]
