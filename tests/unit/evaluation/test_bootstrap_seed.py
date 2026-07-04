from phentrieve.evaluation._extraction_types import ExtractionResult
from phentrieve.evaluation.extraction_metrics import CorpusExtractionMetrics


def _results():
    return [
        ExtractionResult(f"d{i}", [(f"HP:{i}", "PRESENT")], [(f"HP:{i}", "PRESENT")])
        for i in range(10)
    ] + [ExtractionResult("dx", [("HP:99", "PRESENT")], [("HP:1", "PRESENT")])]


def test_seeded_bootstrap_is_deterministic():
    m = CorpusExtractionMetrics()
    a = m.bootstrap_confidence_intervals(_results(), n_bootstrap=200, seed=7)
    b = m.bootstrap_confidence_intervals(_results(), n_bootstrap=200, seed=7)
    assert a == b
