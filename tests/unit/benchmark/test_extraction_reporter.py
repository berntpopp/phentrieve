import pytest

from phentrieve.benchmark.extraction_reporter import ExtractionReporter

pytestmark = pytest.mark.unit


def test_markdown_report_includes_ontology_metric_columns():
    reporter = ExtractionReporter(output_format="markdown")

    report = reporter.generate_report(
        [
            {
                "metadata": {"model": "demo-model"},
                "corpus_metrics": {
                    "micro": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
                    "macro": {"f1": 0.45},
                    "ontology_metrics": {
                        "soft": {"micro": {"f1": 0.75}},
                        "partial": {"micro": {"f1": 0.80}},
                    },
                },
            }
        ]
    )

    assert "Soft Micro F1" in report
    assert "Partial Micro F1" in report
    assert "| demo-model | 0.550 | 0.450 | 0.500 | 0.600 | 0.750 | 0.800 |" in report
