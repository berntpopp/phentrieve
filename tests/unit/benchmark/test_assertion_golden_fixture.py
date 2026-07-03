from pathlib import Path

from phentrieve.benchmark.data_loader import load_benchmark_data, parse_gold_terms

FIXTURE = Path("tests/data/benchmarks/en/assertion_edge_cases.json")


def test_fixture_parses_assertion_field_not_assertion_status():
    data = load_benchmark_data(FIXTURE, dataset="all")
    docs = {d["id"]: d for d in data["documents"]}
    gold = dict(parse_gold_terms(docs["no_nystagmus"]["gold_hpo_terms"]))
    assert gold == {
        "HP:0000639": "ABSENT"
    }  # proves .assertion is read, default not applied


def test_present_case_reads_present():
    data = load_benchmark_data(FIXTURE, dataset="all")
    docs = {d["id"]: d for d in data["documents"]}
    gold = dict(parse_gold_terms(docs["plain_present"]["gold_hpo_terms"]))
    assert gold == {"HP:0001250": "PRESENT"}
