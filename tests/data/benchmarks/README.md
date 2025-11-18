# Benchmark Evaluation Datasets

Curated datasets for evaluating HPO retrieval performance.

## Structure

- `german/` - German-language datasets (current)
- `en/` - English-language datasets (future)
- Other languages as needed

## Datasets

| File | Cases | Variant | Description |
|------|-------|---------|-------------|
| tiny_v1.json | 9 | Manual | Quick testing, default dataset |
| small_v1.json | 9 | Manual | Small evaluation set |
| 70cases_gemini_v1.json | 70 | AI-translated | Gemini 2.5 translated |
| 70cases_o3_v1.json | 70 | AI-translated | O3 translated |
| 200cases_gemini_v1.json | 200 | AI-translated | Large Gemini set |
| 200cases_o3_v1.json | 200 | AI-translated | Large O3 set |

## Format

```json
[
  {
    "description": "Optional human-readable description",
    "text": "Clinical text in target language",
    "expected_hpo_ids": ["HP:0001234", "HP:0005678"]
  }
]
```

## Usage

### CLI
```bash
# Default (uses tiny_v1.json)
phentrieve benchmark run

# Specify dataset
phentrieve benchmark run --test-file german/70cases_gemini_v1.json
```

### Python
```python
from phentrieve.data_processing.test_data_loader import load_test_data

# Relative path from project root
dataset = load_test_data("tests/data/benchmarks/german/tiny_v1.json")
```

### Tests
```python
def test_something(benchmark_data_dir):
    dataset = load_test_data(str(benchmark_data_dir / "german/tiny_v1.json"))
    assert len(dataset) == 9
```

## Adding Datasets

1. Add JSON file to appropriate directory
2. Update table above
3. Verify with: `pytest tests/unit/cli/test_benchmark_commands.py::TestBenchmarkDataLoading`

## Version History

- **v1** (2025-11-18): Initial migration from `data/test_cases/`
