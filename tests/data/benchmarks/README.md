# Benchmark Evaluation Datasets

Curated datasets for evaluating HPO (Human Phenotype Ontology) retrieval performance across different models and languages.

## Overview

These datasets test the system's ability to:
- Map clinical text to accurate HPO terms
- Handle multilingual medical terminology
- Perform consistently across different text lengths and complexities
- Retrieve semantically similar phenotype terms

## Directory Structure

```
tests/data/benchmarks/
├── README.md (this file)
└── german/
    ├── tiny_v1.json           # 9 cases - Quick testing
    ├── small_v1.json          # 9 cases - Validation
    ├── 70cases_gemini_v1.json # 70 cases - Medium evaluation
    ├── 70cases_o3_v1.json     # 70 cases - Medium evaluation
    ├── 200cases_gemini_v1.json # 200 cases - Comprehensive
    └── 200cases_o3_v1.json    # 200 cases - Comprehensive
```

## Dataset Details

### German Language Datasets

All datasets contain German clinical text with English HPO term mappings.

| File | Cases | Source | Purpose | Use When |
|------|-------|--------|---------|----------|
| **tiny_v1.json** | 9 | Manual curation | Quick smoke tests, CI/CD | Testing changes rapidly |
| **small_v1.json** | 9 | Manual curation | Validation set | Cross-validation |
| **70cases_gemini_v1.json** | 70 | Gemini 2.5 AI translation | Medium-scale evaluation | Comparing translation quality |
| **70cases_o3_v1.json** | 70 | OpenAI O3 translation | Medium-scale evaluation | Cross-model validation |
| **200cases_gemini_v1.json** | 200 | Gemini 2.5 AI translation | Comprehensive benchmarking | Full model evaluation |
| **200cases_o3_v1.json** | 200 | OpenAI O3 translation | Comprehensive benchmarking | Production readiness testing |

### Dataset Characteristics

**Manual Datasets (tiny_v1, small_v1)**:
- Hand-crafted clinical scenarios
- Carefully selected HPO term mappings
- High confidence in ground truth
- Diverse phenotype categories
- Contains German umlauts and special characters

**AI-Translated Datasets (70cases, 200cases)**:
- Generated from English clinical vignettes
- Professionally translated by large language models
- Larger sample sizes for statistical significance
- Include edge cases and rare phenotypes
- Variants from different AI models for comparison

## JSON Format Specification

```json
[
  {
    "description": "Hypertrophic cardiomyopathy with septal hypertrophy",
    "text": "Hypertrophe Kardiomyopathie mit Septumhypertrophie",
    "expected_hpo_ids": ["HP:0001639", "HP:0001712"]
  }
]
```

**Required Fields**:
- `text` (string): Clinical text in the target language
- `expected_hpo_ids` (array): List of HPO term IDs (format: `HP:\d{7}`)

**Optional Fields**:
- `description` (string): Human-readable description (usually English)

**Validation Rules**:
1. HPO IDs must match pattern `HP:\d{7}` (e.g., `HP:0001234`)
2. Text must not be empty
3. At least one expected HPO ID required
4. UTF-8 encoding for special characters

## Usage Examples

### CLI Commands

```bash
# Quick test with default dataset (tiny_v1.json)
phentrieve benchmark run

# Test with specific model
phentrieve benchmark run \
  --model-name FremyCompany/BioLORD-2023-M \
  --cpu

# Medium-scale evaluation
phentrieve benchmark run \
  --test-file german/70cases_gemini_v1.json \
  --model-name jinaai/jina-embeddings-v2-base-de

# Comprehensive benchmark with all models
phentrieve benchmark run \
  --test-file german/200cases_o3_v1.json \
  --all-models

# Compare results across datasets
phentrieve benchmark compare

# Generate visualizations
phentrieve benchmark visualize
```

### Python API

```python
from phentrieve.data_processing.test_data_loader import load_test_data
from pathlib import Path

# Load a specific dataset
project_root = Path(__file__).parent.parent  # Adjust as needed
dataset_path = project_root / "tests/data/benchmarks/german/tiny_v1.json"
dataset = load_test_data(str(dataset_path))

# Iterate through test cases
for i, case in enumerate(dataset):
    print(f"Case {i+1}:")
    print(f"  Text: {case['text']}")
    print(f"  Expected HPO IDs: {case['expected_hpo_ids']}")
    if 'description' in case:
        print(f"  Description: {case['description']}")
```

### pytest Fixtures

```python
def test_benchmark_with_dataset(benchmark_data_dir):
    """Example test using the benchmark_data_dir fixture."""
    from phentrieve.data_processing.test_data_loader import load_test_data

    # Load dataset
    dataset_path = benchmark_data_dir / "german" / "tiny_v1.json"
    dataset = load_test_data(str(dataset_path))

    # Verify structure
    assert len(dataset) == 9
    for case in dataset:
        assert "text" in case
        assert "expected_hpo_ids" in case
        assert len(case["expected_hpo_ids"]) > 0
```

## Best Practices

### When to Use Each Dataset

1. **Development & Debugging** → `tiny_v1.json`
   - Fast iteration cycles
   - Quick sanity checks
   - CI/CD smoke tests

2. **Cross-Validation** → `small_v1.json`
   - Separate validation set
   - Prevent overfitting to tiny_v1
   - Independent quality check

3. **Model Comparison** → `70cases_*_v1.json`
   - Compare Gemini vs O3 translations
   - Evaluate translation quality impact
   - Medium sample size for significance

4. **Production Evaluation** → `200cases_*_v1.json`
   - Comprehensive model assessment
   - Statistical significance
   - Final validation before deployment

### Benchmark Workflow

```bash
# 1. Quick development check
phentrieve benchmark run --test-file german/tiny_v1.json

# 2. Validate with separate set
phentrieve benchmark run --test-file german/small_v1.json

# 3. Comprehensive evaluation
phentrieve benchmark run --test-file german/200cases_o3_v1.json --all-models

# 4. Compare results
phentrieve benchmark compare

# 5. Generate visualizations
phentrieve benchmark visualize
```

## Adding New Datasets

### 1. Prepare Your Data

Create a JSON file following the format:

```json
[
  {
    "description": "Optional description",
    "text": "Clinical text in target language",
    "expected_hpo_ids": ["HP:0000001", "HP:0000002"]
  }
]
```

### 2. Choose Appropriate Location

- **German datasets** → `tests/data/benchmarks/german/`
- **English datasets** → `tests/data/benchmarks/en/` (create directory)
- **Other languages** → Create language-specific subdirectory

### 3. Follow Naming Convention

```
{size}_{source}_{version}.json

Examples:
- 10cases_manual_v1.json
- 50cases_gpt4_v2.json
- 100cases_human_expert_v1.json
```

### 4. Validate Your Dataset

```bash
# Run integration tests
pytest tests/unit/cli/test_benchmark_integration.py -v

# Specifically test data loading
pytest tests/unit/cli/test_benchmark_integration.py::TestBenchmarkDataLoading -v
```

### 5. Update Documentation

1. Add entry to the dataset table in this README
2. Document the source and purpose
3. Note any special characteristics
4. Add to CLAUDE.md if needed

### 6. Test with Actual Benchmark

```bash
# Verify it works end-to-end
phentrieve benchmark run --test-file <language>/<your_file>.json --debug
```

## Metrics Calculated

Benchmarks calculate the following metrics:

- **MRR (Mean Reciprocal Rank)**: Average inverse rank of first correct result
- **Hit Rate @ K**: Percentage of cases with correct result in top K
- **MaxOntSim**: Semantic similarity using HPO ontology structure
- **Avg Semantic Similarity**: Cosine similarity between embeddings

## Quality Assurance

### Automated Checks

The integration test suite (`test_benchmark_integration.py`) verifies:

1. ✅ All datasets are loadable
2. ✅ JSON structure is valid
3. ✅ Required fields present
4. ✅ HPO IDs match pattern `HP:\d{7}`
5. ✅ Case counts match expectations
6. ✅ Non-ASCII characters preserved (for multilingual)

### Manual Review Checklist

Before adding a new dataset:

- [ ] Clinical text is accurate and realistic
- [ ] HPO term mappings are correct
- [ ] Language-specific characters preserved (ä, ö, ü, ß, etc.)
- [ ] Diverse phenotype categories represented
- [ ] Edge cases included (negations, uncertainties)
- [ ] No protected health information (PHI)
- [ ] Proper attribution for data sources

## Troubleshooting

### Common Issues

**"File not found" error**:
```bash
# Use relative path from project root
phentrieve benchmark run --test-file german/tiny_v1.json

# NOT absolute path
phentrieve benchmark run --test-file /full/path/to/file.json
```

**"Invalid HPO ID format"**:
- Ensure IDs match pattern: `HP:0001234` (7 digits)
- Check for typos: `HP:001234` ❌ vs `HP:0001234` ✅

**"Empty dataset"**:
- Verify JSON is valid (use `jq` or JSON validator)
- Check file encoding is UTF-8
- Ensure no BOM (Byte Order Mark) in file

**"Results differ from expected"**:
- Verify HPO data is up-to-date: `phentrieve data prepare`
- Rebuild indexes: `phentrieve index build`
- Check model matches expected (BioLORD vs other models)

## Version History

### v1 (2025-11-18)

**Initial Release**
- Migrated from `data/test_cases/` to `tests/data/benchmarks/`
- Reorganized with language-specific subdirectories
- Renamed files with simplified convention
- Added comprehensive documentation
- Fixed path resolution in benchmark orchestrator

**Changes from previous location**:
- `sample_test_cases.json` → `german/small_v1.json`
- `test_cases_small.json` → `german/tiny_v1.json`
- `expanded_test_70cases_gemini25translated.json` → `german/70cases_gemini_v1.json`
- `expanded_test_70cases_o3translated.json` → `german/70cases_o3_v1.json`
- `expanded_test_200cases_gemini25translated.json` → `german/200cases_gemini_v1.json`
- `expanded_test_200cases_o3translated.json` → `german/200cases_o3_v1.json`

## References

- [Human Phenotype Ontology (HPO)](https://hpo.jax.org/)
- [HPO GitHub Repository](https://github.com/obophenotype/human-phenotype-ontology)
- [Project Documentation](../../../CLAUDE.md)
- [Testing Plan](../../../plan/02-completed/TESTING-MODERNIZATION-PLAN.md)

## Support

For questions or issues:
1. Check [CLAUDE.md](../../../CLAUDE.md) for usage examples
2. Review [STATUS.md](../../../plan/STATUS.md) for project status
3. Open an issue on GitHub with `benchmark` label
