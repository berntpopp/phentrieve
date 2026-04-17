# Running Tests

This page explains how to run the test suite and add new tests to the Phentrieve project.

## Testing Framework

Phentrieve uses pytest as its testing framework. Tests are organized into:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test interactions between components
- **End-to-end tests**: Test complete workflows

## Running Tests

### Running the Full Test Suite

Use the repository standard commands from the project root:

```bash
make test
make check
make typecheck-fast
```

### Running Specific Test Categories

```bash
uv run pytest tests/unit/api/ -n 0 -v
uv run pytest tests/ --collect-only -q
```

### Test Coverage

Coverage is configured in the default pytest settings. The full suite writes HTML
and XML reports automatically when you run the standard test commands.

For targeted local work, prefer focused commands:

```bash
uv run pytest tests/unit/api/test_text_processing_router.py -n 0 -v --no-cov
```

The HTML report is written to the `htmlcov` directory during the default
coverage-enabled runs.

## Writing Tests

### Unit Test Example

Here's an example of a unit test for a Phentrieve component:

```python
import pytest
from phentrieve.embeddings import EmbeddingModel

def test_embedding_model_initialization():
    # Arrange
    model_name = "FremyCompany/BioLORD-2023-M"

    # Act
    model = EmbeddingModel(model_name)

    # Assert
    assert model.name == model_name
    assert model.dimension > 0
    assert callable(model.embed)

def test_embedding_generation():
    # Arrange
    model = EmbeddingModel("FremyCompany/BioLORD-2023-M")
    text = "Microcephaly"

    # Act
    embedding = model.embed(text)

    # Assert
    assert len(embedding) == model.dimension
```

### Integration Test Example

```python
import pytest
from phentrieve.data_processing import HPOProcessor
from phentrieve.indexing import IndexBuilder

def test_index_building_with_processor():
    # Arrange
    processor = HPOProcessor()
    builder = IndexBuilder("test_model")

    # Act
    terms = processor.process()
    builder.build_index(terms)

    # Assert
    assert builder.index_exists()
    assert builder.index_size() > 0
```

### Test Fixtures

Phentrieve uses pytest fixtures for setting up test dependencies:

```python
import pytest
from phentrieve.embeddings import EmbeddingModel

@pytest.fixture
def test_model():
    """Provides a test embedding model."""
    return EmbeddingModel("FremyCompany/BioLORD-2023-M")

def test_with_fixture(test_model):
    # The test_model fixture is automatically injected
    assert test_model.name == "FremyCompany/BioLORD-2023-M"
```

## Test Configuration

Tests are configured in the repository root `pyproject.toml` file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

## Mocking

For tests that require external dependencies, use the `unittest.mock` module or `pytest-mock`:

```python
def test_with_mock(mocker):
    # Mock the ChromaDB client
    mock_client = mocker.patch("phentrieve.indexing.index_builder.chromadb.Client")
    mock_client.return_value.get_collection.return_value.count.return_value = 42

    # Act
    builder = IndexBuilder("test_model")
    count = builder.get_document_count()

    # Assert
    assert count == 42
```

## Continuous Integration

Tests run automatically in GitHub Actions for pull requests and pushes. The main
Python workflow is defined in `.github/workflows/ci.yml`.
