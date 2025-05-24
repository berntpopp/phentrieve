# Running Tests

This page explains how to run the test suite and add new tests to the Phentrieve project.

## Testing Framework

Phentrieve uses pytest as its testing framework. Tests are organized into:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test interactions between components
- **End-to-end tests**: Test complete workflows

## Running Tests

### Running the Full Test Suite

To run all tests:

```bash
# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests
pytest
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests for a specific module
pytest tests/unit/test_embeddings.py
```

### Test Coverage

To run tests with coverage reporting:

```bash
# Run tests with coverage
pytest --cov=phentrieve

# Generate an HTML coverage report
pytest --cov=phentrieve --cov-report=html
```

The HTML report will be available in the `htmlcov` directory.

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

Tests are configured in the `pytest.ini` file:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
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

Tests are automatically run on GitHub Actions for every pull request and push to the main branch. The CI configuration is in `.github/workflows/tests.yml`.
