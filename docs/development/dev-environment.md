# Development Environment

This page explains how to set up a development environment for contributing to the Phentrieve project.

## Prerequisites

- Python 3.8 or higher
- Git for version control
- A code editor or IDE (VSCode, PyCharm, etc.)
- Docker (optional, for containerized development)
- Node.js and NPM (for frontend development)

## Setting Up the Python Environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve
```

### Step 2: Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install in Development Mode

```bash
# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Frontend Development Setup

For working on the frontend:

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Create a development environment file
echo "VUE_APP_API_URL=http://localhost:8000" > .env.development

# Start the development server
npm run serve
```

## IDE Configuration

### VSCode

For Visual Studio Code, recommended settings:

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "python.testing.pytestEnabled": true
}
```

Recommended extensions:
- Python
- Pylance
- Vetur (for Vue.js)
- Docker

### PyCharm

For PyCharm:

1. Open the Phentrieve project folder
2. Go to Settings → Project → Python Interpreter
3. Create a new virtual environment or select the existing one
4. Install project dependencies

## Working with Git

### Branching Strategy

Phentrieve uses a feature branch workflow:

1. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

3. Push your branch and create a pull request:
   ```bash
   git push -u origin feature/your-feature-name
   ```

## Environment Variables

For development, you can use a `.env` file in the project root:

```
# Data directories
PHENTRIEVE_DATA_DIR=./data
PHENTRIEVE_HPO_DATA_DIR=./data/hpo_core_data
PHENTRIEVE_INDEX_DIR=./data/indexes
PHENTRIEVE_RESULTS_DIR=./data/results

# Logging
PHENTRIEVE_LOG_LEVEL=DEBUG

# Development settings
PHENTRIEVE_DEV_MODE=true
```

## Pre-commit Hooks

To ensure code quality, set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This will run linters and formatters automatically before each commit.

## GPU Development

For development with GPU support:

1. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
   ```

2. Set the appropriate environment variables:
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
   ```

Based on our project memories, Phentrieve has been updated to support GPU acceleration with CUDA when available and gracefully fall back to CPU when unavailable.
