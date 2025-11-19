# Installation

This guide covers the installation of Phentrieve for both users and developers.

## Prerequisites

*   **Python 3.10+** (Required for modern type hinting features)
*   **Git**
*   **Docker & Docker Compose** (Recommended for running the full stack)
*   **uv** (Recommended for fast Python dependency management)

## Method 1: Production/User Installation (Docker)

The easiest way to run Phentrieve is using the provided Docker configuration, which handles the API, Frontend, and database dependencies securely.

```bash
# Clone the repository
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve

# Copy the environment template
cp .env.docker.example .env.docker

# Edit .env.docker to set your data directory
# PHENTRIEVE_HOST_DATA_DIR=/absolute/path/to/your/data

# Start the services
docker-compose up -d
```

The Docker deployment includes:
- **API Server**: FastAPI backend with automatic reloading
- **Frontend**: Vue 3 application served via Nginx
- **Security Hardening**: Non-root users, read-only filesystems, dropped capabilities
- **Resource Limits**: Memory and CPU constraints to prevent resource exhaustion

See [Docker Deployment](../deployment/local-docker.md) for detailed configuration options.

## Method 2: Developer Installation (Local)

For developers contributing to the codebase, we use `uv` for fast package management.

### 1. Install uv
If you haven't installed `uv` yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, using pip:
```bash
pip install uv
```

### 2. Setup the Environment
We provide a `Makefile` to automate environment setup:

```bash
# Clone the repository
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve

# Install dependencies and create virtual environment automatically
make install-dev
```

This command runs `uv sync --all-extras`, installing:
- Core package dependencies
- API dependencies (FastAPI, ChromaDB, SQLite)
- Text processing extras (spaCy models for multiple languages)
- Development tools (Ruff, mypy, pytest)

### 3. Verify Installation
Activate the virtual environment and check the CLI:

```bash
source .venv/bin/activate
phentrieve --version
```

You should see output like:
```
Phentrieve version 0.1.0
```

## Alternative Installation Methods

### Using pip (Not Recommended)

While `uv` is strongly recommended for its speed (10-100x faster than pip), you can still use pip:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from source
pip install -e ".[all]"
```

**Note**: This method is significantly slower and does not benefit from uv's advanced dependency resolution.

## Next Steps

After successful installation:

1. Proceed to [Initial Setup](initial-setup.md) to prepare the HPO data and build the necessary indexes
2. Explore the [CLI Usage Guide](../user-guide/cli-usage.md) to learn about available commands
3. Review [Development Environment](../development/dev-environment.md) for contributing guidelines

!!! note "GPU Acceleration"
    Phentrieve supports GPU acceleration with CUDA for improved performance. If you have a compatible NVIDIA GPU, Phentrieve will automatically detect and use it. The Docker images include CUDA support when available.

!!! tip "Performance"
    Using `uv` for dependency management provides:
    - 10-100x faster package installation
    - More reliable dependency resolution
    - Automatic virtual environment management
    - Better reproducibility via lockfiles
