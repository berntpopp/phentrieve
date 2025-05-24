# Installation

This guide will walk you through the process of installing Phentrieve and its dependencies.

## Prerequisites

Phentrieve requires:

* Python 3.8 or higher
* Pip package manager
* Virtual environment (recommended)
* Git (for cloning the repository)

## Installation Methods

### Method 1: Using pip (Recommended)

The simplest way to install Phentrieve is using pip:

```bash
# Create and activate a virtual environment (recommended)
python -m venv phentrieve-env
source phentrieve-env/bin/activate  # On Windows: phentrieve-env\Scripts\activate

# Install Phentrieve
pip install phentrieve
```

### Method 2: From Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/berntpopp/phentrieve.git
cd phentrieve

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Verifying Installation

After installation, verify that Phentrieve is correctly installed:

```bash
phentrieve --version
```

You should see the current version number of the Phentrieve package.

## Next Steps

After successful installation:

1. Proceed to [Initial Setup](initial-setup.md) to prepare the HPO data and build the necessary indexes
2. Explore the [CLI Usage Guide](../user-guide/cli-usage.md) to learn about available commands

!!! note "GPU Acceleration"
    Phentrieve supports GPU acceleration with CUDA for improved performance. If you have a compatible NVIDIA GPU, Phentrieve will automatically detect and use it.
