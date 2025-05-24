# Configuration Profiles

This page explains how to use and customize configuration profiles in Phentrieve.

## Introduction

Phentrieve uses configuration profiles to manage different settings for various use cases. These profiles help you customize the behavior of the system without modifying the code.

## Default Configuration

The default configuration is defined in `phentrieve/config.py` and includes settings for:

- Data directories
- Default models
- Processing parameters
- Logging levels

## Environment Variables

You can override configuration settings using environment variables:

```bash
# Set data directory
export PHENTRIEVE_DATA_DIR=/path/to/your/data

# Set default model
export PHENTRIEVE_DEFAULT_MODEL="FremyCompany/BioLORD-2023-M"
```

## Configuration Files

For more permanent configuration, you can create configuration profiles:

1. Create a `.phentrieve` directory in your home folder
2. Add a `config.yaml` file with your custom settings

Example `config.yaml`:

```yaml
data:
  base_dir: /path/to/data
  hpo_data_dir: ${data.base_dir}/hpo_core_data
  index_dir: ${data.base_dir}/indexes
  results_dir: ${data.base_dir}/results

models:
  default: "FremyCompany/BioLORD-2023-M"
  reranker: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

processing:
  chunking_strategy: "semantic"
  min_confidence: 0.4
  window_size: 128
  step_size: 64
```

## Profile Selection

You can switch between different configuration profiles:

```bash
# Use a specific configuration profile
phentrieve --config-profile clinical query --interactive

# Specify config file directly
phentrieve --config-file /path/to/custom-config.yaml query --interactive
```

## Configuration Hierarchy

Phentrieve uses the following hierarchy for configuration (later sources override earlier ones):

1. Default configuration in code
2. Configuration files
3. Environment variables
4. Command-line arguments
