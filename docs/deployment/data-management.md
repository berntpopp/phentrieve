# Data Management

This page explains how Phentrieve manages data in different deployment scenarios and how to configure data directories.

## Data Directory Structure

Phentrieve organizes its data in a structured directory hierarchy:

```text
/your/data/dir/
├── hpo_core_data/    # HPO source files (hp.json, etc.)
├── indexes/          # ChromaDB persistent storage
│   ├── model_name_1/ # Vector indexes for specific models
│   ├── model_name_2/
│   └── ...
├── results/          # Benchmark results
│   ├── summaries/    # JSON summaries per run/model
│   ├── visualizations/ # Plot images
│   └── detailed/     # Detailed CSV results per run
└── hpo_translations/ # Translation files (if used)
```

## Configuration Options

### Environment Variables

You can configure data directories using environment variables:

- `PHENTRIEVE_DATA_DIR`: Base directory for all Phentrieve data
- `PHENTRIEVE_HPO_DATA_DIR`: Directory for HPO data files
- `PHENTRIEVE_INDEX_DIR`: Directory for vector indexes
- `PHENTRIEVE_RESULTS_DIR`: Directory for benchmark results
- `PHENTRIEVE_TRANSLATIONS_DIR`: Directory for translation files (if used)

### Docker Environment

For Docker deployments, specify the host data directory in your `.env.docker` file:

```
PHENTRIEVE_HOST_DATA_DIR=/path/to/your/data
```

This directory will be mounted into the containers at `/app/data`.

## Data Management Commands

### Preparing HPO Data

```bash
# Download and process HPO data
phentrieve data prepare
```

This command:
1. Downloads the official HPO data in JSON format
2. Processes it into the format required by Phentrieve
3. Stores the processed data in the configured data directory

### Building Indexes

```bash
# Build index for a specific model
phentrieve index build --model-name "FremyCompany/BioLORD-2023-M"

# Build indexes for all supported models
phentrieve index build --all-models
```

These commands create vector indexes in the configured index directory.

### Data Cleanup

```bash
# Clean HPO data (removes downloaded and processed files)
phentrieve data clean

# Clean indexes (removes all vector stores)
phentrieve index clean
```

!!! warning "Data Loss"
    These commands permanently delete data. Make sure you have backups if needed.

## Persistent Storage Considerations

### Local Deployments

For local deployments, data is stored in the configured directories on your filesystem. By default, these are subdirectories of the current working directory or the location specified by `PHENTRIEVE_DATA_DIR`.

### Docker Deployments

For Docker deployments, data is persisted through volume mounts:

```yaml
volumes:
  - ${PHENTRIEVE_HOST_DATA_DIR}:/app/data
```

This ensures that data remains available between container restarts and updates.

### Backup Recommendations

Regular backups of the data directories are recommended, especially:

- HPO core data after processing
- Vector indexes after building (these can be time-consuming to rebuild)
- Benchmark results

You can use standard filesystem backup tools for this purpose.
