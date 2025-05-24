# Frontend Usage

This page explains how to use the Phentrieve web frontend interface.

## Overview

Phentrieve includes a Vue.js-based web frontend that provides a user-friendly interface for interacting with the system. The frontend allows you to:

- Query for HPO terms based on text input
- View detailed information about matched HPO terms
- Configure various settings and models

## Accessing the Frontend

Once deployed, the frontend is accessible at:

- **Local development**: http://localhost:8080
- **Docker deployment**: http://localhost:8080 or at your configured domain

## Query Interface

The main interface consists of:

1. **Query Input**: A text area where you can enter clinical text
2. **Settings Panel**: Configuration options for models and parameters
3. **Results Display**: A panel showing matched HPO terms with details

## Features

### Basic Querying

1. Enter clinical text in the query input area
2. Click "Submit" to process the query
3. View matched HPO terms in the results panel

### Model Selection

The frontend allows you to select from available embedding models:

1. Open the settings panel
2. Select a model from the dropdown menu
3. The selected model will be used for subsequent queries

### Reranking Options

Enable cross-encoder reranking for improved precision:

1. Check the "Enable Reranker" option in the settings panel
2. Select a reranking mode (crosslingual or monolingual)
3. Adjust any other reranking parameters as needed

### Results Export

Based on our project memories, the Phentrieve frontend integrates with the @berntpopp/phenopackets-js library to enable exporting HPO terms as valid GA4GH Phenopacket v2 JSON:

1. Enter subject information (optional):
   - Subject ID
   - Sex
   - Date of birth

2. Click "Export as Phenopacket (JSON)" to generate a properly formatted Phenopacket containing:
   - Subject information (if provided)
   - Identified phenotypic features with HPO terms
   - Metadata including creation timestamps
   - Phentrieve resource information

The system also preserves the original text export functionality.

## Advanced Features

### Assertion Status Display

The frontend displays the assertion status of each HPO term:

- **Affirmed**: Terms explicitly mentioned as present
- **Negated**: Terms explicitly mentioned as absent
- **Uncertain**: Terms mentioned with uncertainty
- **Normal**: Terms described as normal or within normal limits

### Confidence Filtering

Adjust the confidence threshold for displayed results:

1. Use the confidence slider in the settings panel
2. Only HPO terms with similarity scores above the threshold will be shown

### Language Support

The frontend supports multilingual input thanks to the underlying embedding models:

1. Enter text in any supported language
2. The system will process it directly without requiring translation
