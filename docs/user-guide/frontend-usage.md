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

### URL Query Parameters

Phentrieve supports URL query parameters that allow pre-filling the search form and automatically triggering searches. This is particularly useful for integrating Phentrieve with other systems or creating bookmarks to specific searches.

The following URL parameters are supported:

| Parameter | Type | Description | Example |
|-----------|------|-------------|--------|
| `text` | string | Pre-fills the query text field | `text=Patient%20has%20headache` |
| `model` | string | Sets the embedding model | `model=FremyCompany/BioLORD-2023-M` |
| `threshold` | float | Sets the similarity threshold (0.0-1.0) | `threshold=0.4` |
| `reranker` | boolean | Enables/disables the reranker | `reranker=true` |
| `rerankerMode` | string | Sets the reranker mode | `rerankerMode=cross-lingual` |
| `autoSubmit` | boolean | Automatically submits the query if true | `autoSubmit=true` |

#### Examples

**Pre-fill search text only:**
```
http://localhost:8080/?text=Patient%20has%20headache
```

**Pre-fill and auto-submit:**
```
http://localhost:8080/?text=Patient%20has%20headache&autoSubmit=true
```

**Complete configuration with specific model and settings:**
```
http://localhost:8080/?text=Patient%20has%20headache&model=FremyCompany/BioLORD-2023-M&threshold=0.4&reranker=true&rerankerMode=cross-lingual&autoSubmit=true
```

#### Notes

- When providing the `text` parameter, the search will automatically be submitted without needing the `autoSubmit` parameter
- The `autoSubmit` parameter is removed from the URL after submission to prevent re-submission on page refresh
- Model names must exactly match one of the available models
- The threshold must be a number between 0.0 and 1.0
- Reranker mode is only applied if the reranker is enabled
