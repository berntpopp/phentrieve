# API Usage

This page explains how to use the Phentrieve API for integrating HPO term mapping into your applications.

## API Overview

Phentrieve provides a FastAPI-based REST API that exposes core functionality through a set of endpoints. The API allows you to:

- Query for HPO terms based on text input
- Process clinical text to extract HPO terms
- Manage indexes and data

## API Endpoints

### Main Query Endpoint

```
POST /api/v1/query/
```

This endpoint accepts a JSON payload with the following parameters:

```json
{
  "text": "Der Patient zeigt Mikrozephalie und Krampfanfälle",
  "model_name": "FremyCompany/BioLORD-2023-M",
  "num_results": 5,
  "similarity_threshold": 0.3,
  "enable_reranker": true
}
```

#### Response Format

```json
{
  "results": [
    {
      "id": "HP:0000252",
      "name": "Microcephaly",
      "similarity": 0.85,
      "definition": "A condition in which head circumference is smaller than normal...",
      "synonyms": ["Abnormally small skull", "Decreased head circumference", "..."]
    },
    {
      "id": "HP:0001250",
      "name": "Seizures",
      "similarity": 0.78,
      "definition": "Seizures are an intermittent abnormality of the central nervous system...",
      "synonyms": ["Convulsions", "Fits", "..."]
    }
  ]
}
```

### Text Processing Endpoint

```
POST /api/v1/text/process/
```

This endpoint processes clinical text and extracts HPO terms:

```json
{
  "text": "The patient exhibits microcephaly and frequent seizures.",
  "model_name": "FremyCompany/BioLORD-2023-M",
  "strategy": "semantic",
  "min_confidence": 0.4,
  "top_term_per_chunk": false
}
```

## Authentication

By default, the API does not require authentication for local usage. For production deployments, you can enable authentication by setting the following environment variables:

- `PHENTRIEVE_API_AUTH_ENABLED=true`
- `PHENTRIEVE_API_KEY=your_secret_key`

Then include the API key in your requests:

```bash
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key" \
  -d '{"text": "Microcephaly"}'
```

## API Documentation

When running the API, full OpenAPI documentation is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Example Code

### Python Client

```python
import requests
import json

def query_hpo_terms(text, model_name="FremyCompany/BioLORD-2023-M"):
    url = "http://localhost:8000/api/v1/query/"
    payload = {
        "text": text,
        "model_name": model_name,
        "num_results": 5,
        "similarity_threshold": 0.3
    }
    response = requests.post(url, json=payload)
    return response.json()

results = query_hpo_terms("The patient exhibits microcephaly")
for result in results["results"]:
    print(f"{result['id']} - {result['name']}: {result['similarity']}")
```

### JavaScript Client

```javascript
async function queryHpoTerms(text, modelName = "FremyCompany/BioLORD-2023-M") {
  const url = "http://localhost:8000/api/v1/query/";
  const payload = {
    text: text,
    model_name: modelName,
    num_results: 5,
    similarity_threshold: 0.3
  };
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  return await response.json();
}

// Usage
queryHpoTerms("The patient exhibits microcephaly")
  .then(data => {
    data.results.forEach(result => {
      console.log(`${result.id} - ${result.name}: ${result.similarity}`);
    });
  });
```
