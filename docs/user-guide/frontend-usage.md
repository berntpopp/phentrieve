# Frontend Usage

This page covers the concrete frontend flow for standard and LLM full-text
extraction.

## Accessing the Frontend

- Development: `http://localhost:5734`
- API for the frontend: `http://localhost:8734`

## Running an LLM Full-Text Analysis

```bash
phentrieve text process --extraction-backend llm note.txt
```

1. Start the API with your LLM environment configured.
2. Open the frontend at `http://localhost:5734`.
3. Paste research phenotype text into the main input.
4. Open Advanced Options.
5. Select the `llm` extraction backend.
6. The shared default model is `gemini-3.1-flash-lite-preview`; override it only for targeted experiments.
7. Keep `two_phase` as the mode.
8. Submit the analysis.

The results view keeps the same overall structure as standard extraction and
adds LLM metadata so you can verify which model and mode produced the output.

## Example API Environment for Frontend Validation

```bash
export PHENTRIEVE_ENV=production
export PHENTRIEVE_TRUSTED_PROXY_CIDRS="127.0.0.1/32,10.0.0.0/8"
export PHENTRIEVE_LLM_DAILY_LIMIT=3
export PHENTRIEVE_LLM_QUOTA_DB_PATH="../data/app/llm_quota.db"
```

Use those settings when you want the frontend to exercise the same proxy and
quota behavior as the production API. The benchmark CLI does not use this quota
path.
