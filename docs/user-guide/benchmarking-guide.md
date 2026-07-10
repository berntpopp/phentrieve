# Benchmarking Guide

This page shows the concrete benchmark commands for retrieval and LLM full-text
validation.

## Retrieval Benchmark

```bash
phentrieve benchmark run \
  --test-file tests/data/benchmarks/german/tiny_v1.json \
  --model-name "sentence-transformers/LaBSE"
```

## LLM Full-Text Benchmark

The primary LLM benchmark workflow uses the converted PhenoBERT full-text
corpus under `tests/data/en/phenobert/`. The benchmark instantiates the LLM
pipeline directly and does not go through the FastAPI quota layer.

```bash
phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-model gemini-2.5-flash
```

The converted corpus contains these dataset subsets:

- `GSC_plus`
- `ID_68`
- `GeneReviews`
- `all`

The output JSON includes:

- `cases`
- `dataset`
- `llm_model`
- `llm_mode`
- `dataset_metadata`
- `metrics`
- `results`
- `output_path`

## Corpus Acquisition And Conversion

If you need to rebuild the corpus, use the reproducible PhenoBERT download and
conversion workflow already documented in this repo:

- `scripts/PHENOBERT-DOWNLOAD-GUIDE.md`
- `scripts/README.md`
- `scripts/convert_phenobert_data.py`

Typical conversion flow:

```bash
python scripts/convert_phenobert_data.py \
  --phenobert-data /path/to/PhenoBERT/phenobert/data \
  --output tests/data/en/phenobert \
  --hpo-data data/hpo_core_data
```

Use a specific upstream PhenoBERT commit for reproducibility and keep the
generated `conversion_report.json`.

## Legacy Smoke Datasets

The small JSON files under `tests/data/benchmarks/` remain useful for quick
smoke validation, but they are not the primary full-text benchmark workflow.

```bash
phentrieve benchmark llm \
  --test-file tests/data/benchmarks/german/tiny_v1.json \
  --llm-model gemini-2.5-flash
```

Provider and model selection are CLI parameters. Use `.env` for keys, not for
per-run model switching:

```bash
uv run --env-file .env phentrieve benchmark llm \
  --test-file tests/data/benchmarks/german/tiny_v1.json \
  --llm-provider openrouter \
  --llm-model meta-llama/llama-3.1-70b-instruct
```

For multi-model smoke runs, keep one model id per line and let the helper call
the same benchmark command for each model:

```text
# models.txt
meta-llama/llama-3.1-70b-instruct
google/gemini-3.1-flash-lite
```

```bash
uv run python scripts/run_llm_model_benchmarks.py \
  --test-file tests/data/benchmarks/german/tiny_v1.json \
  --models-file models.txt \
  --output-dir data/results/openrouter-smoke \
  -- --language en
```

Token cost estimates are already integrated into the LLM benchmark. For
`--llm-provider openrouter`, Phentrieve fetches current model pricing from
OpenRouter's Models API when no manual pricing is supplied. The fetched
per-token `prompt`, `completion`, and `input_cache_read` prices are converted to
Phentrieve's per-1M-token accounting fields.

You can override pricing directly:

```bash
uv run --env-file .env phentrieve benchmark llm \
  --test-file tests/data/benchmarks/german/tiny_v1.json \
  --llm-provider openrouter \
  --llm-model meta-llama/llama-3.1-70b-instruct \
  --input-cost-per-1m-tokens "$INPUT_PRICE_PER_1M" \
  --output-cost-per-1m-tokens "$OUTPUT_PRICE_PER_1M"
```

or with `--pricing-config path/to/pricing.json`. Manual pricing and pricing
config files take precedence over the OpenRouter fetch. If the fetch fails, the
benchmark still runs and cost estimates remain `null`.

## Example CLI LLM Run

```bash
phentrieve text process clinical_note.txt \
  --extraction-backend llm \
  --llm-model gemini-3.1-flash-lite-preview
```

## API Quota Environment

These variables matter for API and frontend validation. They do not gate the
direct benchmark command above.

```bash
export PHENTRIEVE_ENV=production
export PHENTRIEVE_TRUSTED_PROXY_CIDRS="127.0.0.1/32,10.0.0.0/8"
export PHENTRIEVE_LLM_DAILY_LIMIT=3
export PHENTRIEVE_LLM_QUOTA_DB_PATH="../data/app/llm_quota.db"
```
