# Benchmarking Guide

This page shows the concrete benchmark commands for retrieval, document
extraction, and LLM full-text validation.

## Retrieval Benchmark

Use the canonical German single-term datasets for official retrieval
comparisons:

```bash
phentrieve benchmark run \
  --test-file tests/data/benchmarks/german/570terms_german.json \
  --model-name "FremyCompany/BioLORD-2023-M" \
  --results-dir results
```

```bash
phentrieve benchmark run \
  --test-file tests/data/benchmarks/german/200cases_o3_v1.json \
  --model-name "FremyCompany/BioLORD-2023-M" \
  --results-dir results
```

The default benchmark remains the small smoke dataset for fast local checks:

```bash
phentrieve benchmark run \
  --test-file tests/data/benchmarks/german/tiny_v1.json \
  --model-name "sentence-transformers/LaBSE" \
  --results-dir results
```

Each model run receives its own directory. Use `--run-id baseline` for a stable
name. Reusing that name requires the explicit `--overwrite` option.

## GSC And CSC Extraction Benchmarks

The canonical text-fragment corpora are the converted RAG-HPO `GSC` and `CSC`
sets under `tests/data/en/raghpo_paper/`. They are legacy polarity-blind
corpora, so use `present-only` scoring while leaving assertion detection
enabled:

```bash
phentrieve benchmark extraction run tests/data/en/raghpo_paper \
  --dataset GSC \
  --model "FremyCompany/BioLORD-2023-M" \
  --language en \
  --scoring-mode present-only \
  --output-dir results
```

```bash
phentrieve benchmark extraction run tests/data/en/raghpo_paper \
  --dataset CSC \
  --model "FremyCompany/BioLORD-2023-M" \
  --language en \
  --scoring-mode present-only \
  --output-dir results
```

Use `GSC_plus`, `ID_68`, and `GeneReviews` under
`tests/data/en/phenobert/` for regression and diagnostic runs.

## Run Artifacts

Retrieval, extraction, and LLM full-text runs use this hierarchy:

```text
results/
  retrieval/<dataset>/<model>/<run-id>/
  extraction/<dataset>/<model>/<run-id>/
  llm/<dataset>/<model>/<run-id>/
```

The important files are:

- `manifest.json`: run identity, dataset checksum, model, configuration, status,
  and machine-readable artifact locations.
- `summary.json`: aggregate metrics and per-case arrays used by comparisons and
  the standard benchmark graphics.
- `terms.jsonl`: ranked or extracted HPO terms, scores, gold membership, and
  TP/FP/FN outcomes. Extraction records distinguish pipeline predictions from
  evaluated predictions and identify whether a term was removed by the chunk
  threshold, chunk selection, aggregation confidence, or scoring mode. Use
  this for term-level analysis.
- `cases.jsonl`: query/document inputs, predictions, per-case metrics, timings,
  and errors. Use this for statistical analysis.
- `diagnostics/chunks.jsonl`: extraction chunks and every returned top-N
  candidate before thresholding. This is diagnostic material rather than the
  primary result. Extraction only; absent for retrieval and LLM runs.
- `legacy/`: the previous summary JSON, CSV, and extraction result formats.
  Retrieval and extraction only.
- `predictions/<llm-mode>/`, `traces/<llm-mode>/`, and `metrics/`: per-document
  LLM predictions, per-document pipeline traces, and an aggregate metrics JSON.
  LLM runs only. Traces are always written; `--capture-phase1-debug` adds the
  phase-1 retrieval detail to them rather than enabling them.

Retrieval and extraction currently publish manifest schema v1. LLM runs publish
schema v2, whose canonical inventory contains the SHA-256 digest of every owned
file. The direct `summary`, `checkpoint`, `metrics`, `term_results`, and
`case_results` keys and the `llm_predictions` / `llm_traces` directory keys are
compatibility aliases for v1 consumers. Code that iterates every v2 artifact
entry must deduplicate by `path`; aliases are lookup entries, not additional
files.

LLM manifests separate identities according to what they describe:

- `source_sha256` hashes the selected source files exactly as stored.
- `input_sha256` hashes the model-visible document identifiers and text after
  loading and selection.
- the gold hashes identify positive-only, assertion-aware, and ID-only scoring
  labels independently of model input.
- `projection_sha256` identifies the effective dataset assertion projection.
- `execution_fingerprint` identifies inputs that can change inference, including
  document order, prompts, resolved provider/model settings, seed, and retrieval
  assets.
- `scoring_fingerprint` identifies gold labels, selected documents, and scoring
  projection. Producer version and Git provenance are recorded separately.

Comparison and visualization commands discover structured runs recursively and
still accept old flat summary directories. Both compare dense retrieval metrics,
so they only ever load `retrieval` runs; pointing them at a results root that
also holds extraction and LLM runs is safe:

```bash
phentrieve benchmark compare --summaries-dir results
phentrieve benchmark visualize --summaries-dir results
```

When `--results-dir` is omitted for retrieval, `PHENTRIEVE_RESULTS_DIR` or the
configured user results directory is used. Extraction defaults to the local
`results/` root.

## LLM Full-Text Benchmark

The primary LLM benchmark workflow uses the converted PhenoBERT full-text
corpus under `tests/data/en/phenobert/`. The benchmark instantiates the LLM
pipeline directly and does not go through the FastAPI quota layer.

An LLM benchmark requires an installed retrieval bundle with a valid
`manifest.json`, embedding-model identity, and HPO version. Install a published
bundle and inspect it before starting a run:

```bash
phentrieve data download --model bge-m3 --hpo-version v2026-06-23
phentrieve data status
```

Use the bundle appropriate for the experiment. To assert the expected ontology
explicitly, pass `--evaluation-hpo-version`; the command rejects a value that
does not match the installed retrieval bundle:

```bash
phentrieve benchmark llm \
  --test-file tests/data/en/phenobert \
  --dataset GeneReviews \
  --llm-model gemini-2.5-flash \
  --evaluation-hpo-version v2026-06-23
```

The converted corpus contains these dataset subsets:

- `GSC_plus`
- `ID_68`
- `GeneReviews`
- `all`

The RAG-HPO `GSC` and `CSC` sets under `tests/data/en/raghpo_paper/` are also
supported (see [GSC And CSC Extraction Benchmarks](#gsc-and-csc-extraction-benchmarks)
for background on the corpora).

Like retrieval and extraction runs, each invocation writes to its own
`results/llm/<dataset>/<model>/<run-id>/` directory (see
[Run Artifacts](#run-artifacts)): `manifest.json`, `summary.json`,
`terms.jsonl`, `cases.jsonl`, plus the LLM-specific `predictions/`, `traces/`,
and `metrics/` artifacts. Use `--run-id` for a stable name; reusing that name
requires the explicit `--overwrite` option. For LLM runs, `--overwrite` means
resume only when the preserved checkpoint has matching fingerprints and full
configuration. Validation happens before existing artifacts are cleared. A
pre-v2 or otherwise incompatible checkpoint is rejected without modifying the
run; choose a new `--run-id` or deliberately remove the old run directory to
start fresh. `--output-dir` (default `results`) sets the result root.

Provider seeds are best-effort because support depends on the selected provider
and model. When supplied through `--llm-seed`, the resolved seed is included in
the execution fingerprint and passed to the runtime provider.

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
