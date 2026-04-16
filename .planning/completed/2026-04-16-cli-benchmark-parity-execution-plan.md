# CLI Benchmark Parity Execution Plan

**Status:** Completed
**Completed:** 2026-04-16
**Outcome:** GeneReviews full benchmark improved to assertion-aware micro F1 0.7629 with durable tracing, better scoring projection, and passing CI.

> Behavioral reference: `../phentrieve-bench`, with feature deltas from branches `feature/llm-annotation-system` and `feature/agentic-judge-mode`.
>
> Priority: CLI and benchmark first. API and frontend are explicitly out of scope until CLI parity is competitive on real benchmarks.

## Goal

Reach CLI benchmark parity or near-parity with `../phentrieve-bench` on real Gemini-backed full-text extraction, starting with GeneReviews, while keeping prompts and operational parameters configurable and preserving production-quality boundaries.

## Slice Decomposition

### Slice 1: Provider, tool executor, prompt loader, and config packaging

Scope:
- `phentrieve/llm/provider.py`
- `phentrieve/llm/prompts/loader.py`
- `phentrieve/llm/config.py`
- `phentrieve/llm/types.py`
- `phentrieve/llm/utils.py`
- `phentrieve/llm/__init__.py`
- `phentrieve/llm/prompts/templates/**`
- `tests/unit/llm/test_provider.py`
- `tests/unit/llm/test_prompts.py`
- `tests/unit/llm/test_types.py`
- `tests/unit/llm/test_utils.py`

Target:
- Port the benchmark-grade provider abstraction and tool executor shape from the bench reference branches.
- Keep prompts/configs file-backed and configurable.
- Do not hardcode prompts or tuning constants beyond named config values.

### Slice 2: CLI runtime and benchmark-grade two-phase execution path

Scope:
- `phentrieve/llm/pipeline.py`
- `phentrieve/cli/text_commands.py`
- `phentrieve/text_processing/full_text_service.py` only if required for clean integration
- `tests/unit/llm/test_pipeline.py`
- `tests/unit/llm/test_two_phase.py`
- `tests/unit/cli/test_text_commands.py`

Target:
- Port the strongest CLI-facing benchmark mode required for parity first, centered on `two_phase`.
- Keep runtime behavior aligned with the bench implementation, with prompt/provider dependencies injected through configurable boundaries.

### Slice 3: Benchmark harness, dataset flow, metrics, and reporting compatibility

Scope:
- `phentrieve/benchmark/data_loader.py`
- `phentrieve/benchmark/llm_benchmark.py`
- `phentrieve/benchmark/llm_cli.py`
- `phentrieve/benchmark/extraction_benchmark.py` only if compatibility requires it
- `tests/unit/test_llm_benchmark.py`
- `tests/unit/cli/test_benchmark_commands.py`

Target:
- Make benchmark execution compatible with the bench reference datasets, result structure, and metrics path needed for real Gemini reruns.
- Prefer GeneReviews-first flows and reporting that supports direct comparison against bench outputs.

## Execution Rules

- Every slice uses TDD: failing targeted test first, then minimal implementation, then refactor.
- After each slice:
  1. Implementer pass
  2. Spec-compliance review
  3. Code-quality review
- After each major slice integration, rerun a real Gemini benchmark on the proper full-text dataset, starting with GeneReviews.
- Do not widen into API or frontend work until CLI benchmark parity is competitive.
- Do not open or update a PR until the parity objective is met and full verification passes.

## Integration Order

1. Freeze reference behavior from the bench branches for the three slices above.
2. Land Slice 1, review it, and verify targeted unit coverage.
3. Land Slice 2, review it, and verify targeted CLI and pipeline coverage.
4. Land Slice 3, review it, and verify targeted benchmark coverage.
5. Run live GeneReviews Gemini benchmark, diagnose gaps, and iterate on the narrowest failing slice.
6. Only after benchmark competitiveness is achieved, run repository-wide required checks.
