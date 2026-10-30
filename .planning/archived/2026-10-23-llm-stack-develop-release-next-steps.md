# LLM phase-1 stack on `develop`: next steps to release

Date: 2026-10-23
Status: archived 2026-10-30 - the stack shipped in v0.30.0 (#457). The full CSC/GSC run
was not done; the release PR records a 13-document sample instead.

## Where we are

All of the following are merged into `develop` (merge commits, nothing on `main`
yet, so no Docker image build has run):

| PR | Content |
|---|---|
| #373 | fsspec 2026.9.0 (CVE-2026-104851) |
| #386 | lockfile security update: pillow 13, mcp 1.30, opentelemetry 1.45.1, certifi, virtualenv |
| #374 | grounded phase-1 prompt without empty clinical-text block; system prompt valid for legacy mode (prompt v3.2.0) |
| #376 | chunk-proportional phase-1 output budget, no identical retry after truncation |
| #377 | phase-1 chunk index restores stripped negation cues (`prompt_text`) |
| #384 | evidence offsets computed in code (`offset_basis`), mapped into cleaned chunk text |
| #385 | prompt v3.3.0 axes contract: negation scope, uncertain, experiencer `other`, temporality |

Every PR had full CI green against `develop` before merging, plus local
`make check`, `make typecheck-fast`, `make test`. Review findings and their
resolution are in the PR comments and the "Review follow-up" sections of the PR
descriptions.

Why `develop`: every push to `main` runs `docker-publish.yml` (API image 20+
min, pushed as `:latest`) and a `v*` tag builds again. CI, CodeQL and security
scans run on PRs to and pushes to `develop`; image builds do not.

## Next steps

### 1. Benchmark sample (minutes)

Run `phentrieve benchmark llm` (see `--help` for exact arguments) on `develop`
with a fixed model and `--llm-seed`, restricted with repeated `--doc-id` to
10-20 CSC/GSC documents covering the review risk areas:

- coordinated negation lists ("no A, B or C", also German "keine ... oder ...")
  - the #375 sentence is the reference case
- onset / inheritance / age / measurement phrases (must stay category Other,
  never become patient phenotypes)
- family-history and other-person findings (neighbour, donor, ...)
- long, dense notes (output budget of #376; watch for grouped fallbacks)
- one run with `--llm-internal-mode whole_document_legacy` (prompt fix of #374)
- if a reasoning model is used anywhere in production: a few long documents
  with it (hidden reasoning tokens count against the 8,192-token floor)

Check by hand: negated list items are absent, no metadata in the term list,
`other_person_findings` populated where expected, no duplicate rows,
highlight spans (`text_attributions`) point at the right text.

### 2. Full CSC/GSC run (once)

One full run on `develop` (CSC 112 docs, GSC 114 docs) with the baseline model,
compared against `.planning/analysis/2026-07-12-llm-benchmark-csc-gsc-f1-regression-check.md`:

| Model | Dataset | P | R | F1 (baseline) |
|---|---|---:|---:|---:|
| Gemini 3.1 Flash Lite | CSC | 0.742 | 0.687 | 0.713 |
| Gemini 3.1 Flash Lite | GSC | 0.810 | 0.698 | 0.750 |

- Roughly +-1 F1 point is noise.
- On a regression, do not rerun the full benchmark per PR; bisect with the
  sample from step 1 on the intermediate merge commits (#374 `038df20`,
  #377 `bfd3f4b`, #385 `c9c054a`).
- Record the result as a new note in `.planning/analysis/`.

Open decisions before running: model (default proposal: Gemini 3.1 Flash Lite
via OpenRouter, as in the baseline), provider credentials in the environment.

### 3. Release

1. If the benchmark is acceptable: release commit on `develop` (CHANGELOG
   `[Unreleased]` -> version, bump CLI / API / frontend versions together).
   Add a CHANGELOG line for #386 (it has none yet).
2. One PR `develop` -> `main`, merge with a merge commit (no squash). This is
   the single image build.
3. Tag `v*` afterwards (second build; see optional item below).
4. Decide whether to keep `develop` as the integration branch or delete it.
   If kept: add the same required status checks as `main` in branch
   protection, and periodically merge `main` into `develop` (Dependabot PRs
   still target `main`).

## Lessons from the merge sequence

- Before merging a stacked PR and deleting its branch, retarget the next PR to
  `develop` first. Deleting the base branch makes GitHub close the dependent
  PR (happened with #376; fixed by re-pushing the base branch, reopening and
  retargeting).
- Changing a PR's base does not trigger CI. Merge `origin/develop` into the
  branch and push to get a run against the new base (no rebase, no force-push).
- A re-run of a `pull_request` workflow reuses the old merge commit; it does
  not pick up a newer base.

## Follow-ups (not blocking the release)

- #374: align the phase-1 few-shot examples with the chunk-index input shape
  (prompt change -> bundle with the next benchmark).
- #377 / #385: carrying a negation cue across coordinated chunks is handled by
  the v3.3.0 prompt; verify in the benchmark sample.
- #384: phase-1 dedup still compares model-reported offsets.
- #385: phase timings, `phase2b_llm` trace and benchmark counts for the
  other-person path; frontend does not display `other_person_findings` or
  `temporality` yet.
- CI: `Docker Build Test` was skipped for the lockfile-only #386 although the
  API image installs those dependencies; consider triggering it on `uv.lock`.
- Optional: avoid the double image build per release (tag build re-tags the
  `main` image, or build only on tags).
- Local cleanup: worktrees under `../phentrieve-worktrees/` for the merged
  branches; `deps-security-2026-10` failed to delete because of Windows path
  length in `.venv` (needs `git config core.longpaths true` or manual removal,
  then `git worktree prune`).
