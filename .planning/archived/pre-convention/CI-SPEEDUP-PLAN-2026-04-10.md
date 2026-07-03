# CI Pipeline Speed-up Plan

**Created**: 2026-04-10
**Branch**: `improve/code-quality-2026-04` (extends PR #191, NOT a new branch)
**Goal**: Reduce full CI wall-clock from ~15-20 min to ~5-8 min without sacrificing quality, and make local `make` commands match CI exactly so developers get the same feedback loop.

**Status source**: Research synthesis from three parallel agents (2026-04-10). See `plan/05-analysis/CODE-QUALITY-REVIEW-2026-04-09.md` for the broader context.

**Research sources (reputable 2026 references)**:
- [Trail of Bits — Making PyPI's test suite 81% faster](https://blog.trailofbits.com/2025/05/01/making-pypis-test-suite-81-faster/)
- [astral-sh/setup-uv official GitHub Action](https://github.com/astral-sh/setup-uv)
- [Vitest: Improving Performance](https://vitest.dev/guide/improving-performance)
- [Vite: Build Options](https://vite.dev/config/build-options)
- [ESLint v9.34: Multithread Linting](https://eslint.org/blog/2025/08/multithread-linting/)
- [Prettier CLI docs — `--cache`](https://prettier.io/docs/cli)
- [Docker cache backends — GHA](https://docs.docker.com/build/cache/backends/gha/)
- [AustinScola/mypy-cache-github-action](https://github.com/AustinScola/mypy-cache-github-action)
- [GitHub Actions concurrency control](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)

---

## Why now

PR #191 (`improve/code-quality-2026-04`) just landed 14 commits of Phase 2 work on top of Streams A/B/C. All 17 CI checks are green. The branch is in a stable state to extend with infra improvements before merge. Doing this on the same branch (not a new one) means:
- The speedups take effect the moment PR #191 merges to main
- No second PR-cycle review churn
- Each speedup commit can be verified against the now-green baseline

## Scope boundaries

**In scope** (Tier 1 + Tier 2 from the research synthesis):
- Local + CI Python tooling speed (ruff, mypy, pytest, uv)
- Local + CI frontend tooling speed (ESLint, Prettier, Vitest, Vite build)
- GitHub Actions workflow optimization (caching, concurrency, parallelism)
- Docker build cache parity between API and frontend
- Makefile ↔ CI parity fixes

**Out of scope** (defer to a future plan):
- `pytest-split` test sharding across CI jobs — over-engineered for current suite size (927 tests)
- Migration from `npm` to `pnpm` — Tier 3 risk, defer
- `ty` (Astral Rust type checker) — still beta in April 2026, no plugin system for Pydantic
- Turborepo — only 2 packages, not worth the overhead
- Vitest browser mode — `happy-dom` is sufficient
- Switching from `mypy` to `pyright` — different error set needs separate tuning pass

---

## Baseline (capture before any changes)

| Metric | Current |
|---|---|
| Python CI (3.10/3.11/3.12) wall-clock | ~8-15 min per version (matrix parallelism ⇒ ~15 min overall) |
| Frontend CI wall-clock | ~5-8 min |
| Docker API build | ~20-30 min (ML deps) |
| Docker frontend build | ~3-5 min |
| Full CI wall-clock (longest pole) | ~15-20 min |
| `make test` (local, no coverage) | ~28s for 927 tests |
| `make frontend-test` (local) | ~1.2s for 107 tests |
| `make frontend-build` (local) | ~30-45s (terser + brotli compression at level 11) |

**Target after Tier 1**: full CI wall-clock ~5-8 min; local `make test` ~10-15s; local `make frontend-build` ~10-15s.

---

## Task List

### Tier 1 — Drop-in wins (8 tasks, LOW risk, ~30-45 min total work)

Each task is a standalone atomic commit. Dispatch order matters only for Task 0 (baseline measurement) which must come first.

### Task 0: Capture baseline timings

**Why**: The claim "CI went from 20 min to 8 min" is only credible if we record the starting point.

**Files**: Create `plan/01-active/CI-SPEEDUP-BASELINE-2026-04-10.md` (new, non-committed — just record locally for comparison)

**Steps**:

- [ ] **Step 1: Record current CI durations from recent PR #191 runs**

```bash
gh run list --workflow="CI - Continuous Integration" --limit 5 --json databaseId,displayTitle,createdAt,updatedAt,conclusion | python3 -c "
import json, sys
from datetime import datetime
runs = json.load(sys.stdin)
for r in runs:
    if r.get('conclusion') != 'SUCCESS':
        continue
    start = datetime.fromisoformat(r['createdAt'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(r['updatedAt'].replace('Z', '+00:00'))
    dur = (end - start).total_seconds() / 60
    print(f\"{r['displayTitle'][:60]:60} {dur:.1f} min\")
"
```

Record the results in a local note.

- [ ] **Step 2: Record local timings**

```bash
time make check                    # ruff format + lint
time make typecheck-fast           # dmypy
time make test                     # pytest with coverage
time make frontend-lint            # ESLint
time make frontend-test            # Vitest
time make frontend-build           # Vite build
```

Record each wall-clock time for before/after comparison.

**Commit**: none — baseline is for local reference only.

---

### Task 1: Enable `pytest-xdist -n auto`

**Why**: `pytest-xdist` is already in `dev` dependencies but never invoked. A 4-core GitHub-hosted runner gets 3-4× parallelization for free. 927 tests in 28s → expected ~8-12s.

**Files**:
- Modify: `pyproject.toml` (addopts)

**Safety check first**: Ensure tests don't share filesystem state that would collide under parallelism.

- [ ] **Step 1: Audit test isolation**

```bash
grep -rn "tmp_path\|tempfile\|/tmp/\|os.getcwd\|chdir" tests/ --include="*.py" | grep -v "conftest\|\.pyc" | head -30
```

Look for tests that write to hardcoded paths (not `tmp_path` fixtures). `tmp_path` is worker-safe; hardcoded `/tmp/...` paths are not.

Expected: all test writes go through `tmp_path` or similar fixtures. If any test uses a shared path, STOP and report — that test needs a fixture conversion before enabling xdist.

- [ ] **Step 2: Try `-n auto` locally first to catch hidden state issues**

```bash
uv run pytest tests/ -n auto --no-cov 2>&1 | tail -10
```

Expected: 927 passed in ~8-12s. If any tests fail that passed serially, investigate and fix the offending fixtures before modifying `pyproject.toml`.

- [ ] **Step 3: Add `-n auto` to `pyproject.toml` addopts**

Read `pyproject.toml` first. In the `[tool.pytest.ini_options]` section's `addopts` list, add `"-n", "auto"` after `"--tb=short"`:

```toml
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "-n", "auto",   # pytest-xdist parallelization across CPU cores

    # Coverage (80% target - temporarily disabled during migration)
    "--cov=phentrieve",
    # ... rest unchanged
]
```

- [ ] **Step 4: Run the full suite to verify**

```bash
make test 2>&1 | tail -5
```

Expected: 927 passed, same count, ~8-12s wall-clock.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
perf(test): enable pytest-xdist parallel execution

pytest-xdist (3.8.0) was already in dev dependencies but never
invoked. Adding `-n auto` to pytest addopts uses all available CPU
cores for parallel test execution.

Expected gain: 927 tests in ~28s → ~8-12s locally; similar speedup
in CI (3-4x on GitHub-hosted 4-core runners).

Verified test isolation before enabling — all writes use tmp_path
or similar worker-safe fixtures. No hardcoded /tmp paths in tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Cache `.mypy_cache` in CI + skip mypy on non-Python changes

**Why**: Fresh CI runners rebuild the mypy cache from scratch every run. Caching `.mypy_cache/` keyed on Python version + pyproject.toml hash cuts typecheck time by 40-60%.

**Files**:
- Modify: `.github/workflows/ci.yml` — add cache step before the mypy invocation

- [ ] **Step 1: Read the current CI workflow**

```bash
grep -n "mypy\|typecheck" .github/workflows/ci.yml
```

Locate the Python CI job and the exact step that runs `mypy`.

- [ ] **Step 2: Add the cache step immediately before mypy**

Using `AustinScola/mypy-cache-github-action@v1` is the cleanest one-liner, but `actions/cache@v4` is more explicit. Use `actions/cache@v4`:

```yaml
      - name: Cache mypy
        uses: actions/cache@v4
        with:
          path: .mypy_cache
          key: mypy-${{ matrix.python-version }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}
          restore-keys: |
            mypy-${{ matrix.python-version }}-

      - name: Run mypy
        run: uv run mypy phentrieve/ api/
```

(Adjust indentation to match the existing workflow file.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(ci): cache .mypy_cache across runs

mypy's incremental cache was rebuilt from scratch on every CI run.
Caching .mypy_cache/ keyed on python-version + hash of pyproject.toml
+ uv.lock lets subsequent runs skip already-checked modules.

Expected gain: ~40-60% reduction in mypy step time.

Cache key invalidation: any pyproject.toml or uv.lock change triggers
a full re-check (correct behavior). Restore-keys provides partial
fallback across minor changes.

Source: https://github.com/AustinScola/mypy-cache-github-action

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `COVERAGE_CORE=sysmon` for Python 3.12 CI

**Why**: Python 3.12's `sys.monitoring` API is ~53% faster than trace-based coverage (Trail of Bits PyPI benchmark). Only applies on 3.12+; older versions fall back automatically.

**Files**:
- Modify: `.github/workflows/ci.yml` — add env var to the pytest step

**Constraint**: sysmon doesn't support branch coverage on Python 3.12/3.13 (works in 3.14+). Check if phentrieve's coverage config enables branch coverage first.

- [ ] **Step 1: Check for branch coverage in config**

```bash
grep -rn "branch" pyproject.toml .coveragerc 2>/dev/null
```

If `[tool.coverage.run] branch = true` is set, STOP and either disable branch coverage in the sysmon-activated runs OR skip this task. phentrieve likely does NOT use branch coverage (not flagged in the Task 9 caching audit), so this should be clear.

- [ ] **Step 2: Add `COVERAGE_CORE` to the CI env**

Find the "Run Python tests" or equivalent step in `.github/workflows/ci.yml` and add an `env:` block:

```yaml
      - name: Run Python tests
        env:
          COVERAGE_CORE: sysmon
        run: uv run pytest tests/ -v -m "not e2e" --cov=phentrieve --cov=api --cov-report=xml --cov-report=term
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(ci): enable COVERAGE_CORE=sysmon on Python 3.12+

Uses Python 3.12's sys.monitoring API for coverage instrumentation
instead of the legacy trace-based path.

Expected gain: ~53% reduction in coverage-instrumented test time
(Trail of Bits benchmark on PyPI: 58s → 27s).

Python 3.10/3.11 ignore the env var and fall back to the trace-based
path automatically. sysmon currently lacks branch-coverage support
on 3.12/3.13, but phentrieve does not enable branch coverage.

Source: https://blog.trailofbits.com/2025/05/01/making-pypis-test-suite-81-faster/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Switch Vite minifier from Terser to Oxc (default)

**Why**: Oxc is Vite's default since v6 and is 30-90× faster than Terser with only 0.5-2% worse compression. phentrieve explicitly opted into `minify: 'terser'` — removing that opt-in is a pure win.

**Files**:
- Modify: `frontend/vite.config.js`

**Gotcha**: Terser was being used for `drop_console: true` and `drop_debugger: true`. Oxc supports these via a different config key.

- [ ] **Step 1: Read the current vite.config.js build section**

```bash
grep -A 20 "^  build:" frontend/vite.config.js
```

Current shape:
```js
  build: {
    target: 'es2015',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    // ...
  }
```

- [ ] **Step 2: Replace with Oxc equivalent**

Remove `minify: 'terser'` and the entire `terserOptions` block. Add `esbuild` config to handle console/debugger stripping (Vite uses esbuild for the minify phase when `minify` is unset or 'esbuild'; `minify: 'oxc'` is the modern explicit form in Vite 7+).

For phentrieve's Vite version (check `frontend/package.json` for the installed version), the correct approach depends on what's installed:

- **Vite 6.x**: leave `minify` unset (defaults to esbuild, which is fast) and use `esbuild: { drop: ['console', 'debugger'] }` at the top level of the config.
- **Vite 7.x** (if upgraded): set `minify: 'oxc'` explicitly.

Since phentrieve is on Vite 7.x (verify with `grep '"vite":' frontend/package.json`), use:

```js
  build: {
    target: 'es2015',
    // Oxc is the default minifier since Vite 6; it's 30-90x faster than terser
    // with negligible compression loss. No explicit config needed.
    commonjsOptions: {
      // ... existing content unchanged
    },
    rollupOptions: {
      // ... existing content unchanged
    },
    chunkSizeWarningLimit: 600,
    reportCompressedSize: false,  // skip gzip calculation on every build (CI speedup)
  },
```

And add at the top level of `defineConfig({...})`, alongside `plugins:`:

```js
  esbuild: {
    drop: ['console', 'debugger'],
  },
```

- [ ] **Step 3: Verify build still works and console/debugger are stripped**

```bash
make frontend-build 2>&1 | tail -10
```

Expected: successful build, significantly faster than before (previously ~30-45s, now ~10-20s).

Then verify console stripping:
```bash
grep -l "console.log" frontend/dist/assets/*.js 2>&1 | head -5
```

Expected: no matches (console calls stripped).

- [ ] **Step 4: Run frontend tests to make sure nothing depends on terser**

```bash
make frontend-test 2>&1 | tail -5
```

Expected: 107 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/vite.config.js
git commit -m "$(cat <<'EOF'
perf(frontend): use Vite default (esbuild/oxc) minifier instead of terser

phentrieve's vite.config.js explicitly opted into terser for
console/debugger stripping. Vite's default minifier (esbuild/oxc) is
30-90x faster than terser with only 0.5-2% worse compression, and
supports the same console/debugger dropping via esbuild.drop.

Also enables reportCompressedSize: false to skip the post-build gzip
calculation which adds 2-5s for no CI value.

Expected gain: frontend build ~30-45s → ~10-20s.

Console.log / debugger statements are still stripped in production
builds via esbuild.drop.

Sources:
- https://vite.dev/config/build-options
- https://dev.to/perisicnikola37/optimize-vite-build-time-a-comprehensive-guide-4c99

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Gate `viteCompression` and `visualizer` behind `!CI`

**Why**: Brotli compression at level 11 is the slowest compression tier available and is pointless in CI builds (CI verifies the build, it doesn't deploy). `visualizer` also writes a stats.html file that nobody reads in CI.

**Files**:
- Modify: `frontend/vite.config.js`

- [ ] **Step 1: Read the current plugins block**

```bash
grep -A 25 "plugins:" frontend/vite.config.js | head -30
```

- [ ] **Step 2: Wrap the two plugins in a conditional**

Change:

```js
  plugins: [
    vue(),
    commonjs({...}),
    iconOptimizer(),
    viteCompression({
      algorithm: 'brotliCompress',
      ext: '.br',
      threshold: 1024,
      compressionOptions: { level: 11 },
      deleteOriginFile: false,
    }),
    visualizer({
      filename: 'dist/stats.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
```

To:

```js
  plugins: [
    vue(),
    commonjs({...}),
    iconOptimizer(),
    // Brotli compression and bundle visualizer are deploy-time concerns.
    // Skip in CI to save 10-30s per build; keep for local builds so the
    // dev can inspect stats.html and see the actual deploy-ready payload.
    !process.env.CI &&
      viteCompression({
        algorithm: 'brotliCompress',
        ext: '.br',
        threshold: 1024,
        compressionOptions: { level: 11 },
        deleteOriginFile: false,
      }),
    !process.env.CI &&
      visualizer({
        filename: 'dist/stats.html',
        open: false,
        gzipSize: true,
        brotliSize: true,
      }),
  ].filter(Boolean),
```

The `.filter(Boolean)` at the end removes the `false` entries when `CI=true`.

- [ ] **Step 3: Verify local build still runs compression**

```bash
make frontend-build 2>&1 | tail -10
ls frontend/dist/assets/*.br 2>&1 | head -3
```

Expected: `.br` files present (local build still compresses).

- [ ] **Step 4: Verify CI-mode build skips compression**

```bash
CI=true make frontend-build 2>&1 | tail -10
ls frontend/dist/assets/*.br 2>&1 | head -3
```

Expected: successful build, NO `.br` files (compression skipped).

- [ ] **Step 5: Reset by running local build once more to restore normal state**

```bash
make frontend-build 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add frontend/vite.config.js
git commit -m "$(cat <<'EOF'
perf(frontend): skip brotli compression + visualizer in CI builds

Brotli compression at level 11 is the slowest tier available and is
a deploy-time concern. The visualizer plugin writes stats.html which
nobody reads in a CI context. Both add 10-30s to every CI build for
zero CI value.

Gating both behind `!process.env.CI` preserves the full local build
(dev can still inspect stats.html and verify the compressed payload)
while speeding up CI.

Expected CI gain: ~10-30s per frontend build.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add ESLint `--cache` with content strategy

**Why**: ESLint's built-in cache skips files that haven't changed since the last run. With CI caching the `.eslintcache` file, warm runs go from ~5 min to ~10-30s on unchanged files.

**Files**:
- Modify: `frontend/package.json` (lint script)
- Modify: `frontend/.gitignore` (add `.eslintcache`)
- Modify: `.github/workflows/ci.yml` (cache step)

- [ ] **Step 1: Update the lint script in `frontend/package.json`**

Read current `scripts` block. Change:
```json
"lint": "eslint ."
```
To:
```json
"lint": "eslint . --cache --cache-strategy content --cache-location .eslintcache"
```

`--cache-strategy content` hashes file content (not mtime), which is correct for CI where git doesn't preserve mtimes.

- [ ] **Step 2: Add `.eslintcache` to `.gitignore`**

Read `frontend/.gitignore` and append `.eslintcache` if not already present.

- [ ] **Step 3: Add cache step to CI**

In `.github/workflows/ci.yml`, find the `frontend-ci` job and add a cache step before `npm run lint`:

```yaml
      - name: Cache ESLint
        uses: actions/cache@v4
        with:
          path: frontend/.eslintcache
          key: eslint-${{ hashFiles('frontend/package-lock.json', 'frontend/eslint.config.*') }}
          restore-keys: |
            eslint-
```

- [ ] **Step 4: Verify locally**

```bash
cd frontend && rm -f .eslintcache && time npm run lint 2>&1 | tail -5
time npm run lint 2>&1 | tail -5
```

First run: baseline speed. Second run: should be significantly faster (near-instant on unchanged files).

- [ ] **Step 5: Commit**

```bash
cd /home/bernt-popp/development/phentrieve
git add frontend/package.json frontend/.gitignore .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(frontend): enable ESLint cache with content strategy

ESLint's --cache flag skips files that haven't changed since the last
run. Using --cache-strategy content (not mtime) is correct for CI
where git checkouts don't preserve mtimes, so unchanged files are
reliably cached across runs.

Changes:
- frontend/package.json: add --cache --cache-strategy content
  --cache-location .eslintcache to the lint script
- frontend/.gitignore: exclude .eslintcache
- .github/workflows/ci.yml: cache .eslintcache across CI runs, keyed
  on package-lock.json + eslint.config.* hash

Expected gain: warm ESLint runs drop from ~5 min to ~10-30s for
unchanged files (real-world case study).

Source: https://www.charpeni.com/blog/speeding-up-eslint-even-on-ci

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add Prettier `--cache`

**Why**: Prettier 3.0+ supports `--cache` which skips unchanged files. Same rationale as ESLint caching.

**Files**:
- Modify: `frontend/package.json` (format:check script)
- (Cache is stored in `frontend/node_modules/.cache/prettier/` — not tracked, already inside node_modules tree, picked up by existing npm cache)

- [ ] **Step 1: Update the format:check script**

Change:
```json
"format:check": "prettier --check \"src/**/*.{js,vue,css,scss,html}\""
```
To:
```json
"format:check": "prettier --check --cache --cache-strategy content \"src/**/*.{js,vue,css,scss,html}\""
```

Also update the write variant for local parity:
```json
"format": "prettier --write --cache --cache-strategy content \"src/**/*.{js,vue,css,scss,html}\""
```

- [ ] **Step 2: Verify locally**

```bash
cd frontend && time npm run format:check 2>&1 | tail -3
time npm run format:check 2>&1 | tail -3
```

Expected: second run is near-instant.

- [ ] **Step 3: Commit**

```bash
cd /home/bernt-popp/development/phentrieve
git add frontend/package.json
git commit -m "$(cat <<'EOF'
perf(frontend): enable Prettier --cache with content strategy

Prettier 3.0+ supports --cache for skipping unchanged files. Using
--cache-strategy content is correct for CI (git doesn't preserve
mtimes). Cache file lives in node_modules/.cache/prettier and is
automatically covered by the existing npm cache in CI.

Applied to both format:check (CI) and format (local write).

Expected gain: warm format runs drop from seconds to <1s for
unchanged files.

Source: https://prettier.io/docs/cli

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Concurrency groups with `cancel-in-progress` on PRs

**Why**: When a developer pushes a new commit to a PR, the old CI run becomes obsolete but continues consuming runner minutes until timeout. A concurrency group with `cancel-in-progress: true` on pull_request events kills the stale run immediately.

**Files**:
- Modify: all workflow files in `.github/workflows/` that run on pull_request events

- [ ] **Step 1: List affected workflows**

```bash
grep -l "pull_request" .github/workflows/*.yml
```

- [ ] **Step 2: For each workflow, add a top-level concurrency block**

At the top of each workflow file (after `name:` and `on:`), before `jobs:`:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.run_id }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

The `cancel-in-progress` expression is true only for PRs — main branch pushes run to completion (you don't want to cancel a release build).

Do this for: `ci.yml`, any security workflows that run on PRs. Do NOT add to `docker-publish.yml` if it only runs on tags/main push — cancelling release pipelines is bad.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/
git commit -m "$(cat <<'EOF'
perf(ci): add concurrency groups with cancel-in-progress on PRs

When a developer force-pushes or amends a PR commit, the old CI run
becomes obsolete. Without a concurrency group, it continues running
and consumes GitHub Actions runner minutes for no reason.

Adds a concurrency group keyed on workflow + head_ref that cancels
in-progress runs on new PR pushes. The cancel-in-progress flag is
gated to pull_request events only, so main branch push builds and
release pipelines still run to completion.

Expected gain: eliminates wasted runner minutes on superseded commits.

Source: https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Use `astral-sh/setup-uv@v5` with built-in cache

**Why**: The official Astral action has a first-class uv cache that beats generic `actions/cache@v4`. It automatically computes the key from `uv.lock`, manages the cache dir, and runs `uv cache prune --ci` at job end.

**Files**:
- Modify: `.github/workflows/ci.yml` — replace `actions/setup-python@v5` + manual uv install with `astral-sh/setup-uv@v5`

**Check first**: what's the current Python setup in the workflow?

- [ ] **Step 1: Read the current Python setup steps in ci.yml**

```bash
grep -B 1 -A 10 "setup-python\|astral-sh\|install uv\|pipx install uv" .github/workflows/ci.yml
```

- [ ] **Step 2: Replace with the Astral action**

Find the current "Set up Python" step in the python-ci job. Replace the sequence with:

```yaml
      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
          python-version: ${{ matrix.python-version }}
```

This one step replaces both the Python setup AND uv installation.

If the workflow currently does `pip install uv` or similar, remove that step. If it uses a separate `actions/setup-python@v5`, replace that too.

- [ ] **Step 3: Verify the workflow still passes linting**

```bash
gh workflow view ci.yml 2>&1 | head -20
```

Expected: workflow parses cleanly. Any syntax error will fail push.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(ci): switch to astral-sh/setup-uv@v5 with enable-cache

Replace manual `actions/setup-python@v5` + `pip install uv` (or
equivalent) with the official Astral action. setup-uv@v5:

- Installs uv in <1s (no pip compilation)
- Installs Python via uv (matches project's tooling)
- Auto-caches ~/.cache/uv keyed on uv.lock hash
- Runs `uv cache prune --ci` at job end to keep cache lean

Expected gain: dependency install on cache hit goes from ~11s to
near 0ms. Cold install (180+ packages) completes in ~60s.

Source: https://github.com/astral-sh/setup-uv

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Vitest `pool: 'threads'` config + skip coverage on PR runs

**Why**:
1. Default Vitest `forks` pool spawns child processes; `threads` uses Worker threads (lower per-file overhead).
2. Coverage instrumentation adds 15-25% overhead. PR runs don't need coverage upload; main branch merges do.

**Files**:
- Modify: `frontend/vite.config.js`
- Modify: `frontend/package.json` (add `test:ci` script)
- Modify: `.github/workflows/ci.yml` (use `test:ci` for PR runs)

- [ ] **Step 1: Add `pool: 'threads'` to the test section of vite.config.js**

Read the test block first. Add:

```js
  test: {
    globals: true,
    environment: 'happy-dom',
    pool: 'threads',
    setupFiles: './src/test/setup.js',
    // ... rest unchanged
  }
```

- [ ] **Step 2: Add a `test:ci` script in `frontend/package.json`**

Add a new script alongside the existing `test:run` and `test:coverage`:

```json
    "test:ci": "vitest run --no-coverage --reporter=default"
```

Keep `test:run` and `test:coverage` — they're used by local devs and by the main-branch coverage upload.

- [ ] **Step 3: Update CI to use `test:ci` for PR runs**

In `.github/workflows/ci.yml`, find the frontend test step. Change from `npm run test:coverage` (if that's what it uses) to a conditional:

```yaml
      - name: Run frontend tests (with coverage on main, no coverage on PRs)
        working-directory: frontend
        run: |
          if [ "${{ github.event_name }}" = "pull_request" ]; then
            npm run test:ci
          else
            npm run test:coverage
          fi
```

- [ ] **Step 4: Verify locally**

```bash
cd frontend && time npm run test:run 2>&1 | tail -5
time npm run test:ci 2>&1 | tail -5
```

Expected: `test:ci` is 15-25% faster than `test:run` (or similar, depending on whether `test:run` already skips coverage).

- [ ] **Step 5: Commit**

```bash
cd /home/bernt-popp/development/phentrieve
git add frontend/vite.config.js frontend/package.json .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(frontend): Vitest threads pool + skip coverage on PR CI

Two changes:

1. Add `pool: 'threads'` to vite.config.js test section. The default
   forks pool spawns child processes per test file; threads uses
   Worker threads with lower per-file overhead. Expected gain: 10-30%
   on larger suites.

2. Add a new `test:ci` npm script that runs vitest with
   --no-coverage. Use it for pull_request CI runs; keep
   test:coverage for main branch merges (Codecov upload still
   happens).

Coverage instrumentation adds 15-25% overhead. PR runs don't need
the coverage data — they just need a pass/fail signal. Main branch
merges still upload coverage to Codecov.

Expected gain: 107 tests ~1.2s → ~0.9s locally; more in CI.

Sources:
- https://vitest.dev/guide/improving-performance
- https://vitest.dev/config/pool

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Verification gate after Tier 1 (Task 11)

### Task 11: Run full local + CI verification

**Why**: All 10 changes should not regress any check. Run the full quality gate before committing Tier 2.

- [ ] **Step 1: Full local verification**

```bash
make check typecheck-fast test
cd frontend && npm run lint && npm run format:check && npm run test:ci && npm run build
```

Expected: all green.

- [ ] **Step 2: Compare to baseline timings**

```bash
time make check
time make typecheck-fast
time make test
cd frontend && time npm run lint && time npm run format:check && time npm run test:ci && time npm run build
```

Compare each to the baseline captured in Task 0. Expected: all significantly faster.

- [ ] **Step 3: Push and verify CI**

```bash
git push origin improve/code-quality-2026-04
sleep 60
gh pr view 191 --json statusCheckRollup 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for c in data.get('statusCheckRollup', []):
    s = c.get('conclusion') or c.get('status', '?')
    print(f'  {s:15} {c.get(\"name\")}')
"
```

Wait 5-10 minutes and re-check. Expected: 17/17 green.

- [ ] **Step 4: Compare CI run durations**

```bash
gh run list --workflow="CI - Continuous Integration" --limit 3 --json databaseId,displayTitle,createdAt,updatedAt,conclusion | python3 -c "
import json, sys
from datetime import datetime
runs = json.load(sys.stdin)
for r in runs:
    start = datetime.fromisoformat(r['createdAt'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(r['updatedAt'].replace('Z', '+00:00'))
    dur = (end - start).total_seconds() / 60
    print(f\"{r['conclusion']:10} {dur:.1f} min  {r['displayTitle'][:60]}\")
"
```

Compare the new run duration to the pre-Tier-1 baseline. Target: ~40-60% reduction in total CI wall-clock.

---

## Tier 2 — Medium-effort wins (7 tasks, MEDIUM effort, LOW-MEDIUM risk)

These are opt-in — they require more care but yield additional savings. Dispatch only after Tier 1 is verified green.

### Task 12: Frontend Docker registry cache (mirror API strategy)

**Why**: The API Docker image already uses `cache-from: type=gha, mode=max`; the frontend image does not. Adding the same pattern to the frontend image eliminates repeated `npm ci` on every Docker frontend build.

**Files**:
- Modify: `.github/workflows/docker-publish.yml` (frontend build step)

- [ ] **Step 1: Read the current docker-publish.yml**

```bash
grep -B 2 -A 15 "Build.*frontend\|frontend.*build\|docker/build-push-action" .github/workflows/docker-publish.yml
```

- [ ] **Step 2: Mirror the API caching strategy**

Find the frontend build step and add:

```yaml
          cache-from: type=gha,scope=frontend
          cache-to: type=gha,mode=max,scope=frontend
```

Use `scope=frontend` to avoid mixing cache with the API scope.

The API build should already have `scope=api` or similar. Add if missing.

- [ ] **Step 3: Ensure `docker/setup-buildx-action@v4` is used**

GitHub Cache API v1 was sunset April 2025. Verify the workflow uses `docker/setup-buildx-action@v4` or later (which ships Buildx ≥0.21 / BuildKit ≥0.20).

```bash
grep "docker/setup-buildx-action" .github/workflows/docker-publish.yml
```

If it's still on `@v3` or earlier, upgrade to `@v4`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/docker-publish.yml
git commit -m "$(cat <<'EOF'
perf(docker): add GHA registry cache for frontend image build

The API Docker image build already uses cache-from/cache-to with
the GHA backend. The frontend image did not, meaning every Docker
frontend build re-ran `npm ci` from scratch, downloading ~200MB of
node_modules on every CI run.

Adds the same cache strategy with scope=frontend to keep the frontend
and API caches separate.

Also bumps docker/setup-buildx-action to @v4 if needed (the GitHub
Cache API v1 that older versions use was sunset in April 2025).

Expected gain: frontend Docker build ~3-5 min → ~30-60s on cache hit.

Source: https://docs.docker.com/build/cache/backends/gha/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Gate i18n check behind locale/template path filter

**Why**: `make frontend-i18n-check` runs on every frontend CI run but only matters when locales or Vue templates change. `dorny/paths-filter` is already in use — extend it.

**Files**:
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Locate the paths-filter job in ci.yml**

```bash
grep -B 2 -A 20 "dorny/paths-filter" .github/workflows/ci.yml
```

- [ ] **Step 2: Add an `i18n` output to the filter**

In the `changes` job (or wherever paths-filter is configured), add a new filter:

```yaml
            i18n:
              - 'frontend/src/locales/**'
              - 'frontend/src/**/*.vue'
```

And expose it as an output of the job.

- [ ] **Step 3: Gate the i18n check step on that output**

In the frontend-ci job, find the i18n check step and add a conditional:

```yaml
      - name: Run i18n validation
        if: needs.changes.outputs.i18n == 'true'
        working-directory: frontend
        run: npm run i18n:check
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(ci): gate i18n check behind locale/template path filter

make frontend-i18n-check runs vue-i18n-extract which scans all
locale JSON files and all .vue templates. It only produces new
information when a locale or template changes.

Uses the existing dorny/paths-filter infrastructure to skip the
i18n step when only backend or unrelated frontend files change.

Expected gain: i18n step skipped on ~80% of PRs (most commits don't
touch locales or templates).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: ESLint `--concurrency=auto` (requires ESLint ≥9.34)

**Why**: ESLint 9.34+ supports multithread linting. ~1.3× baseline speedup on multicore runners.

**Files**:
- Modify: `frontend/package.json` (lint script)

**Precondition**: verify ESLint version is ≥9.34.0.

- [ ] **Step 1: Check ESLint version**

```bash
grep '"eslint"' frontend/package.json
```

If `<9.34.0`, stop and upgrade first (or skip this task).

- [ ] **Step 2: Add `--concurrency=auto` to the lint script**

```json
"lint": "eslint . --cache --cache-strategy content --cache-location .eslintcache --concurrency=auto"
```

**Gotcha**: ESLint multithread requires that the flat config only contain serializable values (no function-based rules passed inline). phentrieve's flat config should already be compatible — verify by running the command.

- [ ] **Step 3: Verify locally**

```bash
cd frontend && rm -f .eslintcache && time npm run lint 2>&1 | tail -5
```

Expected: faster than baseline, especially on large initial runs.

If it fails with a serialization error, revert the flag and skip this task.

- [ ] **Step 4: Commit**

```bash
cd /home/bernt-popp/development/phentrieve
git add frontend/package.json
git commit -m "$(cat <<'EOF'
perf(frontend): enable ESLint --concurrency=auto

ESLint 9.34+ supports multithread linting across CPU cores. Adds
--concurrency=auto to the lint script.

Expected gain: ~1.3x on multicore runners for cold runs; cached runs
remain instant because they hit the content cache first.

Precondition: ESLint ≥9.34.0 (verified in package.json).

Source: https://eslint.org/blog/2025/08/multithread-linting/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Add `timeout-minutes` to every job

**Why**: Default GitHub Actions timeout is 360 minutes. A hung process (e.g. model download that stalls) consumes full runner credit. Per-job `timeout-minutes` caps the damage at 2× normal run time.

**Files**:
- Modify: all workflow files in `.github/workflows/`

- [ ] **Step 1: Add `timeout-minutes` to each job**

Suggested values:
- `python-ci`: `timeout-minutes: 20`
- `frontend-ci`: `timeout-minutes: 15`
- `docker-build-test` (API): `timeout-minutes: 60` (ML deps legitimately take this long)
- `docker-build-test` (frontend): `timeout-minutes: 15`
- `changes` / `ci-summary`: `timeout-minutes: 5`

Example:
```yaml
  python-ci:
    needs: changes
    if: needs.changes.outputs.python == 'true'
    runs-on: ubuntu-latest
    timeout-minutes: 20  # ← add this line
    strategy:
      # ...
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/
git commit -m "$(cat <<'EOF'
perf(ci): cap per-job runtime with timeout-minutes

Default GitHub Actions timeout is 360 minutes per job. A hung process
(e.g. a stalled model download) can consume full runner credit before
the job is killed.

Adds timeout-minutes to every job, sized to approximately 2x normal
completion time:
- python-ci: 20 min
- frontend-ci: 15 min
- docker-build-test (api): 60 min (ML deps legitimately take 20-40 min)
- docker-build-test (frontend): 15 min
- changes/summary: 5 min

Caps runaway cost without affecting legitimate runs.

Source: https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/setting-a-timeout-for-a-workflow-or-job

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: `matrix: fail-fast: false` on Python version matrix

**Why**: By default, a failure in any matrix cell cancels sibling cells. This hides cross-version failures (a bug that only affects 3.10 is masked if 3.11 completes first and fails too, and xyz). `fail-fast: false` lets all matrix cells run to completion regardless.

**Files**:
- Modify: `.github/workflows/ci.yml` (python-ci matrix)

- [ ] **Step 1: Add `fail-fast: false` to the Python matrix**

```yaml
  python-ci:
    strategy:
      fail-fast: false   # ← add this line
      matrix:
        python-version: ['3.10', '3.11', '3.12']
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(ci): set fail-fast: false on Python matrix

Default matrix behavior cancels sibling cells on first failure. This
hides cross-version regressions where a bug affects, say, 3.10 and
3.12 but not 3.11 — the devs only see the first failure and miss
the second.

fail-fast: false lets all three Python versions run to completion
regardless. Slightly more runner minutes on failed runs, but much
more diagnostic information.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Fix Makefile ↔ CI parity drifts

**Why**: Several `make` commands don't match what CI actually runs. Fixes reduce "works on my machine" surprises.

**Files**:
- Modify: `Makefile`

Drifts to fix (from the audit):

1. **`make frontend-build` doesn't set `VITE_API_URL`** — CI hardcodes `/api/v1`; local build uses whatever Vite picks. Add `VITE_API_URL=/api/v1` as an explicit env export.

2. **Local `make test` runs with all markers; CI uses `-m "not e2e"`** — add a `test-ci` target that matches CI exactly.

3. **Local `make format` writes; CI runs `--check`** — add an explicit `format-check` target that mirrors CI.

- [ ] **Step 1: Read the current Makefile**

```bash
grep -n "^frontend-build:\|^test:\|^format:" Makefile
```

- [ ] **Step 2: Add explicit parity targets**

Append to `Makefile`:

```makefile
format-check: ## Check Python formatting (CI mode, read-only)
	uv run ruff format --check phentrieve/ api/ tests/

test-ci: ## Run Python tests exactly as CI does
	uv run pytest tests/ -v -m "not e2e" --cov=phentrieve --cov=api --cov-report=xml --cov-report=term

frontend-build-ci: ## Build frontend exactly as CI does
	cd frontend && VITE_API_URL=/api/v1 CI=true npm run build
```

Also update `frontend-build` to NOT export `CI=true` — keep it as the dev-mode full build with compression.

- [ ] **Step 3: Add a `ci-local` meta-target**

```makefile
ci-local: format-check lint typecheck test-ci frontend-lint frontend-format-check frontend-test-ci frontend-build-ci ## Run every check CI runs, locally
```

Where `frontend-format-check` is `cd frontend && npm run format:check` and `frontend-test-ci` is `cd frontend && npm run test:ci` (the script added in Task 10).

- [ ] **Step 4: Verify `make ci-local` works**

```bash
make ci-local 2>&1 | tail -10
```

Expected: all steps pass.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "$(cat <<'EOF'
chore: add Makefile targets matching CI exactly

Three drifts between local `make` commands and CI were causing
"works on my machine" failures:

1. `make frontend-build` didn't set VITE_API_URL; CI hardcodes
   /api/v1. Fixed via a new frontend-build-ci target.
2. `make test` ran all markers; CI runs -m "not e2e". Fixed via
   a new test-ci target that matches the CI pytest invocation.
3. `make format` writes; CI runs --check. Fixed via a new
   format-check target.

Also adds a `make ci-local` meta-target that runs everything CI
runs, in the same order, so a dev can reproduce CI on the first
try.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Cache `HF_HOME` for SBERT/spaCy model downloads

**Why**: phentrieve loads BioLORD (~500MB) and spaCy `en_core_web_sm` in integration tests. These downloads happen fresh on every CI run.

**Files**:
- Modify: `.github/workflows/ci.yml`

**Caveat**: GitHub-hosted runners have a 10 GB cache limit per repository. BioLORD at ~500MB is safe; combined with other caches it's still well under 10 GB.

- [ ] **Step 1: Add `HF_HOME` env and cache step to CI**

In `.github/workflows/ci.yml`, at the job level of `python-ci`, add an env block:

```yaml
    env:
      HF_HOME: ${{ github.workspace }}/.cache/huggingface
```

Then add a cache step before the test invocation:

```yaml
      - name: Cache HuggingFace models
        uses: actions/cache@v4
        with:
          path: ${{ github.workspace }}/.cache/huggingface
          key: hf-${{ hashFiles('phentrieve/config.py') }}-BioLORD
          restore-keys: |
            hf-
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
perf(ci): cache HF_HOME for SBERT + spaCy model downloads

phentrieve tests that load the BioLORD SBERT model (~500MB) or
spaCy models previously re-downloaded on every CI run. Caching
HF_HOME keyed on phentrieve/config.py hash + model name lets
subsequent runs skip the download entirely.

Key invalidation: any change to phentrieve/config.py (where default
models are declared) forces a fresh download. restore-keys provide
partial fallback across minor config changes.

Expected gain: 2-5 minutes saved per CI run on cache hit.

Source: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification Gate (Task 19)

### Task 19: Push all Tier 1 + Tier 2 changes and verify

- [ ] **Step 1: Full local verification**

```bash
make ci-local 2>&1 | tail -10
```

Expected: all green.

- [ ] **Step 2: Push**

```bash
git push origin improve/code-quality-2026-04
```

- [ ] **Step 3: Wait for CI and capture final timings**

```bash
sleep 300
gh run list --workflow="CI - Continuous Integration" --limit 3 --json databaseId,displayTitle,createdAt,updatedAt,conclusion
```

Compare to baseline. Target:
- Full CI wall-clock: ~15-20 min → ~5-8 min
- Python CI per version: ~8-15 min → ~3-6 min
- Frontend CI: ~5-8 min → ~1-3 min
- Docker frontend build on cache hit: ~3-5 min → ~30-60s

- [ ] **Step 4: Update the review dashboard**

Add a new "CI speedup phase" row or section to `plan/05-analysis/CODE-QUALITY-REVIEW-2026-04-09.md` documenting the improvements. Commit.

---

## Expected outcome

After Tier 1 + Tier 2 (18 tasks, ~15-18 commits):

| Metric | Before | After Tier 1 | After Tier 2 |
|---|---|---|---|
| Python CI per version | 8-15 min | 3-6 min | 3-5 min |
| Frontend CI | 5-8 min | 1-3 min | <1 min |
| Docker frontend (cache hit) | 3-5 min | 3-5 min | 30-60s |
| Full CI wall-clock | 15-20 min | 5-8 min | 4-6 min |
| Local `make test` | 28s | 8-12s | 8-12s |
| Local `make frontend-build` | 30-45s | 10-20s | 10-20s |
| Local `make ci-local` | N/A | N/A | ~90s (matches CI) |

**Review ROI**: ~7-14 min saved per CI run × every PR × every force-push. Over a typical 10-PR week with ~4 pushes each, this saves 2-4 hours of runner minutes and compresses developer feedback loops.

## Execution handoff

**Recommended approach**: Use `superpowers:subagent-driven-development` (same as Phase 2) — each task is small, atomic, and independently verifiable. Dispatch one implementer subagent per task with the exact steps above, run two-stage review (spec + code quality) between tasks, and proceed through Tier 1 first.

**Alternative**: Batch-execute locally (skill `superpowers:executing-plans`). Faster but less review-rich.

Task 0 (baseline capture) is done by the controller before dispatching the first implementer.

## Out of scope — revisit if needed later

- **`pytest-split` test sharding** across CI jobs — revisit when suite exceeds 1500 tests
- **npm → pnpm migration** — medium migration effort, ~34s saved per job; worth it when other tasks are done
- **`ty` (Astral type checker)** — in beta, no plugin system; wait for 1.0
- **Turborepo remote cache** — only 2 packages, marginal gain
- **Vitest browser mode** — `happy-dom` is sufficient for current test needs
- **`pyright` as mypy supplement or replacement** — different error set needs dedicated tuning pass
