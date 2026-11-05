# Acceptance checklist: issue batch of 2026-10-27 (PRs #437-#444)

Status: partly done. Results of the UI acceptance on 2026-10-30 are under
"Results 2026-10-30". Sections 1-5 and 7 were not repeated; the release PR #457
records what ran for 0.30.0. Tick items as you go; record results inline.

## Why this exists

PRs #437-#444 fix issues from the v0.28.4 code review and the `develop` review:
#394-#398, #401, #402, #404, #405, #407, #408, #410, #411, #414-#420, #426-#433.

All eight are green in CI after merging `develop` (0.28.5-0.28.7), and pass
`make check`, `make typecheck-fast` and `make test` locally. Four kinds of check
never ran on them:

- E2E tests (`tests/e2e/`). PR CI runs `-m "not slow and not e2e"`.
- The API image. The CI image job builds only when `api/Dockerfile`,
  `pyproject.toml`, `uv.lock` or compose files change, and none of these PRs
  touches them.
- `slow` tests.
- A real LLM. All LLM tests use `FakeProvider`.

The sandbox could not close these gaps: HuggingFace, apt, sqlite.org and ghcr.io
downloads were blocked there.

## Setup

One branch carries all eight PRs merged on top of `develop`. It exists for local
testing only; do not merge it.

```bash
git fetch origin
git checkout test/integration-2026-10-27   # develop + #437..#444
make install-dev && make frontend-install
```

Delete any stale setuptools `build/` directory before testing (`rm -rf build`).
#414 deletes modules, and a stale `build/lib` keeps them in locally built wheels. That makes
`test_runtime_source_hash_matches_built_wheel` fail with a hash mismatch.

To test a single PR, check out its branch instead:

| PR | Branch |
|---|---|
| #437 | `fix/assertion-export-auth` |
| #438 | `fix/llm-evidence-offsets` |
| #439 | `fix/api-query-path` |
| #440 | `fix/api-text-path` |
| #441 | `fix/llm-pipeline-robustness` |
| #442 | `fix/frontend-batch` |
| #443 | `fix/cleanup-and-perf` |
| #444 | `fix/retrieval-integrity` |

You need:

- the real data bundle (`phentrieve data download` or the existing `data/`);
- Docker;
- for section 4, an LLM provider key in `.env` (`PHENTRIEVE_LLM_PROVIDER`,
  `PHENTRIEVE_LLM_MODEL` and the provider's `*_API_KEY`).

Baseline for comparisons: `develop` at `284ce3e`, which has none of the eight PRs.

## 1. Fast gate (about 5 min)

- [ ] `make check && make typecheck-fast && make test` on the integration branch.
  Expected: no failures. Some tests needed HuggingFace downloads in the sandbox
  and should pass here.
- [ ] `make frontend-lint && make frontend-test-ci && make frontend-build-ci`
- [ ] `uv run pytest tests/ -m slow -q`. The slow tests never ran on these PRs.

## 2. Docker image and E2E (about 20 min)

Covers: #443 (health route), #439 (lifespan warm-up, `/query` off the event
loop), #444 (Chroma batching), and image regressions in general.

- [ ] `make test-e2e` (builds the images, runs `tests/e2e/`: security, health,
  API). Expected: all green.
- [ ] Health route (#433 in #443). With `make docker-dev`, the API is on host port
  8001 (`API_PORT_HOST`):
  - [ ] `curl -s localhost:8001/api/v1/health` returns 200 directly, with no
    307 redirect, and body `{"status":"healthy","service":...,"version":...,"timestamp":...}`.
  - [ ] `curl -s localhost:8001/api/v1/health/` and
    `curl -s localhost:8001/api/v1/system/health` still return 200 with the same
    body. **The `/health/` body changed from `{"status":"ok"}`. Check external
    monitors that match `"ok"`.**
  - [ ] `docker ps` shows the API container as `healthy`.
- [ ] Startup warm-up (#419 in #439). In `make docker-logs`, the startup logs
  show the warm-up query and no traceback. The first `/api/v1/query` after a
  restart should be about as fast as later ones.
- [ ] `phentrieve --version` inside the API container still works:
  `docker compose run --rm --entrypoint phentrieve phentrieve_api --version`.

## 3. API behaviour against the real index (about 15 min)

Start the API with `make dev-api` (port 8734) or use `make docker-dev` (port 8001).
Set `API=http://localhost:8734/api/v1`.

- [ ] Long note no longer returns zero terms (#402 in #444). Send a note with
  more than 240 chunks. The issue used the same paragraph repeated to 5,190+
  characters.

  ```bash
  python - <<'EOF' > /tmp/long.json
  import json
  p = ("The patient has recurrent seizures, global developmental delay and "
       "hypotonia. No hearing loss. ")
  print(json.dumps({"text": p * 60, "language": "en"}))
  EOF
  curl -s -X POST $API/text/process -H 'content-type: application/json' \
    -d @/tmp/long.json | jq '.aggregated_hpo_terms | length'
  ```

  Expected: well above 0. On `develop`, the server log showed `too many SQL
  variables` and the response had 0 terms. Also check the server log has no
  `Error in batch query`.
- [ ] `/query` no longer blocks the server (#405 in #439). Run 16 parallel
  queries while timing `/health`:

  ```bash
  for i in $(seq 16); do curl -s "$API/query/?text=seizures%20and%20ataxia" >/dev/null & done
  time curl -s $API/health >/dev/null; wait
  ```

  Expected: `/health` answers in milliseconds. On `develop` it took about 1.5 s.
- [ ] GET `/query` validation (#408 in #439):
  - `curl -s -o /dev/null -w '%{http_code}' "$API/query/?text=x&num_results=100"` returns `422`.
  - `curl -s -o /dev/null -w '%{http_code}' "$API/query/?text=x&query_assertion_preference=bogus"` returns `422`.
  - The server log shows no `coroutine ... was never awaited`.
- [ ] Similarity (#415, #417 in #444). Both surfaces return the same values for
  the same pair, for example `HP:0001250` vs `HP:0002069`. REST:
  `curl -s "$API/similarity/HP:0001250/HP:0002069"`. MCP: the
  `phentrieve_compare_hpo_terms` tool. Then rename `data/hpo_data.db`, restart,
  call once (expect 503), restore the file, and call again: expect 200 without
  a restart.
- [ ] Bundle verification (#398 in #444). Checksums are verified when a bundle
  is installed (`phentrieve data download`) and at the start of every
  `phentrieve benchmark llm` run (`run_identity`). Expected: both pass on the
  installed bundle; the LLM benchmark in section 4 covers the second. **Note:**
  `content_sha256` changes once, so benchmark checkpoints from before this
  change will not resume.
- [ ] Phenopacket export of negations (#407 in #437). Use the MCP export tool,
  or CLI/library export, on terms with assertion `denied`, `no` or `negative`,
  or with `excluded: true`. Expected: `excluded: true` in the packet.
- [ ] MCP budget (#396 in #437). Call the MCP extract tool in `minimal` mode on a
  note with many family or other-person findings. Expected: the patient's
  `aggregated_hpo_terms` are still present, and `_meta` reports what was
  truncated.

## 4. LLM path with a real provider (about 30 min plus provider cost)

Covers #438 (evidence offsets, prompt change), #441 (truncation retry,
`incomplete` flag) and #440 (fallback, retriever reuse).

- [ ] **Benchmark before/after.** Note: `PROMPT_BEHAVIOR_VERSION` goes 4 -> 5 in
  #438, so old checkpoints will not match.

  ```bash
  # baseline on develop
  git checkout develop
  phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews \
    --llm-model "$PHENTRIEVE_LLM_MODEL" --run-id accept-base
  # candidate
  git checkout test/integration-2026-10-27
  phentrieve benchmark llm --test-file tests/data/en/phenobert --dataset GeneReviews \
    --llm-model "$PHENTRIEVE_LLM_MODEL" --run-id accept-cand
  ```

  Expected: no drop in present-term F1 or assertion accuracy beyond noise.
  Other-person findings now appear in the assertion distribution (#401).
- [ ] **Highlights in the UI** (#394, #397, #411 in #438). In the frontend
  (`make dev-frontend`, port 5734), run LLM full-text extraction on:
  - `Intro.\n\n   Her   roommate  has   hypothyroidism.`: the highlight
    covers exactly "hypothyroidism" (#394).
  - `İnfant with seizures`: the highlight covers "seizures", not "eizures" (#411).
  - A paraphrased finding: no whole-chunk highlight. Either an exact quote is
    highlighted, or the UI falls back to the label (#397).
  - Text pasted from a PDF with `Hämo-\nglobin` or soft hyphens: the evidence is
    found (#397).
- [ ] **Truncation retry and `incomplete` flag** (#395 in #441). Send a dense
  note with a long feature list (about 40 findings in one chunk) via the LLM
  backend. Expected: complete results. If a group still fails, the REST `meta`
  shows `incomplete: true` and `incomplete_reason`. Check latency and token cost
  of the 65,536-token retry in the provider dashboard. The frontend does not
  show the flag yet; that is a follow-up.
- [ ] **REST fallback** (#418 in #440). Point the LLM at a bad key or an
  unreachable `PHENTRIEVE_LLM_BASE_URL`, then POST `/text/process` with
  `extraction_backend: "llm"` and `allow_standard_fallback: true`. Expected:
  200, `meta.fallback_reason == "llm_backend_error"`, and LLM quota unchanged.
  Without the flag: 500 as before.
- [ ] **Model choice** (#420 in #440). An LLM request with a non-default
  `retrieval_model_name` now uses that model for tool retrieval, not always
  `DEFAULT_MODEL`. Confirm this is intended.

## 5. Text processing performance and identity (#410 in #443, about 15 min)

Batching must not change results.

- [ ] Run the standard extraction benchmark on `develop` and on the candidate:

  ```bash
  phentrieve benchmark extraction run tests/data/en/phenobert/ --dataset GeneReviews --run-id perf-base   # on develop
  phentrieve benchmark extraction run tests/data/en/phenobert/ --dataset GeneReviews --run-id perf-cand   # on candidate
  phentrieve benchmark extraction compare <base-result> <cand-result>
  ```

  Expected: identical metrics. Floating-point noise in the last digit is
  acceptable. Wall time should be noticeably lower; the issue measured about 78%
  of request CPU in encode and spaCy calls.
- [ ] Spot-check one German note (`-l de`) for identical output between the two
  branches.

## 6. Frontend walkthrough (#404, #426, #430, #431 in #442; about 10 min)

`make dev-api` + `make dev-frontend`, then open http://localhost:5734.

- [ ] **Clinical, #404:** run full-text (standard backend) on
  `Normal intellect. Recurrent seizures.` and use "add all" to the collection.
  Expected: "intellect" or the normal finding shows as **negated/excluded**,
  seizures as affirmed. In the exported Phenopacket the normal finding has
  `excluded: true`.
- [ ] **QueryInterface after the `<script setup>` migration, #430.** Check:
  - a simple query;
  - URL parameters with auto-submit, e.g. `/?text=seizures&autoSubmit=true`;
  - switching to full text;
  - the advanced options panel;
  - the PII review flow if enabled;
  - conversation history and scrolling.
- [ ] **Advanced options default, #426:** the similarity threshold starts at 0.3.
- [ ] **Health poll, #431:** with DevTools Network open, switching to another tab
  stops `/health` polling, and switching back resumes it.

## 7. CLI (#414, #429 in #443)

- [ ] `phentrieve text process "Patient has seizures." --xlhpo` prints a
  deprecation warning and still runs. The same applies to `--use-cached-index`
  and `--chunk-confidence`.
- [ ] `phentrieve text process` with a semantic chunking strategy still loads
  the semantic model; with a simple strategy it does not (#429).

## Results 2026-10-30

Tested in a real browser on v0.30.0 plus the #458 fix, with the local dev API and
frontend. The LLM ran through OpenRouter with `google/gemini-3.1-flash-lite`, set by a
throwaway launcher outside the repo. LLM results are single samples.

Section 6 and highlights:

| Check | Result |
|---|---|
| #458 "Auto" in full text, German note | Pass: detected as `de`, the seizure shows as Negated |
| #404 normal finding to Phenopacket | Pass with lowered thresholds: Negated in the list and collection, `excluded: true` in the file. With the UI defaults (0.7 / 0.75) the checklist text produces no normal term at all |
| #430 auto-submit, mode switch, advanced panel, PII dialog, history | Pass |
| #426 threshold default | Pass: 0.30 |
| #431 health poll | Pass: no call during 34 s hidden, polling resumes at the next tick |
| #394 highlight with extra whitespace | API pass: offsets 13-27 are "hypothyroidism". UI not testable: the finding is in `other_person_findings`, so the UI shows 0 findings and no highlight |
| #411 `İnfant with seizures` | Pass: highlight is exactly "seizures" |
| #397 paraphrase | Pass: three exact spans, no whole-chunk highlight |
| #397 PDF text | Fail: `Hämo-\nglobin` finds the term but has no highlight, and `sei­zures` was not extracted |

Regression of the 2026-10-27 UI review (`ui-ux-review/C-test-protocol.md`):

- Pass: N05, L05, S05, D05, D06, C09, I01, T01, K03, A01, M01, NW06 (cancel is available).
  C07 also looks fixed: one click does not clear the collection.
- Still failing:
  - N04 (P2): "seizures were ruled out" is reported as Affirmed, with an absence seizure as
    top term. This is a clinical error.
  - C10 (P2): the drawer's scrim blocks adding a second term.
  - D07 (P3): reloading `/?text=...` submits the query again.
  - K04 (P3): Esc does not close the reset dialog.
  - K06 (P3): Esc does not close the tutorial.
- Not repeated: D03, K01, S06.

New findings:

- The REST LLM path returns 500 when `PHENTRIEVE_LLM_BASE_URL` is set: `run_llm_backend`
  passes the environment value as an explicit argument, and the resolver rejects it for
  the fixed `gemini` target.
- `other_person_findings`, `temporality` and `family_history_findings` are not shown in
  the frontend. A note that only names findings of other people reports "0 findings".
- `Normal ...` chunks and German standard-backend results can show findings without any
  highlight. Confirmed for normal findings: `text_attributions` is empty.
- "Low Hämo-\nglobin ..." triggers the PII dialog with Name (1), probably a false positive.
- "Recurrent seizures." ranks Seizure cluster and Status epilepticus first in the standard
  backend, not Seizure.
- `FullTextResponseReceipt.vue` hardcodes "findings across ... document sections" in
  English, which also shows in the German UI.
- In `make test` on Windows with a local `data/`, four tests in
  `tests/unit/cli/test_similarity_commands.py` fail because they assume no `hpo_data.db`
  exists. `make frontend-i18n-check` fails on Windows with "vueFiles glob has no files".

## Follow-up fixes 2026-10-30

Branch `fix/acceptance-followups-2026-10-30`. Issues #459-#472 record all findings above.

Fixed and checked in the browser, in English and German: #458, #459, #460, #461, #462,
#467, #471, plus the `incomplete` warning and `/health/ready`. In the query path and
standard full text, "ruled out" reads as negated and "cannot be ruled out" as open.

Benchmarks, `main` against the branch, GeneReviews (10 documents):

| Run | Micro F1 main | Micro F1 branch | Note |
|---|---|---|---|
| Standard backend, BioLORD | 0.2463 | 0.2463 | Identical metrics |
| LLM, Gemini 3.1 Flash Lite via OpenRouter | 0.8266 | 0.8266 | 274 terms each, one differs per side |

These runs use no fixed seed, and GeneReviews has few of the affected phrases, so they
show no regression but do not measure the gain. The full CSC/GSC run is still open.

Still open: #463, #464, #465, #466, #468, #469, #470, #472. The E2E suite was not run.

## Sign-off

| PR | Result | Notes |
|---|---|---|
| #437 | | |
| #438 | | |
| #439 | | |
| #440 | | |
| #441 | | |
| #442 | | |
| #443 | | |
| #444 | | |

Open follow-ups, not covered by these PRs:

- Frontend display of `incomplete` (#441).
- `docker-compose.yml` does not pass `PHENTRIEVE_MAX_CONCURRENT_INFERENCE` to the
  container.
- #406 follow-up: a worker thread keeps running after a `wait_for` timeout
  (from #440; #406 was closed on `develop`).
- After #438 and #441 merge: close #401, and remove the last three dead helpers in
  `phentrieve/llm/pipeline.py` (#414).
- #399, #400 (export of uncertain and temporal findings; standard-pipeline
  negation lists) remain open and unassigned.
