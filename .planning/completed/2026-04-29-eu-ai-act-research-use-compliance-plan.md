# EU AI Act Research-Use Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align Phentrieve with a defensible EU research-only posture for open, free public availability by removing contradictory clinical-use claims, adding user-facing and API-level guardrails, documenting data flows, and creating a maintainable classification record.

**Architecture:** Treat compliance posture as a product surface, not only legal copy. The frontend, API, CLI, MCP, docs, deployment guidance, and tests must all express the same intended purpose: research, benchmarking, education, and research data standardisation only. Hosted-service operation gets additional gates and privacy/provider controls because a public service can be "put into service" even if it is free.

**Tech Stack:** Python 3.10+, FastAPI, Typer CLI, Vue 3/Vuetify/Pinia/Vue I18n, MkDocs, pytest, Vitest, Ruff, mypy.

**Legal Baseline Used:** Regulation (EU) 2024/1689 Article 2(6), Article 2(7), Article 2(8), Article 6, Article 50, Article 53-55, Commission AI Act timeline and FAQ. This plan improves engineering evidence and documentation; counsel must validate the final public service terms before launch.

---

## Non-Negotiable Compliance Position

This plan cannot honestly produce "100% legal compliance" by code changes alone. It can produce a consistent, auditable, research-only posture that is ready for legal review.

The target public statement is:

> Phentrieve is open-source research software for mapping clinical or biomedical research text to Human Phenotype Ontology terms for exploration, benchmarking, education, and research data standardisation. Phentrieve is not a medical device, is not CE marked, and is not intended to diagnose, treat, prevent, monitor, predict, or manage disease, nor to triage patients or make or support patient-care decisions. Outputs are algorithmic suggestions that may be wrong or incomplete and must be independently reviewed before any downstream use. Public demo instances must not be used with identifiable patient data.

## File Structure

Create:

- `docs/compliance/index.md` - Compliance landing page linked from docs nav.
- `docs/compliance/research-use.md` - Canonical intended-purpose and prohibited-use statement.
- `docs/compliance/eu-ai-act-classification.md` - Public classification rationale and operator checklist.
- `docs/compliance/privacy-and-data-processing.md` - Plain-language privacy/provider data-flow notice.
- `docs/compliance/model-and-provider-inventory.md` - Inventory of local embedding models and optional LLM providers.
- `tests/unit/api/test_research_use_guard.py` - API guard tests for hosted research acknowledgement.
- `tests/unit/cli/test_research_use_notice.py` - CLI notice tests.
- `frontend/src/test/complianceCopy.test.js` - Frontend copy regression tests.
- `frontend/src/constants/compliance.js` - Shared frontend compliance text keys if needed.

Modify:

- `README.md` - Top-level research-only banner, remove production/diagnostic ambiguity.
- `docs/index.md` - Replace diagnosis framing with research-domain framing.
- `mkdocs.yml` - Add compliance docs to navigation.
- `docs/DOCKER-DEPLOYMENT.md` - Rename posture to self-hosted research deployment and add clinical-use warning.
- `docs/deployment/index.md`, `docs/deployment/security.md`, `docs/user-guide/api-usage.md`, `docs/user-guide/frontend-usage.md`, `docs/user-guide/cli-usage.md`, `docs/mcp-server.md` - Add intended-use, privacy, and provider warnings where relevant.
- `api/main.py`, `api/routers/query_router.py`, `api/routers/text_processing_router.py`, `api/routers/phenopacket_router.py`, `api/mcp/server.py` - Add OpenAPI/tool descriptions and optional hosted acknowledgement enforcement.
- `api/config.py`, `api/api.yaml`, `api/api.yaml.template` - Add public-hosted compliance flags.
- `phentrieve/cli/query_commands.py`, `phentrieve/cli/text_commands.py`, `phentrieve/cli/mcp_commands.py`, `phentrieve/cli/benchmark_commands.py` - Add CLI warnings and output behavior.
- `frontend/src/locales/en.json`, `frontend/src/locales/de.json`, `frontend/src/locales/fr.json`, `frontend/src/locales/es.json`, `frontend/src/locales/nl.json` - Align user-facing copy.
- `frontend/src/config/faqConfig.json` - Align or delete stale FAQ source if unused.
- `frontend/src/components/DisclaimerDialog.vue`, `frontend/src/components/QueryInterface.vue`, `frontend/src/components/PhenotypeCollectionPanel.vue`, `frontend/src/components/FullTextAnnotationWorkspace.vue` - Add visible guardrails near data entry/export.
- `frontend/src/stores/disclaimer.js` - Version disclaimer acknowledgement so changed terms are shown again.
- `frontend/src/services/PhentrieveService.js` - Send hosted research-use acknowledgement header when user accepted the current disclaimer.
- `SECURITY.md` - Strengthen PHI/PII statement and link privacy notice.
- `.planning/analysis/2026-04-29-eu-ai-act-research-use-review.md` - Add completion links after implementation.

## Compliance Configuration

Add these API settings:

- `PHENTRIEVE_PUBLIC_HOSTED_MODE=false` by default.
- `PHENTRIEVE_REQUIRE_RESEARCH_ACK=false` by default for local/self-hosted installs.
- `PHENTRIEVE_ENABLE_EXTERNAL_LLM=false` by default for public hosted mode.
- `PHENTRIEVE_ALLOW_IDENTIFIABLE_PATIENT_DATA=false` always for public hosted mode.

Behavior:

- Local/self-hosted research users can run without a blocking header, but docs and OpenAPI still show the intended-use limitation.
- Public hosted mode requires `X-Phentrieve-Research-Use-Acknowledged: true` for text-bearing API endpoints.
- Public hosted mode rejects LLM extraction unless external LLM use is explicitly enabled and the privacy/provider notice says so.
- Public hosted mode never stores submitted clinical text beyond the request lifecycle unless a future explicit research protocol config is added.

---

### Task 1: Canonical Compliance Documentation

**Files:**
- Create: `docs/compliance/index.md`
- Create: `docs/compliance/research-use.md`
- Create: `docs/compliance/eu-ai-act-classification.md`
- Create: `docs/compliance/privacy-and-data-processing.md`
- Create: `docs/compliance/model-and-provider-inventory.md`
- Modify: `mkdocs.yml`
- Modify: `README.md`
- Modify: `SECURITY.md`

- [ ] **Step 1: Create compliance landing page**

Add `docs/compliance/index.md`:

```markdown
# Compliance

Phentrieve is open-source research software for mapping clinical or biomedical
research text to Human Phenotype Ontology terms.

Phentrieve is not a medical device, is not CE marked, and is not intended for
diagnosis, treatment, triage, patient management, or other patient-care
decisions.

Start here:

- [Research Use](research-use.md)
- [EU AI Act Classification](eu-ai-act-classification.md)
- [Privacy and Data Processing](privacy-and-data-processing.md)
- [Model and Provider Inventory](model-and-provider-inventory.md)
```

- [ ] **Step 2: Create canonical research-use page**

Add `docs/compliance/research-use.md` with the target public statement from this plan, plus explicit prohibited uses:

```markdown
# Research Use

Phentrieve is intended for research, benchmarking, education, and research data
standardisation.

Do not use Phentrieve for:

- diagnosis, treatment, prevention, monitoring, prediction, or management of
  disease;
- patient triage, referral, eligibility, or prioritisation;
- clinical decision support;
- replacing qualified professional review;
- generating or updating clinical records for patient care;
- processing identifiable patient data in a public demo instance.

Outputs are algorithmic suggestions. They may be wrong, incomplete, or biased by
input text, model choice, ontology coverage, retrieval thresholds, and language
handling. Any downstream research use requires independent review.
```

- [ ] **Step 3: Create EU AI Act classification record**

Add `docs/compliance/eu-ai-act-classification.md`:

```markdown
# EU AI Act Classification

## Intended Purpose

Phentrieve maps research text to HPO terms for exploration, benchmarking,
education, and research data standardisation.

## Excluded Uses

Phentrieve is not intended for diagnosis, treatment, monitoring, prediction,
patient triage, patient management, clinical decision support, or operation as
software in or as a medical device.

## Current Classification Posture

For the open-source repository and research-only self-hosted deployments,
Phentrieve is positioned as research software. Public hosted deployments must
maintain the research-only controls documented here.

If an operator changes the intended purpose or integrates Phentrieve into
patient-care workflows, the operator must perform a new EU AI Act, GDPR, and
MDR/IVDR assessment before use.

## Evidence Controls

- Research-use notices in README, docs, UI, API, CLI, and MCP.
- Public hosted API acknowledgement header.
- Public hosted warning against identifiable patient data.
- External LLM providers disabled by default for public hosted mode.
- No submitted text storage by default.
- Human review warnings near output and export surfaces.

## Review

Owner: project maintainer.

Review cadence: before each public release and whenever intended use, hosted
operation, model providers, or data retention changes.
```

- [ ] **Step 4: Create privacy and data-processing notice**

Add `docs/compliance/privacy-and-data-processing.md`:

```markdown
# Privacy and Data Processing

Phentrieve processes text submitted by the user to produce HPO term suggestions.

Public demo instances must be used only with synthetic, de-identified, or
research-consented text. Do not submit names, dates of birth, addresses, record
numbers, contact details, or other identifiers.

By default, Phentrieve does not store submitted text. Server logs should contain
operational metadata only, such as request outcome, text length, model name, and
timing. Operators must not enable raw-text logging for public demos.

When local embedding models are used, text is processed within the running
Phentrieve environment.

When external LLM providers are enabled, submitted text may be sent to the
configured provider. Operators are responsible for provider terms, data
processing agreements, transfer assessments, retention settings, and user notice.
External LLM providers are disabled by default for public hosted mode.
```

- [ ] **Step 5: Create model and provider inventory**

Add `docs/compliance/model-and-provider-inventory.md` listing:

- default embedding model from `phentrieve/config.py`;
- optional benchmark embedding models;
- optional LLM providers from `phentrieve/llm/config.py`;
- whether each provider is local-only or third-party;
- what input text each provider may receive;
- data-retention responsibility.

- [ ] **Step 6: Add MkDocs navigation**

Modify `mkdocs.yml` to add:

```yaml
- Compliance:
  - Overview: compliance/index.md
  - Research Use: compliance/research-use.md
  - EU AI Act Classification: compliance/eu-ai-act-classification.md
  - Privacy and Data Processing: compliance/privacy-and-data-processing.md
  - Model and Provider Inventory: compliance/model-and-provider-inventory.md
```

- [ ] **Step 7: Update README banner**

At the top of `README.md`, directly after the logo, add:

```markdown
> **Research use only:** Phentrieve is open-source research software. It is not a
> medical device, is not CE marked, and must not be used for diagnosis,
> treatment, triage, patient management, or other patient-care decisions. Do not
> submit identifiable patient data to public demo instances.
```

Replace "production environments" with "self-hosted research deployments" in
the Docker section.

- [ ] **Step 8: Update SECURITY.md**

Replace the current "No PHI/PII stored by default" line with:

```markdown
- No PHI/PII is stored by default. Public demo instances must not be used with
  identifiable patient data. See `docs/compliance/privacy-and-data-processing.md`.
```

- [ ] **Step 9: Verify docs compile and checks pass**

Run:

```bash
make check
```

Expected: Ruff passes.

Run:

```bash
uv run mkdocs build --strict
```

Expected: MkDocs builds with no missing links.

### Task 2: Remove Diagnostic And Clinical-Use Claims

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/user-guide/api-usage.md`
- Modify: `docs/user-guide/frontend-usage.md`
- Modify: `docs/user-guide/cli-usage.md`
- Modify: `docs/DOCKER-DEPLOYMENT.md`
- Modify: `docs/deployment/index.md`
- Modify: `docs/deployment/security.md`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/de.json`
- Modify: `frontend/src/locales/fr.json`
- Modify: `frontend/src/locales/es.json`
- Modify: `frontend/src/locales/nl.json`
- Modify or delete if unused: `frontend/src/config/faqConfig.json`

- [ ] **Step 1: Replace diagnosis framing in docs index**

In `docs/index.md`, replace:

```markdown
In clinical genomics and rare disease diagnosis, identifying phenotypic abnormalities in patient descriptions is a critical step.
```

with:

```markdown
In clinical genomics and rare disease research, standardising phenotypic
descriptions is important for dataset curation, literature review, benchmarking,
and exploratory analysis.
```

- [ ] **Step 2: Replace frontend English diagnostic copy**

In `frontend/src/locales/en.json`, replace:

- "streamlining phenotype annotation for research and diagnostics"
- "both research and clinical diagnostics"
- "making it faster and more accurate"
- "Clinical Geneticists: Research phenotyping and literature review (NOT for patient diagnosis)" if the surrounding text implies clinical users as a target group

Use:

```json
"welcomeText": "Phentrieve maps clinical or biomedical research text to standardized Human Phenotype Ontology (HPO) terms for research, benchmarking, education, and data standardization."
```

For FAQ "What is Phentrieve?", use:

```html
<p>Phentrieve is an open-source research tool that maps clinical or biomedical research text to standardized Human Phenotype Ontology (HPO) terms. It supports phenotype annotation for research datasets, benchmarking, education, and exploratory analysis.</p>
```

- [ ] **Step 3: Align all locales**

Update `de`, `fr`, `es`, and `nl` locale files with equivalent meaning. Do not
use wording equivalent to "diagnosis", "diagnostics", "treatment", "patient
care", or "clinical decision support" except in negative/prohibited-use
statements.

- [ ] **Step 4: Fix or remove stale FAQ config**

Check whether `frontend/src/config/faqConfig.json` is still imported.

Run:

```bash
rg -n "faqConfig" frontend/src
```

If unused, delete it and add a test proving FAQ uses i18n JSON. If used, replace
the diagnostic claims with research-only wording.

- [ ] **Step 5: Rename deployment posture**

In `docs/DOCKER-DEPLOYMENT.md`, replace:

- "production environments"
- "production-ready Docker images"
- "Production Deployment"

with:

- "self-hosted research deployments"
- "hardened Docker images"
- "Self-Hosted Research Deployment"

Add a warning block:

```markdown
!!! warning "Research use only"
    This deployment guide does not authorize clinical operation. Operators who
    use Phentrieve for patient-care workflows must perform their own EU AI Act,
    GDPR, and MDR/IVDR assessment before use.
```

- [ ] **Step 6: Add compliance link to usage docs**

At the top of API, CLI, frontend, and MCP usage docs, add:

```markdown
!!! warning "Research use only"
    Phentrieve is not intended for diagnosis, treatment, triage, patient
    management, or clinical decision support. Do not submit identifiable patient
    data to public demo instances. See [Research Use](../compliance/research-use.md).
```

Adjust relative paths per document location.

- [ ] **Step 7: Add copy regression test**

Create `frontend/src/test/complianceCopy.test.js`:

```javascript
import en from '../locales/en.json';

describe('compliance copy', () => {
  it('does not make affirmative diagnostic claims in primary English UI copy', () => {
    const serialized = JSON.stringify(en).toLowerCase();
    expect(serialized).not.toContain('for research and diagnostics');
    expect(serialized).not.toContain('clinical diagnostics');
    expect(serialized).not.toContain('making it faster and more accurate');
  });

  it('retains research-use disclaimer language', () => {
    expect(en.disclaimerDialog.mainHeader.toLowerCase()).toContain('research');
    expect(en.disclaimerDialog.limitations.item1.toLowerCase()).toContain('not a clinical diagnostic tool');
  });
});
```

- [ ] **Step 8: Run frontend copy tests**

Run:

```bash
cd frontend && npm test -- complianceCopy.test.js --runInBand
```

If the project's Vitest command does not support `--runInBand`, use:

```bash
cd frontend && npm test -- complianceCopy.test.js
```

Expected: test passes.

### Task 3: Frontend Guardrails And Disclaimer Versioning

**Files:**
- Modify: `frontend/src/components/DisclaimerDialog.vue`
- Modify: `frontend/src/stores/disclaimer.js`
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/components/PhenotypeCollectionPanel.vue`
- Modify: `frontend/src/components/FullTextAnnotationWorkspace.vue`
- Modify: `frontend/src/services/PhentrieveService.js`
- Modify: `frontend/src/locales/*.json`
- Test: existing component/service tests plus new tests as needed

- [ ] **Step 1: Version disclaimer acknowledgement**

In `frontend/src/stores/disclaimer.js`, add:

```javascript
const CURRENT_DISCLAIMER_VERSION = '2026-04-29-research-use-v1';
const disclaimerVersion = ref(null);

const isCurrentVersionAcknowledged = computed(() => {
  return isAcknowledged.value && disclaimerVersion.value === CURRENT_DISCLAIMER_VERSION;
});
```

Update `saveAcknowledgment()` to set `disclaimerVersion.value`.

Persist `disclaimerVersion`.

- [ ] **Step 2: Use current-version acknowledgement in App**

In `frontend/src/App.vue`, show the disclaimer when
`!disclaimerStore.isCurrentVersionAcknowledged` rather than only
`!disclaimerStore.isAcknowledged`.

- [ ] **Step 3: Add visible data-entry warning**

Near the main text input in `QueryInterface.vue`, add a compact warning using
locale text:

```text
Research use only. Use synthetic, de-identified, or research-consented text.
Do not enter identifiable patient data in public demos.
```

Use Vuetify alert or text consistent with existing UI.

- [ ] **Step 4: Add Phenopacket export warning**

In `PhenotypeCollectionPanel.vue`, place this near export controls:

```text
Phenopacket export is a research interoperability artifact, not a clinical record.
```

Rename visible labels:

- `Patient Information (Optional)` -> `Research Subject Information (Optional)`
- `Patient ID` -> `Research Subject ID`

- [ ] **Step 5: Add LLM provider warning**

In `FullTextAnnotationWorkspace.vue` or the advanced options area where LLM
backend is selected, add:

```text
External LLM providers may receive submitted text if enabled by this deployment.
Use local LLM mode or synthetic/de-identified text for public demos.
```

Only show this when `extraction_backend === 'llm'` or LLM option is visible.

- [ ] **Step 6: Send acknowledgement header**

In `frontend/src/services/PhentrieveService.js`, include:

```javascript
'X-Phentrieve-Research-Use-Acknowledged': 'true'
```

for text-bearing requests after current disclaimer acknowledgement. Pass the
acknowledgement state from the app/store into the service using the existing
service pattern; do not import Pinia directly into a low-level service unless
that is already local style.

- [ ] **Step 7: Update tests**

Add/update tests to prove:

- old disclaimer versions trigger the dialog again;
- current acknowledgement suppresses it;
- text process requests include the research-use header;
- visible UI no longer uses patient wording except in prohibited-use warnings.

Run:

```bash
make frontend-test-ci
```

Expected: frontend tests pass.

### Task 4: API Hosted-Mode Acknowledgement And LLM Controls

**Files:**
- Modify: `api/config.py`
- Modify: `api/api.yaml`
- Modify: `api/api.yaml.template`
- Modify: `api/routers/query_router.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `api/routers/phenopacket_router.py`
- Modify: `api/main.py`
- Create: `tests/unit/api/test_research_use_guard.py`

- [ ] **Step 1: Add config constants**

In `api/config.py`, add defaults and exports:

```python
PHENTRIEVE_PUBLIC_HOSTED_MODE: bool = os.getenv(
    "PHENTRIEVE_PUBLIC_HOSTED_MODE", "false"
).lower() == "true"
PHENTRIEVE_REQUIRE_RESEARCH_ACK: bool = os.getenv(
    "PHENTRIEVE_REQUIRE_RESEARCH_ACK", "false"
).lower() == "true"
PHENTRIEVE_ENABLE_EXTERNAL_LLM: bool = os.getenv(
    "PHENTRIEVE_ENABLE_EXTERNAL_LLM", "false"
).lower() == "true"
```

Effective acknowledgement requirement:

```python
REQUIRE_RESEARCH_ACK_EFFECTIVE = (
    PHENTRIEVE_PUBLIC_HOSTED_MODE or PHENTRIEVE_REQUIRE_RESEARCH_ACK
)
```

- [ ] **Step 2: Add reusable guard helper**

In a new or existing API helper module, define:

```python
from fastapi import HTTPException, Request, status

RESEARCH_ACK_HEADER = "x-phentrieve-research-use-acknowledged"

def require_research_use_acknowledgement(request: Request) -> None:
    if not api_config.REQUIRE_RESEARCH_ACK_EFFECTIVE:
        return
    if request.headers.get(RESEARCH_ACK_HEADER, "").lower() == "true":
        return
    raise HTTPException(
        status_code=status.HTTP_428_PRECONDITION_REQUIRED,
        detail={
            "error_message": "Research-use acknowledgement is required.",
            "required_header": "X-Phentrieve-Research-Use-Acknowledged",
            "required_value": "true",
        },
    )
```

Prefer a dedicated `api/research_use.py` if that keeps routers clean.

- [ ] **Step 3: Apply guard to text-bearing endpoints**

Apply the guard to:

- `POST /api/v1/query/`
- `GET /api/v1/query/`
- `POST /api/v1/text/process`
- `POST /api/v1/phenopackets/export`

Do not apply it to health, config, version, docs, or static metadata endpoints.

- [ ] **Step 4: Block external LLM in public hosted mode**

In `text_processing_router.py`, before LLM quota logic:

```python
if (
    request.extraction_backend == "llm"
    and api_config.PHENTRIEVE_PUBLIC_HOSTED_MODE
    and not api_config.PHENTRIEVE_ENABLE_EXTERNAL_LLM
):
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "LLM extraction is disabled for this public research demo unless "
            "the operator explicitly enables external LLM processing and "
            "publishes the required privacy/provider notice."
        ),
    )
```

- [ ] **Step 5: Add OpenAPI descriptions**

Update route descriptions to include:

```text
Research use only. Not for diagnosis, treatment, triage, patient management, or
clinical decision support. Do not submit identifiable patient data to public
demo instances.
```

For text-bearing endpoints, document the optional/required hosted header.

- [ ] **Step 6: Add API tests**

Create `tests/unit/api/test_research_use_guard.py` with tests for:

- no acknowledgement needed when hosted mode and explicit require flag are false;
- 428 returned when required and header is missing;
- request succeeds when required and header is true;
- public hosted LLM request is 403 when external LLM is disabled;
- standard backend still works with acknowledgement.

Use FastAPI `TestClient` and monkeypatch config constants.

- [ ] **Step 7: Run API tests**

Run:

```bash
uv run pytest tests/unit/api/test_research_use_guard.py -v
```

Expected: tests pass.

### Task 5: CLI And MCP Research-Use Notices

**Files:**
- Modify: `phentrieve/cli/query_commands.py`
- Modify: `phentrieve/cli/text_commands.py`
- Modify: `phentrieve/cli/mcp_commands.py`
- Modify: `phentrieve/cli/benchmark_commands.py`
- Modify: `api/mcp/server.py`
- Modify: `docs/mcp-server.md`
- Create: `tests/unit/cli/test_research_use_notice.py`
- Modify MCP tests if needed: `tests/unit/mcp/test_mcp_server.py`

- [ ] **Step 1: Add shared CLI notice helper**

Create or add to `phentrieve/cli/utils.py`:

```python
RESEARCH_USE_NOTICE = (
    "Research use only. Phentrieve is not a medical device and must not be "
    "used for diagnosis, treatment, triage, patient management, or clinical "
    "decision support. Do not submit identifiable patient data to public demos."
)

def emit_research_use_notice() -> None:
    typer.secho(RESEARCH_USE_NOTICE, fg=typer.colors.YELLOW, err=True)
```

- [ ] **Step 2: Emit notice for text-bearing CLI commands**

Call `emit_research_use_notice()` at the start of:

- `phentrieve query`
- `phentrieve query --interactive`
- `phentrieve text process`
- `phentrieve text interactive`
- LLM benchmark commands when they send text to providers

Do not emit for pure config/data/index commands unless they process user text.

- [ ] **Step 3: Add quiet/ack option if needed**

If existing tests or UX require less noise, add:

```text
--research-use-acknowledged
```

to suppress the notice after printing it once per invocation, not to remove the
intended-use terms.

- [ ] **Step 4: Update MCP tool descriptions**

In `api/mcp/server.py`, ensure text-processing tools include:

```text
Research use only. Not for diagnosis, treatment, triage, patient management, or
clinical decision support. Do not send identifiable patient data to public demo
instances.
```

- [ ] **Step 5: Add CLI tests**

Create tests asserting:

- `phentrieve text process ...` emits "Research use only" to stderr;
- `phentrieve query ...` emits "Research use only" to stderr;
- non-text commands do not emit the notice.

- [ ] **Step 6: Run CLI and MCP tests**

Run:

```bash
uv run pytest tests/unit/cli/test_research_use_notice.py tests/unit/mcp/test_mcp_server.py -v
```

Expected: tests pass.

### Task 6: Logging And Data-Retention Safety Tests

**Files:**
- Modify: `api/routers/query_router.py`
- Modify: `api/routers/text_processing_router.py`
- Modify: `phentrieve/text_processing/full_text_service.py`
- Modify: `phentrieve/llm/provider.py`
- Create: `tests/unit/api/test_no_raw_text_logging.py`

- [ ] **Step 1: Audit logging statements**

Run:

```bash
rg -n "logger\\.(debug|info|warning|error|exception).*text|query_text|request\\.text|input_text|clinical" api phentrieve
```

Classify each hit:

- allowed: text length, model name, mode, counts, status;
- disallowed: raw submitted text, source document text, prompt payload, evidence text in logs.

- [ ] **Step 2: Remove or sanitize raw text logs**

Replace any raw text logging with counts:

```python
logger.info("Processing text request: text_chars=%d", len(request.text))
```

Do not log prompt bodies for LLM providers in public/default mode.

- [ ] **Step 3: Add no-raw-text logging tests**

Create tests that use `caplog` with a unique sentinel:

```python
sentinel = "PATIENT_NAME_SENTINEL_12345"
```

Run representative query/text process paths with mocked retrieval/model calls and
assert the sentinel is absent from logs.

- [ ] **Step 4: Run safety tests**

Run:

```bash
uv run pytest tests/unit/api/test_no_raw_text_logging.py -v
```

Expected: sentinel text is not present in captured logs.

### Task 7: Phenopacket Research Framing

**Files:**
- Modify: `api/schemas/phenopacket_schemas.py`
- Modify: `api/routers/phenopacket_router.py`
- Modify: `frontend/src/components/PhenotypeCollectionPanel.vue`
- Modify: `frontend/src/locales/*.json`
- Modify tests: `tests/unit/api/test_phenopacket_router.py`, frontend component tests

- [ ] **Step 1: Update schema descriptions**

Replace "patient" terminology in schema descriptions with "research subject"
where the API controls the description.

Keep standards terms only where required by Phenopacket spec, but explain
research context in descriptions.

- [ ] **Step 2: Add export metadata**

Include an external reference or annotation in exported bundles:

```json
{
  "id": "phentrieve:intended_use",
  "description": "Research use only; not a clinical record or diagnostic output."
}
```

- [ ] **Step 3: Update UI labels**

Apply the label changes from Task 3:

- Research Subject Information
- Research Subject ID
- Research Phenopacket Export

- [ ] **Step 4: Update tests**

Add API assertion that exported Phenopacket metadata contains
`phentrieve:intended_use`.

Add frontend assertion that export panel contains "research" and does not show
"Patient Information".

- [ ] **Step 5: Run targeted tests**

Run:

```bash
uv run pytest tests/unit/api/test_phenopacket_router.py -v
make frontend-test-ci
```

Expected: tests pass.

### Task 8: Public Demo Deployment Defaults

**Files:**
- Modify: `.env.docker.template`
- Modify: `docker-compose.yml`
- Modify: `docs/DOCKER-DEPLOYMENT.md`
- Modify: `api/docker-entrypoint.sh` if environment validation belongs there
- Test: existing Docker/security tests if applicable

- [ ] **Step 1: Add public-demo env template entries**

In `.env.docker.template`, add:

```bash
# Compliance posture
PHENTRIEVE_PUBLIC_HOSTED_MODE=false
PHENTRIEVE_REQUIRE_RESEARCH_ACK=false
PHENTRIEVE_ENABLE_EXTERNAL_LLM=false
PHENTRIEVE_ALLOW_IDENTIFIABLE_PATIENT_DATA=false
```

- [ ] **Step 2: Wire env vars into docker-compose**

In `docker-compose.yml`, pass these variables into the API service.

- [ ] **Step 3: Add deployment docs matrix**

In `docs/DOCKER-DEPLOYMENT.md`, add:

| Scenario | PUBLIC_HOSTED_MODE | REQUIRE_RESEARCH_ACK | ENABLE_EXTERNAL_LLM |
| --- | --- | --- | --- |
| Local research | false | false | operator choice |
| Private research team | false | true recommended | operator choice |
| Public EU demo | true | true | false by default |
| Clinical use | not covered | not covered | not covered |

- [ ] **Step 4: Validate environment behavior**

Run targeted API tests from Task 4.

Run:

```bash
make check
```

Expected: check passes.

### Task 9: Release Checklist And Legal Review Gate

**Files:**
- Create: `.planning/active/2026-04-29-eu-ai-act-release-checklist.md`
- Modify: `.planning/README.md`
- Modify: `CHANGELOG.md` when release is prepared

- [ ] **Step 1: Create release checklist**

Add checklist:

```markdown
# EU Research-Use Release Checklist

- [ ] Diagnostic/clinical-use copy scan passes.
- [ ] Public hosted API requires research acknowledgement.
- [ ] Public hosted LLM is disabled unless privacy/provider review is complete.
- [ ] Privacy and data-processing notice is published.
- [ ] EU AI Act classification page is reviewed.
- [ ] README links compliance docs.
- [ ] Frontend disclaimer version updated.
- [ ] API/CLI/MCP notices are visible.
- [ ] No raw text logging tests pass.
- [ ] Counsel reviewed public intended-purpose wording.
- [ ] Maintainer approved release.
```

- [ ] **Step 2: Add final search gates**

Run:

```bash
rg -n "clinical diagnostics|for research and diagnostics|diagnostic tool|healthcare professionals|patient information|patient id|production-ready|production deployment" README.md docs frontend/src api phentrieve
```

Expected: no affirmative clinical-use hits. Negative/prohibited-use statements
may remain if they explicitly say "not" or "do not".

- [ ] **Step 3: Add release-note entry**

In `CHANGELOG.md`, add an unreleased entry:

```markdown
- Aligned Phentrieve's public documentation, UI, API, CLI, and MCP surfaces with
  a research-use-only compliance posture for EU public demo operation.
```

### Task 10: Full Verification

**Files:** no new files; verification only.

- [ ] **Step 1: Run required repo checks**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected:

- Ruff formatting/checking passes.
- mypy daemon reports no issues.
- pytest passes.

If `make test` fails due CUDA out-of-memory unrelated to these changes, rerun
CPU-only or single-threaded verification and document exact evidence:

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -n 0 -v
```

- [ ] **Step 2: Run frontend parity checks**

Because this changes frontend copy and service headers, run:

```bash
make frontend-test-ci
make frontend-build-ci
```

Expected: frontend tests and build pass.

- [ ] **Step 3: Run docs build**

Run:

```bash
uv run mkdocs build --strict
```

Expected: docs build without warnings or broken links.

- [ ] **Step 4: Run final compliance scans**

Run:

```bash
rg -n "clinical diagnostics|for research and diagnostics|patient information|patient id|production deployment|production-ready Docker" README.md docs frontend/src api phentrieve
```

Expected: zero affirmative prohibited-positioning hits.

Run:

```bash
rg -n "Research use only|not a medical device|not CE marked|diagnosis, treatment, triage" README.md docs frontend/src api phentrieve
```

Expected: hits in README, compliance docs, frontend disclaimer/copy, API
descriptions, CLI notice helper, and MCP tool descriptions.

## Implementation Order

1. Task 1: canonical docs first, because every other surface links to them.
2. Task 2: remove contradictory claims before adding enforcement.
3. Task 3: frontend guardrails and acknowledgement versioning.
4. Task 4: API hosted-mode enforcement.
5. Task 5: CLI/MCP notices.
6. Task 6: raw-text logging tests.
7. Task 7: Phenopacket research framing.
8. Task 8: deployment defaults.
9. Task 9: release checklist.
10. Task 10: full verification.

## Definition Of Done

- No public-facing affirmative diagnostic/clinical-decision claims remain.
- Research-only intended use is visible in README, docs, UI, API, CLI, and MCP.
- Public hosted API mode requires explicit acknowledgement for text-bearing endpoints.
- Public hosted LLM mode is disabled by default.
- Privacy/provider notice is published and linked.
- Phenopacket export is framed as a research artifact.
- Tests cover copy regressions, API acknowledgement, CLI notices, and no raw text logging.
- Required checks have been run and results are recorded.
- Legal counsel has reviewed the final public-facing intended-purpose language before public EU launch.
