# EU AI Act Research-Use Review

Date: 2026-04-29

Scope reviewed: repository documentation, frontend copy, API contracts, deployment
docs, and code paths that expose HPO term mapping, full-text extraction,
LLM-backed extraction, and Phenopacket export.

This is an engineering and documentation risk review, not legal advice. A lawyer
should validate the final intended-purpose statement before public EU operation.

## Executive Summary

Phentrieve is close to a defensible research-tool posture in some places, but
the current repository does not consistently support "research use only" as the
sole intended purpose.

The strongest evidence for research-only positioning is the frontend disclaimer:
`frontend/src/locales/en.json:53-66` states that Phentrieve is for research and
informational purposes only, is not a clinical diagnostic tool, may be inaccurate,
and must not replace professional medical advice.

The strongest contrary evidence is also user-facing:

- `frontend/src/locales/en.json:71` says the product streamlines phenotype
  annotation "for research and diagnostics".
- `frontend/src/locales/en.json:381` says it supports "both research and
  clinical diagnostics".
- `frontend/src/config/faqConfig.json:10` and `:18` repeat clinical diagnostic
  and healthcare-professional positioning.
- `docs/index.md:15` frames phenotype identification as critical in rare disease
  diagnosis.
- `README.md:68-86` and `docs/DOCKER-DEPLOYMENT.md:1-33` describe production
  deployment without a research-only boundary.
- The API itself is described as processing "clinical text" to "extract HPO
  terms" (`api/routers/text_processing_router.py:482-527`) and provides
  GET/POST query endpoints for "clinical text" (`api/routers/query_router.py:84-113`).

Recommended classification posture:

1. Treat the open-source repository as a research and evaluation tool.
2. Treat any hosted public instance as a separate service that needs stricter
   intended-purpose controls, privacy controls, and EU AI Act classification
   documentation.
3. Do not claim that open source or free availability alone makes the hosted
   service exempt. Under the AI Act, free/open-source exclusions are limited and
   do not protect high-risk systems or certain transparency/prohibited-practice
   cases.

## Legal Baseline Checked

Primary sources checked:

- Regulation (EU) 2024/1689, official EUR-Lex text:
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Commission AI Act overview and timeline:
  https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- European Commission "Navigating the AI Act" FAQ:
  https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act

Key source points:

- The AI Act excludes AI systems or models specifically developed and put into
  service for the sole purpose of scientific research and development
  (EUR-Lex Article 2(6), visible at lines 1243-1245 in the official text).
- Recital 25 clarifies that the regulation should not undermine research, but
  the exclusion does not apply once a resulting AI system is placed on the market
  or put into service outside the sole R&D purpose (EUR-Lex lines 471-475).
- The AI Act also excludes some free/open-source AI systems, but not where they
  are placed on the market or put into service as high-risk systems, or where
  Article 5 or Article 50 applies. The official EUR-Lex search snippet for
  Article 2(12) states this directly.
- General-purpose AI model obligations began applying on 2 August 2025, and the
  general application date is 2 August 2026, with some high-risk product rules
  applying later. The Commission timeline states this at the AI Act policy page
  lines 150-156 and FAQ lines 238-242.
- Providers and deployers of high-risk AI systems have quality management,
  registration, monitoring, human oversight, and incident duties; the Commission
  FAQ summarizes these obligations at lines 155-168.
- Providers of open-source GPAI models get a limited exception from some
  technical-documentation obligations only when the model is released under a
  qualifying free/open-source licence and parameters, weights, architecture, and
  usage information are public, unless the model presents systemic risk
  (EUR-Lex Article 54(6), lines 2797-2798).

## Current Product Facts

### What Phentrieve Does

The codebase exposes AI-assisted phenotype annotation:

- `api/schemas/query_schemas.py:10-13` accepts clinical text for HPO retrieval.
- `api/schemas/text_processing_schemas.py:20-28` accepts raw clinical text and
  supports both standard and LLM extraction backends.
- `api/routers/text_processing_router.py:482-527` exposes a `/api/v1/text/process`
  endpoint for HPO extraction with assertion detection and chunking.
- `phentrieve/llm/provider.py` supports Gemini, Ollama, Anthropic, and OpenAI
  providers; non-local providers may receive submitted clinical text.
- `api/routers/phenopacket_router.py:144-158` exports extracted phenotypes and
  source text into a Phenopacket bundle.
- `frontend/src/locales/en.json:84-113` collects optional patient/subject fields
  and supports Phenopacket export.

This is not merely a static ontology browser. It is an AI system that processes
free text and produces structured phenotype outputs that could influence clinical
documentation or downstream rare-disease analysis if used that way.

### Existing Research-Only Controls

Positive controls already present:

- Persistent disclaimer dialog at first use:
  `frontend/src/components/DisclaimerDialog.vue:1-66`.
- Disclaimer state stored locally:
  `frontend/src/stores/disclaimer.js:15-99`.
- English disclaimer text is clear that the tool is research/informational only
  and not diagnostic: `frontend/src/locales/en.json:53-66`.
- FAQ includes a research-only answer in the i18n file:
  `frontend/src/locales/en.json:387-389`.
- Security policy says no PHI/PII is stored by default:
  `SECURITY.md:31-34`.
- Production LLM quota exists for anonymous LLM access:
  `api/routers/text_processing_router.py:549-606`.

### Controls Missing Or Weak

Missing or weak controls:

- No single canonical `INTENDED_USE.md`, `TERMS.md`, or `RESEARCH_USE.md`.
- No API-level intended-use acknowledgement. CLI and API users can bypass the
  frontend disclaimer entirely.
- No request-level prohibition on real patient data in public demos.
- No public privacy notice describing what happens to submitted clinical text,
  especially when LLM providers are configured.
- No model/provider data-processing warning for Gemini/OpenAI/Anthropic paths.
- No EU AI Act classification file documenting why the project is research-only,
  non-diagnostic, and non-medical-device software.
- No monitoring or UI banner that distinguishes "research demo" from
  production/clinical deployment.
- "Production" docs imply operational use without research-only constraints.

## EU AI Act Risk Assessment

### Repository As Open-Source Research Software

Likely lower risk if all of these remain true:

- The repository is released as software for research, benchmarking, and
  evaluation.
- It is not marketed for diagnosis, treatment, triage, patient management, or
  clinical decision support.
- It does not include a hosted service positioned for patient care.
- It does not monetize support, hosting, or service access in a way that changes
  the open-source/research posture.
- It clearly warns against submitting real patient data to public demos.

The current MIT license supports open-source distribution, but that alone is
not enough. The user-facing wording must align with the intended purpose.

### Public Hosted Demo Or Free EU Service

Higher risk than the repository alone. A free hosted service can still be
"put into service." If the public service accepts clinical notes and returns HPO
terms, regulators may look at actual and foreseeable use, not just the license.

To keep a public EU instance in a research/demo posture:

- Require explicit research-only acknowledgement before API and UI use.
- Remove "diagnostics" and "healthcare professional documentation" claims.
- Use synthetic examples by default.
- Add strong "do not enter identifiable or real patient data" text.
- Disable third-party hosted LLM providers by default on the public demo, or make
  them local-only / opt-in with a separate privacy notice.
- Log only operational metadata, not submitted text.
- Publish a classification rationale and residual-risk statement.

### Clinical Or Hospital Use

If used for patient diagnosis, clinical documentation, treatment selection,
triage, or rare-disease diagnostic work-up, Phentrieve could become a high-risk
AI system and may also raise MDR/IVDR software-as-medical-device questions.

Current wording creates this risk because "clinical diagnostics", "patient",
"case", "Phenopacket", and "healthcare professionals" appear in user-facing
copy. Those are not automatically forbidden in a research tool, but they need
careful wording:

- Use "research subject", "example case", or "study participant" where possible.
- Avoid "patient diagnosis" and "clinical diagnostics" except in explicit
  negative statements.
- Make Phenopacket export a research interoperability feature, not a clinical
  record feature.

## Specific Findings

### High: Intended Purpose Is Contradictory

Evidence:

- Research-only disclaimer: `frontend/src/locales/en.json:53-66`.
- Diagnostic claim: `frontend/src/locales/en.json:71`.
- FAQ diagnostic claim: `frontend/src/locales/en.json:381`.
- Legacy FAQ config diagnostic claim: `frontend/src/config/faqConfig.json:10`.
- Healthcare-professional/clinical genetics target users:
  `frontend/src/config/faqConfig.json:18`.

Impact:

Contradictory intended-use statements weaken the argument that the public tool is
solely for scientific R&D or research information. "Diagnostics" is especially
sensitive under both AI Act and medical-device analysis.

Recommended fix:

Replace all affirmative diagnostic/clinical-use claims with research-only
language. Keep clinical context only as the domain of research text annotation.

### High: Public Service Framing Conflicts With Research-Only Exemption

Evidence:

- `README.md:68-86` says to deploy with Docker Compose for production
  environments.
- `docs/DOCKER-DEPLOYMENT.md:1-33` says the deployment guide covers production
  environments and production-ready Docker images.
- `docs/user-guide/api-usage.md:45-63` documents production LLM handling.
- `docs/user-guide/frontend-usage.md:29-40` documents production-like frontend
  validation.

Impact:

The scientific R&D exclusion is narrow. Public production operation, even free,
can be "put into service" and may be regulated based on intended purpose and
actual use.

Recommended fix:

Rename public docs from "production deployment" to "self-hosted research
deployment" unless you are ready to operate under a broader compliance posture.
Add a separate section: "Not for clinical operation or patient-care workflows."

### High: Real Clinical Text And Third-Party LLM Providers Create Privacy Risk

Evidence:

- API accepts raw clinical text: `api/schemas/text_processing_schemas.py:20-25`.
- LLM backend can send text to configured providers:
  `phentrieve/text_processing/full_text_service.py:671-713`.
- Supported external providers include Gemini, Anthropic, and OpenAI:
  `phentrieve/llm/config.py:8` and `phentrieve/llm/provider.py`.
- Docs demonstrate patient-like text examples:
  `docs/user-guide/api-usage.md:20-39`.

Impact:

AI Act compliance does not replace GDPR. Clinical notes can contain health data
and identifiers. A research-only posture is not enough if public users submit
real patient data to third-party providers without a privacy basis and processor
terms.

Recommended fix:

Add a privacy and data-processing notice before public launch. For the public EU
demo, prefer local/Ollama-only LLM mode or disable LLM extraction unless a
controller/processor setup is documented.

### Medium: API/CLI Bypass The Disclaimer

Evidence:

- Frontend disclaimer exists, but API schemas and OpenAPI routes do not require
  an acknowledgement token or header.
- CLI usage in `README.md:38-46` and `docs/user-guide/api-usage.md:12-39` does
  not include research-only warnings.

Impact:

The disclaimer is only a UI gate. API, CLI, and MCP users can use the tool
without seeing the intended-use limitation.

Recommended fix:

Add intended-use warnings to CLI help, API OpenAPI descriptions, README quick
start, and MCP tool descriptions. For hosted APIs, require an
`X-Phentrieve-Research-Use-Acknowledged: true` header or equivalent.

### Medium: Phenopacket Export Can Look Clinical

Evidence:

- UI text offers "Patient Information (Optional)" and "Patient ID":
  `frontend/src/locales/en.json:84-87`.
- UI exports Phenopackets: `frontend/src/locales/en.json:97` and `:113`.
- API exports subject metadata and source text:
  `api/routers/phenopacket_router.py:104-158`.

Impact:

Phenopacket export is useful for research interoperability, but "patient" and
"case" language can support clinical-use interpretation.

Recommended fix:

Rename visible fields to "Research subject ID" and "Study subject metadata".
Warn that exports are research artifacts and not clinical records.

### Medium: Open-Source Exemption Should Not Be Overclaimed

Evidence:

- MIT license permits use, modification, and distribution.
- The project does not provide public technical documentation for all model
  weights/architectures/training data because it consumes third-party embedding
  and LLM models.

Impact:

The repository may benefit from free/open-source treatment for software
components, but not necessarily for hosted services, high-risk systems,
third-party GPAI models, or Article 50 transparency cases.

Recommended fix:

State: "Phentrieve is open-source research software. Operators of hosted or
clinical deployments are responsible for their own EU AI Act, GDPR, MDR/IVDR,
and provider-model compliance." Avoid saying "open source means exempt."

## Recommended Remediation Plan

### Phase 1: Documentation And Copy Alignment

1. Create `RESEARCH_USE.md` or `docs/compliance/research-use.md`.
2. Add a README top-banner:
   "Research and informational use only. Not a medical device. Not for diagnosis,
   treatment, triage, or other patient-care decisions. Do not submit identifiable
   patient data to public demos."
3. Remove affirmative "diagnostics" claims from:
   - `README.md`
   - `docs/index.md`
   - `frontend/src/locales/*.json`
   - `frontend/src/config/faqConfig.json`
4. Add a "Regulatory Status" section:
   - No CE marking.
   - Not validated for clinical diagnosis.
   - Not intended for patient-specific decision-making.
   - Research outputs require independent expert review.
5. Update deployment docs to say "self-hosted research deployment" and add a
   warning that clinical operation requires separate compliance assessment.

### Phase 2: API/CLI Guardrails

1. Add a hosted-service research-use acknowledgement header for public API use.
2. Add API OpenAPI descriptions that repeat the intended-use limitation.
3. Add CLI startup/help warnings for `query`, `text process`, and LLM modes.
4. Add MCP tool descriptions that state research-only use.
5. Add config to disable LLM mode or require explicit enablement on public demos.

### Phase 3: Privacy And Provider Controls

1. Publish a privacy notice covering:
   - what input text is processed,
   - whether text is stored,
   - logging behavior,
   - third-party LLM provider transfer,
   - public demo prohibition on real patient data,
   - controller/contact details.
2. Add a UI banner near the text input:
   "Use synthetic, de-identified, or research-consented text only."
3. Add docs for local-only LLM operation.
4. Add tests that ensure raw submitted text is not logged by default.

### Phase 4: Classification Record

Create `.planning/analysis/eu-ai-act-classification-record.md` or a public
`docs/compliance/eu-ai-act.md` with:

- intended purpose,
- excluded uses,
- foreseeable misuse,
- user groups,
- model inventory,
- data flow,
- whether public hosted instance is offered,
- whether LLM providers are enabled,
- rationale for non-high-risk/research-only posture,
- residual risks and mitigations,
- review date and owner.

## Suggested Canonical Intended-Purpose Text

> Phentrieve is open-source research software for mapping clinical or biomedical
> research text to Human Phenotype Ontology terms for exploration, benchmarking,
> education, and research data standardisation. Phentrieve is not a medical
> device, is not CE marked, and is not intended to diagnose, treat, prevent,
> monitor, predict, or manage disease, nor to triage patients or make or support
> patient-care decisions. Outputs are algorithmic suggestions that may be wrong
> or incomplete and must be independently reviewed before any downstream use.
> Public demo instances must not be used with identifiable patient data.

## Go / No-Go For A Free EU Public Service

No-go until these are fixed:

- Remove diagnostic claims from UI/docs.
- Add public privacy notice.
- Add research-use notice to API/CLI/MCP, not only frontend.
- Decide whether public LLM mode is disabled or covered by provider/privacy terms.
- Add explicit "no identifiable patient data" warning.

Go, limited research demo posture, after fixes:

- Free public service can be offered as an open research demo if it is clearly
  positioned and technically constrained as research/informational use only.
- Keep usage low-risk: synthetic/de-identified inputs, no clinical claims, no
  patient-care workflow integration, no clinical reliability claims, no
  diagnostic recommendations.

## Open Questions For Maintainer / Counsel

- Will the public EU service process real patient or identifiable health data?
- Will external LLM providers be enabled on the public service?
- Will any healthcare organisation be encouraged to use it in routine care?
- Will support, hosting, consulting, or commercial services be monetized around
  Phentrieve?
- Will outputs be integrated into EHRs, laboratory systems, clinical decision
  support, or diagnostic pipelines?
- Who is the legal operator/controller for a public hosted instance?
