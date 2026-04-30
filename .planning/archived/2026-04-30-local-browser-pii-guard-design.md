# Local Browser PII Guard Design

## Issue

- GitHub: https://github.com/berntpopp/phentrieve/issues/249
- Scope: Vue frontend text submission paths for both short Query mode and Full
  Text mode.
- Languages in scope: English, German, French, Spanish, and Dutch.

## Goal

Add a browser-only PII/PHI warning and redaction gate before user text is sent to
Phentrieve APIs. The gate must scan both short phenotype queries and full
clinical notes locally, redact high-confidence identifiers before submission,
and require explicit user acknowledgement when lower-confidence findings remain.

The feature is a safety aid for public research use. It must not claim HIPAA
Safe Harbor de-identification, GDPR anonymisation, or complete PII detection.

## Non-Goals

- Do not send raw text to a backend, cloud PII API, LLM, analytics service, or
  log sink for pre-submit PII detection.
- Do not implement backend enforcement in this iteration. Frontend checks guide
  users but are not a security boundary.
- Do not add a heavy browser NLP or NER model in the first implementation.
- Do not claim that redacted text is anonymous or legally de-identified.
- Do not inspect image, PDF, DICOM, uploaded binary, or phenopacket export
  content in this issue.
- Do not persist detected snippets, finding offsets, or pre-redaction text.

## Standards And Research Baseline

The practical identifier categories should align with HIPAA Safe Harbor without
claiming compliance. HHS identifies two HIPAA de-identification routes: Expert
Determination and Safe Harbor. Safe Harbor focuses on removing 18 identifier
classes, including names, sub-state geographic details, date elements except
year for dates related to an individual, contact details, record/account
numbers, device/vehicle identifiers, URLs, IP addresses, biometrics, face
images, and other unique identifying codes.

NIST SP 800-188 frames de-identification as privacy-risk reduction and cautions
that traditional de-identification has inherent limits. This supports UI copy
that treats Phentrieve's PII guard as a warning and redaction aid, not as a
guarantee.

GDPR/EDPB guidance distinguishes pseudonymisation from anonymisation. Because
the frontend can only remove or replace visible identifiers and cannot assess
full re-identification risk, Phentrieve must avoid saying user text is
anonymised.

Cloud and server-side products such as AWS Comprehend Medical DetectPHI, Google
Sensitive Data Protection, Azure AI Language PII, and Microsoft Presidio are
useful comparison points but are not runtime choices for this issue. They either
send text out of the browser or require backend infrastructure before the first
network request, which conflicts with the acceptance criteria.

## Design Principles

1. Local first: detection, review, and high-confidence redaction happen before
   any network request.
2. Config-driven: language and country-specific rules live in data/config, not
   hard-coded branches spread through UI code.
3. Conservative redaction: high-confidence direct identifiers are always
   replaced locally before submission.
4. Transparent uncertainty: lower-confidence findings trigger review and can be
   acknowledged, but the UI must not expose raw snippets.
5. Utility preservation: redaction tokens preserve broad category information
   such as `[REDACTED_DATE]` or `[REDACTED_MRN]` so downstream HPO extraction is
   not confused by silent deletion.
6. No raw-text telemetry: logs may include counts and categories, never matched
   text, offsets, or full submitted text.
7. Expandability: adding a future locale should require a locale config and
   tests, not changes to the detector core.

## Architecture

### PII Rule Engine

Create a small frontend PII module under `frontend/src/pii/`.

Recommended files:

- `frontend/src/pii/types.js` - JSDoc typedefs for rule, finding, scan result,
  and redaction result shapes.
- `frontend/src/pii/ruleConfig.js` - global rules and locale overlays for
  `en`, `de`, `fr`, `es`, and `nl`.
- `frontend/src/pii/validators.js` - checksum and format validators such as
  Spanish DNI/NIE and Dutch BSN.
- `frontend/src/pii/detector.js` - applies configured rules and produces
  sanitized findings.
- `frontend/src/pii/redactor.js` - applies high-confidence redactions and
  optional review-confidence redactions.
- `frontend/src/pii/index.js` - public frontend PII API.

The detector core accepts:

```js
scanPii(text, {
  locale,
  enabledLocales,
  includeGlobalRules,
});
```

It returns category-level findings without raw snippets:

```js
{
  hasFindings: true,
  findings: [
    {
      id: "finding-1",
      ruleId: "global.email",
      category: "email",
      confidence: "high",
      start: 10,
      end: 26,
      redactionToken: "[REDACTED_EMAIL]"
    }
  ],
  summary: {
    high: { email: 1 },
    review: {}
  }
}
```

Offsets may exist in memory so redaction can be applied, but they must not be
shown in the UI, saved to stores, or sent to logs.

### Rule Configuration

Rules should use a common shape:

```js
{
  id: "fr.nir",
  category: "national_identifier",
  confidence: "high",
  locales: ["fr"],
  redactionToken: "[REDACTED_NATIONAL_ID]",
  enabled: true,
  patterns: [/.../gu],
  contextKeywords: ["nir", "sécurité sociale", "numéro de sécurité sociale"],
  validator: "validateFrenchNir"
}
```

Global rules run for every language:

- Email addresses.
- URLs.
- IPv4 and common IPv6 forms.
- International and locale-specific phone numbers through `libphonenumber-js`.
- Exact date patterns and date-of-birth labels.
- MRN/record/account/accession/sample identifiers with localizable labels.
- Address-like lines when a street keyword, house number, and postal-code-like
  token occur together.

Locale overlays add language-specific labels and identifiers:

- `en`: `DOB`, `date of birth`, `MRN`, `medical record number`, `patient ID`,
  `NHS number` as a configurable label pattern rather than a hard-coded UK-only
  validation guarantee.
- `de`: `Geburtsdatum`, `Versichertennummer`,
  `Krankenversichertennummer`, `Patienten-ID`, `Aktenzeichen`, German street
  terms, 5-digit postal-code address context, and KVNR-like values with one
  uppercase letter followed by nine digits.
- `fr`: `date de naissance`, `NIR`, `numéro de sécurité sociale`,
  `identifiant patient`, French street terms, 5-digit postal-code address
  context, and NIR-like values with 13 digits plus optional 2-digit key.
- `es`: `fecha de nacimiento`, `DNI`, `NIE`, `NIF`, `número de historia
  clínica`, Spanish street terms, 5-digit postal-code address context, and
  DNI/NIE checksum validation when the candidate shape allows it.
- `nl`: `geboortedatum`, `BSN`, `burgerservicenummer`,
  `patiëntnummer`, `polisnummer`, Dutch street terms, Dutch postcode plus house
  number context, and BSN-like values with modulus-11 validation.

Postal codes alone are review-confidence because they often appear in research
context. Postal code plus street/house/address context is high-confidence.

### Confidence And Redaction Policy

High-confidence findings are always redacted before submission:

- Email.
- Phone/fax.
- URL.
- IP address.
- SSN/national ID with a recognized shape or checksum.
- MRN, account, accession, sample, or patient ID with label context.
- DOB-labeled dates.
- Exact dates directly attached to admission, discharge, death, visit, birth,
  appointment, or patient-care labels.
- Address-like lines with street/address context.

Review-confidence findings trigger a modal and can remain after explicit user
acknowledgement:

- Possible person names found through local context labels such as `Name:`,
  `Patient`, `Paciente`, `Nom`, `Naam`, or `Name`.
- Standalone exact dates without patient-care context.
- Standalone postal codes.
- Organization or place-like text detected only through labels.

On override, high-confidence findings are redacted silently and locally. The
modal must explain this before submission:

> High-confidence identifiers will be redacted locally before submission. Other
> possible identifiers may remain if you continue.

The user must never be offered a path that sends high-confidence identifiers
unchanged.

### Submission Flow

Update `frontend/src/composables/useQueryInterfaceController.js`.

Current submission flow constructs the conversation turn, clears the input, logs
metadata, and then calls `queryHpo()` or `processText()`. The PII guard must run
before those actions.

New flow:

1. Trim submitted text.
2. Determine active submission mode: Query or Full Text.
3. Scan local text using current UI locale and all global rules.
4. If no findings exist, continue with current flow.
5. If findings exist, open a PII review dialog and stop before adding a
   conversation turn or calling the API.
6. If the user cancels, keep the original text in the input field and do not log
   findings beyond category counts.
7. If the user chooses redaction, apply high-confidence redaction and optionally
   review-confidence redaction, update the input field, and do not submit until
   the user clicks submit again.
8. If the user chooses override/continue, apply high-confidence redaction in
   memory and submit that redacted text. Review-confidence text remains only
   after acknowledgement.

Auto-submit from URL parameters must not bypass the gate. If `autoSubmit=true`
and PII findings exist, the dialog opens and no API call happens until the user
acts.

### UI Component

Create `frontend/src/components/PiiReviewDialog.vue`.

The dialog should show:

- concise title: `Possible identifiers detected`
- categories and counts, not snippets
- confidence grouping: `Will be redacted` and `Needs review`
- short public-demo warning aligned with existing research-use posture
- actions:
  - `Review text` or `Cancel` keeps text local and closes the dialog
  - `Redact in text` replaces all findings in the input and does not submit
  - `Continue with local redaction` submits with high-confidence redactions and
    acknowledged review-confidence findings

The dialog should use existing Vuetify patterns and i18n keys. It must work at
mobile widths without overflowing.

### I18n

Add UI copy to all current locale files:

- `frontend/src/locales/en.json`
- `frontend/src/locales/de.json`
- `frontend/src/locales/fr.json`
- `frontend/src/locales/es.json`
- `frontend/src/locales/nl.json`

Locale-specific detector labels belong in `ruleConfig.js` rather than the UI
translation files. UI translation files are for user-facing dialog text only.

### Logging And Privacy

Update frontend logging around submission so raw text is not logged.

Current `submitQuery()` logs request objects for both Query and Full Text paths.
That is incompatible with this feature. Replace those logs with metadata-only
events:

- mode
- text length
- extraction backend for Full Text
- selected language
- category counts from PII scan
- whether high-confidence redaction was applied

Do not log:

- raw query text
- raw full note text
- matched snippets
- finding offsets
- redacted text

### Documentation

Update `docs/compliance/privacy-and-llm-processing.md` with:

- browser-side PII warning and redaction behavior
- five-language scope
- high-confidence local redaction on override
- limitation that this is not a guarantee and not legal de-identification
- reminder not to submit identifiable patient data to public demos

Update frontend usage docs if a user-facing frontend guide already describes
submission or public demo privacy behavior.

## Testing Strategy

### Unit Tests

Add tests under `frontend/src/test/pii/`.

Detector tests:

- high-confidence email, phone, URL, IP, MRN, sample/accession IDs
- exact dates with and without DOB/admission/discharge context
- no false positive for common HPO IDs such as `HP:0001250`
- no false positive for model identifiers or threshold values
- overlapping findings are merged deterministically
- redaction preserves surrounding text and uses category tokens

Locale tests:

- English: `DOB`, `MRN`, `Patient ID`
- German: `Geburtsdatum`, `Krankenversichertennummer`, KVNR-like value
- French: `date de naissance`, NIR-like value
- Spanish: `fecha de nacimiento`, DNI/NIE checksum-positive value
- Dutch: `geboortedatum`, BSN checksum-positive value, Dutch postcode plus
  house number

Validator tests:

- Spanish DNI/NIE accepts valid checksums and rejects invalid checksums.
- Dutch BSN accepts valid modulus-11 numbers and rejects invalid numbers.
- Validator failures downgrade or discard findings according to rule config.

### Component And Flow Tests

Extend `frontend/src/test/components/QueryInterface.test.js` or add a focused
PII submission test file.

Required cases:

- Query mode with PII opens dialog and does not call `queryHpo()`.
- Full Text mode with PII opens dialog and does not call `processText()`.
- Auto-submit with PII opens dialog and performs no network call before user
  action.
- `Continue with local redaction` submits redacted text.
- `Redact in text` updates the input and does not submit immediately.
- Cancel keeps original text local.
- Logs contain counts and categories but not raw snippets.

### Documentation And I18n Checks

Run locale validation for UI copy changes:

- `make frontend-i18n-check`

Run frontend parity checks:

- `make frontend-test-ci`
- `make frontend-build-ci`

Before completion of implementation, run repository-required checks:

- `make check`
- `make typecheck-fast`
- `make test`

## Open Risks

- Names are difficult to detect with deterministic browser rules, especially
  across five languages. The first version should be conservative and use name
  label context rather than broad capitalized-word heuristics.
- Exact dates are clinically useful. Redacting all standalone dates would reduce
  utility, so the first version treats context-labeled patient-care dates as
  high-confidence and standalone dates as review-confidence.
- Frontend-only checks are bypassable. This design is appropriate for the
  issue's browser-before-network acceptance criteria, but backend operators
  should still avoid raw-text logging and may add server-side controls later.
- Locale configuration can grow large. Keep the first rule set focused on direct
  identifiers and high-signal context patterns.

## Acceptance Criteria Mapping

- Local browser-only detection: `frontend/src/pii/*` runs before service calls.
- Category review and local redaction: `PiiReviewDialog.vue` shows counts and
  redaction actions without snippets.
- Override behavior: users can continue only after high-confidence findings are
  locally redacted and review-confidence findings are acknowledged.
- No network before review: controller flow stops before `queryHpo()` or
  `processText()` when findings exist.
- Tests: detector, redactor, locale, validator, component, auto-submit, and
  logging tests cover the behavior.
- Documentation: compliance docs and UI copy state that this is a safety aid,
  not a guarantee.
