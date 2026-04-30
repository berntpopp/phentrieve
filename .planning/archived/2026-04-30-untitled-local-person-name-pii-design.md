# Untitled Local Person Name PII Design

## Goal

Improve the browser-only PII guard so it can flag likely person names without
titles, such as `Bernt Popp`, before any query or full-text network request.

## Scope

This is a local, deterministic safety aid. It does not attempt guaranteed
de-identification or anonymisation, and it does not call a server or external
model. Untitled person-name matches are review-confidence only.

## Approach

Add a config-driven `possible_name` recognizer in the frontend PII layer. The
recognizer generates name-shaped candidates from adjacent capitalized tokens and
scores them with locale-aware context, particles, and blocklists.

The recognizer runs across all configured PII locales, matching the current
cross-locale PII behavior. It emits `person_name` findings with
`confidence: "review"` and `redactionToken: "[REDACTED_NAME]"`.

## Detection Rules

A candidate is eligible when it contains at least two name-shaped tokens, with
optional locale particles between them. Examples:

- `Bernt Popp`
- `Jean Dupont`
- `María García`
- `Jan van Dijk`

The score starts from token shape and is adjusted using config:

- Add score for explicit nearby context such as `name`, `patient`, `Patient`,
  `Name`, `Vorname`, `Nachname`, `Paciente`, `Nom`, `Naam`.
- Add score for valid locale particles such as `van`, `von`, `de`, `del`,
  `da`, `le`.
- Subtract score or reject candidates containing blocked medical/domain terms,
  HPO IDs, model-like tokens, all-caps identifiers, or known ontology/model
  words.

The default threshold flags full-name-shaped candidates like `Bernt Popp ist
dumm`, while avoiding common clinical/domain phrases like `Pectus Carinatum`,
`Down Syndrome`, `BioLORD Model`, and `HP:0002779 Tracheomalacia`.

## UI Behavior

Untitled name findings use the existing PII review dialog. They appear under
review-confidence `Name` findings. The user can choose `Redact in text` to
replace review findings locally. Continuing without manual redaction still only
auto-redacts high-confidence findings, preserving current behavior.

## Non-Goals

- No Transformers.js or model-based NER in this change.
- No backend PII detector.
- No high-confidence auto-redaction for untitled names.
- No raw matched text, snippets, offsets beyond existing local-only offsets, or
  redacted text in logs.

## Tests

Add detector tests for:

- Untitled two-token names across all supported locales.
- Particle names such as `Jan van Dijk`.
- Cross-locale detection while selected language is English.
- Non-matches for clinical/domain phrases and IDs.

Add QueryInterface tests for:

- Query mode opens PII review and sends no network request for an untitled name.
- Full Text mode opens PII review and sends no network request for an untitled
  name.
