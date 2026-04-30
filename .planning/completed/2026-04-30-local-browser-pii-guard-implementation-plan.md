# Local Browser PII Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser-only, five-language PII/PHI warning and redaction gate before both Query and Full Text submissions.

**Architecture:** Build a config-driven frontend PII module with global rules plus `en`, `de`, `fr`, `es`, and `nl` locale overlays. Integrate it into `QueryInterface` before conversation turns, logs, or API calls are created, and use a Vuetify review dialog to let users cancel, redact locally, or continue with mandatory high-confidence redaction.

**Tech Stack:** Vue 3, Vuetify, Pinia, Vue I18n, Vitest, `libphonenumber-js`, existing `PhentrieveService` and `useQueryInterfaceController`.

---

## File Structure

- Create: `frontend/src/pii/types.js` - JSDoc typedefs for PII rules, findings, scan results, pending submissions, and redaction results.
- Create: `frontend/src/pii/validators.js` - checksum and format validators for Spanish DNI/NIE, Dutch BSN, French NIR-like values, and German KVNR-like values.
- Create: `frontend/src/pii/ruleConfig.js` - global rules and locale overlays for English, German, French, Spanish, and Dutch.
- Create: `frontend/src/pii/detector.js` - config-driven scanner that produces in-memory findings and category summaries without snippets.
- Create: `frontend/src/pii/redactor.js` - deterministic redaction with overlap handling and category tokens.
- Create: `frontend/src/pii/index.js` - public PII API exports.
- Create: `frontend/src/components/PiiReviewDialog.vue` - Vuetify dialog for category review and local redaction actions.
- Create: `frontend/src/test/pii/validators.test.js` - validator coverage.
- Create: `frontend/src/test/pii/detector.test.js` - global and locale detector coverage.
- Create: `frontend/src/test/pii/redactor.test.js` - overlap and token redaction coverage.
- Modify: `frontend/package.json` and `frontend/package-lock.json` - add `libphonenumber-js`.
- Modify: `frontend/src/components/QueryInterface.vue` - mount the dialog, store pending PII submission state, and pass controller hooks.
- Modify: `frontend/src/composables/useQueryInterfaceController.js` - run PII scan before submission and add a resume path after user acknowledgement.
- Modify: `frontend/src/plugins/vuetify.js` - register any Vuetify components used by the dialog that are not already registered.
- Modify: `frontend/src/plugins/icons.js` - add PII dialog icons used in templates.
- Modify: `frontend/src/locales/en.json`, `de.json`, `fr.json`, `es.json`, `nl.json` - add matching UI strings.
- Modify: `frontend/src/test/components/QueryInterface.test.js` - cover no-network-before-review and redacted submission.
- Modify: `frontend/src/test/services/PhentrieveService.test.js` or `frontend/src/test/services/PhentrieveService.researchUse.test.js` only if service logging assertions need adjustment.
- Modify: `docs/compliance/privacy-and-llm-processing.md` - document local warning/redaction behavior and limitations.

## Task 1: Dependency And Validator Foundation

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Create: `frontend/src/pii/types.js`
- Create: `frontend/src/pii/validators.js`
- Create: `frontend/src/test/pii/validators.test.js`

- [ ] **Step 1: Add the phone parsing dependency**

Run:

```bash
cd frontend
npm install libphonenumber-js
```

Expected: `frontend/package.json` and `frontend/package-lock.json` include `libphonenumber-js` under dependencies.

- [ ] **Step 2: Create failing validator tests**

Create `frontend/src/test/pii/validators.test.js`:

```js
import { describe, expect, it } from 'vitest';
import {
  normalizeIdentifierCandidate,
  validateDutchBsn,
  validateFrenchNir,
  validateGermanKvnr,
  validateSpanishDniNie,
} from '../../pii/validators';

describe('PII validators', () => {
  it('normalizes separators and case without changing digits or letters', () => {
    expect(normalizeIdentifierCandidate(' x-123 4567-l ')).toBe('X1234567L');
  });

  it('validates Spanish DNI checksum letters', () => {
    expect(validateSpanishDniNie('12345678Z')).toBe(true);
    expect(validateSpanishDniNie('12345678A')).toBe(false);
  });

  it('validates Spanish NIE checksum letters', () => {
    expect(validateSpanishDniNie('X1234567L')).toBe(true);
    expect(validateSpanishDniNie('X1234567A')).toBe(false);
  });

  it('validates Dutch BSN values with the eleven proof', () => {
    expect(validateDutchBsn('111222333')).toBe(true);
    expect(validateDutchBsn('111222334')).toBe(false);
  });

  it('validates German KVNR-like values conservatively', () => {
    expect(validateGermanKvnr('A123456789')).toBe(true);
    expect(validateGermanKvnr('AB23456789')).toBe(false);
  });

  it('validates French NIR-like values with 13 digits and optional key', () => {
    expect(validateFrenchNir('1900675123456')).toBe(true);
    expect(validateFrenchNir('190067512345699')).toBe(true);
    expect(validateFrenchNir('9909975123456')).toBe(false);
  });
});
```

- [ ] **Step 3: Run validator tests and verify they fail**

Run:

```bash
cd frontend
npm run test:run -- src/test/pii/validators.test.js
```

Expected: FAIL because `frontend/src/pii/validators.js` does not exist.

- [ ] **Step 4: Add PII typedefs**

Create `frontend/src/pii/types.js`:

```js
/**
 * @typedef {'high'|'review'} PiiConfidence
 * @typedef {'email'|'phone'|'url'|'ip_address'|'date'|'dob'|'medical_record'|'accession_id'|'sample_id'|'national_identifier'|'address'|'person_name'|'location'|'organization'} PiiCategory
 *
 * @typedef {Object} PiiRule
 * @property {string} id
 * @property {PiiCategory} category
 * @property {PiiConfidence} confidence
 * @property {string[]} locales
 * @property {string} redactionToken
 * @property {boolean} enabled
 * @property {RegExp[]} patterns
 * @property {string[]} [contextKeywords]
 * @property {string} [validator]
 *
 * @typedef {Object} PiiFinding
 * @property {string} id
 * @property {string} ruleId
 * @property {PiiCategory} category
 * @property {PiiConfidence} confidence
 * @property {number} start
 * @property {number} end
 * @property {string} redactionToken
 *
 * @typedef {Object} PiiScanResult
 * @property {boolean} hasFindings
 * @property {PiiFinding[]} findings
 * @property {{ high: Record<string, number>, review: Record<string, number> }} summary
 *
 * @typedef {Object} PiiRedactionResult
 * @property {string} text
 * @property {boolean} changed
 * @property {{ high: Record<string, number>, review: Record<string, number> }} summary
 */

export {};
```

- [ ] **Step 5: Implement validators**

Create `frontend/src/pii/validators.js`:

```js
export function normalizeIdentifierCandidate(value) {
  return String(value ?? '')
    .replace(/[\s.-]/gu, '')
    .toUpperCase();
}

export function validateSpanishDniNie(value) {
  const normalized = normalizeIdentifierCandidate(value);
  const match = /^(?:(\d{8})|([XYZ])(\d{7}))([A-Z])$/u.exec(normalized);
  if (!match) return false;

  const checksumLetters = 'TRWAGMYFPDXBNJZSQVHLCKE';
  const prefixMap = { X: '0', Y: '1', Z: '2' };
  const number = match[1] ?? `${prefixMap[match[2]]}${match[3]}`;
  const expected = checksumLetters[Number(number) % 23];
  return match[4] === expected;
}

export function validateDutchBsn(value) {
  const normalized = normalizeIdentifierCandidate(value);
  if (!/^\d{9}$/u.test(normalized)) return false;

  const digits = [...normalized].map(Number);
  const checksum =
    digits[0] * 9 +
    digits[1] * 8 +
    digits[2] * 7 +
    digits[3] * 6 +
    digits[4] * 5 +
    digits[5] * 4 +
    digits[6] * 3 +
    digits[7] * 2 -
    digits[8];
  return checksum % 11 === 0;
}

export function validateGermanKvnr(value) {
  return /^[A-Z]\d{9}$/u.test(normalizeIdentifierCandidate(value));
}

export function validateFrenchNir(value) {
  const normalized = normalizeIdentifierCandidate(value);
  if (!/^[12]\d{12}(?:\d{2})?$/u.test(normalized)) return false;
  const month = Number(normalized.slice(3, 5));
  return month >= 1 && month <= 12;
}

export const VALIDATORS = Object.freeze({
  validateSpanishDniNie,
  validateDutchBsn,
  validateGermanKvnr,
  validateFrenchNir,
});
```

- [ ] **Step 6: Run validator tests and verify they pass**

Run:

```bash
cd frontend
npm run test:run -- src/test/pii/validators.test.js
```

Expected: PASS.

- [ ] **Step 7: Commit validator foundation**

Run:

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/pii/types.js frontend/src/pii/validators.js frontend/src/test/pii/validators.test.js
git commit -m "feat(frontend): add PII validator foundation"
```

Expected: commit succeeds.

## Task 2: Config-Driven Detector And Redactor

**Files:**
- Create: `frontend/src/pii/ruleConfig.js`
- Create: `frontend/src/pii/detector.js`
- Create: `frontend/src/pii/redactor.js`
- Create: `frontend/src/pii/index.js`
- Create: `frontend/src/test/pii/detector.test.js`
- Create: `frontend/src/test/pii/redactor.test.js`

- [ ] **Step 1: Write failing detector tests**

Create `frontend/src/test/pii/detector.test.js`:

```js
import { describe, expect, it } from 'vitest';
import { scanPii } from '../../pii';

describe('scanPii', () => {
  it('detects global high-confidence identifiers without exposing snippets', () => {
    const result = scanPii('Email jane@example.org or call +1 202 555 0199.', { locale: 'en' });

    expect(result.summary.high.email).toBe(1);
    expect(result.summary.high.phone).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it('detects English medical record labels', () => {
    const result = scanPii('MRN: AB-123456. Patient has seizures.', { locale: 'en' });
    expect(result.summary.high.medical_record).toBe(1);
  });

  it('detects German KVNR and DOB labels', () => {
    const result = scanPii('Geburtsdatum: 12.03.1980. Krankenversichertennummer A123456789.', {
      locale: 'de',
    });
    expect(result.summary.high.dob).toBe(1);
    expect(result.summary.high.national_identifier).toBe(1);
  });

  it('detects French NIR labels', () => {
    const result = scanPii('NIR 1900675123456 pour le patient.', { locale: 'fr' });
    expect(result.summary.high.national_identifier).toBe(1);
  });

  it('detects Spanish DNI/NIE and clinical history labels', () => {
    const result = scanPii('DNI 12345678Z. Número de historia clínica HC-778899.', {
      locale: 'es',
    });
    expect(result.summary.high.national_identifier).toBe(1);
    expect(result.summary.high.medical_record).toBe(1);
  });

  it('detects Dutch BSN and address context', () => {
    const result = scanPii('BSN 111222333. Adres: Hoofdstraat 12, 1234 AB Leiden.', {
      locale: 'nl',
    });
    expect(result.summary.high.national_identifier).toBe(1);
    expect(result.summary.high.address).toBe(1);
  });

  it('does not flag common HPO IDs or model identifiers', () => {
    const result = scanPii('HP:0001250 with model FremyCompany/BioLORD-2023-M.', { locale: 'en' });
    expect(result.hasFindings).toBe(false);
  });

  it('marks standalone exact dates as review confidence', () => {
    const result = scanPii('Follow-up discussed on 12/03/2024.', { locale: 'en' });
    expect(result.summary.review.date).toBe(1);
    expect(result.summary.high.date).toBeUndefined();
  });
});
```

- [ ] **Step 2: Write failing redactor tests**

Create `frontend/src/test/pii/redactor.test.js`:

```js
import { describe, expect, it } from 'vitest';
import { redactPiiFindings, scanPii } from '../../pii';

describe('redactPiiFindings', () => {
  it('redacts high-confidence findings with category tokens', () => {
    const text = 'Email jane@example.org. MRN: AB-123456.';
    const scan = scanPii(text, { locale: 'en' });
    const redacted = redactPiiFindings(text, scan.findings, { includeReviewFindings: false });

    expect(redacted.text).toContain('[REDACTED_EMAIL]');
    expect(redacted.text).toContain('[REDACTED_MRN]');
    expect(redacted.text).not.toContain('jane@example.org');
    expect(redacted.changed).toBe(true);
  });

  it('keeps review-confidence findings unless requested', () => {
    const text = 'Seen on 12/03/2024.';
    const scan = scanPii(text, { locale: 'en' });

    expect(redactPiiFindings(text, scan.findings, { includeReviewFindings: false }).text).toBe(
      text
    );
    expect(
      redactPiiFindings(text, scan.findings, { includeReviewFindings: true }).text
    ).toContain('[REDACTED_DATE]');
  });

  it('handles overlapping findings by keeping the longest span', () => {
    const text = 'Contact jane@example.org';
    const findings = [
      {
        id: 'a',
        ruleId: 'short',
        category: 'person_name',
        confidence: 'review',
        start: 8,
        end: 12,
        redactionToken: '[REDACTED_NAME]',
      },
      {
        id: 'b',
        ruleId: 'email',
        category: 'email',
        confidence: 'high',
        start: 8,
        end: 24,
        redactionToken: '[REDACTED_EMAIL]',
      },
    ];

    expect(redactPiiFindings(text, findings, { includeReviewFindings: true }).text).toBe(
      'Contact [REDACTED_EMAIL]'
    );
  });
});
```

- [ ] **Step 3: Run detector/redactor tests and verify they fail**

Run:

```bash
cd frontend
npm run test:run -- src/test/pii/detector.test.js src/test/pii/redactor.test.js
```

Expected: FAIL because detector/redactor modules do not exist.

- [ ] **Step 4: Add rule configuration**

Create `frontend/src/pii/ruleConfig.js` with global and locale rules:

```js
export const SUPPORTED_PII_LOCALES = Object.freeze(['en', 'de', 'fr', 'es', 'nl']);

const DATE_PATTERN = /\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b/gu;
const ID_VALUE_PATTERN = /\b[A-Z]{0,4}[- ]?\d{4,12}[A-Z0-9-]*\b/giu;

export const GLOBAL_RULES = Object.freeze([
  {
    id: 'global.email',
    category: 'email',
    confidence: 'high',
    locales: ['*'],
    redactionToken: '[REDACTED_EMAIL]',
    enabled: true,
    patterns: [/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu],
  },
  {
    id: 'global.url',
    category: 'url',
    confidence: 'high',
    locales: ['*'],
    redactionToken: '[REDACTED_URL]',
    enabled: true,
    patterns: [/\bhttps?:\/\/[^\s<>"']+|\bwww\.[^\s<>"']+/giu],
  },
  {
    id: 'global.ipv4',
    category: 'ip_address',
    confidence: 'high',
    locales: ['*'],
    redactionToken: '[REDACTED_IP]',
    enabled: true,
    patterns: [/\b(?:\d{1,3}\.){3}\d{1,3}\b/gu],
  },
  {
    id: 'global.date',
    category: 'date',
    confidence: 'review',
    locales: ['*'],
    redactionToken: '[REDACTED_DATE]',
    enabled: true,
    patterns: [DATE_PATTERN],
  },
]);

export const LOCALE_RULES = Object.freeze({
  en: [
    {
      id: 'en.dob',
      category: 'dob',
      confidence: 'high',
      locales: ['en'],
      redactionToken: '[REDACTED_DOB]',
      enabled: true,
      patterns: [/(?:DOB|date of birth|born)\s*[:#-]?\s*(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})/giu],
    },
    {
      id: 'en.medical_record',
      category: 'medical_record',
      confidence: 'high',
      locales: ['en'],
      redactionToken: '[REDACTED_MRN]',
      enabled: true,
      patterns: [/(?:MRN|medical record number|patient ID|NHS number)\s*[:#-]?\s*[A-Z0-9 -]{4,24}/giu],
    },
  ],
  de: [
    {
      id: 'de.dob',
      category: 'dob',
      confidence: 'high',
      locales: ['de'],
      redactionToken: '[REDACTED_DOB]',
      enabled: true,
      patterns: [/(?:Geburtsdatum|geboren)\s*[:#-]?\s*(?:\d{1,2}[.]\d{1,2}[.]\d{2,4}|\d{4}-\d{2}-\d{2})/giu],
    },
    {
      id: 'de.kvnr',
      category: 'national_identifier',
      confidence: 'high',
      locales: ['de'],
      redactionToken: '[REDACTED_NATIONAL_ID]',
      enabled: true,
      patterns: [/(?:Krankenversichertennummer|Versichertennummer|KVNR)\s*[:#-]?\s*[A-Z]\d{9}/giu],
      validator: 'validateGermanKvnr',
    },
  ],
  fr: [
    {
      id: 'fr.nir',
      category: 'national_identifier',
      confidence: 'high',
      locales: ['fr'],
      redactionToken: '[REDACTED_NATIONAL_ID]',
      enabled: true,
      patterns: [/(?:NIR|numéro de sécurité sociale|securite sociale)\s*[:#-]?\s*[12][\d .-]{12,20}/giu],
      validator: 'validateFrenchNir',
    },
  ],
  es: [
    {
      id: 'es.dni_nie',
      category: 'national_identifier',
      confidence: 'high',
      locales: ['es'],
      redactionToken: '[REDACTED_NATIONAL_ID]',
      enabled: true,
      patterns: [/(?:DNI|NIE|NIF)\s*[:#-]?\s*(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])/giu],
      validator: 'validateSpanishDniNie',
    },
    {
      id: 'es.medical_record',
      category: 'medical_record',
      confidence: 'high',
      locales: ['es'],
      redactionToken: '[REDACTED_MRN]',
      enabled: true,
      patterns: [/(?:número de historia clínica|historia clínica|paciente id)\s*[:#-]?\s*[A-Z0-9 -]{4,24}/giu],
    },
  ],
  nl: [
    {
      id: 'nl.bsn',
      category: 'national_identifier',
      confidence: 'high',
      locales: ['nl'],
      redactionToken: '[REDACTED_NATIONAL_ID]',
      enabled: true,
      patterns: [/(?:BSN|burgerservicenummer)\s*[:#-]?\s*\d{9}/giu],
      validator: 'validateDutchBsn',
    },
  ],
});

export const ADDRESS_RULES = Object.freeze({
  de: /\b(?:straße|strasse|str\.|weg|platz|allee)\s+\d+\b.*\b\d{5}\b/giu,
  fr: /\b(?:rue|avenue|av\.|boulevard|bd|chemin|place)\s+[^,\n]+\d+.*\b\d{5}\b/giu,
  es: /\b(?:calle|c\/|avenida|avda\.|paseo|plaza)\s+[^,\n]+\d+.*\b\d{5}\b/giu,
  nl: /\b(?:straat|laan|weg|plein|adres)\s+[^,\n]*\d+[A-Z]?\b.*\b\d{4}\s?[A-Z]{2}\b/giu,
  en: /\b(?:street|st\.|road|rd\.|avenue|ave\.|drive|dr\.)\s+[^,\n]*\d+.*\b[A-Z0-9][A-Z0-9 -]{3,10}\b/giu,
});

export const MEDICAL_ID_VALUE_PATTERN = ID_VALUE_PATTERN;
```

- [ ] **Step 5: Implement detector**

Create `frontend/src/pii/detector.js`:

```js
import { findPhoneNumbersInText } from 'libphonenumber-js';
import { ADDRESS_RULES, GLOBAL_RULES, LOCALE_RULES, SUPPORTED_PII_LOCALES } from './ruleConfig';
import { VALIDATORS } from './validators';

function createSummary() {
  return { high: {}, review: {} };
}

function addSummary(summary, finding) {
  summary[finding.confidence][finding.category] =
    (summary[finding.confidence][finding.category] ?? 0) + 1;
}

function buildFinding({ rule, start, end, index }) {
  return {
    id: `${rule.id}-${index}`,
    ruleId: rule.id,
    category: rule.category,
    confidence: rule.confidence,
    start,
    end,
    redactionToken: rule.redactionToken,
  };
}

function isRuleEnabledForLocale(rule, locale) {
  return rule.enabled && (rule.locales.includes('*') || rule.locales.includes(locale));
}

function candidateForValidation(text, matchText) {
  const parts = String(matchText).split(/[:#-]/u);
  return parts[parts.length - 1] || text;
}

function applyRules(text, rules, locale) {
  const findings = [];
  for (const rule of rules) {
    if (!isRuleEnabledForLocale(rule, locale)) continue;
    for (const pattern of rule.patterns) {
      pattern.lastIndex = 0;
      let match;
      while ((match = pattern.exec(text)) !== null) {
        const matchText = match[0];
        const validator = rule.validator ? VALIDATORS[rule.validator] : null;
        if (validator && !validator(candidateForValidation(text, matchText))) continue;
        findings.push(buildFinding({ rule, start: match.index, end: match.index + matchText.length, index: findings.length }));
      }
    }
  }
  return findings;
}

function applyPhoneRule(text, locale) {
  const region = locale === 'de' ? 'DE' : locale === 'fr' ? 'FR' : locale === 'es' ? 'ES' : locale === 'nl' ? 'NL' : 'US';
  return findPhoneNumbersInText(text, region).map((match, index) => ({
    id: `global.phone-${index}`,
    ruleId: 'global.phone',
    category: 'phone',
    confidence: 'high',
    start: match.startsAt,
    end: match.endsAt,
    redactionToken: '[REDACTED_PHONE]',
  }));
}

function applyAddressRule(text, locale) {
  const pattern = ADDRESS_RULES[locale];
  if (!pattern) return [];
  const findings = [];
  pattern.lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    findings.push({
      id: `address-${locale}-${findings.length}`,
      ruleId: `${locale}.address`,
      category: 'address',
      confidence: 'high',
      start: match.index,
      end: match.index + match[0].length,
      redactionToken: '[REDACTED_ADDRESS]',
    });
  }
  return findings;
}

export function scanPii(text, { locale = 'en', includeGlobalRules = true } = {}) {
  const normalizedLocale = SUPPORTED_PII_LOCALES.includes(locale) ? locale : 'en';
  const source = String(text ?? '');
  const configuredRules = [
    ...(includeGlobalRules ? GLOBAL_RULES : []),
    ...(LOCALE_RULES[normalizedLocale] ?? []),
  ];
  const findings = [
    ...applyRules(source, configuredRules, normalizedLocale),
    ...applyPhoneRule(source, normalizedLocale),
    ...applyAddressRule(source, normalizedLocale),
  ].sort((a, b) => a.start - b.start || b.end - a.end);

  const summary = createSummary();
  findings.forEach((finding) => addSummary(summary, finding));
  return {
    hasFindings: findings.length > 0,
    findings,
    summary,
  };
}
```

- [ ] **Step 6: Implement redactor**

Create `frontend/src/pii/redactor.js`:

```js
function mergeFindings(findings, includeReviewFindings) {
  const selected = findings
    .filter((finding) => finding.confidence === 'high' || includeReviewFindings)
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const merged = [];
  for (const finding of selected) {
    const previous = merged[merged.length - 1];
    if (previous && finding.start < previous.end) {
      const previousLength = previous.end - previous.start;
      const findingLength = finding.end - finding.start;
      if (findingLength > previousLength) {
        merged[merged.length - 1] = finding;
      }
      continue;
    }
    merged.push(finding);
  }
  return merged;
}

function summarize(findings) {
  return findings.reduce(
    (summary, finding) => {
      summary[finding.confidence][finding.category] =
        (summary[finding.confidence][finding.category] ?? 0) + 1;
      return summary;
    },
    { high: {}, review: {} }
  );
}

export function redactPiiFindings(text, findings, { includeReviewFindings = false } = {}) {
  const source = String(text ?? '');
  const selected = mergeFindings(findings, includeReviewFindings);
  if (selected.length === 0) {
    return { text: source, changed: false, summary: { high: {}, review: {} } };
  }

  let cursor = 0;
  let redacted = '';
  for (const finding of selected) {
    redacted += source.slice(cursor, finding.start);
    redacted += finding.redactionToken;
    cursor = finding.end;
  }
  redacted += source.slice(cursor);

  return {
    text: redacted,
    changed: redacted !== source,
    summary: summarize(selected),
  };
}
```

- [ ] **Step 7: Add public PII exports**

Create `frontend/src/pii/index.js`:

```js
export { scanPii } from './detector';
export { redactPiiFindings } from './redactor';
export {
  normalizeIdentifierCandidate,
  validateDutchBsn,
  validateFrenchNir,
  validateGermanKvnr,
  validateSpanishDniNie,
} from './validators';
```

- [ ] **Step 8: Run detector/redactor tests and refine until passing**

Run:

```bash
cd frontend
npm run test:run -- src/test/pii/validators.test.js src/test/pii/detector.test.js src/test/pii/redactor.test.js
```

Expected: PASS. If a regex overmatches, adjust only `ruleConfig.js` and keep the API stable.

- [ ] **Step 9: Commit detector/redactor**

Run:

```bash
git add frontend/src/pii frontend/src/test/pii
git commit -m "feat(frontend): add local PII detector and redactor"
```

Expected: commit succeeds.

## Task 3: PII Review Dialog And I18n

**Files:**
- Create: `frontend/src/components/PiiReviewDialog.vue`
- Modify: `frontend/src/plugins/vuetify.js`
- Modify: `frontend/src/plugins/icons.js`
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/de.json`
- Modify: `frontend/src/locales/fr.json`
- Modify: `frontend/src/locales/es.json`
- Modify: `frontend/src/locales/nl.json`
- Create: `frontend/src/test/components/PiiReviewDialog.test.js`

- [ ] **Step 1: Write failing dialog tests**

Create `frontend/src/test/components/PiiReviewDialog.test.js`:

```js
import { describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import { createI18n } from 'vue-i18n';
import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import en from '../../locales/en.json';
import PiiReviewDialog from '../../components/PiiReviewDialog.vue';

const vuetify = createVuetify({ components, directives });
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } });

function mountDialog(props = {}) {
  return mount(PiiReviewDialog, {
    props: {
      modelValue: true,
      summary: { high: { email: 1, medical_record: 1 }, review: { date: 1 } },
      ...props,
    },
    global: { plugins: [vuetify, i18n] },
  });
}

describe('PiiReviewDialog', () => {
  it('shows category counts without raw snippets', () => {
    const wrapper = mountDialog();
    expect(wrapper.text()).toContain('Email');
    expect(wrapper.text()).toContain('Medical record');
    expect(wrapper.text()).toContain('Date');
    expect(wrapper.text()).not.toContain('jane@example.org');
  });

  it('emits cancel, redact, and continue actions', async () => {
    const wrapper = mountDialog();

    await wrapper.find('[data-testid="pii-cancel"]').trigger('click');
    await wrapper.find('[data-testid="pii-redact"]').trigger('click');
    await wrapper.find('[data-testid="pii-continue"]').trigger('click');

    expect(wrapper.emitted('cancel')).toHaveLength(1);
    expect(wrapper.emitted('redact')).toHaveLength(1);
    expect(wrapper.emitted('continue')).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run dialog tests and verify they fail**

Run:

```bash
cd frontend
npm run test:run -- src/test/components/PiiReviewDialog.test.js
```

Expected: FAIL because `PiiReviewDialog.vue` does not exist.

- [ ] **Step 3: Add English i18n keys**

In `frontend/src/locales/en.json`, add under `queryInterface`:

```json
"piiReview": {
  "title": "Possible identifiers detected",
  "description": "Phentrieve found possible identifiers before submitting this text.",
  "willRedact": "Will be redacted locally",
  "needsReview": "Needs review",
  "overrideNotice": "High-confidence identifiers will be redacted locally before submission. Other possible identifiers may remain if you continue.",
  "safetyAid": "This is a safety aid and cannot guarantee all PII is detected.",
  "cancel": "Review text",
  "redact": "Redact in text",
  "continue": "Continue with local redaction",
  "categories": {
    "email": "Email",
    "phone": "Phone",
    "url": "URL",
    "ip_address": "IP address",
    "date": "Date",
    "dob": "Date of birth",
    "medical_record": "Medical record",
    "accession_id": "Accession ID",
    "sample_id": "Sample ID",
    "national_identifier": "National identifier",
    "address": "Address",
    "person_name": "Name",
    "location": "Location",
    "organization": "Organization"
  }
}
```

Add equivalent keys with translated values to `de.json`, `fr.json`, `es.json`, and `nl.json`. Keep the key structure identical.

- [ ] **Step 4: Implement dialog**

Create `frontend/src/components/PiiReviewDialog.vue`:

```vue
<template>
  <v-dialog
    v-model="dialogVisible"
    max-width="560"
    persistent
    role="alertdialog"
    aria-labelledby="pii-review-title"
  >
    <v-card>
      <v-card-title id="pii-review-title" class="d-flex align-center text-warning">
        <v-icon class="mr-2" aria-hidden="true">mdi-shield-alert-outline</v-icon>
        {{ $t('queryInterface.piiReview.title') }}
      </v-card-title>

      <v-card-text>
        <p class="text-body-1 text-high-emphasis mb-3">
          {{ $t('queryInterface.piiReview.description') }}
        </p>

        <v-alert type="warning" variant="tonal" density="comfortable" class="mb-4">
          {{ $t('queryInterface.piiReview.overrideNotice') }}
        </v-alert>

        <section v-if="hasHighFindings" class="mb-4">
          <h3 class="text-subtitle-2 mb-2">
            {{ $t('queryInterface.piiReview.willRedact') }}
          </h3>
          <div class="d-flex flex-wrap ga-2">
            <v-chip
              v-for="item in highCategoryItems"
              :key="`high-${item.category}`"
              color="warning"
              variant="tonal"
              size="small"
            >
              {{ getCategoryLabel(item.category) }}: {{ item.count }}
            </v-chip>
          </div>
        </section>

        <section v-if="hasReviewFindings" class="mb-4">
          <h3 class="text-subtitle-2 mb-2">
            {{ $t('queryInterface.piiReview.needsReview') }}
          </h3>
          <div class="d-flex flex-wrap ga-2">
            <v-chip
              v-for="item in reviewCategoryItems"
              :key="`review-${item.category}`"
              color="info"
              variant="tonal"
              size="small"
            >
              {{ getCategoryLabel(item.category) }}: {{ item.count }}
            </v-chip>
          </div>
        </section>

        <p class="text-caption text-medium-emphasis mb-0">
          {{ $t('queryInterface.piiReview.safetyAid') }}
        </p>
      </v-card-text>

      <v-card-actions class="px-6 pb-4 flex-wrap ga-2">
        <v-spacer />
        <v-btn data-testid="pii-cancel" variant="text" @click="$emit('cancel')">
          {{ $t('queryInterface.piiReview.cancel') }}
        </v-btn>
        <v-btn data-testid="pii-redact" variant="tonal" color="primary" @click="$emit('redact')">
          {{ $t('queryInterface.piiReview.redact') }}
        </v-btn>
        <v-btn
          data-testid="pii-continue"
          variant="elevated"
          color="warning"
          @click="$emit('continue')"
        >
          {{ $t('queryInterface.piiReview.continue') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script>
function toCategoryItems(record) {
  return Object.entries(record ?? {}).map(([category, count]) => ({ category, count }));
}

export default {
  name: 'PiiReviewDialog',
  props: {
    modelValue: {
      type: Boolean,
      default: false,
    },
    summary: {
      type: Object,
      default: () => ({ high: {}, review: {} }),
    },
  },
  emits: ['update:modelValue', 'cancel', 'redact', 'continue'],
  computed: {
    dialogVisible: {
      get() {
        return this.modelValue;
      },
      set(value) {
        this.$emit('update:modelValue', value);
      },
    },
    highCategoryItems() {
      return toCategoryItems(this.summary.high);
    },
    reviewCategoryItems() {
      return toCategoryItems(this.summary.review);
    },
    hasHighFindings() {
      return this.highCategoryItems.length > 0;
    },
    hasReviewFindings() {
      return this.reviewCategoryItems.length > 0;
    },
  },
  methods: {
    getCategoryLabel(category) {
      return this.$t(`queryInterface.piiReview.categories.${category}`);
    },
  },
};
</script>
```

- [ ] **Step 5: Register icon**

In `frontend/src/plugins/icons.js`, add:

```js
'mdi-shield-alert-outline',
```

`VAlert`, `VChip`, `VDialog`, `VCard`, `VCardTitle`, `VCardText`, `VCardActions`, `VSpacer`, `VBtn`, and `VIcon` are already registered in `frontend/src/plugins/vuetify.js`; no new Vuetify component registration should be needed.

- [ ] **Step 6: Run dialog and i18n checks**

Run:

```bash
cd frontend
npm run test:run -- src/test/components/PiiReviewDialog.test.js
npm run i18n:check
```

Expected: PASS.

- [ ] **Step 7: Commit dialog and i18n**

Run:

```bash
git add frontend/src/components/PiiReviewDialog.vue frontend/src/test/components/PiiReviewDialog.test.js frontend/src/locales frontend/src/plugins/icons.js
git commit -m "feat(frontend): add PII review dialog"
```

Expected: commit succeeds.

## Task 4: QueryInterface Integration And No-Network Gate

**Files:**
- Modify: `frontend/src/components/QueryInterface.vue`
- Modify: `frontend/src/composables/useQueryInterfaceController.js`
- Modify: `frontend/src/test/components/QueryInterface.test.js`

- [ ] **Step 1: Write failing QueryInterface flow tests**

Append to `frontend/src/test/components/QueryInterface.test.js`:

```js
it('opens PII review before query-mode network submission', async () => {
  const wrapper = await mountQueryInterface();
  await flushPromises();

  await setVmState(wrapper, {
    queryText: 'MRN: AB-123456 with seizures',
    forceEndpointMode: 'query',
  });

  await wrapper.vm.submitQuery();

  expect(PhentrieveService.queryHpo).not.toHaveBeenCalled();
  expect(wrapper.findComponent({ name: 'PiiReviewDialog' }).exists()).toBe(true);
  expect(wrapper.vm.piiReviewDialogVisible).toBe(true);
});

it('opens PII review before full-text network submission', async () => {
  const wrapper = await mountQueryInterface();
  await flushPromises();

  await setVmState(wrapper, {
    queryText: 'DOB: 12/03/1980. Patient had seizures.',
    forceEndpointMode: 'textProcess',
  });

  await wrapper.vm.submitQuery();

  expect(PhentrieveService.processText).not.toHaveBeenCalled();
  expect(wrapper.vm.piiReviewDialogVisible).toBe(true);
});

it('continues with local redaction and submits redacted query text', async () => {
  const wrapper = await mountQueryInterface();
  await flushPromises();

  await setVmState(wrapper, {
    queryText: 'MRN: AB-123456 with seizures',
    forceEndpointMode: 'query',
  });

  await wrapper.vm.submitQuery();
  await wrapper.vm.continueWithPiiRedaction();

  expect(PhentrieveService.queryHpo).toHaveBeenCalledWith(
    expect.objectContaining({ text: expect.stringContaining('[REDACTED_MRN]') })
  );
});

it('redacts in the input without submitting', async () => {
  const wrapper = await mountQueryInterface();
  await flushPromises();

  await setVmState(wrapper, {
    queryText: 'Email jane@example.org and seizures',
    forceEndpointMode: 'query',
  });

  await wrapper.vm.submitQuery();
  await wrapper.vm.redactPiiInInput();

  expect(wrapper.vm.queryText).toContain('[REDACTED_EMAIL]');
  expect(PhentrieveService.queryHpo).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Run QueryInterface tests and verify they fail**

Run:

```bash
cd frontend
npm run test:run -- src/test/components/QueryInterface.test.js
```

Expected: FAIL because `PiiReviewDialog` is not integrated and controller has no PII gate.

- [ ] **Step 3: Add PII dialog state and handlers to QueryInterface**

Modify `frontend/src/components/QueryInterface.vue`:

```vue
<PiiReviewDialog
  v-model="piiReviewDialogVisible"
  :summary="pendingPiiSubmission?.scanResult?.summary || { high: {}, review: {} }"
  @cancel="cancelPiiReview"
  @redact="redactPiiInInput"
  @continue="continueWithPiiRedaction"
/>
```

Add imports:

```js
import PiiReviewDialog from './PiiReviewDialog.vue';
```

Register component:

```js
components: {
  ResultsDisplay,
  ConversationSkeleton,
  AdvancedOptionsPanel,
  PhenotypeCollectionPanel,
  FullTextResponseReceipt,
  PiiReviewDialog,
},
```

Add data:

```js
piiReviewDialogVisible: false,
pendingPiiSubmission: null,
```

Add controller context hooks:

```js
openPiiReview(pendingSubmission) {
  vm.pendingPiiSubmission = pendingSubmission;
  vm.piiReviewDialogVisible = true;
},
```

Add methods:

```js
cancelPiiReview() {
  this.piiReviewDialogVisible = false;
  this.pendingPiiSubmission = null;
},
redactPiiInInput() {
  return this.queryInterfaceController.redactPiiInInput();
},
continueWithPiiRedaction() {
  return this.queryInterfaceController.continueWithPiiRedaction();
},
```

- [ ] **Step 4: Refactor controller submission flow**

Modify `frontend/src/composables/useQueryInterfaceController.js`:

```js
import { redactPiiFindings, scanPii } from '../pii';
```

Add helper:

```js
function getSanitizedSubmissionLog({ mode, text, latestState, piiScanResult, redactionApplied }) {
  return {
    mode,
    textLength: text.length,
    selectedLanguage: latestState.selectedLanguage,
    extractionBackend: mode === 'textProcess' ? latestState.textProcessOptions.extractionBackend : null,
    piiSummary: piiScanResult?.summary ?? { high: {}, review: {} },
    piiRedactionApplied: Boolean(redactionApplied),
  };
}
```

Split the existing API-submission body into an internal function:

```js
async function submitQueryText({ currentQuery, useTextProcessMode, isAutoSubmit, piiScanResult = null, redactionApplied = false }) {
  const context = getContextOrThrow(getContext);
  const latestState = context.getState();

  context.setState({
    isLoading: true,
    shouldScrollToTop: true,
    userHasScrolled: false,
  });

  const queryId = context.conversationStore.addQuery({
    query: currentQuery,
    loading: true,
    type: useTextProcessMode ? 'textProcess' : 'query',
  });

  if (useTextProcessMode) {
    context.setExpandedUserNote(queryId, true);
  }

  if (!isAutoSubmit) {
    context.setState({ queryText: '' });
  }

  try {
    let response;
    logService.info(
      'Submitting text request',
      getSanitizedSubmissionLog({
        mode: useTextProcessMode ? 'textProcess' : 'query',
        text: currentQuery,
        latestState,
        piiScanResult,
        redactionApplied,
      })
    );

    if (useTextProcessMode) {
      response = await service.processText({
        text: currentQuery,
        extractionBackend: latestState.textProcessOptions.extractionBackend,
        llmModel: latestState.textProcessOptions.llmModel,
        llmMode: latestState.textProcessOptions.llmMode,
        language: latestState.selectedLanguage,
        chunkingStrategy: latestState.chunkingStrategy,
        windowSize: latestState.windowSize,
        stepSize: latestState.stepSize,
        splitThreshold: latestState.splitThreshold,
        minSegmentLength: latestState.minSegmentLength,
        semanticModelForChunking: latestState.semanticModelForChunking || latestState.selectedModel,
        retrievalModelForTextProcess: latestState.retrievalModelForTextProcess || latestState.selectedModel,
        trustRemoteCode: true,
        chunkRetrievalThreshold: latestState.chunkRetrievalThreshold,
        numResultsPerChunk: latestState.numResultsPerChunk,
        noAssertionDetectionForTextProcess: latestState.noAssertionDetectionForTextProcess,
        assertionPreferenceForTextProcess: latestState.assertionPreferenceForTextProcess,
        aggregatedTermConfidence: latestState.aggregatedTermConfidence,
        topTermPerChunkForAggregation: latestState.topTermPerChunkForAggregation,
        includeDetails: latestState.includeDetails,
      });
    } else {
      response = await service.queryHpo({
        text: currentQuery,
        model_name: latestState.selectedModel,
        language: latestState.selectedLanguage,
        num_results: latestState.numResults,
        similarity_threshold: latestState.similarityThreshold,
        query_assertion_language: latestState.selectedLanguage,
        detect_query_assertion: true,
        include_details: latestState.includeDetails,
      });
    }

    context.conversationStore.updateQueryResponse(queryId, response);
    if (useTextProcessMode) {
      context.fullTextWorkspaceStore.initializeTurn(queryId);
      context.fullTextWorkspaceStore.setExpanded(queryId, true);
    }
  } catch (error) {
    context.conversationStore.updateQueryResponse(queryId, null, error);
    logService.error('Error submitting query/processing text', error);
  } finally {
    context.setState({ isLoading: false });
  }
}
```

Update public `submitQuery()` to scan before calling the internal function:

```js
async function submitQuery(isAutoSubmit = false) {
  const context = getContextOrThrow(getContext);
  const state = context.getState();
  const queryTextTrimmed = typeof state.queryText === 'string' ? state.queryText.trim() : '';

  if (!queryTextTrimmed) {
    logService.warn('Empty query submission prevented');
    return;
  }

  const useTextProcessMode = state.isTextProcessModeActive;
  const scanResult = scanPii(queryTextTrimmed, { locale: state.selectedLanguage || 'en' });
  if (scanResult.hasFindings) {
    context.openPiiReview({
      text: queryTextTrimmed,
      useTextProcessMode,
      isAutoSubmit,
      scanResult,
    });
    logService.info('PII review required before submission', {
      mode: useTextProcessMode ? 'textProcess' : 'query',
      textLength: queryTextTrimmed.length,
      piiSummary: scanResult.summary,
    });
    return;
  }

  return submitQueryText({
    currentQuery: queryTextTrimmed,
    useTextProcessMode,
    isAutoSubmit,
    piiScanResult: scanResult,
    redactionApplied: false,
  });
}
```

Add controller methods:

```js
async function continueWithPiiRedaction() {
  const context = getContextOrThrow(getContext);
  const pending = context.getState().pendingPiiSubmission;
  if (!pending) return;

  const redaction = redactPiiFindings(pending.text, pending.scanResult.findings, {
    includeReviewFindings: false,
  });
  context.setState({ piiReviewDialogVisible: false, pendingPiiSubmission: null });

  return submitQueryText({
    currentQuery: redaction.text,
    useTextProcessMode: pending.useTextProcessMode,
    isAutoSubmit: pending.isAutoSubmit,
    piiScanResult: pending.scanResult,
    redactionApplied: redaction.changed,
  });
}

function redactPiiInInput() {
  const context = getContextOrThrow(getContext);
  const pending = context.getState().pendingPiiSubmission;
  if (!pending) return;

  const redaction = redactPiiFindings(pending.text, pending.scanResult.findings, {
    includeReviewFindings: true,
  });
  context.setState({
    queryText: redaction.text,
    piiReviewDialogVisible: false,
    pendingPiiSubmission: null,
  });
}
```

Return the methods:

```js
return {
  fetchAvailableModels,
  setFallbackModel,
  applyUrlParametersAndAutoSubmit,
  submitQuery,
  continueWithPiiRedaction,
  redactPiiInInput,
};
```

Ensure `getState()` in `QueryInterface.vue` returns `pendingPiiSubmission`.

- [ ] **Step 5: Preserve existing URL cleanup behavior**

Move the existing `autoSubmit` route cleanup into a helper and call it from `submitQueryText()` finally block:

```js
function clearAutoSubmitQueryParamIfNeeded(context, isAutoSubmit) {
  if (!isAutoSubmit && hasOwnQueryParam(context.routeQuery, 'autoSubmit')) {
    const newRouteQuery = { ...context.routeQuery };
    delete newRouteQuery.autoSubmit;
    Promise.resolve(context.replaceRouteQuery(newRouteQuery)).catch((error) => {
      if (error.name !== 'NavigationDuplicated' && error.name !== 'NavigationCancelled') {
        logService.warn('Error clearing autoSubmit from URL:', error);
      }
    });
  }
}
```

Call:

```js
clearAutoSubmitQueryParamIfNeeded(context, isAutoSubmit);
```

- [ ] **Step 6: Run QueryInterface tests**

Run:

```bash
cd frontend
npm run test:run -- src/test/components/QueryInterface.test.js src/test/components/PiiReviewDialog.test.js src/test/pii
```

Expected: PASS.

- [ ] **Step 7: Commit integration**

Run:

```bash
git add frontend/src/components/QueryInterface.vue frontend/src/composables/useQueryInterfaceController.js frontend/src/test/components/QueryInterface.test.js
git commit -m "feat(frontend): gate submissions with local PII review"
```

Expected: commit succeeds.

## Task 5: Logging Tests And Service Privacy Cleanup

**Files:**
- Modify: `frontend/src/test/components/QueryInterface.test.js`
- Modify: `frontend/src/services/PhentrieveService.js` only if tests reveal raw text logging there.
- Modify: `frontend/src/test/services/PhentrieveService.test.js` only if service behavior changes.

- [ ] **Step 1: Add a log privacy regression test**

In `frontend/src/test/components/QueryInterface.test.js`, import the mocked log service:

```js
import { logService } from '../../services/logService';
```

Add:

```js
it('does not log raw text or detected snippets during PII review and submit', async () => {
  const wrapper = await mountQueryInterface();
  await flushPromises();

  await setVmState(wrapper, {
    queryText: 'Email jane@example.org with seizures',
    forceEndpointMode: 'query',
  });

  await wrapper.vm.submitQuery();
  await wrapper.vm.continueWithPiiRedaction();

  const serializedLogs = JSON.stringify([
    ...logService.info.mock.calls,
    ...logService.debug.mock.calls,
    ...logService.warn.mock.calls,
    ...logService.error.mock.calls,
  ]);

  expect(serializedLogs).not.toContain('jane@example.org');
  expect(serializedLogs).not.toContain('Email jane');
  expect(serializedLogs).toContain('piiSummary');
});
```

- [ ] **Step 2: Run the log privacy test and verify failure or pass**

Run:

```bash
cd frontend
npm run test:run -- src/test/components/QueryInterface.test.js -t "does not log raw text"
```

Expected: PASS if Task 4 removed raw request-object logging. If it fails, remove remaining raw text from controller logs.

- [ ] **Step 3: Inspect service logs for raw text**

Check `frontend/src/services/PhentrieveService.js`. `queryHpo()` and `processText()` should log only `textLength`, request size, model/backend metadata, and response sizes. Do not log full payloads.

If raw text appears, replace it with:

```js
logService.info('Querying HPO API', {
  textLength: queryData.text?.length || 0,
  numResults: queryData.num_results ?? queryData.numResults,
  model: queryData.model_name ?? queryData.modelName,
});
```

and:

```js
logService.info('Calling Text Processing API', {
  requestSize: JSON.stringify(normalizedPayload).length,
  textLength: normalizedPayload.text?.length || 0,
  backend: normalizedPayload.extraction_backend,
  llmTarget:
    normalizedPayload.extraction_backend === 'llm'
      ? 'server-owned'
      : normalizedPayload.retrieval_model_name,
});
```

- [ ] **Step 4: Run service and component tests**

Run:

```bash
cd frontend
npm run test:run -- src/test/services/PhentrieveService.test.js src/test/services/PhentrieveService.researchUse.test.js src/test/components/QueryInterface.test.js
```

Expected: PASS.

- [ ] **Step 5: Commit logging cleanup**

Run:

```bash
git add frontend/src/test/components/QueryInterface.test.js frontend/src/services/PhentrieveService.js frontend/src/test/services/PhentrieveService.test.js frontend/src/test/services/PhentrieveService.researchUse.test.js
git commit -m "test(frontend): verify PII guard avoids raw text logs"
```

Expected: commit succeeds. If no service files changed, omit them from `git add`.

## Task 6: Documentation And Planning Index

**Files:**
- Modify: `docs/compliance/privacy-and-llm-processing.md`
- Modify: `.planning/README.md`

- [ ] **Step 1: Update privacy documentation**

In `docs/compliance/privacy-and-llm-processing.md`, add after `## User Data Guidance`:

```markdown
## Browser-Side PII Warning

The frontend includes a local browser-side warning and redaction aid for English,
German, French, Spanish, and Dutch submissions. Before Query or Full Text input
is sent to the API, the browser checks for common direct identifiers such as
contact details, record numbers, national identifier patterns, exact date
labels, and address-like text.

High-confidence identifiers are redacted locally before submission when the user
continues. Lower-confidence findings require explicit acknowledgement and may
remain if the user decides to proceed.

This feature is a safety aid. It is not a guarantee that all PII or PHI is
detected, and it does not make submitted text HIPAA Safe Harbor de-identified,
GDPR-anonymised, or suitable for identifiable patient data in public demos.
Users should review and remove identifiers before submitting research text.
```

- [ ] **Step 2: Update planning index**

In `.planning/README.md`, add under `Current Active Work`:

```markdown
- `completed/2026-04-30-local-browser-pii-guard-implementation-plan.md` -
  implementation plan for GitHub issue #249, covering local browser-side PII
  detection and redaction before Query and Full Text submissions.
```

- [ ] **Step 3: Run documentation-adjacent checks**

Run:

```bash
make frontend-i18n-check
```

Expected: PASS.

- [ ] **Step 4: Commit docs**

Run:

```bash
git add docs/compliance/privacy-and-llm-processing.md .planning/README.md
git commit -m "docs: document local PII guard behavior"
```

Expected: commit succeeds.

## Task 7: Final Frontend And Repository Verification

**Files:**
- No new files unless verification reveals defects.

- [ ] **Step 1: Run focused frontend tests**

Run:

```bash
make frontend-test-ci
```

Expected: PASS.

- [ ] **Step 2: Run frontend build**

Run:

```bash
make frontend-build-ci
```

Expected: PASS.

- [ ] **Step 3: Run repository-required checks**

Run:

```bash
make check
make typecheck-fast
make test
```

Expected: all PASS.

- [ ] **Step 4: Inspect git diff for privacy regressions**

Run:

```bash
git diff --check
rg -n "Sending to /query API|Sending to /text/process API|queryData|textProcessData|jane@example|AB-123456" frontend/src frontend/src/test docs .planning
```

Expected:

- `git diff --check` produces no output.
- `rg` finds only test fixtures or planned documentation references, not runtime raw-payload logging.

- [ ] **Step 5: Commit final fixes if needed**

If verification required fixes, run:

```bash
git add <changed-files>
git commit -m "fix(frontend): stabilize local PII guard verification"
```

Expected: commit succeeds. If no fixes were needed, skip this step.

## Self-Review Checklist

- Spec coverage: the plan implements local-only detection, both Query and Full
  Text modes, mandatory high-confidence redaction, override acknowledgement,
  five language overlays, config-driven expansion, tests, logs, and docs.
- Placeholder scan: no steps use placeholder markers or unspecified error handling.
- Type consistency: public functions are `scanPii()` and `redactPiiFindings()`;
  controller methods are `continueWithPiiRedaction()` and `redactPiiInInput()`;
  dialog events are `cancel`, `redact`, and `continue`.
- Scope check: backend enforcement and cloud PII APIs remain out of scope.

## Execution Options

Plan complete and saved to `.planning/completed/2026-04-30-local-browser-pii-guard-implementation-plan.md`.

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
