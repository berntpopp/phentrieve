# Untitled Local Person Name PII Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable browser-only review-confidence detection for untitled person names.

**Architecture:** Keep deterministic detection in `frontend/src/pii/`. Add config data in `ruleConfig.js`, scanning logic in `detector.js`, and tests in existing Vitest suites. No network or model dependency is introduced.

**Tech Stack:** Vue 3 frontend, Vitest, existing PII detector/redactor modules.

---

### Task 1: Add Failing Detector Tests

**Files:**
- Modify: `frontend/src/test/pii/detector.test.js`

- [ ] Add tests for untitled names:
  - `Bernt Popp ist dumm` with locale `en`
  - `Jean Dupont a des crises` with locale `fr`
  - `María García tiene convulsiones` with locale `es`
  - `Jan van Dijk heeft aanvallen` with locale `nl`

- [ ] Add false-positive tests:
  - `Pectus Carinatum was noted.`
  - `Down Syndrome was discussed.`
  - `BioLORD Model returned results.`
  - `HP:0002779 Tracheomalacia`

- [ ] Run:
  `npm run test:run -- src/test/pii/detector.test.js -t "untitled"`

- [ ] Expected: FAIL because untitled name detection does not exist yet.

### Task 2: Implement Config-Driven Untitled Name Detection

**Files:**
- Modify: `frontend/src/pii/ruleConfig.js`
- Modify: `frontend/src/pii/detector.js`

- [ ] Add exported config for possible names:
  - `UNTITLED_NAME_RULE_CONFIG`
  - locale particles
  - locale context words
  - blocked phrase words
  - minimum score

- [ ] Add detector helpers:
  - extract capitalized-token candidate spans
  - score candidates from config
  - reject blocked/domain/identifier candidates
  - emit `person_name` review findings only

- [ ] Run:
  `npm run test:run -- src/test/pii/detector.test.js -t "untitled"`

- [ ] Expected: PASS.

### Task 3: Add Submission-Guard Tests

**Files:**
- Modify: `frontend/src/test/components/QueryInterface.test.js`

- [ ] Add Query mode test that `Bernt Popp ist dumm` opens PII review and does not call `queryHpo`.
- [ ] Add Full Text mode test that `Bernt Popp ist dumm` opens PII review and does not call `processText`.
- [ ] Run:
  `npm run test:run -- src/test/components/QueryInterface.test.js -t "untitled name"`
- [ ] Expected: PASS after Task 2.

### Task 4: Focused Verification and Commit

**Files:**
- All modified frontend PII files and planning files.

- [ ] Run:
  `npm run test:run -- src/test/pii src/test/components/QueryInterface.test.js`
- [ ] Run:
  `npm run lint -- src/pii/detector.js src/pii/ruleConfig.js src/test/pii/detector.test.js src/test/components/QueryInterface.test.js`
- [ ] Run:
  `git diff --check`
- [ ] Commit:
  `git commit -m "feat(frontend): detect untitled local person names"`
