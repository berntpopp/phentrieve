# i18n Translation Validation Plan

**Issue:** Ensure translation file completeness and congruence across all locales
**Status:** Planning
**Priority:** Medium
**Complexity:** Low
**Estimated Effort:** 2-3 hours

---

## Executive Summary

The frontend supports 5 locales (en, de, es, fr, nl) with ~250 translation keys each. Currently, there's **no automated validation** to ensure:
- All locales have the same keys (congruence)
- Parameter placeholders (e.g., `{id}`, `{label}`) match across locales
- No missing or unused translation keys

**Recent Issue:** Missing i18n keys in `en.json` caused runtime warnings (fixed in commit ad298e5), highlighting the need for automated validation.

**Solution:** Implement automated i18n validation using industry-standard tools with minimal complexity.

---

## Design Principles

This plan adheres to:
- ✅ **KISS**: Simple solution with minimal commands (2, not 5)
- ✅ **DRY**: Single validation script with modular functions (not multiple files)
- ✅ **SOLID**: Single responsibility per function, extensible design
- ✅ **Modular**: Internal functions can be tested and extended independently

---

## Current State Analysis

### Locale Files Structure

```
frontend/src/locales/
├── en.json  (18 KB, ~247 keys) - Base locale
├── de.json  (20 KB, ~247 keys) - German
├── es.json  (20 KB, ~247 keys) - Spanish
├── fr.json  (20 KB, ~247 keys) - French
└── nl.json  (19 KB, ~247 keys) - Dutch
```

### Known Issues

1. **No structural validation**: Keys can be added to one locale and forgotten in others
2. **No parameter validation**: `{param}` placeholders can mismatch (e.g., `{id}` vs `{hpoId}`)
3. **No usage validation**: Can't detect unused keys or missing translations
4. **Manual review required**: Time-consuming and error-prone

---

## Proposed Solution

### Architecture (KISS-Compliant)

```
┌─────────────────────────────────────┐
│   make frontend-i18n-check          │  ← User runs this
└──────────────┬──────────────────────┘
               │
               ├─→ vue-i18n-extract (npm package)
               │   ├─ Scans .vue/.js/.ts files
               │   ├─ Finds $t() calls
               │   └─ Detects missing/unused keys
               │
               └─→ validate-i18n.js (custom script)
                   ├─ checkStructureCongruence()
                   │  └─ All locales have same keys
                   ├─ checkParameterConsistency()
                   │  └─ {param} placeholders match
                   └─ formatReport()
                      └─ Beautiful terminal output
```

**Key Decision: Single Script vs Multiple Files**
- ❌ **Rejected**: Separate `validate-structure.js` + `validate-params.js` (DRY violation)
- ✅ **Adopted**: One `validate-i18n.js` with internal modular functions (KISS + DRY)

---

## Implementation Details

### 1. Install Dependency

**Package:** `vue-i18n-extract`
- Industry standard for Vue i18n validation
- 500K+ downloads/month
- Actively maintained
- Zero config required

```bash
npm install --save-dev vue-i18n-extract
```

**Size Impact:** ~2 MB (dev dependency only)

### 2. Create Validation Script

**File:** `frontend/scripts/validate-i18n.js`

**Responsibilities:**
1. Load all locale JSON files
2. Extract and flatten keys
3. Compare structures across locales
4. Validate parameter consistency
5. Format and display results

**Modular Functions:**

```javascript
// Single Responsibility: Each function does ONE thing

function loadLocales(localesDir) {
  // Returns: { en: {...}, de: {...}, ... }
}

function flattenKeys(obj, prefix = '') {
  // Converts nested object to flat array: ['key1', 'key1.nested', ...]
  // Single responsibility: Key flattening only
}

function checkStructureCongruence(locales) {
  // Returns: { locale: { missing: [...], extra: [...] } }
  // Single responsibility: Structure validation only
}

function extractParameters(value) {
  // Finds {param} placeholders using regex
  // Returns: Set(['id', 'label', ...])
  // Single responsibility: Parameter extraction only
}

function checkParameterConsistency(locales) {
  // Returns: [{ locale, key, expected: [...], actual: [...] }]
  // Single responsibility: Parameter validation only
}

function formatReport(structureIssues, paramIssues) {
  // Beautiful colored terminal output
  // Returns: exit code (0 = success, 1 = failure)
  // Single responsibility: Reporting only
}

function main() {
  // Orchestrates all checks
  // CLI entry point
}
```

**Error Handling:**
- Graceful failure if locale file missing
- Clear error messages for JSON parse errors
- Exit code 0 (success) or 1 (failure) for CI/CD

### 3. Add npm Scripts

**File:** `frontend/package.json`

```json
{
  "scripts": {
    "i18n:check": "node scripts/validate-i18n.js && vue-i18n-extract --vueFiles './src/**/*.{vue,js,ts}' --languageFiles './src/locales/**/*.json'",
    "i18n:report": "vue-i18n-extract --output json --vueFiles './src/**/*.{vue,js,ts}' --languageFiles './src/locales/**/*.json' > i18n-report.json"
  }
}
```

**Design Decision: Chained Commands**
- `&&` ensures custom validation runs before vue-i18n-extract
- If structure checks fail, don't run expensive extraction
- Fails fast for quicker feedback

### 4. Add Makefile Targets

**File:** `Makefile` (frontend section)

```makefile
frontend-i18n-check: ## Validate i18n translation completeness and congruence
	@echo "🌍 Validating i18n translations..."
	cd frontend && npm run i18n:check
	@echo "✅ i18n validation complete"

frontend-i18n-report: ## Generate detailed i18n validation report (JSON)
	@echo "📊 Generating i18n validation report..."
	cd frontend && npm run i18n:report
	@echo "✅ Report saved to: frontend/i18n-report.json"
```

**KISS Principle Applied:**
- Only 2 commands (not 5)
- Clear, single-purpose targets
- Minimal abstraction

---

## User Experience

### Success Case

```bash
$ make frontend-i18n-check

🌍 Validating i18n translations...

✓ Checking locale structure congruence...
  ✓ en.json: 247 keys
  ✓ de.json: 247 keys
  ✓ es.json: 247 keys
  ✓ fr.json: 247 keys
  ✓ nl.json: 247 keys

✓ Validating parameter consistency...
  ✓ All {param} placeholders match across locales

✓ Checking translation coverage...
  ✓ 0 missing translations
  ✓ 0 unused keys

✅ All i18n validation checks passed!
```

### Failure Case

```bash
$ make frontend-i18n-check

🌍 Validating i18n translations...

❌ Locale Structure Issues:

en.json - Missing keys (present in other locales):
  - resultsDisplay.exportTooltip
  - queryInterface.advancedSettings.maxResults

de.json - Extra keys (not in other locales):
  - deprecated.oldFeatureLabel

❌ Parameter Consistency Issues:

Key: resultsDisplay.addToCollectionAriaLabel
  en.json: {label}, {id} ✓
  de.json: {id}, {label}, {extra} ✗ Mismatch!
  Expected: {label}, {id}

❌ Translation Coverage Issues:

Missing keys (used in code but not in translations):
  - ResultsDisplay.vue:150 → resultsDisplay.newFeature

Unused keys (in translations but not in code):
  - resultsDisplay.deprecatedTooltip

✗ i18n validation failed with 6 issues
```

---

## Testing Strategy

### Manual Testing

```bash
# Test with current locale files (should pass)
make frontend-i18n-check

# Temporarily break a locale file to verify detection
# Remove a key from en.json
make frontend-i18n-check  # Should fail with clear error

# Restore and verify
git restore frontend/src/locales/en.json
make frontend-i18n-check  # Should pass
```

### Integration Testing

Add to development workflow:
```bash
# Before committing i18n changes
make frontend-i18n-check
git add frontend/src/locales/
git commit -m "feat: Add new translation keys"
```

### CI/CD Integration (Optional Future Enhancement)

**File:** `.github/workflows/ci.yml`

```yaml
- name: Validate i18n translations
  working-directory: ./frontend
  run: npm run i18n:check
```

**Decision: NOT implemented initially**
- Rationale: Start simple, add CI later if needed
- YAGNI principle (You Ain't Gonna Need It)

---

## Benefits

### Immediate

1. ✅ **Prevent runtime errors**: Catch missing keys before deployment
2. ✅ **Maintain consistency**: All locales stay synchronized
3. ✅ **Save time**: Automated vs manual review (5 min → 2 sec)
4. ✅ **Clear feedback**: Actionable error messages

### Long-term

1. ✅ **Scalability**: Easy to add 6th, 7th locale
2. ✅ **Confidence**: Refactor without fear of breaking translations
3. ✅ **Documentation**: Script self-documents translation structure
4. ✅ **Onboarding**: New devs can validate their changes instantly

---

## Edge Cases & Limitations

### Handled

- ✅ Nested translation keys (e.g., `queryInterface.tooltips.advancedOptions`)
- ✅ Dynamic parameters with different orders (e.g., `{id}, {label}` vs `{label}, {id}`)
- ✅ Special characters in translations
- ✅ Empty string values (warns but doesn't fail)

### Not Handled (Out of Scope)

- ❌ **Translation quality**: Doesn't validate if German translation is correct
- ❌ **Pluralization**: vue-i18n plural forms not validated (future enhancement)
- ❌ **HTML in translations**: Doesn't check for XSS vulnerabilities
- ❌ **RTL languages**: Not applicable (no Arabic/Hebrew locales)

**Rationale:** Focus on structural validation first. Quality checks are separate concern.

---

## Migration Path

### Phase 1: Initial Setup (This Plan)
1. Install `vue-i18n-extract`
2. Create `validate-i18n.js`
3. Add Makefile commands
4. Run initial validation
5. Fix any detected issues

### Phase 2: Developer Adoption (Optional)
1. Add to CONTRIBUTING.md
2. Mention in pre-commit checklist
3. Train team on usage

### Phase 3: CI/CD Integration (Optional Future)
1. Add to GitHub Actions workflow
2. Require passing validation for PRs
3. Generate reports on merge to main

**Current Plan: Phase 1 Only**
- Keep it simple
- Prove value before expanding

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| False positives (flags correct translations as errors) | Medium | Low | Comprehensive testing before relying on it |
| Developers skip validation | High | Medium | Make it fast (<2s), easy (`make` command) |
| Maintenance burden (script breaks) | Low | Low | Use stable tool (vue-i18n-extract), simple script |
| Over-engineering (too complex for needs) | Medium | Medium | **KISS principle applied: 1 script, 2 commands** |

---

## Success Criteria

### Must Have (MVP)
- ✅ Detects missing keys in any locale
- ✅ Detects parameter mismatches
- ✅ Runs in <5 seconds
- ✅ Clear, actionable error messages
- ✅ Zero false positives on current locale files

### Nice to Have (Future)
- 🔄 CI/CD integration
- 🔄 Auto-fix for simple issues
- 🔄 Translation coverage metrics
- 🔄 VS Code extension integration (i18n-ally)

---

## Implementation Checklist

- [ ] Install `vue-i18n-extract` as dev dependency
- [ ] Create `frontend/scripts/validate-i18n.js` with modular functions
- [ ] Add npm scripts to `frontend/package.json`
- [ ] Add Makefile targets
- [ ] Test against current locale files
- [ ] Fix any detected issues in current translations
- [ ] Document usage in CLAUDE.md
- [ ] Commit and create PR

**Estimated Time:** 2-3 hours

---

## References

- [vue-i18n-extract GitHub](https://github.com/pixari/vue-i18n-extract)
- [Vue i18n Best Practices](https://vue-i18n.intlify.dev/guide/best-practices.html)
- [i18n Ally VS Code Extension](https://github.com/lokalise/i18n-ally)
- [KISS Principle](https://en.wikipedia.org/wiki/KISS_principle)
- [DRY Principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)

---

## Appendix: Alternative Approaches Considered

### Rejected: Multiple Validation Scripts

**Approach:**
- `validate-structure.js` - Structure checks
- `validate-params.js` - Parameter checks
- `validate-coverage.js` - Usage checks

**Rejected Because:**
- ❌ Violates DRY (similar code in each file)
- ❌ More complex (3 scripts vs 1)
- ❌ Harder to maintain
- ❌ Slower (multiple node processes)

### Rejected: Manual Pre-commit Hook

**Approach:**
- Git pre-commit hook runs validation automatically

**Rejected Because:**
- ❌ Can't be skipped when needed
- ❌ Slows down all commits
- ❌ Hard to debug when fails
- ✅ Developer can run `make frontend-i18n-check` manually

### Rejected: Complex Configuration File

**Approach:**
- `.i18nrc.json` with rules, thresholds, etc.

**Rejected Because:**
- ❌ Over-engineering for 5 locale files
- ❌ More to learn and maintain
- ✅ Convention over configuration

---

**Plan Status:** Ready for Implementation
**Next Step:** Review and approve, then implement Phase 1
