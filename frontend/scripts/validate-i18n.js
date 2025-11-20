#!/usr/bin/env node

/**
 * i18n Translation Validation Script
 *
 * Validates translation file completeness and congruence across all locales.
 * Follows SOLID principles with modular, single-responsibility functions.
 *
 * Usage:
 *   node validate-i18n.js
 *   npm run i18n:check
 *   make frontend-i18n-check
 *
 * @module validate-i18n
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// ESM equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ANSI color codes for beautiful terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

/**
 * Load all locale JSON files from the locales directory
 *
 * Single Responsibility: File loading only
 *
 * @param {string} localesDir - Path to locales directory
 * @returns {Object} Map of locale name to locale data
 * @throws {Error} If locales directory doesn't exist or JSON parse fails
 */
function loadLocales(localesDir) {
  if (!fs.existsSync(localesDir)) {
    throw new Error(`Locales directory not found: ${localesDir}`);
  }

  const files = fs.readdirSync(localesDir).filter((f) => f.endsWith('.json'));

  if (files.length === 0) {
    throw new Error(`No locale JSON files found in: ${localesDir}`);
  }

  const locales = {};

  files.forEach((file) => {
    const locale = path.parse(file).name;
    const filePath = path.join(localesDir, file);

    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      locales[locale] = JSON.parse(content);
    } catch (error) {
      throw new Error(`Failed to parse ${file}: ${error.message}`);
    }
  });

  return locales;
}

/**
 * Flatten nested object keys into dot-notation array
 *
 * Single Responsibility: Key flattening only
 *
 * @param {Object} obj - Nested object to flatten
 * @param {string} prefix - Current key prefix (for recursion)
 * @returns {string[]} Array of flattened keys (e.g., ['key1', 'key1.nested'])
 *
 * @example
 * flattenKeys({ a: { b: 'value' } }) // Returns: ['a.b']
 */
function flattenKeys(obj, prefix = '') {
  const keys = [];

  Object.entries(obj).forEach(([key, value]) => {
    const fullKey = prefix ? `${prefix}.${key}` : key;

    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      // Recurse into nested objects
      keys.push(...flattenKeys(value, fullKey));
    } else {
      // Leaf node - add to keys
      keys.push(fullKey);
    }
  });

  return keys;
}

/**
 * Check structural congruence across all locales
 *
 * Single Responsibility: Structure validation only
 *
 * @param {Object} locales - Map of locale name to locale data
 * @returns {Object} Issues found: { locale: { missing: [...], extra: [...] } }
 *
 * @example
 * const issues = checkStructureCongruence({ en: {...}, de: {...} });
 * // Returns: { de: { missing: ['key1'], extra: ['key2'] } }
 */
function checkStructureCongruence(locales) {
  const localeNames = Object.keys(locales);

  if (localeNames.length === 0) {
    return {};
  }

  // Use first locale as reference (typically 'en')
  const referenceLocale = localeNames[0];
  const referenceKeys = flattenKeys(locales[referenceLocale]).sort();

  const issues = {};

  // Check each locale against reference
  localeNames.forEach((locale) => {
    if (locale === referenceLocale) return; // Skip reference itself

    const localeKeys = flattenKeys(locales[locale]).sort();

    // Find missing keys (in reference but not in this locale)
    const missing = referenceKeys.filter((key) => !localeKeys.includes(key));

    // Find extra keys (in this locale but not in reference)
    const extra = localeKeys.filter((key) => !referenceKeys.includes(key));

    if (missing.length > 0 || extra.length > 0) {
      issues[locale] = { missing, extra };
    }
  });

  return issues;
}

/**
 * Extract parameter placeholders from a translation string
 *
 * Single Responsibility: Parameter extraction only
 *
 * @param {string} value - Translation string (e.g., "Hello {name}, you have {count} messages")
 * @returns {Set<string>} Set of parameter names (e.g., Set(['name', 'count']))
 *
 * @example
 * extractParameters("Add {label} ({id})") // Returns: Set(['label', 'id'])
 */
function extractParameters(value) {
  if (typeof value !== 'string') {
    return new Set();
  }

  // Match {param} placeholders
  const paramPattern = /\{(\w+)\}/g;
  const matches = [...value.matchAll(paramPattern)].map((m) => m[1]);

  return new Set(matches);
}

/**
 * Get all parameters from a nested locale object
 *
 * Single Responsibility: Recursive parameter collection
 *
 * @param {Object} obj - Locale object
 * @param {string} prefix - Current key prefix
 * @returns {Object} Map of key to Set of parameters
 *
 * @example
 * getAllParameters({ msg: "Hello {name}" })
 * // Returns: { "msg": Set(['name']) }
 */
function getAllParameters(obj, prefix = '') {
  const params = {};

  function traverse(val, path) {
    if (typeof val === 'string') {
      const extractedParams = extractParameters(val);
      if (extractedParams.size > 0) {
        params[path] = extractedParams;
      }
    } else if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
      Object.entries(val).forEach(([k, v]) => {
        traverse(v, path ? `${path}.${k}` : k);
      });
    }
  }

  traverse(obj, prefix);
  return params;
}

/**
 * Check parameter consistency across all locales
 *
 * Single Responsibility: Parameter validation only
 *
 * @param {Object} locales - Map of locale name to locale data
 * @returns {Array} Issues found: [{ locale, key, expected: [...], actual: [...] }]
 *
 * @example
 * const issues = checkParameterConsistency({ en: {...}, de: {...} });
 * // Returns: [{ locale: 'de', key: 'msg', expected: ['id'], actual: ['id', 'extra'] }]
 */
function checkParameterConsistency(locales) {
  const localeNames = Object.keys(locales);

  if (localeNames.length === 0) {
    return [];
  }

  // Use first locale as reference
  const referenceLocale = localeNames[0];
  const referenceParams = getAllParameters(locales[referenceLocale]);

  const issues = [];

  // Check each locale against reference
  localeNames.forEach((locale) => {
    if (locale === referenceLocale) return;

    const localeParams = getAllParameters(locales[locale]);

    // Check each key that has parameters in reference
    Object.entries(referenceParams).forEach(([key, refParams]) => {
      const locParams = localeParams[key];

      if (!locParams) {
        // Key exists in reference but locale doesn't have parameters
        // This is caught by structure check, skip here
        return;
      }

      // Compare parameter sets
      const refParamsArray = [...refParams].sort();
      const locParamsArray = [...locParams].sort();

      if (JSON.stringify(refParamsArray) !== JSON.stringify(locParamsArray)) {
        issues.push({
          locale,
          key,
          expected: refParamsArray,
          actual: locParamsArray,
        });
      }
    });
  });

  return issues;
}

/**
 * Format and display validation report
 *
 * Single Responsibility: Reporting only
 *
 * @param {Object} locales - Map of locale name to locale data
 * @param {Object} structureIssues - Structure validation results
 * @param {Array} paramIssues - Parameter validation results
 * @returns {number} Exit code (0 = success, 1 = failure)
 */
function formatReport(locales, structureIssues, paramIssues) {
  const localeNames = Object.keys(locales);
  const referenceLocale = localeNames[0];

  console.log(`\n${colors.bright}${colors.cyan}🌍 i18n Validation Report${colors.reset}`);
  console.log('='.repeat(50));

  // Show locale summary
  console.log(`\n${colors.bright}Locale Files:${colors.reset}`);
  localeNames.forEach((locale) => {
    const keys = flattenKeys(locales[locale]);
    const isReference = locale === referenceLocale;
    const marker = isReference ? ' (reference)' : '';
    console.log(`  ${colors.green}✓${colors.reset} ${locale}.json: ${keys.length} keys${marker}`);
  });

  // Check for structure issues
  const hasStructureIssues = Object.keys(structureIssues).length > 0;

  if (hasStructureIssues) {
    console.log(`\n${colors.bright}${colors.red}❌ Locale Structure Issues:${colors.reset}\n`);

    Object.entries(structureIssues).forEach(([locale, { missing, extra }]) => {
      if (missing.length > 0) {
        console.log(`${colors.bright}${locale}.json${colors.reset} - Missing keys (present in ${referenceLocale}.json):`);
        missing.forEach((key) => {
          console.log(`  ${colors.red}✗${colors.reset} ${key}`);
        });
        console.log('');
      }

      if (extra.length > 0) {
        console.log(`${colors.bright}${locale}.json${colors.reset} - Extra keys (not in ${referenceLocale}.json):`);
        extra.forEach((key) => {
          console.log(`  ${colors.yellow}⚠${colors.reset} ${key}`);
        });
        console.log('');
      }
    });
  } else {
    console.log(`\n${colors.green}✓ Checking locale structure congruence...${colors.reset}`);
    console.log(`  ${colors.green}✓${colors.reset} All locales have matching keys`);
  }

  // Check for parameter issues
  const hasParamIssues = paramIssues.length > 0;

  if (hasParamIssues) {
    console.log(`\n${colors.bright}${colors.red}❌ Parameter Consistency Issues:${colors.reset}\n`);

    paramIssues.forEach(({ locale, key, expected, actual }) => {
      console.log(`${colors.bright}Key:${colors.reset} ${key}`);
      console.log(`  ${referenceLocale}.json: {${expected.join('}, {')}} ${colors.green}✓${colors.reset}`);
      console.log(`  ${locale}.json: {${actual.join('}, {')}} ${colors.red}✗ Mismatch!${colors.reset}`);
      console.log(`  ${colors.yellow}Expected: {${expected.join('}, {')}${colors.reset}\n`);
    });
  } else {
    console.log(`\n${colors.green}✓ Validating parameter consistency...${colors.reset}`);
    console.log(`  ${colors.green}✓${colors.reset} All {param} placeholders match across locales`);
  }

  // Final summary
  const totalIssues = Object.keys(structureIssues).length + paramIssues.length;

  if (totalIssues === 0) {
    console.log(`\n${colors.bright}${colors.green}✅ All i18n validation checks passed!${colors.reset}\n`);
    return 0; // Success
  } else {
    console.log(`\n${colors.bright}${colors.red}✗ i18n validation failed with ${totalIssues} issue(s)${colors.reset}\n`);
    return 1; // Failure
  }
}

/**
 * Main entry point
 *
 * Single Responsibility: Orchestration only
 */
function main() {
  try {
    // Locate locales directory
    const localesDir = path.join(__dirname, '..', 'src', 'locales');

    console.log(`${colors.bright}${colors.blue}Loading locale files...${colors.reset}`);

    // Load all locales
    const locales = loadLocales(localesDir);

    console.log(`${colors.green}✓${colors.reset} Loaded ${Object.keys(locales).length} locale files\n`);

    // Run validations
    const structureIssues = checkStructureCongruence(locales);
    const paramIssues = checkParameterConsistency(locales);

    // Display report and exit
    const exitCode = formatReport(locales, structureIssues, paramIssues);
    process.exit(exitCode);
  } catch (error) {
    console.error(`\n${colors.red}${colors.bright}Fatal Error:${colors.reset} ${error.message}\n`);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

// Export for testing
export {
  loadLocales,
  flattenKeys,
  checkStructureCongruence,
  extractParameters,
  getAllParameters,
  checkParameterConsistency,
  formatReport,
};
