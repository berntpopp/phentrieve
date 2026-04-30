import { findPhoneNumbersInText } from 'libphonenumber-js';
import {
  ADDRESS_RULES,
  GLOBAL_RULES,
  LOCALE_RULES,
  SUPPORTED_PII_LOCALES,
  UNTITLED_NAME_RULE_CONFIG,
} from './ruleConfig';
import { VALIDATORS } from './validators';

const WORD_PATTERN = /[A-Za-zÀ-ÖØ-öø-ÿß'-]+/gu;
const NAME_TOKEN_PATTERN = /^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿß'-]*$/u;
const CODE_LIKE_PATTERN = /(?:\d|[A-Z]{2,}[:_-]?\d|[a-z][A-Z])/u;
const ALL_CAPS_TOKEN_PATTERN = /^[A-ZÀ-ÖØ-Þ]{2,}$/u;

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

function candidateForValidation(matchText) {
  const normalizedText = String(matchText);
  const identifierMatch = /[A-Z]?\d[\d .-]{6,20}[A-Z]?/iu.exec(normalizedText);
  return identifierMatch?.[0] ?? normalizedText;
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
        if (validator && !validator(candidateForValidation(matchText))) continue;
        findings.push(
          buildFinding({
            rule,
            start: match.index,
            end: match.index + matchText.length,
            index: findings.length,
          })
        );
      }
    }
  }
  return findings;
}

function regionForLocale(locale) {
  switch (locale) {
    case 'de':
      return 'DE';
    case 'fr':
      return 'FR';
    case 'es':
      return 'ES';
    case 'nl':
      return 'NL';
    default:
      return 'US';
  }
}

function applyPhoneRule(text, locale) {
  return findPhoneNumbersInText(text, regionForLocale(locale)).map((match, index) => ({
    id: `global.phone-${locale}-${index}-${match.startsAt}-${match.endsAt}`,
    ruleId: 'global.phone',
    category: 'phone',
    confidence: 'high',
    start: match.startsAt,
    end: match.endsAt,
    redactionToken: '[REDACTED_PHONE]',
  }));
}

function applyAddressRule(text, locale) {
  let pattern;
  switch (locale) {
    case 'de':
      pattern = ADDRESS_RULES.de;
      break;
    case 'fr':
      pattern = ADDRESS_RULES.fr;
      break;
    case 'es':
      pattern = ADDRESS_RULES.es;
      break;
    case 'nl':
      pattern = ADDRESS_RULES.nl;
      break;
    default:
      pattern = ADDRESS_RULES.en;
  }
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

function normalizeNameToken(token) {
  return String(token ?? '').toLowerCase();
}

function tokenizeWords(text) {
  const tokens = [];
  WORD_PATTERN.lastIndex = 0;
  let match;
  while ((match = WORD_PATTERN.exec(text)) !== null) {
    tokens.push({
      value: match[0],
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return tokens;
}

function isNameToken(token, config) {
  const value = token.value;
  return (
    value.length >= 2 &&
    NAME_TOKEN_PATTERN.test(value) &&
    !CODE_LIKE_PATTERN.test(value) &&
    !ALL_CAPS_TOKEN_PATTERN.test(value) &&
    !config.blockedTerms.includes(normalizeNameToken(value))
  );
}

function isNameParticle(token, config) {
  return config.particles.includes(normalizeNameToken(token.value));
}

function hasOnlyNameSeparators(text, previous, next) {
  return /^[\s'-]+$/u.test(text.slice(previous.end, next.start));
}

function hasNameContext(text, candidate, config) {
  const start = Math.max(0, candidate.start - config.contextWindowChars);
  const end = Math.min(text.length, candidate.end + config.contextWindowChars);
  const context = text.slice(start, end).toLowerCase();
  return config.contextWords.some((word) => context.includes(word.toLowerCase()));
}

function scoreNameCandidate(text, candidate, config) {
  const normalizedValues = candidate.tokens.map((token) => normalizeNameToken(token.value));
  if (normalizedValues.some((value) => config.blockedTerms.includes(value))) {
    return 0;
  }

  let score = candidate.nameTokenCount >= config.minNameTokens ? 2 : 0;
  if (candidate.hasParticle) score += 1;
  if (hasNameContext(text, candidate, config)) score += 2;
  return score;
}

function buildUntitledNameFinding(candidate, index, config) {
  return {
    id: `${config.id}-${index}`,
    ruleId: config.id,
    category: config.category,
    confidence: config.confidence,
    start: candidate.start,
    end: candidate.end,
    redactionToken: config.redactionToken,
  };
}

function applyUntitledNameRule(text, config = UNTITLED_NAME_RULE_CONFIG) {
  if (!config.enabled) return [];

  const tokens = tokenizeWords(text);
  const findings = [];
  for (const [index, firstToken] of tokens.entries()) {
    if (!isNameToken(firstToken, config)) continue;

    let candidate = null;
    let nameTokenCount = 1;
    let hasParticle = false;
    let previous = firstToken;
    const candidateTokens = [firstToken];

    for (let nextIndex = index + 1; nextIndex < tokens.length; nextIndex += 1) {
      // eslint-disable-next-line security/detect-object-injection -- Bounded numeric token scan avoids per-token slice allocations.
      const next = tokens[nextIndex];
      if (candidateTokens.length >= config.maxTokens) break;
      if (!hasOnlyNameSeparators(text, previous, next)) break;

      if (isNameParticle(next, config)) {
        hasParticle = true;
        candidateTokens.push(next);
        previous = next;
        continue;
      }

      if (!isNameToken(next, config)) break;

      nameTokenCount += 1;
      candidateTokens.push(next);
      previous = next;
      if (nameTokenCount >= config.minNameTokens) {
        candidate = {
          start: firstToken.start,
          end: next.end,
          tokens: [...candidateTokens],
          nameTokenCount,
          hasParticle,
        };
      }
    }

    if (!candidate) continue;
    if (scoreNameCandidate(text, candidate, config) < config.minScore) continue;

    findings.push(buildUntitledNameFinding(candidate, findings.length, config));
  }

  return findings;
}

function mergeOverlappingFindings(findings) {
  const sorted = [...findings].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];
  for (const finding of sorted) {
    const previous = merged[merged.length - 1];
    if (previous && finding.start < previous.end) {
      const previousLength = previous.end - previous.start;
      const findingLength = finding.end - finding.start;
      previous.end = Math.max(previous.end, finding.end);
      if (
        previous.confidence !== 'high' &&
        (finding.confidence === 'high' || findingLength > previousLength)
      ) {
        previous.ruleId = finding.ruleId;
        previous.category = finding.category;
        previous.confidence = finding.confidence;
        previous.redactionToken = finding.redactionToken;
      }
      continue;
    }
    merged.push({ ...finding });
  }
  return merged;
}

function rulesForLocale(locale) {
  switch (locale) {
    case 'de':
      return LOCALE_RULES.de;
    case 'fr':
      return LOCALE_RULES.fr;
    case 'es':
      return LOCALE_RULES.es;
    case 'nl':
      return LOCALE_RULES.nl;
    default:
      return LOCALE_RULES.en;
  }
}

function localesForScan(locale) {
  return [locale, ...SUPPORTED_PII_LOCALES.filter((supportedLocale) => supportedLocale !== locale)];
}

export function scanPii(text, { locale = 'en', includeGlobalRules = true } = {}) {
  const normalizedLocale = SUPPORTED_PII_LOCALES.includes(locale) ? locale : 'en';
  const source = String(text ?? '');
  const scanLocales = localesForScan(normalizedLocale);
  const findings = mergeOverlappingFindings([
    ...(includeGlobalRules ? applyRules(source, GLOBAL_RULES, normalizedLocale) : []),
    ...scanLocales.flatMap((scanLocale) =>
      applyRules(source, rulesForLocale(scanLocale), scanLocale)
    ),
    ...applyUntitledNameRule(source),
    ...(includeGlobalRules
      ? scanLocales.flatMap((scanLocale) => applyPhoneRule(source, scanLocale))
      : []),
    ...scanLocales.flatMap((scanLocale) => applyAddressRule(source, scanLocale)),
  ]);

  const summary = createSummary();
  findings.forEach((finding) => addSummary(summary, finding));
  return {
    hasFindings: findings.length > 0,
    findings,
    summary,
  };
}
