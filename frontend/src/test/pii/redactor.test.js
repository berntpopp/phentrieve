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

  it('redacts the full union of partially overlapping findings', () => {
    const text = 'abcdefghi';
    const findings = [
      {
        id: 'a',
        ruleId: 'first',
        category: 'medical_record',
        confidence: 'high',
        start: 0,
        end: 5,
        redactionToken: '[REDACTED_MRN]',
      },
      {
        id: 'b',
        ruleId: 'second',
        category: 'medical_record',
        confidence: 'high',
        start: 3,
        end: 9,
        redactionToken: '[REDACTED_MRN]',
      },
    ];

    expect(redactPiiFindings(text, findings).text).toBe('[REDACTED_MRN]');
  });

  it('redacts chain-overlapping findings as one span', () => {
    const text = '0123456789abcdef';
    const findings = [
      {
        id: 'a',
        ruleId: 'first',
        category: 'medical_record',
        confidence: 'high',
        start: 0,
        end: 5,
        redactionToken: '[REDACTED_MRN]',
      },
      {
        id: 'b',
        ruleId: 'second',
        category: 'medical_record',
        confidence: 'high',
        start: 4,
        end: 10,
        redactionToken: '[REDACTED_MRN]',
      },
      {
        id: 'c',
        ruleId: 'third',
        category: 'medical_record',
        confidence: 'high',
        start: 9,
        end: 16,
        redactionToken: '[REDACTED_MRN]',
      },
    ];

    expect(redactPiiFindings(text, findings).text).toBe('[REDACTED_MRN]');
  });
});
