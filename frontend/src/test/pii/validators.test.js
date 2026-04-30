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
