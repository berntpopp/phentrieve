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
  const isValidLength = normalized.length === 13 || normalized.length === 15;
  if (!isValidLength || (normalized[0] !== '1' && normalized[0] !== '2')) return false;
  if (![...normalized.slice(1)].every((character) => character >= '0' && character <= '9')) {
    return false;
  }
  const month = Number(normalized.slice(3, 5));
  return month >= 1 && month <= 12;
}

export const VALIDATORS = Object.freeze({
  validateSpanishDniNie,
  validateDutchBsn,
  validateGermanKvnr,
  validateFrenchNir,
});
