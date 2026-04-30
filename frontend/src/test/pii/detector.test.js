import { describe, expect, it } from 'vitest';
import { scanPii } from '../../pii';

describe('scanPii', () => {
  it('detects global high-confidence identifiers without exposing snippets', () => {
    const result = scanPii('Email jane@example.org or call +1 202 555 0199.', { locale: 'en' });

    expect(result.summary.high.email).toBe(1);
    expect(result.summary.high.phone).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
    for (const finding of result.findings) {
      expect(Object.keys(finding).sort()).toEqual([
        'category',
        'confidence',
        'end',
        'id',
        'redactionToken',
        'ruleId',
        'start',
      ]);
      expect(finding).not.toHaveProperty('snippet');
      expect(finding).not.toHaveProperty('match');
    }
  });

  it('keeps finding IDs unique when phone rules run across locales', () => {
    const result = scanPii('Call +1 202 555 0199 or +31 20 123 4567.', { locale: 'en' });
    const ids = result.findings.map((finding) => finding.id);

    expect(new Set(ids).size).toBe(ids.length);
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
    expect(result.summary.review.date).toBeUndefined();
    expect(result.summary.high.national_identifier).toBe(1);
  });

  it('marks German titled person names as review confidence', () => {
    const result = scanPii('Herr Popp ist dumm', { locale: 'de' });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it('marks configured titled person names even when the selected language differs', () => {
    const result = scanPii('M Dupont a des crises', { locale: 'en' });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it.each([
    ['en', 'Mr Smith has seizures'],
    ['fr', 'M Dupont a des crises'],
    ['es', 'Sr Garcia tiene convulsiones'],
    ['nl', 'Dhr Jansen heeft aanvallen'],
  ])('marks %s titled person names as review confidence', (locale, text) => {
    const result = scanPii(text, { locale });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it.each([
    ['en', 'John Smith has seizures'],
    ['de', 'Bernt Popp ist dumm'],
    ['fr', 'Jean Dupont a des crises'],
    ['es', 'María García tiene convulsiones'],
    ['nl', 'Jan van Dijk heeft aanvallen'],
  ])('marks %s untitled person names as review confidence', (locale, text) => {
    const result = scanPii(text, { locale });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it.each([
    ['de', 'Bernt Popp ist dumm'],
    ['fr', 'Jean Dupont a des crises'],
    ['es', 'María García tiene convulsiones'],
    ['nl', 'Jan van Dijk heeft aanvallen'],
  ])('marks %s untitled person names when selected language is English', (_locale, text) => {
    const result = scanPii(text, { locale: 'en' });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it('keeps context words outside untitled person-name spans', () => {
    const text = 'Patient John Smith has seizures';
    const result = scanPii(text, { locale: 'en' });
    const finding = result.findings.find((item) => item.category === 'person_name');

    expect(finding).toBeDefined();
    expect(finding.start).toBe(text.indexOf('John'));
    expect(finding.end).toBe(text.indexOf(' has seizures'));
  });

  it.each([
    ['Pectus Carinatum was noted.'],
    ['Down Syndrome was discussed.'],
    ['BioLORD Model returned results.'],
    ['Brain MRI was normal.'],
    ['HP:0002779 Tracheomalacia'],
  ])('does not mark clinical or domain phrases as untitled names: %s', (text) => {
    const result = scanPii(text, { locale: 'en' });

    expect(result.summary.review.person_name).toBeUndefined();
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

  it('keeps address matches bounded to one line', () => {
    const text = 'Adres: Hoofdstraat 12, 1234 AB Leiden.\nSeizures described in detail.';
    const result = scanPii(text, { locale: 'nl' });
    const address = result.findings.find((finding) => finding.category === 'address');

    expect(address).toBeDefined();
    expect(address.end).toBeLessThan(text.indexOf('Seizures'));
  });
});
