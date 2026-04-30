export const SUPPORTED_PII_LOCALES = Object.freeze(['en', 'de', 'fr', 'es', 'nl']);

export const DATE_PATTERN = /\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b/gu;
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
    patterns: [/\b\d+\.\d+\.\d+\.\d+\b/gu],
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
      patterns: [
        /(?:DOB|date of birth|born)\s*[:#-]?\s*(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})/giu,
      ],
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
      patterns: [
        /(?:Geburtsdatum|geboren)\s*[:#-]?\s*(?:\d{1,2}[.]\d{1,2}[.]\d{2,4}|\d{4}-\d{2}-\d{2})/giu,
      ],
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
  de: /\b(?:straße|strasse|str\.|weg|platz|allee)\s+\d+\b[^\n]{0,80}\b\d{5}\b/giu,
  fr: /\b(?:rue|avenue|av\.|boulevard|bd|chemin|place)\s+[^\n,]{0,80}\d+[^\n]{0,80}\b\d{5}\b/giu,
  es: /\b(?:calle|c\/|avenida|avda\.|paseo|plaza)\s+[^\n,]{0,80}\d+[^\n]{0,80}\b\d{5}\b/giu,
  nl: /\b(?:straat|laan|weg|plein|adres)\s*[:#-]?\s+[^\n,]{0,80}\d+[A-Z]?\b[^\n]{0,80}\b\d{4}\s?[A-Z]{2}\b/giu,
  en: /\b(?:street|st\.|road|rd\.|avenue|ave\.|drive|dr\.)\s+[^\n,]{0,80}\d+[^\n]{0,80}\b[A-Z0-9][A-Z0-9 -]{3,10}\b/giu,
});

export const MEDICAL_ID_VALUE_PATTERN = ID_VALUE_PATTERN;
