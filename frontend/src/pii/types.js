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
