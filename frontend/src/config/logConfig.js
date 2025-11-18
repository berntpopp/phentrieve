/**
 * Centralized Logging Configuration
 *
 * This module provides configuration constants and utilities for the logging system.
 * Following DRY principle to avoid scattered magic numbers throughout the codebase.
 *
 * @module config/logConfig
 */

/**
 * Core logging configuration constants
 * @const {Object} LOG_CONFIG
 */
export const LOG_CONFIG = {
  /**
   * Maximum log entries for development environment
   * Higher limit for verbose debugging
   * @type {number}
   */
  MAX_ENTRIES_DEV: 100,

  /**
   * Maximum log entries for production environment
   * Lower limit for better performance and memory usage
   * @type {number}
   */
  MAX_ENTRIES_PROD: 50,

  /**
   * Default minimum log level for development
   * Shows all logs including DEBUG
   * @type {string}
   */
  DEFAULT_LEVEL_DEV: 'DEBUG',

  /**
   * Default minimum log level for production
   * Only shows warnings and errors to reduce noise
   * @type {string}
   */
  DEFAULT_LEVEL_PROD: 'WARN',

  /**
   * LocalStorage keys for configuration persistence
   * Prefixed with 'phentrieve-' to avoid conflicts with other apps
   * @type {Object.<string, string>}
   */
  STORAGE_KEYS: {
    MAX_ENTRIES: 'phentrieve-log-max-entries',
    LOG_LEVEL: 'phentrieve-log-level',
    CONSOLE_ECHO: 'phentrieve-console-echo',
  },

  /**
   * Memory warning threshold in bytes
   * Alert user if log storage exceeds this value
   * @type {number}
   */
  MEMORY_WARNING_THRESHOLD: 1024 * 1024, // 1 MB

  /**
   * Slider configuration for UI
   * @type {Object}
   */
  UI_LIMITS: {
    MIN_ENTRIES: 10,
    MAX_ENTRIES: 500,
    STEP: 10,
  },
};

/**
 * Get environment-specific default for maximum log entries
 *
 * @returns {number} Maximum entries (100 for dev, 50 for prod)
 *
 * @example
 * const maxLogs = getDefaultMaxEntries(); // 100 in dev, 50 in prod
 */
export function getDefaultMaxEntries() {
  return import.meta.env.DEV ? LOG_CONFIG.MAX_ENTRIES_DEV : LOG_CONFIG.MAX_ENTRIES_PROD;
}

/**
 * Get environment-specific default log level
 *
 * @returns {string} Log level ('DEBUG' for dev, 'WARN' for prod)
 *
 * @example
 * const level = getDefaultLogLevel(); // 'DEBUG' in dev, 'WARN' in prod
 */
export function getDefaultLogLevel() {
  return import.meta.env.DEV ? LOG_CONFIG.DEFAULT_LEVEL_DEV : LOG_CONFIG.DEFAULT_LEVEL_PROD;
}

/**
 * Validate that a maxEntries value is within acceptable bounds
 *
 * @param {number} value - The value to validate
 * @returns {boolean} True if valid, false otherwise
 *
 * @example
 * isValidMaxEntries(50);   // true
 * isValidMaxEntries(1000); // false (exceeds max)
 * isValidMaxEntries(-10);  // false (negative)
 */
export function isValidMaxEntries(value) {
  return (
    typeof value === 'number' &&
    !isNaN(value) &&
    value >= LOG_CONFIG.UI_LIMITS.MIN_ENTRIES &&
    value <= LOG_CONFIG.UI_LIMITS.MAX_ENTRIES
  );
}

/**
 * Clamp a value to valid maxEntries range
 *
 * @param {number} value - The value to clamp
 * @returns {number} Clamped value within valid range
 *
 * @example
 * clampMaxEntries(1000); // 500 (max)
 * clampMaxEntries(5);    // 10 (min)
 * clampMaxEntries(50);   // 50 (unchanged)
 */
export function clampMaxEntries(value) {
  return Math.max(
    LOG_CONFIG.UI_LIMITS.MIN_ENTRIES,
    Math.min(LOG_CONFIG.UI_LIMITS.MAX_ENTRIES, value)
  );
}
