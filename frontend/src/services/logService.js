/**
 * Unified Logging Service
 *
 * Provides centralized logging with console echoing and store integration.
 * The store (Pinia) is the single source of truth for log management.
 *
 * @module services/logService
 */

// The useLogStore import is used by components that initialize this service
// eslint-disable-next-line no-unused-vars
import { useLogStore } from '../stores/log';
import { LOG_CONFIG, getDefaultMaxEntries } from '../config/logConfig';

/**
 * Log levels enum
 * @enum {string}
 */
export const LogLevel = {
  DEBUG: 'DEBUG',
  INFO: 'INFO',
  WARN: 'WARN',
  ERROR: 'ERROR',
};

/**
 * LogService - Singleton service for centralized logging
 *
 * Features:
 * - Console echoing (configurable, persisted to localStorage)
 * - Pinia store integration for reactive log management
 * - Automatic log rotation handled by store
 * - Configuration persistence
 *
 * @class
 */
class LogService {
  constructor() {
    /**
     * Pinia log store instance (set via initStore)
     * @type {Object|null}
     * @private
     */
    this.store = null;

    /**
     * Enable console echoing (loaded from localStorage)
     * @type {boolean}
     * @private
     */
    this.consoleEcho = this.loadConsoleEchoFromStorage();
  }

  /**
   * Initialize the log store (called after Pinia is available)
   *
   * @param {Object} store - The Pinia log store instance
   *
   * @example
   * import { useLogStore } from '@/stores/log';
   * import { logService } from '@/services/logService';
   *
   * const store = useLogStore();
   * logService.initStore(store);
   */
  initStore(store) {
    this.store = store;
    this.info('LogService initialized', {
      maxEntries: store.getMaxEntries(),
      consoleEcho: this.consoleEcho,
    });
  }

  /**
   * Core logging method (private)
   *
   * @private
   * @param {string} level - Log level
   * @param {string} message - Log message
   * @param {*} [data=null] - Optional data to log
   */
  _log(level, message, data = null) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      // Deep clone data to prevent reference issues
      data: data ? JSON.parse(JSON.stringify(data)) : null,
    };

    // Add to store if initialized
    if (this.store) {
      this.store.addLogEntry(entry);
    }

    // Echo to console if enabled
    if (this.consoleEcho) {
      this._consoleEcho(level, message, data);
    }
  }

  /**
   * Echo log to console with appropriate method
   *
   * @private
   * @param {string} level - Log level
   * @param {string} message - Log message
   * @param {*} data - Optional data
   */
  _consoleEcho(level, message, data) {
    const prefix = `[${level}]`;
    const args = data ? [prefix, message, data] : [prefix, message];

    switch (level) {
      case LogLevel.DEBUG:
        console.debug(...args);
        break;
      case LogLevel.INFO:
        console.info(...args);
        break;
      case LogLevel.WARN:
        console.warn(...args);
        break;
      case LogLevel.ERROR:
        console.error(...args);
        break;
      default:
        console.log(...args);
    }
  }

  /**
   * Log a debug message
   *
   * @param {string} message - Log message
   * @param {*} [data=null] - Optional additional data
   *
   * @example
   * logService.debug('User input validation', { field: 'email', valid: true });
   */
  debug(message, data = null) {
    this._log(LogLevel.DEBUG, message, data);
  }

  /**
   * Log an info message
   *
   * @param {string} message - Log message
   * @param {*} [data=null] - Optional additional data
   *
   * @example
   * logService.info('User logged in', { userId: 123 });
   */
  info(message, data = null) {
    this._log(LogLevel.INFO, message, data);
  }

  /**
   * Log a warning message
   *
   * @param {string} message - Log message
   * @param {*} [data=null] - Optional additional data
   *
   * @example
   * logService.warn('API response slow', { duration: 5000 });
   */
  warn(message, data = null) {
    this._log(LogLevel.WARN, message, data);
  }

  /**
   * Log an error message
   *
   * @param {string} message - Log message
   * @param {*} [data=null] - Optional additional data
   *
   * @example
   * logService.error('API request failed', { statusCode: 500, endpoint: '/api/users' });
   */
  error(message, data = null) {
    this._log(LogLevel.ERROR, message, data);
  }

  /**
   * Set console echo enabled/disabled and persist to localStorage
   *
   * @param {boolean} enabled - Enable console echoing
   *
   * @example
   * logService.setConsoleEcho(false); // Disable console output
   */
  setConsoleEcho(enabled) {
    this.consoleEcho = enabled;
    try {
      localStorage.setItem(LOG_CONFIG.STORAGE_KEYS.CONSOLE_ECHO, JSON.stringify(enabled));
    } catch (error) {
      console.error('Failed to save console echo setting:', error);
    }
  }

  /**
   * Set maximum log entries (delegates to store as source of truth)
   *
   * @param {number} max - Maximum log entries
   *
   * @example
   * logService.setMaxEntries(100); // Keep only 100 most recent logs
   */
  setMaxEntries(max) {
    if (this.store) {
      this.store.setMaxEntries(max);
    }
  }

  /**
   * Get maximum log entries from store
   *
   * @returns {number} Maximum log entries
   */
  getMaxEntries() {
    return this.store ? this.store.getMaxEntries() : getDefaultMaxEntries();
  }

  /**
   * Clear all logs (delegates to store)
   *
   * @returns {number} Number of logs cleared
   *
   * @example
   * const cleared = logService.clear(); // Returns count of cleared logs
   */
  clear() {
    return this.store ? this.store.clearLogs() : 0;
  }

  /**
   * Get logging statistics from store
   *
   * @returns {Object} Statistics object
   *
   * @example
   * const stats = logService.getStatistics();
   * console.log(`${stats.currentCount}/${stats.maxEntries} logs`);
   */
  getStatistics() {
    return this.store ? this.store.getStatistics() : null;
  }

  /**
   * Load console echo setting from localStorage
   *
   * @private
   * @returns {boolean} Console echo enabled (default: true in dev, false in prod)
   */
  loadConsoleEchoFromStorage() {
    try {
      const stored = localStorage.getItem(LOG_CONFIG.STORAGE_KEYS.CONSOLE_ECHO);
      if (stored !== null) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load console echo setting:', error);
    }
    // Default: true in development for debugging, false in production
    return import.meta.env.DEV;
  }
}

// Create and export singleton instance
export const logService = new LogService();
