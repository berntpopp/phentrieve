/**
 * Pinia Store for Log Management
 *
 * Manages reactive log state with automatic rotation, statistics tracking,
 * and configurable memory limits. Follows circular buffer pattern (FIFO).
 *
 * @module stores/log
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { LOG_CONFIG, getDefaultMaxEntries } from '../config/logConfig';

/**
 * Load maxEntries from localStorage or use environment default
 * @private
 * @returns {number} Maximum log entries
 */
function loadMaxEntriesFromStorage() {
  try {
    const stored = localStorage.getItem(LOG_CONFIG.STORAGE_KEYS.MAX_ENTRIES);
    if (stored) {
      const parsed = parseInt(stored, 10);
      if (!isNaN(parsed) && parsed > 0) {
        return parsed;
      }
    }
  } catch (error) {
    console.error('Failed to load max entries from localStorage:', error);
  }
  return getDefaultMaxEntries();
}

/**
 * Save maxEntries to localStorage
 * @private
 * @param {number} value - Maximum log entries
 */
function saveMaxEntriesToStorage(value) {
  try {
    localStorage.setItem(LOG_CONFIG.STORAGE_KEYS.MAX_ENTRIES, value.toString());
  } catch (error) {
    console.error('Failed to save max entries to localStorage:', error);
  }
}

export const useLogStore = defineStore('log', () => {
  // ===========================
  // State
  // ===========================

  /**
   * Array of log entries (reactive)
   * @type {import('vue').Ref<Array<Object>>}
   */
  const logs = ref([]);

  /**
   * LogViewer visibility state
   * @type {import('vue').Ref<boolean>}
   */
  const isViewerVisible = ref(false);

  /**
   * Maximum number of log entries to keep in memory (configurable)
   * Loaded from localStorage or environment default
   * @type {import('vue').Ref<number>}
   */
  const maxEntries = ref(loadMaxEntriesFromStorage());

  /**
   * Statistics tracking (session-only, reset on page refresh)
   * @type {import('vue').Ref<Object>}
   */
  const stats = ref({
    totalLogsReceived: 0,
    totalLogsDropped: 0,
    lastLogTime: null,
    sessionStartTime: new Date().toISOString(),
  });

  // ===========================
  // Computed Properties
  // ===========================

  /**
   * Current number of logs in memory
   * @type {import('vue').ComputedRef<number>}
   */
  const logCount = computed(() => logs.value.length);

  /**
   * Count of error-level logs
   * @type {import('vue').ComputedRef<number>}
   */
  const errorCount = computed(() => logs.value.filter((log) => log.level === 'ERROR').length);

  /**
   * Count of warning-level logs
   * @type {import('vue').ComputedRef<number>}
   */
  const warningCount = computed(() => logs.value.filter((log) => log.level === 'WARN').length);

  /**
   * Count of info-level logs
   * @type {import('vue').ComputedRef<number>}
   */
  const infoCount = computed(() => logs.value.filter((log) => log.level === 'INFO').length);

  /**
   * Count of debug-level logs
   * @type {import('vue').ComputedRef<number>}
   */
  const debugCount = computed(() => logs.value.filter((log) => log.level === 'DEBUG').length);

  // ===========================
  // Actions
  // ===========================

  /**
   * Add a log entry with automatic rotation (circular buffer pattern)
   *
   * Automatically trims oldest logs when exceeding maxEntries limit.
   * Updates statistics for monitoring memory health.
   *
   * @param {Object} entry - Log entry object
   * @param {string} entry.message - Log message
   * @param {string} entry.level - Log level (DEBUG, INFO, WARN, ERROR)
   * @param {string} [entry.timestamp] - ISO timestamp (auto-generated if missing)
   * @param {*} [entry.data] - Optional additional data
   * @param {number|null} [maxEntriesOverride] - Override default maxEntries
   *
   * @example
   * // Basic usage
   * addLogEntry({ message: 'User logged in', level: 'INFO' });
   *
   * @example
   * // With data
   * addLogEntry({
   *   message: 'API error',
   *   level: 'ERROR',
   *   data: { statusCode: 500, endpoint: '/api/users' }
   * });
   *
   * @example
   * // Override max entries for this log
   * addLogEntry({ message: 'Test', level: 'DEBUG' }, 10);
   */
  function addLogEntry(entry, maxEntriesOverride = null) {
    const max = maxEntriesOverride !== null ? maxEntriesOverride : maxEntries.value;

    // Add timestamp if not present
    if (!entry.timestamp) {
      entry.timestamp = new Date().toISOString();
    }

    // Update statistics
    stats.value.totalLogsReceived++;
    stats.value.lastLogTime = entry.timestamp;

    // Create new array with new entry (immutable pattern)
    const newLogs = [...logs.value, entry];

    // AUTOMATIC ROTATION: Trim if exceeding max (circular buffer / FIFO)
    if (newLogs.length > max) {
      const toRemove = newLogs.length - max;
      logs.value = newLogs.slice(toRemove); // Keep most recent logs
      stats.value.totalLogsDropped += toRemove;
    } else {
      logs.value = newLogs;
    }
  }

  /**
   * Clear all logs and update statistics
   *
   * @returns {number} Number of logs that were cleared
   *
   * @example
   * const clearedCount = clearLogs(); // Returns number of logs removed
   */
  function clearLogs() {
    const previousCount = logs.value.length;
    logs.value = [];
    stats.value.totalLogsDropped += previousCount;
    return previousCount;
  }

  /**
   * Internal helper: Perform trimming operation without changing maxEntries
   * @private
   * @param {number} targetMax - Target maximum entries
   */
  function _performTrim(targetMax) {
    if (logs.value.length > targetMax) {
      const toRemove = logs.value.length - targetMax;
      logs.value = logs.value.slice(toRemove);
      stats.value.totalLogsDropped += toRemove;
    }
  }

  /**
   * Trim logs to a new maximum and update statistics
   *
   * @param {number} newMaxEntries - New maximum log entries
   *
   * @example
   * trimLogs(50); // Keep only 50 most recent logs
   */
  function trimLogs(newMaxEntries) {
    _performTrim(newMaxEntries);
    maxEntries.value = newMaxEntries;
    saveMaxEntriesToStorage(newMaxEntries);
  }

  /**
   * Set maximum log entries and persist to localStorage
   *
   * @param {number} value - New maximum entries
   *
   * @example
   * setMaxEntries(100); // Sets max to 100 and saves to localStorage
   */
  function setMaxEntries(value) {
    maxEntries.value = value;
    saveMaxEntriesToStorage(value);
    _performTrim(value); // Trim existing logs if needed (no circular call)
  }

  /**
   * Get current maximum log entries
   *
   * @returns {number} Maximum log entries
   */
  function getMaxEntries() {
    return maxEntries.value;
  }

  /**
   * Show the log viewer
   */
  function showViewer() {
    isViewerVisible.value = true;
  }

  /**
   * Hide the log viewer
   */
  function hideViewer() {
    isViewerVisible.value = false;
  }

  /**
   * Set log viewer visibility
   *
   * @param {boolean} visible - Visibility state
   */
  function setViewerVisibility(visible) {
    isViewerVisible.value = visible;
  }

  /**
   * Toggle log viewer visibility
   */
  function toggleViewer() {
    isViewerVisible.value = !isViewerVisible.value;
  }

  /**
   * Calculate estimated memory usage of logs
   *
   * NOTE: This is a method (not computed) to avoid expensive recalculation
   * on every reactive update. Call this only when needed (e.g., in UI display).
   *
   * @returns {Object} Memory usage statistics
   * @returns {number} return.bytes - Bytes used
   * @returns {string} return.kb - Kilobytes (formatted to 2 decimals)
   * @returns {string} return.mb - Megabytes (formatted to 2 decimals)
   *
   * @example
   * const usage = getMemoryUsage();
   * console.log(`Logs using ${usage.kb} KB`);
   */
  function getMemoryUsage() {
    const jsonSize = JSON.stringify(logs.value).length;
    return {
      bytes: jsonSize,
      kb: (jsonSize / 1024).toFixed(2),
      mb: (jsonSize / (1024 * 1024)).toFixed(2),
    };
  }

  /**
   * Get comprehensive statistics about the logging system
   *
   * PERFORMANCE NOTE: Set includeMemory=true only when needed, as it performs
   * expensive JSON.stringify operation. Default is false to avoid performance
   * impact when called from computed properties.
   *
   * @param {boolean} [includeMemory=false] - Whether to include memory usage (expensive)
   * @returns {Object} Statistics object
   * @returns {number} return.totalLogsReceived - Total logs added this session
   * @returns {number} return.totalLogsDropped - Total logs auto-rotated
   * @returns {string|null} return.lastLogTime - ISO timestamp of last log
   * @returns {string} return.sessionStartTime - ISO timestamp of session start
   * @returns {number} return.currentCount - Current number of logs
   * @returns {number} return.maxEntries - Maximum log entries
   * @returns {Object} [return.memoryUsage] - Memory usage (only if includeMemory=true)
   * @returns {string|null} return.oldestLog - Timestamp of oldest log
   * @returns {string|null} return.newestLog - Timestamp of newest log
   *
   * @example
   * // Lightweight stats (for computed properties)
   * const stats = getStatistics();
   *
   * @example
   * // Full stats with memory (for explicit UI display)
   * const stats = getStatistics(true);
   * console.log(`${stats.currentCount}/${stats.maxEntries} logs, ${stats.memoryUsage.kb} KB`);
   */
  function getStatistics(includeMemory = false) {
    const baseStats = {
      ...stats.value,
      currentCount: logs.value.length,
      maxEntries: maxEntries.value,
      oldestLog: logs.value[0]?.timestamp || null,
      newestLog: logs.value[logs.value.length - 1]?.timestamp || null,
    };

    // Only compute memory usage if explicitly requested (expensive operation)
    if (includeMemory) {
      baseStats.memoryUsage = getMemoryUsage();
    }

    return baseStats;
  }

  // ===========================
  // Return Public API
  // ===========================

  return {
    // State
    logs,
    isViewerVisible,
    maxEntries,

    // Computed
    logCount,
    errorCount,
    warningCount,
    infoCount,
    debugCount,

    // Actions
    addLogEntry,
    clearLogs,
    trimLogs,
    setMaxEntries,
    getMaxEntries,
    showViewer,
    hideViewer,
    setViewerVisibility,
    toggleViewer,
    getMemoryUsage,
    getStatistics,
  };
});
