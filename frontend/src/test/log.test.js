/**
 * Test Suite for Log Store
 *
 * Tests automatic rotation, statistics tracking, localStorage persistence,
 * and all core functionality of the logging system.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useLogStore } from '../stores/log';

describe('Log Store', () => {
  beforeEach(() => {
    // Create a fresh pinia instance for each test
    setActivePinia(createPinia());
    // Clear localStorage before each test
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('Initialization', () => {
    it('should initialize with empty logs', () => {
      const store = useLogStore();
      expect(store.logs).toEqual([]);
      expect(store.isViewerVisible).toBe(false);
    });

    it('should initialize with default maxEntries from environment', () => {
      const store = useLogStore();
      // Default should be 100 in dev, 50 in prod
      expect(store.maxEntries).toBeGreaterThan(0);
    });

    it('should initialize statistics with session start time', () => {
      const store = useLogStore();
      const stats = store.getStatistics();

      expect(stats.sessionStartTime).toBeDefined();
      expect(stats.totalLogsReceived).toBe(0);
      expect(stats.totalLogsDropped).toBe(0);
    });
  });

  describe('Basic Log Operations', () => {
    it('should add log entries', () => {
      const store = useLogStore();
      const logEntry = {
        level: 'INFO',
        message: 'Test log',
      };

      store.addLogEntry(logEntry);
      expect(store.logs).toHaveLength(1);
      expect(store.logs[0].message).toBe('Test log');
    });

    it('should add timestamp if not present', () => {
      const store = useLogStore();
      store.addLogEntry({ message: 'Test', level: 'INFO' });

      expect(store.logs[0].timestamp).toBeDefined();
      expect(new Date(store.logs[0].timestamp)).toBeInstanceOf(Date);
    });

    it('should clear all logs', () => {
      const store = useLogStore();
      store.addLogEntry({ message: 'Log 1', level: 'INFO' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });
      expect(store.logs).toHaveLength(2);

      const cleared = store.clearLogs();
      expect(store.logs).toEqual([]);
      expect(cleared).toBe(2);
    });
  });

  describe('Automatic Log Rotation', () => {
    it('should automatically trim logs when exceeding maxEntries', () => {
      const store = useLogStore();
      store.setMaxEntries(5);

      // Add 10 logs
      for (let i = 0; i < 10; i++) {
        store.addLogEntry({ message: `Log ${i}`, level: 'INFO' });
      }

      // Should keep only last 5
      expect(store.logs).toHaveLength(5);
      expect(store.logs[0].message).toBe('Log 5');
      expect(store.logs[4].message).toBe('Log 9');
    });

    it('should respect maxEntriesOverride parameter', () => {
      const store = useLogStore();
      store.setMaxEntries(100);

      // Add logs with override
      for (let i = 0; i < 20; i++) {
        store.addLogEntry({ message: `Log ${i}`, level: 'INFO' }, 10);
      }

      // Should respect override, not store default
      expect(store.logs).toHaveLength(10);
      expect(store.logs[0].message).toBe('Log 10');
    });

    it('should not trim if under maxEntries', () => {
      const store = useLogStore();
      store.setMaxEntries(10);

      store.addLogEntry({ message: 'Log 1', level: 'INFO' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });

      expect(store.logs).toHaveLength(2);
    });
  });

  describe('Statistics Tracking', () => {
    it('should track total logs received', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Test 1', level: 'INFO' });
      store.addLogEntry({ message: 'Test 2', level: 'INFO' });
      store.addLogEntry({ message: 'Test 3', level: 'INFO' });

      const stats = store.getStatistics();
      expect(stats.totalLogsReceived).toBe(3);
    });

    it('should update statistics when logs are dropped', () => {
      const store = useLogStore();
      store.setMaxEntries(3);

      for (let i = 0; i < 10; i++) {
        store.addLogEntry({ message: `Log ${i}`, level: 'INFO' });
      }

      const stats = store.getStatistics();
      expect(stats.totalLogsReceived).toBe(10);
      expect(stats.totalLogsDropped).toBe(7);
      expect(stats.currentCount).toBe(3);
    });

    it('should track last log time', () => {
      const store = useLogStore();
      const beforeLog = new Date().toISOString();

      store.addLogEntry({ message: 'Test', level: 'INFO' });

      const stats = store.getStatistics();
      expect(stats.lastLogTime).toBeDefined();
      expect(stats.lastLogTime >= beforeLog).toBe(true);
    });

    it('should count logs dropped by clearLogs', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Log 1', level: 'INFO' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });

      store.clearLogs();

      const stats = store.getStatistics();
      expect(stats.totalLogsDropped).toBe(2);
    });
  });

  describe('Manual Trimming', () => {
    it('should trim logs to max entries manually', () => {
      const store = useLogStore();

      // Add 10 log entries
      for (let i = 0; i < 10; i++) {
        store.addLogEntry({ message: `Log ${i}`, level: 'INFO' });
      }
      expect(store.logs).toHaveLength(10);

      // Trim to 5 entries - should keep last 5
      store.trimLogs(5);
      expect(store.logs).toHaveLength(5);
      expect(store.logs[0].message).toBe('Log 5');
      expect(store.logs[4].message).toBe('Log 9');
    });

    it('should not trim logs if under max entries', () => {
      const store = useLogStore();
      store.addLogEntry({ message: 'Log 1', level: 'INFO' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });

      store.trimLogs(10);
      expect(store.logs).toHaveLength(2);
    });
  });

  describe('LocalStorage Persistence', () => {
    it('should load maxEntries from localStorage', () => {
      localStorage.setItem('phentrieve-log-max-entries', '75');

      const store = useLogStore();
      expect(store.maxEntries).toBe(75);
    });

    it('should save maxEntries to localStorage', () => {
      const store = useLogStore();
      store.setMaxEntries(50);

      const stored = localStorage.getItem('phentrieve-log-max-entries');
      expect(stored).toBe('50');
    });

    it('should use default if localStorage is empty', () => {
      const store = useLogStore();
      // Should use environment default (100 in dev, 50 in prod)
      expect(store.maxEntries).toBeGreaterThan(0);
    });

    it('should handle invalid localStorage values gracefully', () => {
      localStorage.setItem('phentrieve-log-max-entries', 'invalid');

      const store = useLogStore();
      // Should fall back to default
      expect(store.maxEntries).toBeGreaterThan(0);
    });
  });

  describe('Computed Properties', () => {
    it('should calculate logCount correctly', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Log 1', level: 'INFO' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });

      expect(store.logCount).toBe(2);
    });

    it('should count error logs correctly', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Log 1', level: 'ERROR' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });
      store.addLogEntry({ message: 'Log 3', level: 'ERROR' });

      expect(store.errorCount).toBe(2);
    });

    it('should count warning logs correctly', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Log 1', level: 'WARN' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });

      expect(store.warningCount).toBe(1);
    });

    it('should count info logs correctly', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Log 1', level: 'INFO' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });
      store.addLogEntry({ message: 'Log 3', level: 'ERROR' });

      expect(store.infoCount).toBe(2);
    });

    it('should count debug logs correctly', () => {
      const store = useLogStore();

      store.addLogEntry({ message: 'Log 1', level: 'DEBUG' });
      store.addLogEntry({ message: 'Log 2', level: 'INFO' });

      expect(store.debugCount).toBe(1);
    });
  });

  describe('Memory Usage', () => {
    it('should calculate memory usage', () => {
      const store = useLogStore();

      for (let i = 0; i < 10; i++) {
        store.addLogEntry({ message: `Test ${i}`, level: 'INFO' });
      }

      const memory = store.getMemoryUsage();

      expect(memory.bytes).toBeGreaterThan(0);
      expect(memory.kb).toBeDefined();
      expect(memory.mb).toBeDefined();
      expect(parseFloat(memory.kb)).toBeGreaterThan(0);
    });
  });

  describe('Viewer Visibility', () => {
    it('should set viewer visibility', () => {
      const store = useLogStore();
      expect(store.isViewerVisible).toBe(false);

      store.setViewerVisibility(true);
      expect(store.isViewerVisible).toBe(true);

      store.setViewerVisibility(false);
      expect(store.isViewerVisible).toBe(false);
    });

    it('should toggle viewer visibility', () => {
      const store = useLogStore();
      expect(store.isViewerVisible).toBe(false);

      store.toggleViewer();
      expect(store.isViewerVisible).toBe(true);

      store.toggleViewer();
      expect(store.isViewerVisible).toBe(false);
    });

    it('should show and hide viewer', () => {
      const store = useLogStore();

      store.showViewer();
      expect(store.isViewerVisible).toBe(true);

      store.hideViewer();
      expect(store.isViewerVisible).toBe(false);
    });
  });

  describe('Statistics Integration', () => {
    it('should provide comprehensive statistics', () => {
      const store = useLogStore();
      store.setMaxEntries(5);

      // Add logs to trigger rotation
      for (let i = 0; i < 10; i++) {
        store.addLogEntry({ message: `Test ${i}`, level: 'INFO' });
      }

      const stats = store.getStatistics(true); // Include memory usage

      expect(stats.totalLogsReceived).toBe(10);
      expect(stats.totalLogsDropped).toBe(5);
      expect(stats.currentCount).toBe(5);
      expect(stats.maxEntries).toBe(5);
      expect(stats.memoryUsage).toBeDefined();
      expect(stats.oldestLog).toBeDefined();
      expect(stats.newestLog).toBeDefined();
      expect(stats.sessionStartTime).toBeDefined();
    });
  });
});
