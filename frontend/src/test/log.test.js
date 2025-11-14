import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useLogStore } from '../stores/log';

describe('Log Store', () => {
  beforeEach(() => {
    // Create a fresh pinia instance for each test
    setActivePinia(createPinia());
  });

  it('should initialize with empty logs', () => {
    const store = useLogStore();
    expect(store.logs).toEqual([]);
    expect(store.isViewerVisible).toBe(false);
  });

  it('should add log entries', () => {
    const store = useLogStore();
    const logEntry = {
      timestamp: Date.now(),
      level: 'info',
      message: 'Test log',
    };

    store.addLogEntry(logEntry);
    expect(store.logs).toHaveLength(1);
    expect(store.logs[0]).toEqual(logEntry);
  });

  it('should clear all logs', () => {
    const store = useLogStore();
    store.addLogEntry({ message: 'Log 1' });
    store.addLogEntry({ message: 'Log 2' });
    expect(store.logs).toHaveLength(2);

    store.clearLogs();
    expect(store.logs).toEqual([]);
  });

  it('should trim logs to max entries', () => {
    const store = useLogStore();
    // Add 10 log entries
    for (let i = 0; i < 10; i++) {
      store.addLogEntry({ message: `Log ${i}` });
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
    store.addLogEntry({ message: 'Log 1' });
    store.addLogEntry({ message: 'Log 2' });

    store.trimLogs(10);
    expect(store.logs).toHaveLength(2);
  });

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
});
