import { describe, it, expect, beforeEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useQueryPreferencesStore } from '../stores/queryPreferences';

describe('queryPreferences store (characterization)', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  describe('initial state', () => {
    it('includeDetails defaults to false', () => {
      const store = useQueryPreferencesStore();
      expect(store.includeDetails).toBe(false);
    });

    it('has no other state properties', () => {
      const store = useQueryPreferencesStore();
      // The store only manages includeDetails — no modelName, numResults, etc.
      // Those live in QueryInterface.vue component data, not in the store.
      expect(store.$state).toEqual({ includeDetails: false });
    });
  });

  describe('actions', () => {
    it('toggleIncludeDetails flips the value', () => {
      const store = useQueryPreferencesStore();
      expect(store.includeDetails).toBe(false);
      store.toggleIncludeDetails();
      expect(store.includeDetails).toBe(true);
      store.toggleIncludeDetails();
      expect(store.includeDetails).toBe(false);
    });

    it('setIncludeDetails sets to true', () => {
      const store = useQueryPreferencesStore();
      store.setIncludeDetails(true);
      expect(store.includeDetails).toBe(true);
    });

    it('setIncludeDetails sets to false', () => {
      const store = useQueryPreferencesStore();
      store.setIncludeDetails(true);
      store.setIncludeDetails(false);
      expect(store.includeDetails).toBe(false);
    });

    it('setIncludeDetails coerces truthy values to boolean', () => {
      const store = useQueryPreferencesStore();
      store.setIncludeDetails(1);
      expect(store.includeDetails).toBe(true);
      store.setIncludeDetails(0);
      expect(store.includeDetails).toBe(false);
    });
  });
});
