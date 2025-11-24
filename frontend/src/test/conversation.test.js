/**
 * Test Suite for Conversation Store
 *
 * Tests query history management, phenotype collection, persistence,
 * and all core functionality of the conversation state management.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useConversationStore } from '../stores/conversation';

describe('Conversation Store', () => {
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
    it('should initialize with empty state', () => {
      const store = useConversationStore();
      expect(store.queryHistory).toEqual([]);
      expect(store.collectedPhenotypes).toEqual([]);
      expect(store.showCollectionPanel).toBe(false);
    });

    it('should initialize with default maxHistoryLength', () => {
      const store = useConversationStore();
      expect(store.maxHistoryLength).toBe(50);
    });

    it('should have correct computed properties initially', () => {
      const store = useConversationStore();
      expect(store.conversationLength).toBe(0);
      expect(store.hasConversation).toBe(false);
      expect(store.phenotypeCount).toBe(0);
      expect(store.hasPhenotypes).toBe(false);
    });
  });

  describe('Query History Operations', () => {
    it('should add query with unique ID', () => {
      const store = useConversationStore();
      const id = store.addQuery({ query: 'test query' });

      expect(store.queryHistory).toHaveLength(1);
      expect(store.queryHistory[0].id).toBe(id);
      expect(store.queryHistory[0].query).toBe('test query');
    });

    it('should add timestamp automatically', () => {
      const store = useConversationStore();
      store.addQuery({ query: 'test' });

      expect(store.queryHistory[0].timestamp).toBeDefined();
      expect(new Date(store.queryHistory[0].timestamp)).toBeInstanceOf(Date);
    });

    it('should add queries in newest-first order (unshift)', () => {
      const store = useConversationStore();
      store.addQuery({ query: 'first' });
      store.addQuery({ query: 'second' });
      store.addQuery({ query: 'third' });

      expect(store.queryHistory[0].query).toBe('third');
      expect(store.queryHistory[1].query).toBe('second');
      expect(store.queryHistory[2].query).toBe('first');
    });

    it('should set default values for query items', () => {
      const store = useConversationStore();
      store.addQuery({ query: 'test' });

      const item = store.queryHistory[0];
      expect(item.loading).toBe(true);
      expect(item.response).toBe(null);
      expect(item.error).toBe(null);
      expect(item.type).toBe('query');
    });

    it('should preserve custom values in query items', () => {
      const store = useConversationStore();
      store.addQuery({
        query: 'test',
        loading: false,
        type: 'textProcess',
      });

      const item = store.queryHistory[0];
      expect(item.loading).toBe(false);
      expect(item.type).toBe('textProcess');
    });

    it('should trim history when exceeds maxHistoryLength', () => {
      const store = useConversationStore();
      store.setMaxHistoryLength(5);

      for (let i = 0; i < 10; i++) {
        store.addQuery({ query: `query ${i}` });
      }

      expect(store.queryHistory).toHaveLength(5);
      // Should keep newest (highest numbers)
      expect(store.queryHistory[0].query).toBe('query 9');
      expect(store.queryHistory[4].query).toBe('query 5');
    });

    it('should update query response by ID', () => {
      const store = useConversationStore();
      const id = store.addQuery({ query: 'test' });

      const responseData = { results: [{ hpo_id: 'HP:0001234' }] };
      store.updateQueryResponse(id, responseData);

      expect(store.queryHistory[0].loading).toBe(false);
      expect(store.queryHistory[0].response).toEqual(responseData);
      expect(store.queryHistory[0].error).toBe(null);
    });

    it('should update query response with error', () => {
      const store = useConversationStore();
      const id = store.addQuery({ query: 'test' });

      store.updateQueryResponse(id, null, 'API error');

      expect(store.queryHistory[0].loading).toBe(false);
      expect(store.queryHistory[0].response).toBe(null);
      expect(store.queryHistory[0].error).toBe('API error');
    });

    it('should not update non-existent query', () => {
      const store = useConversationStore();
      store.addQuery({ query: 'test' });

      // Try to update with wrong ID
      store.updateQueryResponse('non-existent-id', { data: 'test' });

      // Original should remain unchanged
      expect(store.queryHistory[0].loading).toBe(true);
      expect(store.queryHistory[0].response).toBe(null);
    });

    it('should get latest query', () => {
      const store = useConversationStore();
      expect(store.getLatestQuery()).toBe(null);

      store.addQuery({ query: 'first' });
      store.addQuery({ query: 'second' });

      expect(store.getLatestQuery().query).toBe('second');
    });

    it('should clear conversation', () => {
      const store = useConversationStore();
      store.addQuery({ query: 'test 1' });
      store.addQuery({ query: 'test 2' });

      store.clearConversation();

      expect(store.queryHistory).toEqual([]);
      expect(store.conversationLength).toBe(0);
    });
  });

  describe('Phenotype Collection Operations', () => {
    it('should add phenotype to collection', () => {
      const store = useConversationStore();
      const phenotype = { hpo_id: 'HP:0001234', label: 'Test phenotype' };

      const added = store.addPhenotype(phenotype);

      expect(added).toBe(true);
      expect(store.collectedPhenotypes).toHaveLength(1);
      expect(store.collectedPhenotypes[0].hpo_id).toBe('HP:0001234');
    });

    it('should add timestamp when adding phenotype', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });

      expect(store.collectedPhenotypes[0].added_at).toBeDefined();
    });

    it('should prevent duplicate phenotypes', () => {
      const store = useConversationStore();
      const phenotype = { hpo_id: 'HP:0001234', label: 'Test' };

      const first = store.addPhenotype(phenotype);
      const second = store.addPhenotype(phenotype);

      expect(first).toBe(true);
      expect(second).toBe(false);
      expect(store.collectedPhenotypes).toHaveLength(1);
    });

    it('should use assertion status from phenotype object', () => {
      const store = useConversationStore();
      store.addPhenotype({
        hpo_id: 'HP:0001234',
        label: 'Test',
        assertion_status: 'negated',
      });

      expect(store.collectedPhenotypes[0].assertion_status).toBe('negated');
    });

    it('should use assertion status from parameter', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' }, 'negated');

      expect(store.collectedPhenotypes[0].assertion_status).toBe('negated');
    });

    it('should default assertion status to affirmed', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });

      expect(store.collectedPhenotypes[0].assertion_status).toBe('affirmed');
    });

    it('should auto-show collection panel on first phenotype', () => {
      const store = useConversationStore();
      expect(store.showCollectionPanel).toBe(false);

      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });

      expect(store.showCollectionPanel).toBe(true);
    });

    it('should remove phenotype by index', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001111', label: 'First' });
      store.addPhenotype({ hpo_id: 'HP:0002222', label: 'Second' });
      store.addPhenotype({ hpo_id: 'HP:0003333', label: 'Third' });

      const removed = store.removePhenotype(1);

      expect(removed.hpo_id).toBe('HP:0002222');
      expect(store.collectedPhenotypes).toHaveLength(2);
      expect(store.collectedPhenotypes[0].hpo_id).toBe('HP:0001111');
      expect(store.collectedPhenotypes[1].hpo_id).toBe('HP:0003333');
    });

    it('should return null when removing invalid index', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });

      expect(store.removePhenotype(-1)).toBe(null);
      expect(store.removePhenotype(99)).toBe(null);
    });

    it('should toggle assertion status', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });

      expect(store.collectedPhenotypes[0].assertion_status).toBe('affirmed');

      store.toggleAssertionStatus(0);
      expect(store.collectedPhenotypes[0].assertion_status).toBe('negated');

      store.toggleAssertionStatus(0);
      expect(store.collectedPhenotypes[0].assertion_status).toBe('affirmed');
    });

    it('should clear all phenotypes', () => {
      const store = useConversationStore();
      store.addPhenotype({ hpo_id: 'HP:0001111', label: 'First' });
      store.addPhenotype({ hpo_id: 'HP:0002222', label: 'Second' });

      store.clearPhenotypes();

      expect(store.collectedPhenotypes).toEqual([]);
      expect(store.phenotypeCount).toBe(0);
    });

    it('should toggle collection panel visibility', () => {
      const store = useConversationStore();
      expect(store.showCollectionPanel).toBe(false);

      store.toggleCollectionPanel();
      expect(store.showCollectionPanel).toBe(true);

      store.toggleCollectionPanel();
      expect(store.showCollectionPanel).toBe(false);
    });
  });

  describe('Global Actions', () => {
    it('should reset all state', () => {
      const store = useConversationStore();
      store.addQuery({ query: 'test' });
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });
      store.showCollectionPanel = true;

      store.resetAll();

      expect(store.queryHistory).toEqual([]);
      expect(store.collectedPhenotypes).toEqual([]);
      expect(store.showCollectionPanel).toBe(false);
    });

    it('should set max history length', () => {
      const store = useConversationStore();
      store.setMaxHistoryLength(100);
      expect(store.maxHistoryLength).toBe(100);
    });

    it('should trim existing history when reducing max length', () => {
      const store = useConversationStore();
      for (let i = 0; i < 10; i++) {
        store.addQuery({ query: `query ${i}` });
      }
      expect(store.queryHistory).toHaveLength(10);

      store.setMaxHistoryLength(5);

      expect(store.queryHistory).toHaveLength(5);
    });
  });

  describe('Computed Properties', () => {
    it('should compute conversationLength correctly', () => {
      const store = useConversationStore();
      expect(store.conversationLength).toBe(0);

      store.addQuery({ query: 'test 1' });
      store.addQuery({ query: 'test 2' });

      expect(store.conversationLength).toBe(2);
    });

    it('should compute hasConversation correctly', () => {
      const store = useConversationStore();
      expect(store.hasConversation).toBe(false);

      store.addQuery({ query: 'test' });
      expect(store.hasConversation).toBe(true);

      store.clearConversation();
      expect(store.hasConversation).toBe(false);
    });

    it('should compute phenotypeCount correctly', () => {
      const store = useConversationStore();
      expect(store.phenotypeCount).toBe(0);

      store.addPhenotype({ hpo_id: 'HP:0001111', label: 'First' });
      store.addPhenotype({ hpo_id: 'HP:0002222', label: 'Second' });

      expect(store.phenotypeCount).toBe(2);
    });

    it('should compute hasPhenotypes correctly', () => {
      const store = useConversationStore();
      expect(store.hasPhenotypes).toBe(false);

      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });
      expect(store.hasPhenotypes).toBe(true);

      store.clearPhenotypes();
      expect(store.hasPhenotypes).toBe(false);
    });
  });

  describe('Storage Size', () => {
    it('should calculate storage size', () => {
      const store = useConversationStore();

      store.addQuery({ query: 'test query with some text' });
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test phenotype' });

      const size = store.getStorageSize();

      expect(size.bytes).toBeGreaterThan(0);
      expect(size.kb).toBeDefined();
      expect(size.formatted).toBeDefined();
      expect(parseFloat(size.kb)).toBeGreaterThan(0);
    });

    it('should format storage size correctly', () => {
      const store = useConversationStore();

      // Small data should show bytes
      const smallSize = store.getStorageSize();
      expect(smallSize.formatted).toContain('bytes');

      // Add enough data to exceed 1KB
      for (let i = 0; i < 50; i++) {
        store.addQuery({ query: `This is a longer query text for testing ${i}` });
      }

      const largeSize = store.getStorageSize();
      expect(largeSize.formatted).toContain('KB');
    });
  });

  describe('Assertion Status from Query Context', () => {
    it('should use assertion status from latest query response', () => {
      const store = useConversationStore();

      // Add a query with response containing assertion status
      const id = store.addQuery({ query: 'test' });
      store.updateQueryResponse(id, { query_assertion_status: 'negated' });

      // Add phenotype without explicit assertion status
      store.addPhenotype({ hpo_id: 'HP:0001234', label: 'Test' });

      expect(store.collectedPhenotypes[0].assertion_status).toBe('negated');
    });

    it('should prioritize phenotype assertion over query context', () => {
      const store = useConversationStore();

      // Add query with negated response
      const id = store.addQuery({ query: 'test' });
      store.updateQueryResponse(id, { query_assertion_status: 'negated' });

      // Add phenotype with explicit affirmed status
      store.addPhenotype({
        hpo_id: 'HP:0001234',
        label: 'Test',
        assertion_status: 'affirmed',
      });

      expect(store.collectedPhenotypes[0].assertion_status).toBe('affirmed');
    });
  });
});
