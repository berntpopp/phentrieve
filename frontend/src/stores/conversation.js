/**
 * Pinia Store for Conversation State Management
 *
 * Centralizes query history and collected phenotypes with automatic persistence
 * using pinia-plugin-persistedstate v4. Provides state management for the chat
 * interface with configurable history limits.
 *
 * @module stores/conversation
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

/**
 * Generate a unique ID for query items
 * @private
 * @returns {string} UUID v4 string
 */
function generateId() {
  // Use crypto.randomUUID if available, fallback to timestamp-based ID
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
}

export const useConversationStore = defineStore(
  'conversation',
  () => {
    // ===========================
    // State
    // ===========================

    /**
     * Whether the store is currently hydrating from localStorage
     * Used for showing loading skeletons during initial load
     * @type {import('vue').Ref<boolean>}
     */
    const isHydrating = ref(true);

    /**
     * Query history array (newest first)
     * @type {import('vue').Ref<Array<Object>>}
     */
    const queryHistory = ref([]);

    /**
     * Collected phenotypes from query results
     * @type {import('vue').Ref<Array<Object>>}
     */
    const collectedPhenotypes = ref([]);

    /**
     * Maximum number of queries to keep in history
     * @type {import('vue').Ref<number>}
     */
    const maxHistoryLength = ref(50);

    /**
     * Whether the collection panel is visible
     * @type {import('vue').Ref<boolean>}
     */
    const showCollectionPanel = ref(false);

    // ===========================
    // Computed Properties
    // ===========================

    /**
     * Current number of queries in history
     * @type {import('vue').ComputedRef<number>}
     */
    const conversationLength = computed(() => queryHistory.value.length);

    /**
     * Whether there is any conversation history
     * @type {import('vue').ComputedRef<boolean>}
     */
    const hasConversation = computed(() => queryHistory.value.length > 0);

    /**
     * Current number of collected phenotypes
     * @type {import('vue').ComputedRef<number>}
     */
    const phenotypeCount = computed(() => collectedPhenotypes.value.length);

    /**
     * Whether there are any collected phenotypes
     * @type {import('vue').ComputedRef<boolean>}
     */
    const hasPhenotypes = computed(() => collectedPhenotypes.value.length > 0);

    /**
     * Whether the store is ready (hydration complete)
     * @type {import('vue').ComputedRef<boolean>}
     */
    const isReady = computed(() => !isHydrating.value);

    // ===========================
    // Query History Actions
    // ===========================

    /**
     * Add a new query to history
     *
     * @param {Object} queryItem - Query item to add
     * @param {string} queryItem.query - The query text
     * @param {boolean} [queryItem.loading=true] - Loading state
     * @param {Object|null} [queryItem.response=null] - Response data
     * @param {string|null} [queryItem.error=null] - Error message
     * @param {string} [queryItem.type='query'] - Query type ('query' or 'textProcess')
     * @returns {string} The generated ID for the query item
     */
    function addQuery(queryItem) {
      const id = generateId();
      const item = {
        id,
        timestamp: new Date().toISOString(),
        loading: true,
        response: null,
        error: null,
        type: 'query',
        ...queryItem,
      };

      queryHistory.value.unshift(item);

      // Trim history if exceeds max length
      if (queryHistory.value.length > maxHistoryLength.value) {
        queryHistory.value.length = maxHistoryLength.value;
      }

      return id;
    }

    /**
     * Update a query's response by ID
     *
     * @param {string} id - Query item ID
     * @param {Object|null} responseData - Response data
     * @param {string|null} [error=null] - Error message
     */
    function updateQueryResponse(id, responseData, error = null) {
      const item = queryHistory.value.find((q) => q.id === id);
      if (item) {
        item.response = responseData;
        item.error = error;
        item.loading = false;
      }
    }

    /**
     * Get the most recent query item
     *
     * @returns {Object|null} Most recent query or null
     */
    function getLatestQuery() {
      return queryHistory.value.length > 0 ? queryHistory.value[0] : null;
    }

    /**
     * Clear all query history
     */
    function clearConversation() {
      queryHistory.value = [];
    }

    // ===========================
    // Phenotype Collection Actions
    // ===========================

    /**
     * Add a phenotype to the collection
     *
     * @param {Object} phenotype - Phenotype to add
     * @param {string} phenotype.hpo_id - HPO term ID
     * @param {string} phenotype.label - HPO term label
     * @param {string} [queryAssertionStatus] - Assertion status from query context
     * @returns {boolean} True if added, false if duplicate
     */
    function addPhenotype(phenotype, queryAssertionStatus = null) {
      const isDuplicate = collectedPhenotypes.value.some(
        (item) => item.hpo_id === phenotype.hpo_id
      );

      if (isDuplicate) {
        return false;
      }

      // Determine assertion status with priority:
      // 1. From phenotype object directly
      // 2. From queryAssertionStatus parameter
      // 3. From latest query response
      // 4. Default to 'affirmed'
      let assertionStatus = phenotype.assertion_status;

      if (!assertionStatus) {
        const latestQuery = getLatestQuery();
        const responseAssertionStatus = latestQuery?.response?.query_assertion_status || null;
        assertionStatus = queryAssertionStatus || responseAssertionStatus || 'affirmed';
      }

      collectedPhenotypes.value.push({
        ...phenotype,
        added_at: new Date().toISOString(),
        assertion_status: assertionStatus,
      });

      // Auto-show panel when first phenotype is added
      if (collectedPhenotypes.value.length === 1) {
        showCollectionPanel.value = true;
      }

      return true;
    }

    /**
     * Remove a phenotype from the collection by index
     *
     * @param {number} index - Index of phenotype to remove
     * @returns {Object|null} Removed phenotype or null
     */
    function removePhenotype(index) {
      if (index >= 0 && index < collectedPhenotypes.value.length) {
        return collectedPhenotypes.value.splice(index, 1)[0];
      }
      return null;
    }

    /**
     * Toggle assertion status of a phenotype (affirmed <-> negated)
     *
     * @param {number} index - Index of phenotype
     */
    function toggleAssertionStatus(index) {
      const phenotype = collectedPhenotypes.value[index];
      if (phenotype) {
        phenotype.assertion_status =
          phenotype.assertion_status === 'negated' ? 'affirmed' : 'negated';
      }
    }

    /**
     * Clear all collected phenotypes
     */
    function clearPhenotypes() {
      collectedPhenotypes.value = [];
    }

    /**
     * Toggle collection panel visibility
     */
    function toggleCollectionPanel() {
      showCollectionPanel.value = !showCollectionPanel.value;
    }

    // ===========================
    // Global Actions
    // ===========================

    /**
     * Reset all conversation state
     */
    function resetAll() {
      clearConversation();
      clearPhenotypes();
      showCollectionPanel.value = false;
    }

    /**
     * Set maximum history length
     *
     * @param {number} value - New max length
     */
    function setMaxHistoryLength(value) {
      maxHistoryLength.value = value;
      // Trim if current history exceeds new limit
      if (queryHistory.value.length > value) {
        queryHistory.value.length = value;
      }
    }

    /**
     * Get estimated storage size
     *
     * @returns {Object} Storage size info
     */
    function getStorageSize() {
      const size = JSON.stringify({
        queryHistory: queryHistory.value,
        collectedPhenotypes: collectedPhenotypes.value,
      }).length;

      return {
        bytes: size,
        kb: (size / 1024).toFixed(2),
        formatted: size < 1024 ? `${size} bytes` : `${(size / 1024).toFixed(1)} KB`,
      };
    }

    // ===========================
    // Return Public API
    // ===========================

    return {
      // State
      isHydrating,
      queryHistory,
      collectedPhenotypes,
      maxHistoryLength,
      showCollectionPanel,

      // Computed
      isReady,
      conversationLength,
      hasConversation,
      phenotypeCount,
      hasPhenotypes,

      // Query Actions
      addQuery,
      updateQueryResponse,
      getLatestQuery,
      clearConversation,

      // Phenotype Actions
      addPhenotype,
      removePhenotype,
      toggleAssertionStatus,
      clearPhenotypes,
      toggleCollectionPanel,

      // Global Actions
      resetAll,
      setMaxHistoryLength,
      getStorageSize,
    };
  },
  {
    // pinia-plugin-persistedstate v4 configuration
    persist: {
      key: 'phentrieve-conversation',
      storage: localStorage,
      pick: ['queryHistory', 'collectedPhenotypes', 'maxHistoryLength', 'showCollectionPanel'],
      // Signal hydration complete after restore for non-blocking UI
      afterRestore: (ctx) => {
        // Use nextTick to ensure Vue reactivity cycle completes
        // This allows the UI to render first, then show restored data
        requestAnimationFrame(() => {
          ctx.store.isHydrating = false;
        });
      },
    },
  }
);
