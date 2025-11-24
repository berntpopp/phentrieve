/**
 * Pinia Store for Disclaimer State Management
 *
 * Manages disclaimer acknowledgment state with automatic persistence
 * using pinia-plugin-persistedstate v4. Follows the same pattern as
 * the conversation store for consistency.
 *
 * @module stores/disclaimer
 */

import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

export const useDisclaimerStore = defineStore(
  'disclaimer',
  () => {
    // ===========================
    // State
    // ===========================

    /**
     * Whether the disclaimer has been acknowledged
     * @type {import('vue').Ref<boolean>}
     */
    const isAcknowledged = ref(false);

    /**
     * Timestamp when the disclaimer was acknowledged
     * @type {import('vue').Ref<string|null>}
     */
    const acknowledgmentTimestamp = ref(null);

    // ===========================
    // Computed Properties
    // ===========================

    /**
     * Formatted acknowledgment date for display
     * @type {import('vue').ComputedRef<string>}
     */
    const formattedAcknowledgmentDate = computed(() => {
      if (!acknowledgmentTimestamp.value) return '';

      const date = new Date(acknowledgmentTimestamp.value);
      return date.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    });

    // ===========================
    // Actions
    // ===========================

    /**
     * Save disclaimer acknowledgment
     * Sets the acknowledged state and timestamp
     */
    function saveAcknowledgment() {
      isAcknowledged.value = true;
      acknowledgmentTimestamp.value = new Date().toISOString();
      // No manual localStorage - plugin handles persistence automatically
    }

    /**
     * Reset disclaimer acknowledgment
     * Clears the acknowledged state (for testing/debugging)
     */
    function reset() {
      isAcknowledged.value = false;
      acknowledgmentTimestamp.value = null;
      // No manual localStorage - plugin handles persistence automatically
    }

    // ===========================
    // Return Public API
    // ===========================

    return {
      // State
      isAcknowledged,
      acknowledgmentTimestamp,

      // Computed
      formattedAcknowledgmentDate,

      // Actions
      saveAcknowledgment,
      reset,
    };
  },
  {
    // pinia-plugin-persistedstate v4 configuration
    persist: {
      key: 'phentrieve-disclaimer',
      storage: localStorage,
      pick: ['isAcknowledged', 'acknowledgmentTimestamp'],
    },
  }
);
