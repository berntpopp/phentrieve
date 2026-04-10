/**
 * Pinia Store for Query Preferences (setup store / composition API)
 *
 * Migrated from options API to match the pattern used by all other stores
 * (conversation.js, disclaimer.js, log.js). State and behavior are identical.
 *
 * @module stores/queryPreferences
 */

import { ref } from 'vue';
import { defineStore } from 'pinia';

export const useQueryPreferencesStore = defineStore(
  'queryPreferences',
  () => {
    /**
     * Whether to include HPO term details (definitions and synonyms) in results.
     *
     * Implemented internally as `ref(false)` so mutations go through Vue's
     * reactivity system. Pinia setup stores automatically unwrap returned
     * refs, so consumers access it as a plain boolean: `store.includeDetails`.
     *
     * @type {boolean}
     */
    const includeDetails = ref(false);

    /**
     * Toggle the include details preference
     */
    function toggleIncludeDetails() {
      includeDetails.value = !includeDetails.value;
    }

    /**
     * Set the include details preference
     * @param {boolean} value - Whether to include details
     */
    function setIncludeDetails(value) {
      includeDetails.value = Boolean(value);
    }

    return { includeDetails, toggleIncludeDetails, setIncludeDetails };
  },
  {
    /**
     * Enable automatic persistence to localStorage
     * Plugin will automatically save/load state on changes
     */
    persist: true,
  }
);
