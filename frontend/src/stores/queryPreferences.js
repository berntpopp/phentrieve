/**
 * Pinia Store for Query Preferences
 *
 * Manages user preferences for HPO query options with automatic persistence
 * using pinia-plugin-persistedstate. Preferences are stored in localStorage
 * and automatically hydrated on app load.
 *
 * @module stores/queryPreferences
 */

import { defineStore } from 'pinia';

export const useQueryPreferencesStore = defineStore('queryPreferences', {
  state: () => ({
    /**
     * Whether to include HPO term details (definitions and synonyms) in results
     * @type {boolean}
     * @default false
     */
    includeDetails: false,
  }),

  actions: {
    /**
     * Toggle the include details preference
     */
    toggleIncludeDetails() {
      this.includeDetails = !this.includeDetails;
    },

    /**
     * Set the include details preference
     * @param {boolean} value - Whether to include details
     */
    setIncludeDetails(value) {
      this.includeDetails = Boolean(value);
    },
  },

  /**
   * Enable automatic persistence to localStorage
   * Plugin will automatically save/load state on changes
   */
  persist: true,
});
