import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

const STORAGE_KEY = 'phentrieve_disclaimer_acknowledged';

export const useDisclaimerStore = defineStore('disclaimer', () => {
  // State
  const isAcknowledged = ref(false);
  const acknowledgmentTimestamp = ref(null);

  // Computed
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

  // Actions
  function initialize() {
    // Load disclaimer acknowledgment from localStorage
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored);
        isAcknowledged.value = data.acknowledged || false;
        acknowledgmentTimestamp.value = data.timestamp || null;
      }
    } catch (error) {
      console.error('Error loading disclaimer acknowledgment:', error);
      isAcknowledged.value = false;
      acknowledgmentTimestamp.value = null;
    }
  }

  function saveAcknowledgment() {
    // Save disclaimer acknowledgment to localStorage
    const timestamp = new Date().toISOString();
    isAcknowledged.value = true;
    acknowledgmentTimestamp.value = timestamp;

    try {
      const data = {
        acknowledged: true,
        timestamp: timestamp,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      console.error('Error saving disclaimer acknowledgment:', error);
    }
  }

  function reset() {
    // Reset disclaimer acknowledgment (for testing/debugging)
    isAcknowledged.value = false;
    acknowledgmentTimestamp.value = null;
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Error resetting disclaimer acknowledgment:', error);
    }
  }

  return {
    // State
    isAcknowledged,
    acknowledgmentTimestamp,

    // Computed
    formattedAcknowledgmentDate,

    // Actions
    initialize,
    saveAcknowledgment,
    reset,
  };
});
