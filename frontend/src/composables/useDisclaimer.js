/**
 * useDisclaimer.js
 * 
 * Vue composable for managing the disclaimer's acknowledgment state
 * Uses localStorage to store and retrieve the acknowledgment status
 */

import { ref, readonly } from 'vue'

// Storage keys
const DISCLAIMER_KEY = 'phentrieveDisclaimerAcknowledged'
const DISCLAIMER_TIMESTAMP_KEY = 'phentrieveDisclaimerTimestamp'

export function useDisclaimer() {
  // Reactive state
  const isAcknowledged = ref(false)
  const acknowledgmentTimestamp = ref(null)
  
  /**
   * Load the disclaimer status from localStorage
   */
  const loadDisclaimerStatus = () => {
    try {
      const savedAcknowledgment = localStorage.getItem(DISCLAIMER_KEY)
      const savedTimestamp = localStorage.getItem(DISCLAIMER_TIMESTAMP_KEY)
      
      isAcknowledged.value = savedAcknowledgment === 'true'
      acknowledgmentTimestamp.value = savedTimestamp ? parseInt(savedTimestamp) : null
    } catch (error) {
      console.error('Error loading disclaimer status:', error)
      // Default to not acknowledged if there's an error
      isAcknowledged.value = false
      acknowledgmentTimestamp.value = null
    }
  }

  /**
   * Check if the disclaimer has been acknowledged
   * @returns {boolean} Whether the disclaimer is acknowledged
   */
  const checkDisclaimerStatus = () => {
    // Load current status from localStorage each time to ensure it's up to date
    loadDisclaimerStatus()
    return isAcknowledged.value
  }

  /**
   * Save acknowledgment status and timestamp to localStorage
   */
  const saveDisclaimerAcknowledgment = () => {
    try {
      const now = Date.now()
      localStorage.setItem(DISCLAIMER_KEY, 'true')
      localStorage.setItem(DISCLAIMER_TIMESTAMP_KEY, now.toString())
      
      isAcknowledged.value = true
      acknowledgmentTimestamp.value = now
    } catch (error) {
      console.error('Error saving disclaimer acknowledgment:', error)
    }
  }

  /**
   * Get a user-friendly formatted date of acknowledgment
   * @returns {string} Formatted date or empty string if not acknowledged
   */
  const getFormattedAcknowledgmentDate = () => {
    if (!acknowledgmentTimestamp.value) {
      loadDisclaimerStatus()
    }
    
    if (!acknowledgmentTimestamp.value) {
      return ''
    }
    
    try {
      const date = new Date(acknowledgmentTimestamp.value)
      return date.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      })
    } catch (error) {
      console.error('Error formatting acknowledgment date:', error)
      return ''
    }
  }
  
  // Initialize state on creation
  loadDisclaimerStatus()

  // Return public interface
  return {
    isAcknowledged: readonly(isAcknowledged),
    checkDisclaimerStatus,
    saveDisclaimerAcknowledgment,
    getFormattedAcknowledgmentDate
  }
}
