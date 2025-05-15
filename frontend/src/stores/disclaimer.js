/**
 * Disclaimer store
 * Manages disclaimer state using Pinia and persists to localStorage
 */

import { defineStore } from 'pinia'
import { logService } from '../services/logService'

// Storage keys
const DISCLAIMER_KEY = 'phentrieveDisclaimerAcknowledged'
const DISCLAIMER_TIMESTAMP_KEY = 'phentrieveDisclaimerTimestamp'

export const useDisclaimerStore = defineStore('disclaimer', {
  state: () => ({
    isAcknowledged: false,
    acknowledgmentTimestamp: null
  }),

  getters: {
    // Get the formatted acknowledgment date
    formattedAcknowledgmentDate: (state) => {
      if (!state.acknowledgmentTimestamp) {
        return ''
      }

      try {
        const date = new Date(state.acknowledgmentTimestamp)
        return date.toLocaleDateString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric'
        })
      } catch (error) {
        logService.error('Error formatting acknowledgment date', error)
        return ''
      }
    }
  },

  actions: {
    // Initialize the store by loading from localStorage
    initialize() {
      try {
        const savedAcknowledgment = localStorage.getItem(DISCLAIMER_KEY)
        const savedTimestamp = localStorage.getItem(DISCLAIMER_TIMESTAMP_KEY)
        
        this.isAcknowledged = savedAcknowledgment === 'true'
        this.acknowledgmentTimestamp = savedTimestamp ? parseInt(savedTimestamp) : null
      } catch (error) {
        logService.error('Error loading disclaimer status', error)
        // Default to not acknowledged if there's an error
        this.isAcknowledged = false
        this.acknowledgmentTimestamp = null
      }
    },

    // Save acknowledgment to localStorage
    saveAcknowledgment() {
      try {
        const now = Date.now()
        localStorage.setItem(DISCLAIMER_KEY, 'true')
        localStorage.setItem(DISCLAIMER_TIMESTAMP_KEY, now.toString())
        
        this.isAcknowledged = true
        this.acknowledgmentTimestamp = now
      } catch (error) {
        logService.error('Error saving disclaimer acknowledgment', error)
      }
    }
  }
})
